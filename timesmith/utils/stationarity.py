"""Stationarity tests for time series."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from timesmith.typing import SeriesLike

logger = logging.getLogger(__name__)

# Optional statsmodels for advanced tests
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning(
        "statsmodels not available. Advanced stationarity tests will use basic methods. "
        "Install with: pip install statsmodels"
    )


def test_stationarity(
    y: SeriesLike, significance_level: float = 0.05
) -> Dict[str, Any]:
    """Test time series for stationarity using ADF and KPSS tests.

    Args:
        y: Time series data.
        significance_level: Significance level for tests.

    Returns:
        Dictionary with stationarity test results.
    """
    if isinstance(y, pd.Series):
        series = y
    elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        series = y.iloc[:, 0]
    else:
        series = pd.Series(y)

    series_clean = series.dropna()

    if not STATSMODELS_AVAILABLE:
        logger.warning("Statsmodels not available. Using basic stationarity test.")
        return _basic_stationarity_test(series_clean, significance_level)

    results = {}

    # Augmented Dickey-Fuller test
    try:
        adf_result = adfuller(series_clean, autolag="AIC")
        results["adf"] = {
            "statistic": float(adf_result[0]),
            "p_value": float(adf_result[1]),
            "critical_values": {k: float(v) for k, v in adf_result[4].items()},
            "is_stationary": adf_result[1] < significance_level,
        }
    except Exception as e:
        logger.warning(f"ADF test failed: {e}")
        results["adf"] = {"error": str(e)}

    # KPSS test
    try:
        kpss_result = kpss(series_clean, regression="c")
        results["kpss"] = {
            "statistic": float(kpss_result[0]),
            "p_value": float(kpss_result[1]),
            "critical_values": {k: float(v) for k, v in kpss_result[3].items()},
            "is_stationary": kpss_result[1] > significance_level,
        }
    except Exception as e:
        logger.warning(f"KPSS test failed: {e}")
        results["kpss"] = {"error": str(e)}

    # Overall conclusion
    adf_stationary = results.get("adf", {}).get("is_stationary", False)
    kpss_stationary = results.get("kpss", {}).get("is_stationary", False)

    if adf_stationary and kpss_stationary:
        results["conclusion"] = "stationary"
    elif not adf_stationary and not kpss_stationary:
        results["conclusion"] = "non_stationary"
    else:
        results["conclusion"] = "inconclusive"

    return results


def _basic_stationarity_test(
    series: pd.Series, significance_level: float
) -> Dict[str, Any]:
    """Basic stationarity assessment without statsmodels."""
    from scipy import stats

    series_clean = series.dropna()

    if len(series_clean) < 10:
        return {
            "basic_test": {
                "error": "Insufficient data for stationarity test",
            },
            "conclusion": "insufficient_data",
        }

    # Split series into two halves and compare means/variances
    mid_point = len(series_clean) // 2
    first_half = series_clean.iloc[:mid_point]
    second_half = series_clean.iloc[mid_point:]

    # T-test for mean difference
    t_stat, p_value = stats.ttest_ind(first_half, second_half)

    # F-test for variance difference
    var_first = first_half.var()
    var_second = second_half.var()
    f_stat = var_first / var_second if var_second > 0 else np.inf

    likely_stationary = p_value > significance_level and 0.5 < f_stat < 2.0

    return {
        "basic_test": {
            "mean_difference_p_value": float(p_value),
            "variance_ratio": float(f_stat),
            "likely_stationary": likely_stationary,
        },
        "conclusion": (
            "likely_stationary" if likely_stationary else "likely_non_stationary"
        ),
    }

