"""Climatology and seasonal analysis utilities for time series."""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from timesmith.typing import SeriesLike

logger = logging.getLogger(__name__)


def compute_climatology(
    y: SeriesLike, reference_period: Optional[Tuple[str, str]] = None
) -> Dict[str, Any]:
    """Compute climatological statistics for time series.

    Args:
        y: Time series with datetime index.
        reference_period: Optional tuple of (start_date, end_date) for reference period.

    Returns:
        Dictionary with climatology statistics.
    """
    if isinstance(y, pd.Series):
        series = y
    elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        series = y.iloc[:, 0]
    else:
        raise ValueError("y must be Series or single-column DataFrame with datetime index")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Data must have datetime index for climatology analysis")

    # Filter to reference period if specified
    if reference_period:
        start_date, end_date = reference_period
        series = series[start_date:end_date]

    series_clean = series.dropna()

    if len(series_clean) == 0:
        raise ValueError("No valid data for climatology computation")

    # Long-term statistics
    long_term_mean = float(series_clean.mean())
    long_term_std = float(series_clean.std())

    # Monthly climatology
    monthly_climatology = series_clean.groupby(series_clean.index.month).mean()

    # Annual cycle characteristics
    annual_cycle_amplitude = float(monthly_climatology.max() - monthly_climatology.min())
    annual_cycle_phase = int(monthly_climatology.idxmax())

    # Seasonal statistics
    season_definitions = {
        "DJF": [12, 1, 2],  # Winter
        "MAM": [3, 4, 5],  # Spring
        "JJA": [6, 7, 8],  # Summer
        "SON": [9, 10, 11],  # Fall
    }

    seasonal_stats = {}
    for season, months in season_definitions.items():
        seasonal_data = series_clean[series_clean.index.month.isin(months)]

        if len(seasonal_data) > 0:
            seasonal_stats[season] = {
                "mean": float(seasonal_data.mean()),
                "std": float(seasonal_data.std()),
                "min": float(seasonal_data.min()),
                "max": float(seasonal_data.max()),
                "median": float(seasonal_data.median()),
                "q25": float(seasonal_data.quantile(0.25)),
                "q75": float(seasonal_data.quantile(0.75)),
                "n_samples": len(seasonal_data),
                "cv": (
                    float(seasonal_data.std() / seasonal_data.mean())
                    if seasonal_data.mean() != 0
                    else np.inf
                ),
            }

    # Interannual variability
    annual_means = series_clean.groupby(series_clean.index.year).mean()
    interannual_variability = (
        float(annual_means.std()) if len(annual_means) > 1 else 0.0
    )

    return {
        "long_term_mean": long_term_mean,
        "long_term_std": long_term_std,
        "seasonal_stats": seasonal_stats,
        "monthly_climatology": monthly_climatology.to_dict(),
        "annual_cycle_amplitude": annual_cycle_amplitude,
        "annual_cycle_phase": annual_cycle_phase,
        "interannual_variability": interannual_variability,
    }


def compute_anomalies(
    y: SeriesLike,
    climatology: Optional[Dict[str, Any]] = None,
    anomaly_type: str = "absolute",
) -> pd.Series:
    """Compute climatological anomalies.

    Args:
        y: Time series with datetime index.
        climatology: Pre-computed climatology (computed if None).
        anomaly_type: 'absolute', 'relative', or 'standardized'.

    Returns:
        Time series of anomalies.
    """
    if isinstance(y, pd.Series):
        series = y
    elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        series = y.iloc[:, 0]
    else:
        raise ValueError("y must be Series or single-column DataFrame with datetime index")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Data must have datetime index")

    if climatology is None:
        climatology = compute_climatology(series)

    # Create monthly climatology series aligned with data
    monthly_clim_dict = climatology["monthly_climatology"]
    monthly_clim = series.index.month.map(monthly_clim_dict)

    if anomaly_type == "absolute":
        anomalies = series - monthly_clim
    elif anomaly_type == "relative":
        anomalies = (series - monthly_clim) / monthly_clim * 100
    elif anomaly_type == "standardized":
        # Use long-term standard deviation for standardization
        anomalies = (series - monthly_clim) / climatology["long_term_std"]
    else:
        raise ValueError(
            "anomaly_type must be 'absolute', 'relative', or 'standardized'"
        )

    return anomalies


def detect_extreme_events(
    y: SeriesLike,
    threshold_type: str = "percentile",
    threshold_value: float = 5.0,
) -> pd.DataFrame:
    """Detect extreme events in time series.

    Args:
        y: Time series with datetime index.
        threshold_type: 'percentile', 'std_dev', or 'absolute'.
        threshold_value: Threshold value (percentile, std devs, or absolute).

    Returns:
        DataFrame with extreme events.
    """
    if isinstance(y, pd.Series):
        series = y
    elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        series = y.iloc[:, 0]
    else:
        raise ValueError("y must be Series or single-column DataFrame")

    series_clean = series.dropna()

    if len(series_clean) == 0:
        return pd.DataFrame()

    # Determine thresholds
    if threshold_type == "percentile":
        dry_threshold = series_clean.quantile(threshold_value / 100)
        wet_threshold = series_clean.quantile(1 - threshold_value / 100)
    elif threshold_type == "std_dev":
        mean_val = series_clean.mean()
        std_val = series_clean.std()
        dry_threshold = mean_val - threshold_value * std_val
        wet_threshold = mean_val + threshold_value * std_val
    elif threshold_type == "absolute":
        dry_threshold = threshold_value
        wet_threshold = (
            series_clean.max() - threshold_value
        )  # Arbitrary for wet events
    else:
        raise ValueError(
            "threshold_type must be 'percentile', 'std_dev', or 'absolute'"
        )

    # Find extreme events
    extreme_events = []

    for date, value in series_clean.items():
        if value <= dry_threshold:
            extreme_events.append(
                {
                    "date": date,
                    "value": float(value),
                    "type": "dry",
                    "severity": float((dry_threshold - value) / series_clean.std()),
                    "percentile": float(stats.percentileofscore(series_clean.values, value)),
                }
            )
        elif value >= wet_threshold:
            extreme_events.append(
                {
                    "date": date,
                    "value": float(value),
                    "type": "wet",
                    "severity": float((value - wet_threshold) / series_clean.std()),
                    "percentile": float(stats.percentileofscore(series_clean.values, value)),
                }
            )

    return pd.DataFrame(extreme_events)

