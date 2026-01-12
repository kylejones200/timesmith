"""Autocorrelation and partial autocorrelation functions for time series."""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from timesmith.typing import SeriesLike

logger = logging.getLogger(__name__)

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


def autocorrelation(
    data: SeriesLike, max_lag: Optional[int] = None
) -> List[float]:
    """Calculate autocorrelation function (ACF).

    Computes the autocorrelation coefficients for different lags, measuring
    the correlation between a time series and a lagged version of itself.

    Args:
        data: Time series data (Series, DataFrame, or array-like).
        max_lag: Maximum lag to calculate (default: len(data) - 1).

    Returns:
        List of autocorrelation coefficients for each lag (lag 0 to max_lag).

    Raises:
        ValueError: If data is too short or max_lag is invalid.

    Example:
        >>> data = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        >>> acf = autocorrelation(data, max_lag=3)
        >>> len(acf)
        4
        >>> acf[0]  # Lag 0 is always 1.0
        1.0
    """
    # Convert to numpy array
    if isinstance(data, pd.Series):
        data_array = data.values
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            data_array = data.iloc[:, 0].values
        else:
            raise ValueError("DataFrame must have exactly one column")
    else:
        data_array = np.asarray(data, dtype=float)

    if len(data_array) < 2:
        raise ValueError("Data must contain at least 2 values")

    if max_lag is None:
        max_lag = len(data_array) - 1
    elif max_lag < 0 or max_lag >= len(data_array):
        raise ValueError(
            f"max_lag must be between 0 and {len(data_array) - 1}, got {max_lag}"
        )

    mean = np.mean(data_array)
    var = np.var(data_array, ddof=0)  # Population variance

    if var == 0:
        # Constant data: ACF is 1 at lag 0, 0 elsewhere
        return [1.0] + [0.0] * max_lag

    # Use optimized autocorrelation if available
    if HAS_NUMBA and len(data_array) > 100:
        acf_array = _autocorrelation_numba(data_array, mean, var, max_lag)
        acf = acf_array.tolist()
    else:
        acf = []
        for lag in range(max_lag + 1):
            if lag == 0:
                acf.append(1.0)
            else:
                numerator = np.sum(
                    (data_array[:-lag] - mean) * (data_array[lag:] - mean)
                )
                denominator = len(data_array) * var
                acf.append(float(numerator / denominator))

    return acf


@njit(cache=True, fastmath=True)
def _autocorrelation_numba(
    data_array: np.ndarray, mean: float, var: float, max_lag: int
) -> np.ndarray:
    """Numba-optimized autocorrelation computation.
    
    Args:
        data_array: Input data array.
        mean: Mean of the data.
        var: Variance of the data.
        max_lag: Maximum lag.
    
    Returns:
        Array of autocorrelation coefficients.
    """
    n = len(data_array)
    acf = np.zeros(max_lag + 1, dtype=np.float64)
    acf[0] = 1.0  # Lag 0 is always 1.0
    
    for lag in range(1, max_lag + 1):
        numerator = 0.0
        for i in range(lag, n):
            numerator += (data_array[i] - mean) * (data_array[i - lag] - mean)
        denominator = var * n
        acf[lag] = numerator / denominator if denominator != 0 else 0.0
    
    return acf


def partial_autocorrelation(
    data: SeriesLike, max_lag: Optional[int] = None
) -> List[float]:
    """Calculate partial autocorrelation function (PACF).

    Computes the partial autocorrelation coefficients, which measure the
    correlation between observations at different lags while controlling
    for intermediate lags.

    Uses the Durbin-Levinson algorithm for computation.

    Args:
        data: Time series data (Series, DataFrame, or array-like).
        max_lag: Maximum lag to calculate (default: min(len(data)//2, 10)).

    Returns:
        List of partial autocorrelation coefficients (lag 0 to max_lag).

    Raises:
        ValueError: If data is too short.

    Example:
        >>> data = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        >>> pacf = partial_autocorrelation(data, max_lag=3)
        >>> len(pacf)
        4
        >>> pacf[0]  # Lag 0 is always 1.0
        1.0
    """
    # Convert to numpy array
    if isinstance(data, pd.Series):
        data_array = data.values
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            data_array = data.iloc[:, 0].values
        else:
            raise ValueError("DataFrame must have exactly one column")
    else:
        data_array = np.asarray(data, dtype=float)

    if len(data_array) < 2:
        raise ValueError("Data must contain at least 2 values")

    if max_lag is None:
        max_lag = min(len(data_array) // 2, 10)
    elif max_lag < 0 or max_lag >= len(data_array):
        raise ValueError(
            f"max_lag must be between 0 and {len(data_array) - 1}, got {max_lag}"
        )

    # Get ACF values
    acf_values = autocorrelation(data_array, max_lag=max_lag)
    pacf = [1.0]  # PACF at lag 0 is always 1

    # Durbin-Levinson algorithm
    for k in range(1, max_lag + 1):
        if k == 1:
            pacf.append(acf_values[1])
        else:
            # Calculate numerator
            numerator = acf_values[k]
            for j in range(1, k):
                numerator -= pacf[j] * acf_values[k - j]

            # Calculate denominator
            denominator = 1.0
            for j in range(1, k):
                denominator -= pacf[j] * acf_values[j]

            if abs(denominator) < 1e-10:
                pacf.append(0.0)
            else:
                pacf.append(float(numerator / denominator))

    return pacf


def autocorrelation_plot_data(
    data: SeriesLike, max_lag: Optional[int] = None
) -> dict:
    """Calculate ACF and PACF for plotting.

    Convenience function that returns both ACF and PACF along with
    lag indices for easy plotting.

    Args:
        data: Time series data.
        max_lag: Maximum lag to calculate.

    Returns:
        Dictionary with keys:
        - 'lags': Array of lag values
        - 'acf': Autocorrelation values
        - 'pacf': Partial autocorrelation values
    """
    if max_lag is None:
        if isinstance(data, (pd.Series, pd.DataFrame)):
            max_lag = min(len(data) // 2, 40)
        else:
            max_lag = min(len(data) // 2, 40)

    acf = autocorrelation(data, max_lag=max_lag)
    pacf = partial_autocorrelation(data, max_lag=max_lag)

    return {
        "lags": np.arange(len(acf)),
        "acf": np.array(acf),
        "pacf": np.array(pacf),
    }

