"""Sequence creation utilities for time series models (LSTM, RNN, etc.)."""

import logging
from typing import Tuple

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


def create_sequences(
    data: SeriesLike, lookback: int = 5, forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM/RNN models using sliding window.

    Creates input sequences (X) and target sequences (y) from time series data
    using a sliding window approach.

    Args:
        data: Time series data (Series or array-like).
        lookback: Number of time steps to look back (sequence length).
        forecast_horizon: Number of steps ahead to forecast (default: 1).

    Returns:
        Tuple of (X, y) where:
        - X: Array of shape (n_samples, lookback, n_features)
        - y: Array of shape (n_samples, forecast_horizon)

    Example:
        >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> X, y = create_sequences(data, lookback=3, forecast_horizon=1)
        >>> # X[0] = [1, 2, 3], y[0] = [4]
        >>> # X[1] = [2, 3, 4], y[1] = [5]
    """
    if isinstance(data, pd.Series):
        values = data.values
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            values = data.iloc[:, 0].values
        else:
            # Multi-variate: use all columns
            values = data.values
    else:
        values = np.asarray(data)

    # Handle multi-dimensional input
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    n_samples = len(values)
    n_features = values.shape[1]

    if n_samples < lookback + forecast_horizon:
        raise ValueError(
            f"Data length ({n_samples}) must be >= lookback ({lookback}) + "
            f"forecast_horizon ({forecast_horizon})"
        )

    X = []
    y = []

    for i in range(lookback, n_samples - forecast_horizon + 1):
        X.append(values[i - lookback : i])
        y.append(values[i : i + forecast_horizon])

    X = np.array(X)
    y = np.array(y)

    # Reshape y if forecast_horizon == 1
    if forecast_horizon == 1:
        y = y.reshape(-1, n_features)

    return X, y


def create_sequences_with_exog(
    y: SeriesLike,
    X: SeriesLike,
    lookback: int = 5,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences with exogenous variables.

    Creates input sequences for both target (y) and exogenous (X) variables.

    Args:
        y: Target time series.
        X: Exogenous time series (can be multi-variate).
        lookback: Number of time steps to look back.
        forecast_horizon: Number of steps ahead to forecast.

    Returns:
        Tuple of (X_seq, y_seq, y_target) where:
        - X_seq: Exogenous sequences (n_samples, lookback, n_exog_features)
        - y_seq: Target sequences (n_samples, lookback, 1)
        - y_target: Target values (n_samples, forecast_horizon)
    """
    if isinstance(y, pd.Series):
        y_values = y.values.reshape(-1, 1)
    elif isinstance(y, pd.DataFrame):
        y_values = y.values
    else:
        y_values = np.asarray(y).reshape(-1, 1)

    if isinstance(X, pd.Series):
        X_values = X.values.reshape(-1, 1)
    elif isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = np.asarray(X)
        if X_values.ndim == 1:
            X_values = X_values.reshape(-1, 1)

    if len(y_values) != len(X_values):
        raise ValueError("y and X must have the same length")

    n_samples = len(y_values)

    if n_samples < lookback + forecast_horizon:
        raise ValueError(
            f"Data length ({n_samples}) must be >= lookback ({lookback}) + "
            f"forecast_horizon ({forecast_horizon})"
        )

    # Use optimized sequence creation
    if HAS_NUMBA and n_samples > 100:
        X_seq, y_seq, y_target = _create_sequences_with_exog_numba(
            X_values, y_values, lookback, forecast_horizon
        )
    else:
        X_seq = []
        y_seq = []
        y_target = []
        for i in range(lookback, n_samples - forecast_horizon + 1):
            X_seq.append(X_values[i - lookback : i])
            y_seq.append(y_values[i - lookback : i])
            y_target.append(y_values[i : i + forecast_horizon])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        y_target = np.array(y_target)

    # Reshape y_target if forecast_horizon == 1
    if forecast_horizon == 1:
        y_target = y_target.reshape(-1, 1)

    return X_seq, y_seq, y_target


@njit(cache=True, fastmath=True, parallel=True)
def _create_sequences_with_exog_numba(
    X_values: np.ndarray, y_values: np.ndarray, lookback: int, forecast_horizon: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-optimized sequence creation with exogenous variables.
    
    Args:
        X_values: Exogenous variables (n_samples, n_exog_features).
        y_values: Target values (n_samples, 1).
        lookback: Lookback window size.
        forecast_horizon: Forecast horizon.
    
    Returns:
        Tuple of (X_seq, y_seq, y_target) arrays.
    """
    n_samples = len(y_values)
    n_seq = n_samples - lookback - forecast_horizon + 1
    
    if n_seq <= 0:
        n_exog = X_values.shape[1] if X_values.ndim > 1 else 1
        return (
            np.empty((0, lookback, n_exog)),
            np.empty((0, lookback, 1)),
            np.empty((0, forecast_horizon, 1)),
        )
    
    n_exog = X_values.shape[1] if X_values.ndim > 1 else 1
    X_seq = np.zeros((n_seq, lookback, n_exog), dtype=np.float64)
    y_seq = np.zeros((n_seq, lookback, 1), dtype=np.float64)
    y_target = np.zeros((n_seq, forecast_horizon, 1), dtype=np.float64)
    
    for i in prange(n_seq):
        X_seq[i] = X_values[i - lookback : i].reshape(lookback, -1)
        y_seq[i] = y_values[i - lookback : i].reshape(lookback, -1)
        y_target[i] = y_values[i : i + forecast_horizon].reshape(forecast_horizon, -1)
    
    return X_seq, y_seq, y_target

