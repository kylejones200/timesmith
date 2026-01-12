"""Optimized lag feature creation using NumPy and Numba."""

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@njit(cache=True, fastmath=True)
def _create_lags_numba(values: np.ndarray, lags: np.ndarray, n: int) -> np.ndarray:
    """Numba-optimized lag creation.

    Args:
        values: Input array.
        lags: Array of lag values.
        n: Length of input array.

    Returns:
        2D array where each row is a lag feature.
    """
    n_lags = len(lags)
    result = np.full((n_lags, n), np.nan, dtype=np.float64)

    for lag_idx, lag in enumerate(lags):
        lag_val = int(lag)
        if lag_val > 0:
            result[lag_idx, lag_val:] = values[:-lag_val]
        elif lag_val < 0:
            result[lag_idx, :lag_val] = values[-lag_val:]
        else:
            result[lag_idx, :] = values

    return result


@njit(cache=True, fastmath=True)
def _create_diffs_numba(values: np.ndarray, lags: np.ndarray, n: int) -> np.ndarray:
    """Numba-optimized difference creation.

    Args:
        values: Input array.
        lags: Array of lag values.
        n: Length of input array.

    Returns:
        2D array where each row is a diff feature.
    """
    n_lags = len(lags)
    result = np.full((n_lags, n), np.nan, dtype=np.float64)

    for lag_idx, lag in enumerate(lags):
        lag_val = int(lag)
        if lag_val > 0:
            result[lag_idx, lag_val:] = values[lag_val:] - values[:-lag_val]

    return result


@njit(cache=True, fastmath=True)
def _create_pct_changes_numba(
    values: np.ndarray, lags: np.ndarray, n: int
) -> np.ndarray:
    """Numba-optimized percentage change creation.

    Args:
        values: Input array.
        lags: Array of lag values.
        n: Length of input array.

    Returns:
        2D array where each row is a pct_change feature.
    """
    n_lags = len(lags)
    result = np.full((n_lags, n), np.nan, dtype=np.float64)

    for lag_idx, lag in enumerate(lags):
        lag_val = int(lag)
        if lag_val > 0:
            prev_values = values[:-lag_val]
            curr_values = values[lag_val:]
            # Avoid division by zero
            for i in range(len(prev_values)):
                if prev_values[i] != 0.0:
                    result[lag_idx, lag_val + i] = (
                        curr_values[i] - prev_values[i]
                    ) / prev_values[i]

    return result


def create_lags_vectorized(
    values: np.ndarray, lags: list[int], use_numba: bool = True
) -> dict[str, np.ndarray]:
    """Create lag features using vectorized operations.

    Args:
        values: Input array.
        lags: List of lag values.
        use_numba: Whether to use Numba JIT compilation if available.

    Returns:
        Dictionary mapping 'lag_N' to lag arrays.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    lags_arr = np.array(lags, dtype=np.int32)

    if HAS_NUMBA and use_numba and len(lags) > 0:
        lag_matrix = _create_lags_numba(values, lags_arr, n)
        return {f"lag_{lag}": lag_matrix[i] for i, lag in enumerate(lags)}
    else:
        # Fallback to manual implementation
        result = {}
        for lag in lags:
            lagged = np.full(n, np.nan, dtype=np.float64)
            if lag > 0:
                lagged[lag:] = values[:-lag]
            elif lag < 0:
                lagged[:lag] = values[-lag:]
            else:
                lagged = values.copy()
            result[f"lag_{lag}"] = lagged
        return result
