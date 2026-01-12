"""Optimized rolling window operations using NumPy vectorization.

These functions are faster than pandas rolling operations and can be parallelized.
"""

import numpy as np
from typing import Literal, Optional

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define no-op decorators if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def rolling_mean(arr: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    """Compute rolling mean using vectorized NumPy operations.
    
    Args:
        arr: Input array.
        window: Window size.
        min_periods: Minimum number of observations in window.
    
    Returns:
        Array of rolling means.
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    
    if window <= 0 or n < min_periods:
        return np.full(n, np.nan, dtype=np.float64)
    
    # Use cumsum for efficient rolling mean
    cumsum = np.concatenate([[0.0], np.cumsum(arr, dtype=np.float64)])
    result = np.full(n, np.nan, dtype=np.float64)
    
    # Vectorized computation for full windows
    if window <= n:
        result[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
    
    # Handle min_periods for initial values
    for i in range(min_periods - 1, min(window - 1, n)):
        count = i + 1
        result[i] = cumsum[i + 1] / count
    
    return result


def rolling_std(arr: np.ndarray, window: int, min_periods: int = 1, ddof: int = 1) -> np.ndarray:
    """Compute rolling standard deviation using vectorized operations.
    
    Args:
        arr: Input array.
        window: Window size.
        min_periods: Minimum number of observations in window.
        ddof: Delta degrees of freedom.
    
    Returns:
        Array of rolling standard deviations.
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    
    if window <= 0 or n < min_periods:
        return np.zeros(n, dtype=np.float64)
    
    result = np.zeros(n, dtype=np.float64)
    
    # Use cumsum approach for efficiency
    cumsum = np.concatenate([[0.0], np.cumsum(arr, dtype=np.float64)])
    cumsum2 = np.concatenate([[0.0], np.cumsum(arr ** 2, dtype=np.float64)])
    
    # Vectorized computation for full windows
    if window <= n:
        counts = window
        means = (cumsum[window:] - cumsum[:-window]) / counts
        sum_sq = cumsum2[window:] - cumsum2[:-window]
        variances = (sum_sq / counts - means ** 2) * (counts / (counts - ddof))
        result[window - 1:] = np.sqrt(np.maximum(0, variances))
    
    # Handle min_periods for initial values
    for i in range(min_periods - 1, min(window - 1, n)):
        count = i + 1
        mean = cumsum[i + 1] / count
        sum_sq = cumsum2[i + 1]
        variance = (sum_sq / count - mean ** 2) * (count / (count - ddof)) if count > ddof else 0
        result[i] = np.sqrt(max(0, variance))
    
    return result


@njit(cache=True, fastmath=True)
def _rolling_min_max_median_numba(
    arr: np.ndarray, window: int, op: int
) -> np.ndarray:
    """Numba-optimized rolling min/max/median.
    
    Args:
        arr: Input array.
        window: Window size.
        op: Operation (0=min, 1=max, 2=median).
    
    Returns:
        Result array.
    """
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    
    if window <= 0:
        return result
    
    for i in range(window - 1, n):
        window_data = arr[i - window + 1:i + 1]
        if op == 0:  # min
            result[i] = np.nanmin(window_data)
        elif op == 1:  # max
            result[i] = np.nanmax(window_data)
        else:  # median
            result[i] = np.nanmedian(window_data)
    
    return result


def rolling_min(arr: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    """Compute rolling minimum."""
    arr = np.asarray(arr, dtype=np.float64)
    if HAS_NUMBA and len(arr) > 100:
        result = _rolling_min_max_median_numba(arr, window, 0)
        # Handle min_periods
        if min_periods < window:
            result[:min_periods - 1] = np.nan
        return result
    else:
        # Fallback to manual implementation
        n = len(arr)
        result = np.full(n, np.nan, dtype=np.float64)
        for i in range(window - 1, n):
            result[i] = np.nanmin(arr[i - window + 1:i + 1])
        return result


def rolling_max(arr: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    """Compute rolling maximum."""
    arr = np.asarray(arr, dtype=np.float64)
    if HAS_NUMBA and len(arr) > 100:
        result = _rolling_min_max_median_numba(arr, window, 1)
        if min_periods < window:
            result[:min_periods - 1] = np.nan
        return result
    else:
        n = len(arr)
        result = np.full(n, np.nan, dtype=np.float64)
        for i in range(window - 1, n):
            result[i] = np.nanmax(arr[i - window + 1:i + 1])
        return result


def rolling_median(arr: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    """Compute rolling median."""
    arr = np.asarray(arr, dtype=np.float64)
    if HAS_NUMBA and len(arr) > 100:
        result = _rolling_min_max_median_numba(arr, window, 2)
        if min_periods < window:
            result[:min_periods - 1] = np.nan
        return result
    else:
        n = len(arr)
        result = np.full(n, np.nan, dtype=np.float64)
        for i in range(window - 1, n):
            result[i] = np.nanmedian(arr[i - window + 1:i + 1])
        return result


# Vectorized rolling operations for multiple windows
def rolling_statistics(
    arr: np.ndarray,
    windows: list[int],
    functions: list[Literal["mean", "std", "min", "max", "median"]],
    n_jobs: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Compute multiple rolling statistics efficiently with optional parallelization.
    
    Args:
        arr: Input array.
        windows: List of window sizes.
        functions: List of function names.
        n_jobs: Number of parallel jobs. If None, uses all CPUs. If 1, runs serially.
            Only used if joblib is available and more than 4 total operations.
    
    Returns:
        Dictionary mapping 'function_window' to result arrays.
    """
    arr = np.asarray(arr, dtype=np.float64)
    results = {}
    
    func_map = {
        "mean": rolling_mean,
        "std": rolling_std,
        "min": rolling_min,
        "max": rolling_max,
        "median": rolling_median,
    }
    
    # Prepare list of tasks
    tasks = []
    for window in windows:
        for func_name in functions:
            if func_name in func_map:
                tasks.append((f"rolling_{func_name}_{window}", func_map[func_name], window))
    
    # Use parallelization if joblib available and enough tasks
    if HAS_JOBLIB and len(tasks) > 4 and (n_jobs is None or n_jobs > 1):
        def compute_stat(key_func_window):
            key, func, window = key_func_window
            return key, func(arr, window)
        
        parallel_results = Parallel(n_jobs=n_jobs)(delayed(compute_stat)(task) for task in tasks)
        results = dict(parallel_results)
    else:
        # Serial execution
        for key, func, window in tasks:
            results[key] = func(arr, window)
    
    return results

