"""Distance metrics for time series comparison.

These are utility functions for computing distances between time series.
"""

import logging
from typing import Optional

import numpy as np
from scipy import stats
from scipy.signal import correlate

logger = logging.getLogger(__name__)

try:
    from tslearn.metrics import cdist_dtw as _cdist_dtw
    HAS_TSLEARN = True
except ImportError:
    HAS_TSLEARN = False
    _cdist_dtw = None
    logger.warning(
        "tslearn not installed. DTW distance will use fallback implementation. "
        "Install with: pip install tslearn for optimized DTW."
    )


def correlation_distance(
    x: np.ndarray, y: np.ndarray, method: str = "pearson", absolute: bool = False
) -> float:
    """Compute correlation-based distance between two time series.

    Args:
        x: First time series.
        y: Second time series (must have same length as x).
        method: Correlation method ('pearson' or 'spearman').
        absolute: If True, use absolute value of correlation.

    Returns:
        Distance value (0-2 range, where 0 = perfect correlation).
    """
    if len(x) != len(y):
        raise ValueError(f"Series must have same length: {len(x)} != {len(y)}")

    if method == "pearson":
        corr = np.corrcoef(x, y)[0, 1]
    elif method == "spearman":
        corr, _ = stats.spearmanr(x, y)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")

    if absolute:
        corr = np.abs(corr)

    # Convert correlation to distance (0-2 range)
    return float(np.sqrt(2 * (1 - corr)))


def cross_correlation_distance(
    x: np.ndarray, y: np.ndarray, max_lag: int = 10
) -> float:
    """Compute distance using maximum cross-correlation.

    Args:
        x: First time series.
        y: Second time series.
        max_lag: Maximum lag to consider for cross-correlation.

    Returns:
        Distance value based on maximum cross-correlation.
    """
    ccf = correlate(x - x.mean(), y - y.mean(), mode="full")
    lags = np.arange(-(len(x) - 1), len(x))
    mask = (lags >= -max_lag) & (lags <= max_lag)
    max_r = np.max(np.abs(ccf[mask])) / (np.std(x) * np.std(y) * len(x))
    return float(np.sqrt(2 * (1 - max_r)))


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Dynamic Time Warping (DTW) distance.

    Args:
        x: First time series.
        y: Second time series.

    Returns:
        DTW distance value.
    """
    if HAS_TSLEARN and _cdist_dtw is not None:
        # Use optimized tslearn implementation
        X = np.array([x, y])
        D = _cdist_dtw(X)
        return float(D[0, 1])

    # Fallback: simple DTW implementation
    n, m = len(x), len(y)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
            )

    return float(np.sqrt(dtw_matrix[n, m]))


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean distance between two time series.

    Args:
        x: First time series.
        y: Second time series (must have same length as x).

    Returns:
        Euclidean distance.
    """
    if len(x) != len(y):
        raise ValueError(f"Series must have same length: {len(x)} != {len(y)}")
    return float(np.sqrt(np.sum((x - y) ** 2)))


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Manhattan (L1) distance between two time series.

    Args:
        x: First time series.
        y: Second time series (must have same length as x).

    Returns:
        Manhattan distance.
    """
    if len(x) != len(y):
        raise ValueError(f"Series must have same length: {len(x)} != {len(y)}")
    return float(np.sum(np.abs(x - y)))

