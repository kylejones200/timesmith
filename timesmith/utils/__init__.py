"""Utility functions for time series operations."""

from timesmith.utils.ts_utils import (
    detect_anomalies_mad,
    detect_frequency,
    ensure_datetime_index,
    fill_missing_dates,
    load_ts_data,
    remove_outliers_iqr,
    resample_ts,
    split_ts,
)
from timesmith.utils.distances import (
    correlation_distance,
    cross_correlation_distance,
    dtw_distance,
    euclidean_distance,
    manhattan_distance,
)

# Optional matplotlib-dependent imports
try:
    from timesmith.utils.monte_carlo import monte_carlo_simulation, plot_monte_carlo
    HAS_MONTE_CARLO = True
except ImportError:
    HAS_MONTE_CARLO = False
    # Define stubs to avoid import errors
    def monte_carlo_simulation(*args, **kwargs):
        raise ImportError(
            "matplotlib is required for monte_carlo_simulation. "
            "Install with: pip install matplotlib"
        )

    def plot_monte_carlo(*args, **kwargs):
        raise ImportError(
            "matplotlib is required for plot_monte_carlo. "
            "Install with: pip install matplotlib"
        )

# Stationarity tests (optional statsmodels)
try:
    from timesmith.utils.stationarity import test_stationarity
    HAS_STATIONARITY = True
except ImportError:
    HAS_STATIONARITY = False

# Climatology utilities (always available)
from timesmith.utils.climatology import (
    compute_climatology,
    compute_anomalies,
    detect_extreme_events,
)

__all__ = [
    # Time series utilities
    "load_ts_data",
    "ensure_datetime_index",
    "resample_ts",
    "split_ts",
    "detect_frequency",
    "fill_missing_dates",
    "remove_outliers_iqr",
    "detect_anomalies_mad",
    # Monte Carlo
    "monte_carlo_simulation",
    "plot_monte_carlo",
    # Distance metrics
    "correlation_distance",
    "cross_correlation_distance",
    "dtw_distance",
    "euclidean_distance",
    "manhattan_distance",
]

# Conditionally add stationarity tests
if HAS_STATIONARITY:
    __all__.append("test_stationarity")

# Climatology utilities
__all__.extend([
    "compute_climatology",
    "compute_anomalies",
    "detect_extreme_events",
])

