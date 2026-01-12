"""Utility functions for time series operations."""

from timesmith.utils.distances import (
    correlation_distance,
    cross_correlation_distance,
    dtw_distance,
    euclidean_distance,
    manhattan_distance,
)

# Monte Carlo simulation (always available)
from timesmith.utils.monte_carlo import (
    black_scholes_monte_carlo,
    monte_carlo_simulation,
)
from timesmith.utils.ts_utils import (
    detect_anomalies_mad,
    detect_frequency,
    ensure_datetime_index,
    fill_missing_dates,
    load_ts_data,
    remove_outliers_iqr,
    resample_ts,
    split_ts,
    train_test_split,
)

# Optional plotting utilities (requires plotsmith)
try:
    from timesmith.utils.monte_carlo import plot_monte_carlo
    from timesmith.utils.plotting import (
        plot_autocorrelation,
        plot_forecast,
        plot_monte_carlo_paths,
        plot_multiple_series,
        plot_residuals,
        plot_timeseries,
    )

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

    # Define stubs to avoid import errors
    def plot_timeseries(*args, **kwargs):
        raise ImportError(
            "plotsmith is required for plotting. Install with: pip install plotsmith"
        )

    def plot_forecast(*args, **kwargs):
        raise ImportError(
            "plotsmith is required for plotting. Install with: pip install plotsmith"
        )

    def plot_residuals(*args, **kwargs):
        raise ImportError(
            "plotsmith is required for plotting. Install with: pip install plotsmith"
        )

    def plot_multiple_series(*args, **kwargs):
        raise ImportError(
            "plotsmith is required for plotting. Install with: pip install plotsmith"
        )

    def plot_autocorrelation(*args, **kwargs):
        raise ImportError(
            "plotsmith is required for plotting. Install with: pip install plotsmith"
        )

    def plot_monte_carlo_paths(*args, **kwargs):
        raise ImportError(
            "plotsmith is required for plotting. Install with: pip install plotsmith"
        )

    def plot_monte_carlo(*args, **kwargs):
        raise ImportError(
            "plotsmith is required for plotting. Install with: pip install plotsmith"
        )


# Stationarity tests (optional statsmodels)
try:
    from timesmith.utils.stationarity import is_stationary, test_stationarity  # noqa: F401

    HAS_STATIONARITY = True
except ImportError:
    HAS_STATIONARITY = False

# Climatology utilities (always available)
# Autocorrelation utilities (always available)
from timesmith.utils.autocorrelation import (
    autocorrelation,  # noqa: F401
    autocorrelation_plot_data,  # noqa: F401
    partial_autocorrelation,  # noqa: F401
)
from timesmith.utils.climatology import (
    compute_anomalies,  # noqa: F401
    compute_climatology,  # noqa: F401
    detect_extreme_events,  # noqa: F401
)

# Confidence intervals (always available)
from timesmith.utils.confidence_intervals import (
    bootstrap_confidence_intervals,  # noqa: F401
    parametric_confidence_intervals,  # noqa: F401
)

# Sequence creation utilities (always available)
from timesmith.utils.sequences import (
    create_sequences,  # noqa: F401
    create_sequences_with_exog,  # noqa: F401
)

__all__ = [
    # Time series utilities
    "load_ts_data",
    "ensure_datetime_index",
    "resample_ts",
    "split_ts",
    "train_test_split",
    "detect_frequency",
    "fill_missing_dates",
    "remove_outliers_iqr",
    "detect_anomalies_mad",
    # Monte Carlo
    "monte_carlo_simulation",
    "black_scholes_monte_carlo",
    "plot_monte_carlo",
    # Plotting (if plotsmith available)
    # Distance metrics
    "correlation_distance",
    "cross_correlation_distance",
    "dtw_distance",
    "euclidean_distance",
    "manhattan_distance",
]

# Conditionally add stationarity tests
if HAS_STATIONARITY:
    __all__.extend(["test_stationarity", "is_stationary"])

# Climatology utilities
__all__.extend(
    [
        "compute_climatology",
        "compute_anomalies",
        "detect_extreme_events",
    ]
)

# Sequence creation utilities
__all__.extend(
    [
        "create_sequences",
        "create_sequences_with_exog",
    ]
)

# Autocorrelation utilities
__all__.extend(
    [
        "autocorrelation",
        "partial_autocorrelation",
        "autocorrelation_plot_data",
    ]
)

# Confidence intervals
__all__.extend(
    [
        "bootstrap_confidence_intervals",
        "parametric_confidence_intervals",
    ]
)

# Conditionally add plotting functions
if HAS_PLOTTING:
    __all__.extend(
        [
            "plot_timeseries",
            "plot_forecast",
            "plot_residuals",
            "plot_multiple_series",
            "plot_autocorrelation",
            "plot_monte_carlo_paths",
        ]
    )
