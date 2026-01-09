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

# Monte Carlo simulation (always available)
from timesmith.utils.monte_carlo import (
    monte_carlo_simulation,
    black_scholes_monte_carlo,
)

# Optional plotting utilities (requires plotsmith)
try:
    from timesmith.utils.plotting import (
        plot_timeseries,
        plot_forecast,
        plot_residuals,
        plot_multiple_series,
        plot_autocorrelation,
        plot_monte_carlo_paths,
    )
    from timesmith.utils.monte_carlo import plot_monte_carlo
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

# Sequence creation utilities (always available)
from timesmith.utils.sequences import (
    create_sequences,
    create_sequences_with_exog,
)

# Autocorrelation utilities (always available)
from timesmith.utils.autocorrelation import (
    autocorrelation,
    autocorrelation_plot_data,
    partial_autocorrelation,
)

# Confidence intervals (always available)
from timesmith.utils.confidence_intervals import (
    bootstrap_confidence_intervals,
    parametric_confidence_intervals,
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
    __all__.append("test_stationarity")

# Climatology utilities
__all__.extend([
    "compute_climatology",
    "compute_anomalies",
    "detect_extreme_events",
])

# Sequence creation utilities
__all__.extend([
    "create_sequences",
    "create_sequences_with_exog",
])

# Autocorrelation utilities
__all__.extend([
    "autocorrelation",
    "partial_autocorrelation",
    "autocorrelation_plot_data",
])

# Confidence intervals
__all__.extend([
    "bootstrap_confidence_intervals",
    "parametric_confidence_intervals",
])

# Conditionally add plotting functions
if HAS_PLOTTING:
    __all__.extend([
        "plot_timeseries",
        "plot_forecast",
        "plot_residuals",
        "plot_multiple_series",
        "plot_autocorrelation",
        "plot_monte_carlo_paths",
    ])

