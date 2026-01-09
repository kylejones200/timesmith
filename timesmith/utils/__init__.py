"""Utility functions for time series operations."""

from timesmith.utils.monte_carlo import monte_carlo_simulation, plot_monte_carlo
from timesmith.utils.ts_utils import (
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

__all__ = [
    # Time series utilities
    "load_ts_data",
    "ensure_datetime_index",
    "resample_ts",
    "split_ts",
    "detect_frequency",
    "fill_missing_dates",
    "remove_outliers_iqr",
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

