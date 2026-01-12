"""Seasonal baseline anomaly detection for time series."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseDetector
from timesmith.core.tags import set_tags

logger = logging.getLogger(__name__)


class SeasonalBaselineDetector(BaseDetector):
    """Seasonal baseline anomaly detector for time series.

    Calculates seasonal baselines (e.g., weekly, monthly) and flags
    points that deviate significantly from expected seasonal patterns.
    """

    def __init__(
        self,
        seasonality: str = "week",
        threshold_sigma: float = 2.5,
    ):
        """Initialize seasonal baseline detector.

        Args:
            seasonality: Seasonality to use. Options: 'week', 'month', 'day', 'hour'.
            threshold_sigma: Number of standard deviations for threshold.
        """
        super().__init__()
        self.seasonality = seasonality
        self.threshold_sigma = threshold_sigma
        self.seasonal_stats_ = None

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def _get_seasonal_key(self, dates: pd.Series) -> pd.Series:
        """Extract seasonal key from dates based on seasonality type."""
        seasonality_map = {
            "week": lambda d: d.dt.isocalendar().week,
            "month": lambda d: d.dt.month,
            "day": lambda d: d.dt.dayofyear,
            "hour": lambda d: d.dt.hour,
        }
        if self.seasonality not in seasonality_map:
            raise ValueError(f"Unknown seasonality: {self.seasonality}")
        return seasonality_map[self.seasonality](dates)

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "SeasonalBaselineDetector":
        """Fit the detector by computing seasonal baselines.

        Args:
            y: Target time series with datetime index.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError(
                "y must be Series or single-column DataFrame with datetime index"
            )

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(
                "Data must have datetime index for seasonal baseline detection"
            )

        # Use vectorized NumPy operations for seasonal statistics
        values = series.values
        seasonal_keys = self._get_seasonal_key(series.index)

        # Convert keys to integer indices for efficient numpy operations
        unique_keys = pd.Series(seasonal_keys).unique()
        key_to_idx = {key: idx for idx, key in enumerate(unique_keys)}
        key_indices = np.array(
            [key_to_idx[key] for key in seasonal_keys], dtype=np.int32
        )

        n_keys = len(unique_keys)

        # Vectorized computation of mean and std per group
        means = np.zeros(n_keys, dtype=np.float64)
        stds = np.zeros(n_keys, dtype=np.float64)

        for i, key in enumerate(unique_keys):
            mask = key_indices == i
            if mask.sum() > 0:
                group_values = values[mask]
                means[i] = np.mean(group_values)
                std_val = np.std(group_values, ddof=1) if len(group_values) > 1 else 0.0
                stds[i] = std_val if std_val > 0 else 1.0

        # Create DataFrame for compatibility
        seasonal_stats = pd.DataFrame(
            {
                "seasonal_key": unique_keys,
                "mean": means,
                "std": stds,
            }
        )

        self.seasonal_stats_ = seasonal_stats
        self.index_ = series.index
        self.y_ = series.values

        self._is_fitted = True
        return self

    def score(self, y: Any, X: Optional[Any] = None) -> np.ndarray:
        """Compute Z-scores relative to seasonal baseline.

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).

        Returns:
            Array of Z-scores relative to seasonal baseline.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be Series or single-column DataFrame")

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Data must have datetime index")

        # Create DataFrame
        df = pd.DataFrame({"value": series.values, "date": series.index})
        df["seasonal_key"] = self._get_seasonal_key(df["date"])

        # Merge with seasonal stats
        df = df.merge(self.seasonal_stats_, on="seasonal_key", how="left")

        # Compute Z-scores
        z_scores = np.abs((df["value"] - df["mean"]) / df["std"])
        z_scores = z_scores.fillna(0).values

        return z_scores

    def predict(
        self, y: Any, X: Optional[Any] = None, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Predict anomaly flags.

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).
            threshold: Optional threshold (uses self.threshold_sigma if not provided).

        Returns:
            Boolean array with True at anomalies.
        """
        threshold = threshold or self.threshold_sigma
        scores = self.score(y, X)

        flags = np.zeros(len(scores), dtype=bool)
        flags[scores > threshold] = True

        logger.info(
            f"Seasonal baseline detector found {flags.sum()} anomalies "
            f"(seasonality={self.seasonality}, threshold={threshold})"
        )

        return flags
