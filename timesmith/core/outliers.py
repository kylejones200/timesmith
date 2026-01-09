"""Advanced outlier detection transformers for time series."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseTransformer
from timesmith.core.tags import set_tags
from timesmith.typing import SeriesLike

logger = logging.getLogger(__name__)

# Optional sklearn for IsolationForest
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn not available. IsolationForest outlier detection will be unavailable. "
        "Install with: pip install scikit-learn"
    )


class HampelOutlierRemover(BaseTransformer):
    """Remove outliers using Hampel filter (MAD-based).

    Hampel filter uses median absolute deviation (MAD) to detect outliers
    relative to a rolling median baseline. More robust than Z-score methods.
    """

    def __init__(self, window: int = 10, n_sigma: float = 3.0):
        """Initialize Hampel outlier remover.

        Args:
            window: Window size for rolling median.
            n_sigma: Number of standard deviations for threshold.
        """
        super().__init__()
        self.window = window
        self.n_sigma = n_sigma

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "HampelOutlierRemover":
        """Fit the transformer (computes outlier mask).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if isinstance(y, pd.Series):
            self.y_ = y.values
            self.index_ = y.index
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            self.y_ = y.iloc[:, 0].values
            self.index_ = y.index
        else:
            self.y_ = np.asarray(y, dtype=float)
            self.index_ = np.arange(len(self.y_))

        # Remove invalid values
        valid_mask = np.isfinite(self.y_)
        self.y_ = self.y_[valid_mask]
        self.index_ = self.index_[valid_mask]

        if len(self.y_) < self.window:
            raise ValueError(f"Need at least {self.window} data points")

        # Compute rolling median (center=False to avoid future data leakage)
        rolling_median = (
            pd.Series(self.y_)
            .rolling(window=self.window, center=False)
            .median()
            .fillna(pd.Series(self.y_).median())
        )

        # Compute residuals
        residuals = self.y_ - rolling_median.values

        # Compute MAD (Median Absolute Deviation)
        mad = np.median(np.abs(residuals - np.median(residuals)))

        # If MAD is zero or very small, use a small default threshold
        if mad < 1e-10:
            threshold = self.n_sigma * 0.01 * np.median(np.abs(self.y_))
        else:
            # Threshold (using modified Z-score)
            threshold = self.n_sigma * 1.4826 * mad  # 1.4826 makes MAD comparable to std

        # Detect outliers
        self.outlier_mask_ = np.abs(residuals) > threshold

        logger.debug(
            f"Hampel filter detected {self.outlier_mask_.sum()} outliers",
            extra={"n_outliers": int(self.outlier_mask_.sum()), "threshold": threshold},
        )

        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.Series:
        """Remove outliers.

        Args:
            y: SeriesLike data (should match fit data).
            X: Optional exogenous data (ignored).

        Returns:
            SeriesLike data with outliers removed.
        """
        self._check_is_fitted()

        # Remove outliers
        keep_mask = ~self.outlier_mask_
        cleaned_values = self.y_[keep_mask]
        cleaned_index = self.index_[keep_mask]

        return pd.Series(cleaned_values, index=cleaned_index)


class IsolationForestOutlierRemover(BaseTransformer):
    """Remove outliers using IsolationForest.

    Uses feature set: rate, delta rate, rolling median residual.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: Optional[int] = None,
        window: int = 10,
    ):
        """Initialize IsolationForest outlier remover.

        Args:
            contamination: Expected proportion of outliers.
            random_state: Random seed for reproducibility.
            window: Window size for rolling median feature.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for IsolationForestOutlierRemover. "
                "Install with: pip install scikit-learn"
            )

        super().__init__()
        self.contamination = contamination
        self.random_state = random_state
        self.window = window

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "IsolationForestOutlierRemover":
        """Fit the transformer (trains IsolationForest).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if isinstance(y, pd.Series):
            self.y_ = y.values
            self.index_ = y.index
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            self.y_ = y.iloc[:, 0].values
            self.index_ = y.index
        else:
            self.y_ = np.asarray(y, dtype=float)
            self.index_ = np.arange(len(self.y_))

        # Remove invalid values
        valid_mask = np.isfinite(self.y_)
        self.y_ = self.y_[valid_mask]
        self.index_ = self.index_[valid_mask]

        if len(self.y_) < self.window + 5:
            raise ValueError(f"Need at least {self.window + 5} data points")

        # Prepare features
        features = []

        # Feature 1: Rate (normalized)
        rate_norm = (self.y_ - self.y_.mean()) / (self.y_.std() + 1e-10)
        features.append(rate_norm)

        # Feature 2: Delta rate (first difference)
        delta_rate = np.diff(self.y_, prepend=self.y_[0])
        delta_rate_norm = (delta_rate - delta_rate.mean()) / (delta_rate.std() + 1e-10)
        features.append(delta_rate_norm)

        # Feature 3: Rolling median residual
        rolling_median = (
            pd.Series(self.y_)
            .rolling(window=self.window, center=False)
            .median()
            .fillna(pd.Series(self.y_).median())
        )
        residual = self.y_ - rolling_median.values
        residual_norm = (residual - residual.mean()) / (residual.std() + 1e-10)
        features.append(residual_norm)

        # Stack features
        X_features = np.column_stack(features)

        # Fit IsolationForest
        self.iso_forest_ = IsolationForest(
            contamination=self.contamination, random_state=self.random_state
        )
        predictions = self.iso_forest_.fit_predict(X_features)

        # Convert to boolean (1 = inlier, -1 = outlier)
        self.outlier_mask_ = predictions == -1

        logger.debug(
            f"IsolationForest detected {self.outlier_mask_.sum()} outliers",
            extra={"n_outliers": int(self.outlier_mask_.sum()), "contamination": self.contamination},
        )

        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.Series:
        """Remove outliers.

        Args:
            y: SeriesLike data (should match fit data).
            X: Optional exogenous data (ignored).

        Returns:
            SeriesLike data with outliers removed.
        """
        self._check_is_fitted()

        # Remove outliers
        keep_mask = ~self.outlier_mask_
        cleaned_values = self.y_[keep_mask]
        cleaned_index = self.index_[keep_mask]

        return pd.Series(cleaned_values, index=cleaned_index)


class ZScoreOutlierRemover(BaseTransformer):
    """Remove outliers using Z-score on log residual.

    Better for multiplicative errors than linear Z-score.
    """

    def __init__(
        self, window: int = 10, z_threshold: float = 3.0, use_log: bool = True
    ):
        """Initialize Z-score outlier remover.

        Args:
            window: Window size for rolling median baseline.
            z_threshold: Z-score threshold.
            use_log: If True, use log residual (better for multiplicative errors).
        """
        super().__init__()
        self.window = window
        self.z_threshold = z_threshold
        self.use_log = use_log

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "ZScoreOutlierRemover":
        """Fit the transformer (computes outlier mask).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if isinstance(y, pd.Series):
            self.y_ = y.values
            self.index_ = y.index
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            self.y_ = y.iloc[:, 0].values
            self.index_ = y.index
        else:
            self.y_ = np.asarray(y, dtype=float)
            self.index_ = np.arange(len(self.y_))

        # Remove invalid values
        valid_mask = np.isfinite(self.y_)
        self.y_ = self.y_[valid_mask]
        self.index_ = self.index_[valid_mask]

        if len(self.y_) < self.window:
            raise ValueError(f"Need at least {self.window} data points")

        # Compute baseline (center=False to avoid future data leakage)
        rolling_median = (
            pd.Series(self.y_)
            .rolling(window=self.window, center=False)
            .median()
            .fillna(pd.Series(self.y_).median())
        )

        # Compute residuals
        if self.use_log:
            # Use log residual (multiplicative)
            rates_positive = np.maximum(
                self.y_, self.y_[self.y_ > 0].min() if (self.y_ > 0).any() else 1.0
            )
            baseline_positive = np.maximum(rolling_median.values, 1.0)
            residuals = np.log(rates_positive) - np.log(baseline_positive)
        else:
            # Use linear residual (additive)
            residuals = self.y_ - rolling_median.values

        # Compute Z-scores
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)

        if residual_std > 0:
            z_scores = np.abs((residuals - residual_mean) / residual_std)
        else:
            z_scores = np.zeros_like(residuals)

        # Detect outliers
        self.outlier_mask_ = z_scores > self.z_threshold

        logger.debug(
            f"Z-score method detected {self.outlier_mask_.sum()} outliers",
            extra={"n_outliers": int(self.outlier_mask_.sum()), "z_threshold": self.z_threshold},
        )

        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.Series:
        """Remove outliers.

        Args:
            y: SeriesLike data (should match fit data).
            X: Optional exogenous data (ignored).

        Returns:
            SeriesLike data with outliers removed.
        """
        self._check_is_fitted()

        # Remove outliers
        keep_mask = ~self.outlier_mask_
        cleaned_values = self.y_[keep_mask]
        cleaned_index = self.index_[keep_mask]

        return pd.Series(cleaned_values, index=cleaned_index)

