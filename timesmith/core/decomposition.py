"""Time series decomposition transformers.

Provides trend and seasonality detection and removal for time series analysis.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseTransformer
from timesmith.core.tags import set_tags
from timesmith.typing import SeriesLike

logger = logging.getLogger(__name__)

# Optional scipy imports
try:
    from scipy import signal, stats
    from scipy.ndimage import uniform_filter1d

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    signal = None
    stats = None
    uniform_filter1d = None
    logger.warning(
        "scipy not installed. Decomposition functionality will be limited. "
        "Install with: pip install scipy or pip install timesmith[scipy]"
    )


def _detect_seasonal_period(data: np.ndarray, max_period: int = 50) -> Optional[int]:
    """Detect seasonal period using autocorrelation.

    Args:
        data: Time series data.
        max_period: Maximum period to check.

    Returns:
        Detected seasonal period or None.
    """
    n = len(data)
    if n < max_period * 2:
        return None

    # Compute autocorrelation
    autocorr = np.correlate(data, data, mode="full")
    autocorr = autocorr[n - 1 :] / autocorr[n - 1]

    # Find peaks in autocorrelation (potential seasonal periods)
    if not HAS_SCIPY:
        # Fallback: find peaks manually
        peaks = []
        for i in range(1, min(max_period, len(autocorr) - 1)):
            if (
                autocorr[i] > 0.3
                and autocorr[i] > autocorr[i - 1]
                and autocorr[i] > autocorr[i + 1]
            ):
                peaks.append(i)
        peaks = np.array(peaks)
    else:
        peaks, _ = signal.find_peaks(autocorr[1:max_period], height=0.3)

    if len(peaks) > 0:
        # Return first significant peak
        return int(peaks[0] + 1)

    return None


def detect_trend(y: SeriesLike, method: str = "linear") -> Dict[str, Any]:
    """Detect trend in time series data.

    Args:
        y: Time series values.
        method: Trend detection method: 'linear', 'polynomial', or 'moving_average'.

    Returns:
        Dictionary with trend information.
    """
    if isinstance(y, pd.Series):
        y_arr = y.values
    elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y_arr = y.iloc[:, 0].values
    else:
        y_arr = np.asarray(y, dtype=float)

    valid_mask = np.isfinite(y_arr)
    y_arr = y_arr[valid_mask]

    if len(y_arr) < 3:
        raise ValueError("Need at least 3 data points")

    time_arr = np.arange(len(y_arr))

    if method == "linear":
        slope, intercept, r_value, _, _ = np.polyfit(time_arr, y_arr, 1, full=False)
        trend = slope * time_arr + intercept
        strength = abs(r_value)

        return {
            "trend": trend,
            "slope": float(slope),
            "intercept": float(intercept),
            "strength": float(strength),
        }

    elif method == "theil_sen":
        # Theil-Sen estimator: more robust to outliers
        if not HAS_SCIPY:
            logger.warning("scipy not available, falling back to linear trend")
            # Fall back to linear
            slope, intercept, r_value, _, _ = np.polyfit(time_arr, y_arr, 1, full=False)
            trend = slope * time_arr + intercept
            return {
                "trend": trend,
                "slope": float(slope),
                "intercept": float(intercept),
                "strength": float(abs(r_value)),
            }
        else:
            try:
                slope, intercept = stats.theilslopes(y_arr, time_arr)[:2]
                # Approximate correlation for Theil-Sen
                r_value = np.corrcoef(time_arr, y_arr)[0, 1]
                trend = slope * time_arr + intercept

                return {
                    "trend": trend,
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "strength": float(abs(r_value)),
                }
            except Exception as e:
                logger.warning(f"Theil-Sen failed: {e}, falling back to linear")
                # Fall back to linear
                slope, intercept, r_value, _, _ = np.polyfit(
                    time_arr, y_arr, 1, full=False
                )
                trend = slope * time_arr + intercept
                return {
                    "trend": trend,
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "strength": float(abs(r_value)),
                }

    elif method == "polynomial":
        coeffs = np.polyfit(time_arr, y_arr, deg=2)
        trend = np.polyval(coeffs, time_arr)
        # Calculate R-squared as strength
        ss_res = np.sum((y_arr - trend) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        strength = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "trend": trend,
            "coefficients": coeffs.tolist(),
            "strength": float(strength),
        }

    elif method == "moving_average":
        window = max(3, len(y_arr) // 10)
        if not HAS_SCIPY:
            # Fallback: simple moving average
            trend = np.convolve(y_arr, np.ones(window) / window, mode="same")
        else:
            trend = uniform_filter1d(y_arr, size=window, mode="nearest")
        # Calculate trend strength as variance reduction
        var_original = np.var(y_arr)
        var_residual = np.var(y_arr - trend)
        strength = 1 - (var_residual / var_original) if var_original > 0 else 0

        return {
            "trend": trend,
            "strength": float(strength),
        }

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            "Use 'linear', 'theil_sen', 'polynomial', or 'moving_average'"
        )


def detect_seasonality(y: SeriesLike, max_period: int = 50) -> Dict[str, Any]:
    """Detect seasonality in time series data.

    Args:
        y: Time series values.
        max_period: Maximum period to check.

    Returns:
        Dictionary with seasonality information.
    """
    if isinstance(y, pd.Series):
        y_arr = y.values
    elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y_arr = y.iloc[:, 0].values
    else:
        y_arr = np.asarray(y, dtype=float)

    valid_mask = np.isfinite(y_arr)
    y_arr = y_arr[valid_mask]

    if len(y_arr) < max_period * 2:
        return {
            "period": None,
            "strength": 0.0,
            "pattern": None,
        }

    # Remove trend first
    trend_info = detect_trend(y_arr, method="linear")
    detrended = y_arr - trend_info["trend"]

    # Detect seasonal period
    period = _detect_seasonal_period(detrended, max_period)

    if period is None:
        return {
            "period": None,
            "strength": 0.0,
            "pattern": None,
        }

    # Extract seasonal pattern
    n = len(detrended)
    n_periods = n // period
    seasonal_pattern = np.zeros(period)

    for i in range(period):
        indices = np.arange(i, n, period)
        if len(indices) > 0:
            seasonal_pattern[i] = np.mean(detrended[indices])

    # Center pattern
    seasonal_pattern = seasonal_pattern - np.mean(seasonal_pattern)

    # Calculate strength as variance explained
    seasonal_component = np.tile(seasonal_pattern, n_periods + 1)[:n]
    var_seasonal = np.var(seasonal_component)
    var_total = np.var(detrended)
    strength = var_seasonal / var_total if var_total > 0 else 0.0

    return {
        "period": int(period),
        "strength": float(strength),
        "pattern": seasonal_pattern.tolist(),
    }


class DecomposeTransformer(BaseTransformer):
    """Decompose time series into trend, seasonal, and residual components."""

    def __init__(
        self,
        method: str = "moving_average",
        seasonal_period: Optional[int] = None,
        trend_window: Optional[int] = None,
    ):
        """Initialize decomposition transformer.

        Args:
            method: Decomposition method: 'moving_average' or 'stl'.
            seasonal_period: Seasonal period
                (auto-detected if not specified).
            trend_window: Window size for trend extraction
                (auto-determined if not specified).
        """
        super().__init__()
        self.method = method
        self.seasonal_period = seasonal_period
        self.trend_window = trend_window

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "DecomposeTransformer":
        """Fit the decomposition transformer.

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if isinstance(y, pd.Series):
            self.y_ = y.values
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            self.y_ = y.iloc[:, 0].values
        else:
            self.y_ = np.asarray(y, dtype=float)

        # Remove invalid values
        valid_mask = np.isfinite(self.y_)
        self.y_ = self.y_[valid_mask]

        if len(self.y_) < 10:
            raise ValueError("Need at least 10 data points for decomposition")

        # Store decomposition components
        self.components_ = self._decompose()

        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.Series:
        """Return residual component (original - trend - seasonal).

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).

        Returns:
            Residual component as Series.
        """
        self._check_is_fitted()
        return pd.Series(self.components_["residual"])

    def inverse_transform(self, y: Any, X: Optional[Any] = None) -> pd.Series:
        """Reconstruct original from residual by adding trend and seasonal.

        Args:
            y: Residual component.
            X: Optional exogenous data (ignored).

        Returns:
            Reconstructed original series.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            residual = y.values
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            residual = y.iloc[:, 0].values
        else:
            residual = np.asarray(y)

        # Reconstruct: residual + trend + seasonal
        reconstructed = (
            residual + self.components_["trend"] + self.components_["seasonal"]
        )
        return pd.Series(reconstructed)

    def _decompose(self) -> Dict[str, np.ndarray]:
        """Perform decomposition."""
        n = len(self.y_)

        # Auto-determine trend window
        if self.trend_window is None:
            trend_window = max(3, n // 10)
        else:
            trend_window = self.trend_window

        # Extract trend using moving average
        if not HAS_SCIPY:
            # Fallback: simple moving average
            trend = np.convolve(
                self.y_, np.ones(trend_window) / trend_window, mode="same"
            )
        else:
            trend = uniform_filter1d(self.y_, size=trend_window, mode="nearest")

        # Detrend
        detrended = self.y_ - trend

        # Extract seasonal component
        if self.seasonal_period is None:
            seasonal_period = _detect_seasonal_period(detrended)
        else:
            seasonal_period = self.seasonal_period

        seasonal = np.zeros_like(self.y_)
        if seasonal_period and seasonal_period > 1:
            # Average over seasonal periods
            n_periods = n // seasonal_period
            if n_periods > 0:
                seasonal_pattern = np.zeros(seasonal_period)
                for i in range(seasonal_period):
                    indices = np.arange(i, n, seasonal_period)
                    if len(indices) > 0:
                        seasonal_pattern[i] = np.mean(detrended[indices])

                # Center seasonal pattern
                seasonal_pattern = seasonal_pattern - np.mean(seasonal_pattern)

                # Replicate pattern
                for i in range(n):
                    seasonal[i] = seasonal_pattern[i % seasonal_period]

        # Residual
        residual = detrended - seasonal

        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "original": self.y_,
        }

    def get_components(self) -> Dict[str, np.ndarray]:
        """Get decomposition components.

        Returns:
            Dictionary with 'trend', 'seasonal', 'residual', and 'original' components.
        """
        self._check_is_fitted()
        return self.components_


class DetrendTransformer(BaseTransformer):
    """Remove trend from time series."""

    def __init__(self, method: str = "linear"):
        """Initialize detrend transformer.

        Args:
            method: Trend removal method: 'linear', 'polynomial', or 'moving_average'.
        """
        super().__init__()
        self.method = method

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "DetrendTransformer":
        """Fit the detrend transformer.

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if isinstance(y, pd.Series):
            self.y_ = y.values
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            self.y_ = y.iloc[:, 0].values
        else:
            self.y_ = np.asarray(y, dtype=float)

        valid_mask = np.isfinite(self.y_)
        self.y_ = self.y_[valid_mask]

        if len(self.y_) < 3:
            raise ValueError("Need at least 3 data points")

        # Detect and store trend
        trend_info = detect_trend(self.y_, method=self.method)
        self.trend_ = trend_info["trend"]

        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.Series:
        """Remove trend from time series.

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).

        Returns:
            Detrended series.
        """
        self._check_is_fitted()
        detrended = self.y_ - self.trend_
        return pd.Series(detrended)

    def inverse_transform(self, y: Any, X: Optional[Any] = None) -> pd.Series:
        """Add trend back to detrended series.

        Args:
            y: Detrended series.
            X: Optional exogenous data (ignored).

        Returns:
            Series with trend restored.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            detrended = y.values
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            detrended = y.iloc[:, 0].values
        else:
            detrended = np.asarray(y)

        reconstructed = detrended + self.trend_
        return pd.Series(reconstructed)


class DeseasonalizeTransformer(BaseTransformer):
    """Remove seasonality from time series."""

    def __init__(self, seasonal_period: Optional[int] = None, max_period: int = 50):
        """Initialize deseasonalize transformer.

        Args:
            seasonal_period: Seasonal period (auto-detected if not specified).
            max_period: Maximum period to check for auto-detection.
        """
        super().__init__()
        self.seasonal_period = seasonal_period
        self.max_period = max_period

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "DeseasonalizeTransformer":
        """Fit the deseasonalize transformer.

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if isinstance(y, pd.Series):
            self.y_ = y.values
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            self.y_ = y.iloc[:, 0].values
        else:
            self.y_ = np.asarray(y, dtype=float)

        valid_mask = np.isfinite(self.y_)
        self.y_ = self.y_[valid_mask]

        # Remove trend first for better seasonality detection
        trend_info = detect_trend(self.y_, method="linear")
        detrended = self.y_ - trend_info["trend"]

        # Detect seasonal period if not provided
        if self.seasonal_period is None:
            period = _detect_seasonal_period(detrended, self.max_period)
            if period is None:
                # No seasonality detected
                self.seasonal_ = np.zeros_like(self.y_)
                self.seasonal_period = None
            else:
                self.seasonal_period = period
        else:
            period = self.seasonal_period

        # Extract seasonal pattern
        n = len(detrended)
        seasonal = np.zeros_like(self.y_)

        if period and period > 1:
            n_periods = n // period
            if n_periods > 0:
                seasonal_pattern = np.zeros(period)
                for i in range(period):
                    indices = np.arange(i, n, period)
                    if len(indices) > 0:
                        seasonal_pattern[i] = np.mean(detrended[indices])

                # Center pattern
                seasonal_pattern = seasonal_pattern - np.mean(seasonal_pattern)

                # Replicate pattern
                for i in range(n):
                    seasonal[i] = seasonal_pattern[i % period]

        self.seasonal_ = seasonal

        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.Series:
        """Remove seasonality from time series.

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).

        Returns:
            Deseasonalized series.
        """
        self._check_is_fitted()
        deseasonalized = self.y_ - self.seasonal_
        return pd.Series(deseasonalized)

    def inverse_transform(self, y: Any, X: Optional[Any] = None) -> pd.Series:
        """Add seasonality back to deseasonalized series.

        Args:
            y: Deseasonalized series.
            X: Optional exogenous data (ignored).

        Returns:
            Series with seasonality restored.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            deseasonalized = y.values
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            deseasonalized = y.iloc[:, 0].values
        else:
            deseasonalized = np.asarray(y)

        reconstructed = deseasonalized + self.seasonal_
        return pd.Series(reconstructed)
