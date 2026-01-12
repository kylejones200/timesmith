"""Time series filtering transformers."""

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import pandas as pd

from timesmith.core.base import BaseTransformer
from timesmith.core.tags import set_tags

if TYPE_CHECKING:
    from timesmith.typing import SeriesLike, TableLike

logger = logging.getLogger(__name__)

# Optional scipy for advanced filters
try:
    from scipy import signal
    from scipy.signal import butter, filtfilt

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    signal = None
    butter = None
    filtfilt = None
    logger.warning(
        "scipy not available. Advanced filters will use basic methods. "
        "Install with: pip install scipy or pip install timesmith[scipy]"
    )


class ButterworthFilter(BaseTransformer):
    """Butterworth low-pass filter transformer."""

    def __init__(
        self, cutoff_freq: float = 0.1, filter_order: int = 4, btype: str = "low"
    ):
        """Initialize Butterworth filter.

        Args:
            cutoff_freq: Cutoff frequency (normalized, 0-0.5).
            filter_order: Filter order.
            btype: Filter type: 'low', 'high', 'band', 'bandstop'.
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for ButterworthFilter. "
                "Install with: pip install scipy"
            )

        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order
        self.btype = btype

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self,
        y: Union["SeriesLike", Any],
        X: Optional[Union["TableLike", Any]] = None,
        **fit_params: Any,
    ) -> "ButterworthFilter":
        """Fit the filter (no-op, but required by interface).

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

        if len(self.y_) < 10:
            raise ValueError("Need at least 10 data points for filtering")

        # Design filter
        nyquist = 0.5  # Normalized frequency
        normal_cutoff = self.cutoff_freq / nyquist

        if normal_cutoff >= 1.0:
            raise ValueError(
                f"cutoff_freq ({self.cutoff_freq}) must be < 0.5 (Nyquist frequency)"
            )

        self.b_, self.a_ = butter(
            self.filter_order, normal_cutoff, btype=self.btype, analog=False
        )

        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.Series:
        """Apply Butterworth filter to time series.

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).

        Returns:
            Filtered series.
        """
        self._check_is_fitted()

        # Apply filter
        filtered_values = filtfilt(self.b_, self.a_, self.y_)

        return pd.Series(filtered_values, index=self.index_)


class SavitzkyGolayFilter(BaseTransformer):
    """Savitzky-Golay filter transformer for smoothing."""

    def __init__(self, window_length: int = 5, polyorder: int = 2):
        """Initialize Savitzky-Golay filter.

        Args:
            window_length: Window length (must be odd).
            polyorder: Polynomial order.
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for SavitzkyGolayFilter. "
                "Install with: pip install scipy"
            )

        super().__init__()
        self.window_length = window_length
        self.polyorder = polyorder

        # Ensure window_length is odd
        if self.window_length % 2 == 0:
            self.window_length += 1
            logger.warning(
                f"window_length must be odd. Adjusted to {self.window_length}"
            )

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "SavitzkyGolayFilter":
        """Fit the filter (no-op, but required by interface).

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

        if len(self.y_) < self.window_length:
            raise ValueError(
                f"Need at least {self.window_length} data points for filtering"
            )

        # Ensure polyorder is valid
        if self.polyorder >= self.window_length:
            self.polyorder = self.window_length - 1
            logger.warning(f"polyorder adjusted to {self.polyorder}")

        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.Series:
        """Apply Savitzky-Golay filter to time series.

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).

        Returns:
            Filtered series.
        """
        self._check_is_fitted()

        # Apply filter
        filtered_values = signal.savgol_filter(
            self.y_, self.window_length, self.polyorder
        )

        return pd.Series(filtered_values, index=self.index_)
