"""Moving average forecaster implementations."""

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
import pandas as pd

from timesmith.core.base import BaseForecaster
from timesmith.core.tags import set_tags
from timesmith.exceptions import DataError
from timesmith.results.forecast import Forecast
from timesmith.utils.ts_utils import ensure_datetime_index

if TYPE_CHECKING:
    from timesmith.typing import SeriesLike, TableLike

logger = logging.getLogger(__name__)


class SimpleMovingAverageForecaster(BaseForecaster):
    """Simple moving average forecaster.

    Uses the average of the last N values as the forecast.
    """

    def __init__(self, window: int = 7):
        """Initialize simple moving average forecaster.

        Args:
            window: Window size for moving average.
        """
        super().__init__()
        self.window = window
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
            supports_panel=False,
            requires_fh=True,
        )

    def fit(
        self,
        y: Union["SeriesLike", Any],
        X: Optional[Union["TableLike", Any]] = None,
        **fit_params: Any,
    ) -> "SimpleMovingAverageForecaster":
        """Fit the forecaster (computes moving average).

        Args:
            y: Target time series.
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
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        series = ensure_datetime_index(series)

        # Validate data
        if len(series) == 0:
            raise DataError("Input series cannot be empty")

        # Check for window size
        if self.window > len(series):
            raise ValueError(
                f"Window size ({self.window}) cannot be larger than data length ({len(series)})"
            )

        if self.window < 1:
            raise ValueError(f"Window size must be at least 1, got {self.window}")

        self.train_index_ = series.index
        rolling_mean = series.rolling(window=self.window).mean()
        self.last_ma_value_ = rolling_mean.iloc[-1]

        # Check if result is NaN (e.g., all NaN values in window)
        if pd.isna(self.last_ma_value_):
            # Try to use available data
            valid_values = series.dropna()
            if len(valid_values) == 0:
                raise DataError(
                    "All values in input series are NaN. Cannot compute moving average."
                )
            self.last_ma_value_ = valid_values.mean()
            logger.warning(
                f"Moving average resulted in NaN (possibly due to missing values). "
                f"Using mean of available data ({self.last_ma_value_}) instead."
            )

        self._is_fitted = True
        return self

    def predict(
        self,
        fh: Union[int, list, Any],
        X: Optional[Union["TableLike", Any]] = None,
        **predict_params: Any,
    ) -> Forecast:
        """Generate forecast.

        Args:
            fh: Forecast horizon (integer or array).
            X: Optional exogenous data (ignored).
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast object with predictions.
        """
        self._check_is_fitted()

        # Parse forecast horizon and create index
        n_periods = self._parse_forecast_horizon(fh)
        forecast_index = self._create_forecast_index(self.train_index_, n_periods)

        # Forecast is constant (last MA value)
        y_pred = pd.Series([self.last_ma_value_] * n_periods, index=forecast_index)

        return Forecast(y_pred=y_pred, fh=fh)


class ExponentialMovingAverageForecaster(BaseForecaster):
    """Exponential moving average forecaster.

    Uses exponential smoothing with alpha parameter.
    """

    def __init__(self, alpha: float = 0.3):
        """Initialize exponential moving average forecaster.

        Args:
            alpha: Smoothing factor (0 < alpha <= 1).
        """
        super().__init__()
        self.alpha = alpha
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
            supports_panel=False,
            requires_fh=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "ExponentialMovingAverageForecaster":
        """Fit the forecaster (computes EMA).

        Args:
            y: Target time series.
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
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        series = ensure_datetime_index(series)

        # Validate data
        if len(series) == 0:
            raise DataError("Input series cannot be empty")

        # Validate alpha parameter
        if not (0 < self.alpha <= 1):
            raise ValueError(f"Alpha must be in range (0, 1], got {self.alpha}")

        self.train_index_ = series.index
        ema_result = series.ewm(alpha=self.alpha, adjust=False).mean()
        self.last_ema_value_ = ema_result.iloc[-1]

        # Check if result is NaN
        if pd.isna(self.last_ema_value_):
            valid_values = series.dropna()
            if len(valid_values) == 0:
                raise DataError(
                    "All values in input series are NaN. Cannot compute exponential moving average."
                )
            self.last_ema_value_ = valid_values.mean()
            logger.warning(
                f"Exponential moving average resulted in NaN. "
                f"Using mean of available data ({self.last_ema_value_}) instead."
            )

        self._is_fitted = True
        return self

    def predict(
        self,
        fh: Union[int, list, Any],
        X: Optional[Union["TableLike", Any]] = None,
        **predict_params: Any,
    ) -> Forecast:
        """Generate forecast.

        Args:
            fh: Forecast horizon (integer or array).
            X: Optional exogenous data (ignored).
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast object with predictions.
        """
        self._check_is_fitted()

        # Parse forecast horizon and create index
        n_periods = self._parse_forecast_horizon(fh)
        forecast_index = self._create_forecast_index(self.train_index_, n_periods)

        # Forecast is constant (last EMA value)
        y_pred = pd.Series([self.last_ema_value_] * n_periods, index=forecast_index)

        return Forecast(y_pred=y_pred, fh=fh)


class WeightedMovingAverageForecaster(BaseForecaster):
    """Weighted moving average forecaster.

    Uses weighted average of the last N values.
    """

    def __init__(self, window: int = 7, weights: Optional[List[float]] = None):
        """Initialize weighted moving average forecaster.

        Args:
            window: Window size for moving average.
            weights: Optional weights for each position (defaults to equal weights).
        """
        super().__init__()
        self.window = window
        self.weights = weights if weights is not None else [1.0 / window] * window
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
            supports_panel=False,
            requires_fh=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "WeightedMovingAverageForecaster":
        """Fit the forecaster (computes weighted MA).

        Args:
            y: Target time series.
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
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        series = ensure_datetime_index(series)

        # Validate data
        if len(series) == 0:
            raise DataError("Input series cannot be empty")

        # Validate window size
        if self.window > len(series):
            raise ValueError(
                f"Window size ({self.window}) cannot be larger than data length ({len(series)})"
            )

        if self.window < 1:
            raise ValueError(f"Window size must be at least 1, got {self.window}")

        # Validate weights
        if len(self.weights) != self.window:
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match window size ({self.window})"
            )

        self.train_index_ = series.index

        # Compute weighted moving average
        last_window = series.iloc[-self.window :].values

        # Check for NaN values in window
        if np.any(np.isnan(last_window)):
            valid_mask = ~np.isnan(last_window)
            if not np.any(valid_mask):
                raise DataError(
                    "All values in the last window are NaN. Cannot compute weighted moving average."
                )
            # Use only valid values and adjust weights proportionally
            valid_values = last_window[valid_mask]
            valid_weights = np.array(self.weights)[valid_mask]
            valid_weights = valid_weights / valid_weights.sum()  # Normalize
            self.last_wma_value_ = np.dot(valid_values, valid_weights)
            logger.warning(
                f"NaN values found in window. Using weighted average of {len(valid_values)} valid values."
            )
        else:
            self.last_wma_value_ = np.dot(last_window, self.weights)

        self._is_fitted = True
        return self

    def predict(
        self,
        fh: Union[int, list, Any],
        X: Optional[Union["TableLike", Any]] = None,
        **predict_params: Any,
    ) -> Forecast:
        """Generate forecast.

        Args:
            fh: Forecast horizon (integer or array).
            X: Optional exogenous data (ignored).
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast object with predictions.
        """
        self._check_is_fitted()

        # Parse forecast horizon and create index
        n_periods = self._parse_forecast_horizon(fh)
        forecast_index = self._create_forecast_index(self.train_index_, n_periods)

        # Forecast is constant (last WMA value)
        y_pred = pd.Series([self.last_wma_value_] * n_periods, index=forecast_index)

        return Forecast(y_pred=y_pred, fh=fh)
