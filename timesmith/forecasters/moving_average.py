"""Moving average forecaster implementations."""

import logging
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseForecaster
from timesmith.core.tags import set_tags
from timesmith.results.forecast import Forecast
from timesmith.utils.ts_utils import detect_frequency, ensure_datetime_index

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
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
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
        self.train_index_ = series.index
        self.last_ma_value_ = series.rolling(window=self.window).mean().iloc[-1]

        self._is_fitted = True
        return self

    def predict(
        self, fh: Any, X: Optional[Any] = None, **predict_params: Any
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

        # Convert fh to integer
        if isinstance(fh, (list, np.ndarray)):
            n_periods = len(fh)
        elif isinstance(fh, int):
            n_periods = fh
        else:
            n_periods = int(fh)

        # Create forecast index
        last_date = pd.Timestamp(self.train_index_[-1])
        freq = detect_frequency(pd.Series(index=self.train_index_))

        if isinstance(freq, str):
            next_date = last_date + pd.tseries.frequencies.to_offset(freq)
            forecast_index = pd.date_range(
                start=next_date, periods=n_periods, freq=freq
            )
        else:
            if len(self.train_index_) > 1:
                avg_delta = self.train_index_[-1] - self.train_index_[-2]
                next_date = last_date + avg_delta
                forecast_index = pd.date_range(
                    start=next_date, periods=n_periods, freq=avg_delta
                )
            else:
                forecast_index = pd.date_range(
                    start=last_date, periods=n_periods + 1, freq="D"
                )[1:]

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
        self.train_index_ = series.index
        self.last_ema_value_ = (
            series.ewm(alpha=self.alpha, adjust=False).mean().iloc[-1]
        )

        self._is_fitted = True
        return self

    def predict(
        self, fh: Any, X: Optional[Any] = None, **predict_params: Any
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

        # Convert fh to integer
        if isinstance(fh, (list, np.ndarray)):
            n_periods = len(fh)
        elif isinstance(fh, int):
            n_periods = fh
        else:
            n_periods = int(fh)

        # Create forecast index
        last_date = pd.Timestamp(self.train_index_[-1])
        freq = detect_frequency(pd.Series(index=self.train_index_))

        if isinstance(freq, str):
            next_date = last_date + pd.tseries.frequencies.to_offset(freq)
            forecast_index = pd.date_range(
                start=next_date, periods=n_periods, freq=freq
            )
        else:
            if len(self.train_index_) > 1:
                avg_delta = self.train_index_[-1] - self.train_index_[-2]
                next_date = last_date + avg_delta
                forecast_index = pd.date_range(
                    start=next_date, periods=n_periods, freq=avg_delta
                )
            else:
                forecast_index = pd.date_range(
                    start=last_date, periods=n_periods + 1, freq="D"
                )[1:]

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
        self.train_index_ = series.index

        # Compute weighted moving average
        last_window = series.iloc[-self.window :].values
        self.last_wma_value_ = np.dot(last_window, self.weights)

        self._is_fitted = True
        return self

    def predict(
        self, fh: Any, X: Optional[Any] = None, **predict_params: Any
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

        # Convert fh to integer
        if isinstance(fh, (list, np.ndarray)):
            n_periods = len(fh)
        elif isinstance(fh, int):
            n_periods = fh
        else:
            n_periods = int(fh)

        # Create forecast index
        last_date = pd.Timestamp(self.train_index_[-1])
        freq = detect_frequency(pd.Series(index=self.train_index_))

        if isinstance(freq, str):
            next_date = last_date + pd.tseries.frequencies.to_offset(freq)
            forecast_index = pd.date_range(
                start=next_date, periods=n_periods, freq=freq
            )
        else:
            if len(self.train_index_) > 1:
                avg_delta = self.train_index_[-1] - self.train_index_[-2]
                next_date = last_date + avg_delta
                forecast_index = pd.date_range(
                    start=next_date, periods=n_periods, freq=avg_delta
                )
            else:
                forecast_index = pd.date_range(
                    start=last_date, periods=n_periods + 1, freq="D"
                )[1:]

        # Forecast is constant (last WMA value)
        y_pred = pd.Series([self.last_wma_value_] * n_periods, index=forecast_index)

        return Forecast(y_pred=y_pred, fh=fh)
