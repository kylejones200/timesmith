"""Linear trend forecaster for time series."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseForecaster
from timesmith.core.tags import set_tags
from timesmith.results.forecast import Forecast

logger = logging.getLogger(__name__)


class LinearTrendForecaster(BaseForecaster):
    """Forecaster using linear trend extrapolation.

    Fits a linear trend to historical data and extrapolates forward.
    """

    def __init__(self):
        """Initialize linear trend forecaster."""
        super().__init__()

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="ForecastLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "LinearTrendForecaster":
        """Fit linear trend to training data.

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if X is not None:
            logger.warning(
                "Exogenous data X not yet supported in LinearTrendForecaster"
            )

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

        if len(self.y_) < 2:
            raise ValueError("Need at least 2 data points for linear trend")

        # Fit linear trend
        x = np.arange(len(self.y_))
        self.coeffs_ = np.polyfit(x, self.y_, 1)  # [slope, intercept]

        self._is_fitted = True
        return self

    def predict(
        self, fh: Any, X: Optional[Any] = None, **predict_params: Any
    ) -> Forecast:
        """Generate forecasts using linear trend extrapolation.

        Args:
            fh: Forecast horizon (integer or array-like).
            X: Optional exogenous data (ignored).
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast results.
        """
        self._check_is_fitted()

        if X is not None:
            logger.warning(
                "Exogenous data X not yet supported in LinearTrendForecaster"
            )

        # Convert fh to integer
        if isinstance(fh, (int, np.integer)):
            n_steps = int(fh)
            fh_arr = np.arange(1, n_steps + 1)
        elif isinstance(fh, (list, np.ndarray, pd.Index)):
            n_steps = len(fh)
            fh_arr = np.asarray(fh)
        else:
            raise ValueError(f"Unsupported fh type: {type(fh)}")

        # Extrapolate
        future_x = np.arange(len(self.y_), len(self.y_) + n_steps)
        forecast_values = np.polyval(self.coeffs_, future_x)

        # Ensure non-negative (common for production/rate data)
        forecast_values = np.maximum(forecast_values, 0)

        # Convert to Series
        if isinstance(self.index_, pd.DatetimeIndex):
            # Try to infer frequency
            if len(self.index_) > 1:
                freq = pd.infer_freq(self.index_) or pd.Timedelta(days=1)
            else:
                freq = pd.Timedelta(days=1)

            forecast_index = pd.date_range(
                start=self.index_[-1] + freq, periods=n_steps, freq=freq
            )
        else:
            forecast_index = np.arange(len(self.y_), len(self.y_) + n_steps)

        y_pred_series = pd.Series(forecast_values, index=forecast_index)

        return Forecast(
            y_pred=y_pred_series,
            fh=fh,
            metadata={
                "slope": float(self.coeffs_[0]),
                "intercept": float(self.coeffs_[1]),
                "method": "linear_trend",
            },
        )
