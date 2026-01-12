"""Exponential smoothing forecaster implementation."""

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import pandas as pd

from timesmith.core.base import BaseForecaster
from timesmith.core.tags import set_tags
from timesmith.results.forecast import Forecast
from timesmith.utils.ts_utils import ensure_datetime_index

if TYPE_CHECKING:
    from timesmith.typing import SeriesLike, TableLike

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    ExponentialSmoothing = None
    logger.warning(
        "statsmodels not installed. ExponentialSmoothingForecaster will not work. "
        "Install with: pip install statsmodels"
    )


class ExponentialSmoothingForecaster(BaseForecaster):
    """Exponential smoothing forecaster using Holt-Winters method.

    Wraps statsmodels.tsa.holtwinters.ExponentialSmoothing.
    """

    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = "add",
        seasonal_periods: Optional[int] = None,
        optimized: bool = True,
    ):
        """Initialize exponential smoothing forecaster.

        Args:
            trend: Type of trend component ('add', 'mul', None).
            seasonal: Type of seasonal component ('add', 'mul', None).
            seasonal_periods: Number of periods in a season.
            optimized: Whether to optimize parameters.
        """
        if ExponentialSmoothing is None:
            raise ImportError(
                "statsmodels is required for ExponentialSmoothingForecaster. "
                "Install with: pip install statsmodels"
            )

        super().__init__()
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.optimized = optimized

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
    ) -> "ExponentialSmoothingForecaster":
        """Fit exponential smoothing model.

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if X is not None:
            logger.warning(
                "Exogenous variables (X) not yet supported for ExponentialSmoothingForecaster"
            )

        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        series = ensure_datetime_index(series)
        self.train_index_ = series.index

        # Fit model
        self.model_ = ExponentialSmoothing(
            series,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        ).fit(optimized=self.optimized)

        # Store residuals for confidence intervals
        self.residuals_ = series - self.model_.fittedvalues
        self.sigma_ = (
            float(self.residuals_.std(ddof=1)) if len(self.residuals_) > 0 else 0.0
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

        if X is not None:
            logger.warning(
                "Exogenous variables (X) not yet supported for ExponentialSmoothingForecaster"
            )

        # Convert fh to integer
        if isinstance(fh, (list, np.ndarray)):
            n_periods = len(fh)
        elif isinstance(fh, int):
            n_periods = fh
        else:
            n_periods = int(fh)

        # Generate forecast
        forecast = self.model_.forecast(n_periods)

        return Forecast(y_pred=forecast, fh=fh)

    def predict_interval(
        self,
        fh: Any,
        X: Optional[Any] = None,
        coverage: float = 0.9,
        **predict_params: Any,
    ) -> Forecast:
        """Generate forecast with prediction intervals.

        Args:
            fh: Forecast horizon.
            X: Optional exogenous data.
            coverage: Coverage level (e.g., 0.9 for 90%).
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast with intervals.
        """
        self._check_is_fitted()

        # Get point forecast
        forecast = self.predict(fh, X, **predict_params)

        # Calculate intervals using z-score
        from scipy import stats

        z_score = stats.norm.ppf((1 + coverage) / 2)
        margin = z_score * self.sigma_

        y_int = pd.DataFrame(
            {
                "lower": forecast.y_pred - margin,
                "upper": forecast.y_pred + margin,
            },
            index=forecast.y_pred.index,
        )

        return Forecast(y_pred=forecast.y_pred, fh=fh, y_int=y_int)
