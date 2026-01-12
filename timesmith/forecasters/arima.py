"""ARIMA forecaster implementation."""

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import pandas as pd

from timesmith.core.base import BaseForecaster
from timesmith.core.tags import set_tags
from timesmith.results.forecast import Forecast
from timesmith.utils.ts_utils import detect_frequency, ensure_datetime_index

if TYPE_CHECKING:
    from timesmith.typing import SeriesLike, TableLike

logger = logging.getLogger(__name__)

try:
    from pmdarima import auto_arima
except ImportError:
    auto_arima = None
    logger.warning(
        "pmdarima not installed. ARIMAForecaster will not work. "
        "Install with: pip install pmdarima"
    )


class ARIMAForecaster(BaseForecaster):
    """ARIMA forecaster using auto_arima for automatic order selection.

    Wraps pmdarima.auto_arima to provide a BaseForecaster interface.
    """

    def __init__(
        self,
        start_p: int = 0,
        start_q: int = 0,
        max_p: int = 5,
        max_q: int = 5,
        seasonal: bool = False,
        stepwise: bool = True,
        suppress_warnings: bool = True,
        error_action: str = "ignore",
        **kwargs,
    ):
        """Initialize ARIMA forecaster.

        Args:
            start_p: Starting value for p parameter.
            start_q: Starting value for q parameter.
            max_p: Maximum value for p parameter.
            max_q: Maximum value for q parameter.
            seasonal: Whether to include seasonal component.
            stepwise: Whether to use stepwise selection.
            suppress_warnings: Whether to suppress warnings.
            error_action: Action on error ('ignore', 'warn', 'raise').
            **kwargs: Additional arguments passed to auto_arima.
        """
        if auto_arima is None:
            raise ImportError(
                "pmdarima is required for ARIMAForecaster. "
                "Install with: pip install pmdarima"
            )

        super().__init__()
        self.start_p = start_p
        self.start_q = start_q
        self.max_p = max_p
        self.max_q = max_q
        self.seasonal = seasonal
        self.stepwise = stepwise
        self.suppress_warnings = suppress_warnings
        self.error_action = error_action
        self.kwargs = kwargs

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
    ) -> "ARIMAForecaster":
        """Fit ARIMA model to time series.

        Args:
            y: Target time series.
            X: Optional exogenous data (not yet supported).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if X is not None:
            logger.warning(
                "Exogenous variables (X) not yet supported for ARIMAForecaster"
            )

        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        series = ensure_datetime_index(series)
        self.train_index_ = series.index

        # Default auto_arima parameters
        fit_kwargs = {
            "start_p": self.start_p,
            "start_q": self.start_q,
            "max_p": self.max_p,
            "max_q": self.max_q,
            "seasonal": self.seasonal,
            "stepwise": self.stepwise,
            "suppress_warnings": self.suppress_warnings,
            "error_action": self.error_action,
            **self.kwargs,
            **fit_params,
        }

        # Fit model (use values only, not index)
        self.model_ = auto_arima(series.values, **fit_kwargs)
        self.order_ = self.model_.order

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
            X: Optional exogenous data (not yet supported).
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast object with predictions.
        """
        self._check_is_fitted()

        if X is not None:
            logger.warning(
                "Exogenous variables (X) not yet supported for ARIMAForecaster"
            )

        # Convert fh to integer
        if isinstance(fh, (list, np.ndarray)):
            n_periods = len(fh)
        elif isinstance(fh, int):
            n_periods = fh
        else:
            n_periods = int(fh)

        # Generate forecast with confidence intervals
        forecast, conf_int = self.model_.predict(
            n_periods=n_periods,
            return_conf_int=True,
            alpha=0.05,  # 95% confidence interval
        )

        # Create forecast index
        last_date = pd.Timestamp(self.train_index_[-1])
        freq = detect_frequency(pd.Series(index=self.train_index_))

        # Create forecast index
        if isinstance(freq, str):
            next_date = last_date + pd.tseries.frequencies.to_offset(freq)
            forecast_index = pd.date_range(
                start=next_date, periods=n_periods, freq=freq
            )
        else:
            # Fallback: estimate from spacing
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

        y_pred = pd.Series(forecast, index=forecast_index)

        # Create confidence intervals DataFrame
        y_int = pd.DataFrame(
            conf_int,
            index=forecast_index,
            columns=["lower", "upper"],
        )

        return Forecast(y_pred=y_pred, fh=fh, y_int=y_int)

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
        # ARIMA already returns intervals, just adjust alpha
        alpha = 1.0 - coverage
        predict_params["alpha"] = alpha
        return self.predict(fh, X, **predict_params)

    def get_order(self) -> tuple:
        """Get ARIMA order (p, d, q).

        Returns:
            Tuple of (p, d, q) order.
        """
        self._check_is_fitted()
        return self.order_
