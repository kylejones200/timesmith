"""Prophet forecaster implementation."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseForecaster
from timesmith.core.tags import set_tags
from timesmith.results.forecast import Forecast
from timesmith.utils.ts_utils import ensure_datetime_index

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet

    HAS_PROPHET = True
except ImportError:
    Prophet = None
    HAS_PROPHET = False
    logger.warning(
        "prophet not installed. ProphetForecaster will not work. "
        "Install with: pip install prophet"
    )


class ProphetForecaster(BaseForecaster):
    """Prophet forecaster using Facebook's Prophet.

    Wraps prophet.Prophet for automatic forecasting of business time series.
    """

    def __init__(
        self,
        yearly_seasonality: Any = "auto",
        weekly_seasonality: Any = "auto",
        daily_seasonality: bool = False,
        seasonality_mode: str = "additive",
        growth: str = "linear",
        **prophet_params: Any,
    ):
        """Initialize Prophet forecaster.

        Args:
            yearly_seasonality: Fit yearly seasonality ('auto', True, False, or int).
            weekly_seasonality: Fit weekly seasonality ('auto', True, False, or int).
            daily_seasonality: Fit daily seasonality (default: False).
            seasonality_mode: 'additive' or 'multiplicative' (default: 'additive').
            growth: 'linear' or 'logistic' (default: 'linear').
            **prophet_params: Additional Prophet parameters.
        """
        if not HAS_PROPHET:
            raise ImportError(
                "prophet is required for ProphetForecaster. "
                "Install with: pip install prophet"
            )

        super().__init__()
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.growth = growth
        self.prophet_params = prophet_params

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="ForecastLike",
            handles_missing=False,
            requires_sorted_index=True,
            supports_panel=False,
            requires_fh=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "ProphetForecaster":
        """Fit Prophet model.

        Args:
            y: Target time series.
            X: Optional exogenous data (not yet supported).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if X is not None:
            logger.warning(
                "Exogenous variables (X) not yet supported for ProphetForecaster"
            )

        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        series = ensure_datetime_index(series)
        self.train_index_ = series.index

        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({"ds": series.index, "y": series.values})

        # Create and fit Prophet model
        self.model_ = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            growth=self.growth,
            **self.prophet_params,
        )

        self.model_.fit(df)

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

        if X is not None:
            logger.warning(
                "Exogenous variables (X) not yet supported for ProphetForecaster"
            )

        # Convert fh to integer
        if isinstance(fh, (list, np.ndarray)):
            n_periods = len(fh)
            if isinstance(fh[0], (pd.Timestamp, pd.DatetimeIndex)):
                # fh is array of dates
                future_dates = pd.to_datetime(fh)
            else:
                # fh is array of integers - create future dates
                freq = pd.infer_freq(self.train_index_) or "D"
                last_date = self.train_index_[-1]
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=n_periods,
                    freq=freq,
                )
        elif isinstance(fh, int):
            n_periods = fh
            freq = pd.infer_freq(self.train_index_) or "D"
            last_date = self.train_index_[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=n_periods,
                freq=freq,
            )
        else:
            n_periods = int(fh)
            freq = pd.infer_freq(self.train_index_) or "D"
            last_date = self.train_index_[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=n_periods,
                freq=freq,
            )

        # Create future dataframe
        future_df = pd.DataFrame({"ds": future_dates})

        # Generate forecast
        forecast_df = self.model_.predict(future_df)

        # Extract predictions
        y_pred = pd.Series(
            forecast_df["yhat"].values,
            index=future_dates,
        )

        return Forecast(y_pred=y_pred, fh=fh)

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

        # Convert fh to integer
        if isinstance(fh, (list, np.ndarray)):
            n_periods = len(fh)
            if isinstance(fh[0], (pd.Timestamp, pd.DatetimeIndex)):
                future_dates = pd.to_datetime(fh)
            else:
                freq = pd.infer_freq(self.train_index_) or "D"
                last_date = self.train_index_[-1]
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=n_periods,
                    freq=freq,
                )
        elif isinstance(fh, int):
            n_periods = fh
            freq = pd.infer_freq(self.train_index_) or "D"
            last_date = self.train_index_[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=n_periods,
                freq=freq,
            )
        else:
            n_periods = int(fh)
            freq = pd.infer_freq(self.train_index_) or "D"
            last_date = self.train_index_[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=n_periods,
                freq=freq,
            )

        # Create future dataframe
        future_df = pd.DataFrame({"ds": future_dates})

        # Generate forecast with intervals
        forecast_df = self.model_.predict(future_df)

        # Calculate intervals based on coverage
        # Prophet provides yhat_lower and yhat_upper, but we need to adjust for coverage
        lower_col = "yhat_lower"
        upper_col = "yhat_upper"

        # Prophet uses 80% intervals by default, we need to adjust
        # For simplicity, we'll use the provided intervals and scale them
        # In practice, you might want to refit with interval_width parameter
        if coverage != 0.8:
            # Use z-score to approximate intervals
            from scipy import stats

            z_score = stats.norm.ppf((1 + coverage) / 2)
            z_80 = stats.norm.ppf(0.9)  # 80% interval z-score

            # Estimate uncertainty from 80% intervals
            uncertainty = (forecast_df[upper_col] - forecast_df[lower_col]) / (2 * z_80)
            margin = z_score * uncertainty

            y_int = pd.DataFrame(
                {
                    "lower": forecast_df["yhat"] - margin,
                    "upper": forecast_df["yhat"] + margin,
                },
                index=future_dates,
            )
        else:
            y_int = pd.DataFrame(
                {
                    "lower": forecast_df[lower_col].values,
                    "upper": forecast_df[upper_col].values,
                },
                index=future_dates,
            )

        y_pred = pd.Series(
            forecast_df["yhat"].values,
            index=future_dates,
        )

        return Forecast(y_pred=y_pred, fh=fh, y_int=y_int)
