"""Vector Autoregression (VAR) forecaster implementation."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseForecaster
from timesmith.core.tags import set_tags
from timesmith.results.forecast import Forecast

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller

    HAS_STATSMODELS = True
except ImportError:
    VAR = None
    adfuller = None
    HAS_STATSMODELS = False
    logger.warning(
        "statsmodels not installed. VARForecaster will not work. "
        "Install with: pip install statsmodels"
    )


class VARForecaster(BaseForecaster):
    """Vector Autoregression (VAR) forecaster for multivariate time series.

    Fits a VAR model to multiple interdependent time series and generates
    forecasts for all series simultaneously.
    """

    def __init__(
        self,
        maxlags: int = 10,
        ic: str = "aic",
        force_differencing: bool = False,
        verbose: bool = False,
    ):
        """Initialize VAR forecaster.

        Args:
            maxlags: Maximum number of lags to consider for model selection.
            ic: Information criterion for lag selection ('aic', 'bic', 'fpe', 'hqic').
            force_differencing: Whether to force first-order differencing (default: False).
            verbose: Whether to print diagnostic information.
        """
        if not HAS_STATSMODELS:
            raise ImportError(
                "statsmodels is required for VARForecaster. "
                "Install with: pip install statsmodels"
            )

        super().__init__()
        self.maxlags = maxlags
        self.ic = ic
        self.force_differencing = force_differencing
        self.verbose = verbose

        set_tags(
            self,
            scitype_input="PanelLike",
            scitype_output="PanelLike",
            handles_missing=False,
            requires_sorted_index=True,
            supports_panel=True,
            requires_fh=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "VARForecaster":
        """Fit VAR model.

        Args:
            y: Multivariate time series (DataFrame with multiple columns or PanelLike).
            X: Optional exogenous data (not yet supported).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if X is not None:
            logger.warning(
                "Exogenous variables (X) not yet supported for VARForecaster"
            )

        # Convert to DataFrame if needed
        if isinstance(y, pd.Series):
            raise ValueError(
                "VARForecaster requires multivariate data (DataFrame with multiple columns)"
            )
        elif isinstance(y, pd.DataFrame):
            df = y.copy()
        else:
            df = pd.DataFrame(y)

        if df.shape[1] < 2:
            raise ValueError("VARForecaster requires at least 2 time series (columns)")

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if isinstance(df.index, pd.RangeIndex):
                # Create a default datetime index
                df.index = pd.date_range(start="2000-01-01", periods=len(df), freq="D")
            else:
                df.index = pd.to_datetime(df.index)

        self.train_index_ = df.index
        self.column_names_ = df.columns.tolist()

        # Check stationarity and apply differencing if needed
        if self.force_differencing:
            df_stationary = df.diff().dropna()
            self.differenced_ = True
            if self.verbose:
                logger.info("Applied first-order differencing")
        else:
            # Check stationarity for each series
            df_stationary = df.copy()
            self.differenced_ = False
            needs_differencing = False

            for col in df.columns:
                result = adfuller(df[col].dropna())
                if result[1] > 0.05:
                    if self.verbose:
                        logger.info(f"{col}: Non-stationary (p-value={result[1]:.4f})")
                    needs_differencing = True
                elif self.verbose:
                    logger.info(f"{col}: Stationary (p-value={result[1]:.4f})")

            if needs_differencing:
                df_stationary = df.diff().dropna()
                self.differenced_ = True
                if self.verbose:
                    logger.info(
                        "Applied first-order differencing due to non-stationarity"
                    )

        # Fit VAR model
        self.model_ = VAR(df_stationary)
        lag_order = self.model_.select_order(maxlags=self.maxlags)

        # Select optimal lag based on information criterion
        if self.ic == "aic":
            optimal_lag = lag_order.aic
        elif self.ic == "bic":
            optimal_lag = lag_order.bic
        elif self.ic == "fpe":
            optimal_lag = lag_order.fpe
        elif self.ic == "hqic":
            optimal_lag = lag_order.hqic
        else:
            optimal_lag = lag_order.aic

        self.optimal_lag_ = optimal_lag

        if self.verbose:
            logger.info(f"Optimal lag order: {optimal_lag}")

        # Fit model with optimal lag
        self.fitted_model_ = self.model_.fit(optimal_lag)

        if self.verbose:
            logger.info(f"VAR({optimal_lag}) model fitted")

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
            Forecast object with predictions for all series.
        """
        self._check_is_fitted()

        if X is not None:
            logger.warning(
                "Exogenous variables (X) not yet supported for VARForecaster"
            )

        # Convert fh to integer
        if isinstance(fh, (list, np.ndarray)):
            n_periods = len(fh)
        elif isinstance(fh, int):
            n_periods = fh
        else:
            n_periods = int(fh)

        # Generate forecast
        # Use the last optimal_lag_ observations as starting point
        forecast = self.fitted_model_.forecast(
            self.fitted_model_.y[-self.optimal_lag_ :], steps=n_periods
        )

        # Convert to DataFrame
        forecast_df = pd.DataFrame(
            forecast,
            columns=self.column_names_,
        )

        # Create future index
        freq = pd.infer_freq(self.train_index_) or "D"
        last_date = self.train_index_[-1]
        future_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=n_periods,
            freq=freq,
        )
        forecast_df.index = future_index

        # Undo differencing if needed
        if self.differenced_:
            # Get last values before differencing
            last_values = self.fitted_model_.y[-1]
            # Cumulative sum to undo differencing
            forecast_df = forecast_df.cumsum() + last_values

        # Return as Forecast object
        # For multivariate, y_pred is a DataFrame
        return Forecast(y_pred=forecast_df, fh=fh)

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

        # VAR models don't provide built-in prediction intervals
        # We'll estimate them from residuals
        residuals = self.fitted_model_.resid
        n_vars = len(self.column_names_)

        # Calculate standard errors for each variable
        std_errors = {}
        for i, col in enumerate(self.column_names_):
            std_errors[col] = (
                np.std(residuals.iloc[:, i])
                if hasattr(residuals, "iloc")
                else np.std(residuals[:, i])
            )

        # Use z-score for intervals
        from scipy import stats

        z_score = stats.norm.ppf((1 + coverage) / 2)

        # Create intervals DataFrame
        intervals = {}
        for col in self.column_names_:
            margin = z_score * std_errors[col]
            intervals[f"{col}_lower"] = forecast.y_pred[col] - margin
            intervals[f"{col}_upper"] = forecast.y_pred[col] + margin

        y_int = pd.DataFrame(intervals, index=forecast.y_pred.index)

        return Forecast(y_pred=forecast.y_pred, fh=fh, y_int=y_int)
