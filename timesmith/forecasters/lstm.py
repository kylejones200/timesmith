"""LSTM forecaster implementation using Darts."""

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
    from darts import TimeSeries as DartsTimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.models import RNNModel

    HAS_DARTS = True
except ImportError:
    DartsTimeSeries = None
    RNNModel = None
    Scaler = None
    HAS_DARTS = False
    logger.warning(
        "darts not installed. LSTMForecaster will not work. "
        "Install with: pip install darts"
    )


class LSTMForecaster(BaseForecaster):
    """LSTM forecaster using Darts RNNModel.

    Uses Long Short-Term Memory networks for time series forecasting.
    """

    def __init__(
        self,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        n_rnn_layers: int = 2,
        hidden_dim: int = 64,
        n_epochs: int = 100,
        random_state: Optional[int] = None,
        scale: bool = True,
        **darts_params: Any,
    ):
        """Initialize LSTM forecaster.

        Args:
            input_chunk_length: Number of time steps to use as input.
            output_chunk_length: Number of time steps to predict at once.
            n_rnn_layers: Number of RNN layers (default: 2).
            hidden_dim: Hidden dimension size (default: 64).
            n_epochs: Number of training epochs (default: 100).
            random_state: Random seed for reproducibility.
            scale: Whether to scale the data (default: True).
            **darts_params: Additional Darts RNNModel parameters.
        """
        if not HAS_DARTS:
            raise ImportError(
                "darts is required for LSTMForecaster. Install with: pip install darts"
            )

        super().__init__()
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.n_rnn_layers = n_rnn_layers
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.scale = scale
        self.darts_params = darts_params

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
        self,
        y: Union["SeriesLike", Any],
        X: Optional[Union["TableLike", Any]] = None,
        **fit_params: Any,
    ) -> "LSTMForecaster":
        """Fit LSTM model.

        Args:
            y: Target time series.
            X: Optional exogenous data (not yet supported).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if X is not None:
            logger.warning(
                "Exogenous variables (X) not yet supported for LSTMForecaster"
            )

        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        series = ensure_datetime_index(series)
        self.train_index_ = series.index

        # Convert to Darts TimeSeries
        darts_series = DartsTimeSeries.from_series(series)

        # Scale if requested
        if self.scale:
            self.scaler_ = Scaler()
            darts_series = self.scaler_.fit_transform(darts_series)
        else:
            self.scaler_ = None

        # Create and fit model
        pl_trainer_kwargs = {
            "enable_progress_bar": False,
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
        }
        pl_trainer_kwargs.update(self.darts_params.get("pl_trainer_kwargs", {}))

        self.model_ = RNNModel(
            model="LSTM",
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            training_length=max(self.input_chunk_length, 24),
            n_rnn_layers=self.n_rnn_layers,
            hidden_dim=self.hidden_dim,
            n_epochs=self.n_epochs,
            random_state=self.random_state,
            pl_trainer_kwargs=pl_trainer_kwargs,
            **{k: v for k, v in self.darts_params.items() if k != "pl_trainer_kwargs"},
        )

        self.model_.fit(darts_series)

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
                "Exogenous variables (X) not yet supported for LSTMForecaster"
            )

        # Convert fh to integer
        if isinstance(fh, (list, np.ndarray)):
            n_periods = len(fh)
        elif isinstance(fh, int):
            n_periods = fh
        else:
            n_periods = int(fh)

        # Generate forecast
        forecast_darts = self.model_.predict(n_periods)

        # Inverse transform if scaled
        if self.scaler_ is not None:
            forecast_darts = self.scaler_.inverse_transform(forecast_darts)

        # Convert back to pandas Series
        forecast_series = forecast_darts.to_series()

        # Ensure index is correct
        freq = pd.infer_freq(self.train_index_) or "D"
        last_date = self.train_index_[-1]
        expected_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=n_periods,
            freq=freq,
        )

        # Align index if needed
        if len(forecast_series) == n_periods:
            forecast_series.index = expected_index[: len(forecast_series)]

        return Forecast(y_pred=forecast_series, fh=fh)

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

        # Darts doesn't provide built-in prediction intervals for RNNModel
        # We'll estimate them from training residuals
        # This is a simplified approach - in practice, you might want to use
        # quantile regression or other methods

        # Get training predictions for residual calculation
        # Note: This is approximate as we'd need to refit or use a different approach
        # For now, we'll use a simple approach based on forecast uncertainty

        # Estimate uncertainty from point forecast variance
        # This is a heuristic - in practice, you'd want proper uncertainty quantification
        forecast_std = forecast.y_pred.std() * 0.1  # Rough estimate

        from scipy import stats

        z_score = stats.norm.ppf((1 + coverage) / 2)
        margin = z_score * forecast_std

        y_int = pd.DataFrame(
            {
                "lower": forecast.y_pred - margin,
                "upper": forecast.y_pred + margin,
            },
            index=forecast.y_pred.index,
        )

        return Forecast(y_pred=forecast.y_pred, fh=fh, y_int=y_int)
