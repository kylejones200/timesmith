"""Kalman filter forecaster implementation."""

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
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
    HAS_FILTERPY = True
except ImportError:
    KalmanFilter = None
    Q_discrete_white_noise = None
    HAS_FILTERPY = False
    logger.warning(
        "filterpy not installed. KalmanFilterForecaster will not work. "
        "Install with: pip install filterpy"
    )


class KalmanFilterForecaster(BaseForecaster):
    """Kalman filter forecaster for state space models.

    Uses Kalman filtering for time series forecasting with state space models.
    """

    def __init__(
        self,
        state_dimension: int = 2,
        measurement_dimension: int = 1,
        initial_state: Optional[np.ndarray] = None,
        state_transition: Optional[np.ndarray] = None,
        measurement_matrix: Optional[np.ndarray] = None,
        initial_covariance: float = 1000.0,
        measurement_noise: float = 5.0,
        process_noise: float = 0.1,
        dt: float = 1.0,
    ):
        """Initialize Kalman filter forecaster.

        Args:
            state_dimension: Dimension of state vector (default: 2).
            measurement_dimension: Dimension of measurement vector (default: 1).
            initial_state: Initial state vector (default: zeros).
            state_transition: State transition matrix F (default: constant velocity model).
            measurement_matrix: Measurement matrix H (default: [1, 0]).
            initial_covariance: Initial covariance P (default: 1000.0).
            measurement_noise: Measurement noise R (default: 5.0).
            process_noise: Process noise variance (default: 0.1).
            dt: Time step (default: 1.0).
        """
        if not HAS_FILTERPY:
            raise ImportError(
                "filterpy is required for KalmanFilterForecaster. "
                "Install with: pip install filterpy"
            )

        super().__init__()
        self.state_dimension = state_dimension
        self.measurement_dimension = measurement_dimension
        self.initial_state = initial_state
        self.state_transition = state_transition
        self.measurement_matrix = measurement_matrix
        self.initial_covariance = initial_covariance
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.dt = dt

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="ForecastLike",
            handles_missing=False,
            requires_sorted_index=True,
            supports_panel=False,
            requires_fh=True,
        )

    def _create_kalman_filter(self) -> KalmanFilter:
        """Create and configure Kalman filter."""
        kf = KalmanFilter(
            dim_x=self.state_dimension,
            dim_z=self.measurement_dimension
        )

        # Set initial state
        if self.initial_state is not None:
            kf.x = np.array(self.initial_state)
        else:
            kf.x = np.zeros(self.state_dimension)

        # Set state transition matrix F
        if self.state_transition is not None:
            kf.F = np.array(self.state_transition)
        else:
            # Default: constant velocity model
            kf.F = np.array([[1.0, self.dt], [0.0, 1.0]])

        # Set measurement matrix H
        if self.measurement_matrix is not None:
            kf.H = np.array(self.measurement_matrix)
        else:
            # Default: measure position only
            kf.H = np.array([[1.0, 0.0]])

        # Set initial covariance P
        kf.P = np.eye(self.state_dimension) * self.initial_covariance

        # Set measurement noise R
        kf.R = self.measurement_noise

        # Set process noise Q
        kf.Q = Q_discrete_white_noise(
            dim=self.state_dimension,
            dt=self.dt,
            var=self.process_noise,
        )

        return kf

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "KalmanFilterForecaster":
        """Fit Kalman filter to data.

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if X is not None:
            logger.warning("Exogenous variables (X) not yet supported for KalmanFilterForecaster")

        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        series = ensure_datetime_index(series)
        self.train_index_ = series.index
        self.train_values_ = series.values

        # Create and fit Kalman filter
        self.kf_ = self._create_kalman_filter()

        # Apply filter to all measurements
        estimates = []
        for measurement in self.train_values_:
            self.kf_.predict()
            self.kf_.update(measurement)
            estimates.append(self.kf_.x.copy())

        self.estimates_ = np.array(estimates)

        # Store last state for forecasting
        self.last_state_ = self.kf_.x.copy()
        self.last_covariance_ = self.kf_.P.copy()

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
            logger.warning("Exogenous variables (X) not yet supported for KalmanFilterForecaster")

        # Convert fh to integer
        if isinstance(fh, (list, np.ndarray)):
            n_periods = len(fh)
        elif isinstance(fh, int):
            n_periods = fh
        else:
            n_periods = int(fh)

        # Generate forecast by propagating state forward
        predictions = []
        current_state = self.last_state_.copy()
        current_cov = self.last_covariance_.copy()

        for _ in range(n_periods):
            # Predict next state
            current_state = self.kf_.F @ current_state
            current_cov = self.kf_.F @ current_cov @ self.kf_.F.T + self.kf_.Q

            # Extract measurement (position)
            prediction = (self.kf_.H @ current_state)[0]
            predictions.append(prediction)

        # Create forecast Series
        freq = pd.infer_freq(self.train_index_) or "D"
        last_date = self.train_index_[-1]
        future_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=n_periods,
            freq=freq,
        )

        y_pred = pd.Series(predictions, index=future_index)

        return Forecast(y_pred=y_pred, fh=fh)

    def predict_interval(
        self, fh: Any, X: Optional[Any] = None, coverage: float = 0.9, **predict_params: Any
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

        # Calculate intervals from covariance
        predictions = []
        lower_bounds = []
        upper_bounds = []
        current_state = self.last_state_.copy()
        current_cov = self.last_covariance_.copy()

        from scipy import stats
        z_score = stats.norm.ppf((1 + coverage) / 2)

        for _ in range(len(forecast.y_pred)):
            # Predict next state
            current_state = self.kf_.F @ current_state
            current_cov = self.kf_.F @ current_cov @ self.kf_.F.T + self.kf_.Q

            # Extract measurement and uncertainty
            prediction = (self.kf_.H @ current_state)[0]
            measurement_cov = self.kf_.H @ current_cov @ self.kf_.H.T
            std_error = np.sqrt(measurement_cov[0, 0])

            predictions.append(prediction)
            lower_bounds.append(prediction - z_score * std_error)
            upper_bounds.append(prediction + z_score * std_error)

        y_int = pd.DataFrame(
            {
                "lower": lower_bounds,
                "upper": upper_bounds,
            },
            index=forecast.y_pred.index,
        )

        return Forecast(y_pred=forecast.y_pred, fh=fh, y_int=y_int)

