"""Forecaster implementations for time series forecasting."""

from timesmith.forecasters.arima import ARIMAForecaster
from timesmith.forecasters.black_scholes import BlackScholesMonteCarloForecaster
from timesmith.forecasters.exponential_smoothing import ExponentialSmoothingForecaster
from timesmith.forecasters.linear_trend import LinearTrendForecaster
from timesmith.forecasters.monte_carlo import MonteCarloForecaster
from timesmith.forecasters.moving_average import (
    ExponentialMovingAverageForecaster,
    SimpleMovingAverageForecaster,
    WeightedMovingAverageForecaster,
)
from timesmith.forecasters.synthetic_control import SyntheticControlForecaster

# Optional VAR forecaster
try:
    from timesmith.forecasters.var import VARForecaster  # noqa: F401

    HAS_VAR = True
except ImportError:
    HAS_VAR = False

# Optional LSTM forecaster
try:
    from timesmith.forecasters.lstm import LSTMForecaster  # noqa: F401

    HAS_LSTM = True
except ImportError:
    HAS_LSTM = False

# Optional Kalman filter forecaster
try:
    from timesmith.forecasters.kalman import KalmanFilterForecaster  # noqa: F401

    HAS_KALMAN = True
except ImportError:
    HAS_KALMAN = False

# Optional Bayesian forecaster
try:
    from timesmith.forecasters.bayesian import BayesianForecaster  # noqa: F401

    HAS_BAYESIAN = True
except ImportError:
    HAS_BAYESIAN = False

# Optional Ensemble forecaster
try:
    from timesmith.forecasters.ensemble import EnsembleForecaster  # noqa: F401

    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False

__all__ = [
    "ARIMAForecaster",
    "SimpleMovingAverageForecaster",
    "ExponentialMovingAverageForecaster",
    "WeightedMovingAverageForecaster",
    "ExponentialSmoothingForecaster",
    "MonteCarloForecaster",
    "BlackScholesMonteCarloForecaster",
    "LinearTrendForecaster",
    "SyntheticControlForecaster",
]

if HAS_VAR:
    __all__.append("VARForecaster")

if HAS_LSTM:
    __all__.append("LSTMForecaster")

if HAS_KALMAN:
    __all__.append("KalmanFilterForecaster")

if HAS_BAYESIAN:
    __all__.append("BayesianForecaster")

if HAS_ENSEMBLE:
    __all__.append("EnsembleForecaster")
