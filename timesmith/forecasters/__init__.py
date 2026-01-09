"""Forecaster implementations for time series forecasting."""

from timesmith.forecasters.arima import ARIMAForecaster
from timesmith.forecasters.moving_average import (
    ExponentialMovingAverageForecaster,
    SimpleMovingAverageForecaster,
    WeightedMovingAverageForecaster,
)
from timesmith.forecasters.exponential_smoothing import ExponentialSmoothingForecaster
from timesmith.forecasters.monte_carlo import MonteCarloForecaster

# Optional Bayesian forecaster
try:
    from timesmith.forecasters.bayesian import BayesianForecaster
    HAS_BAYESIAN = True
except ImportError:
    HAS_BAYESIAN = False

# Optional Ensemble forecaster
try:
    from timesmith.forecasters.ensemble import EnsembleForecaster
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
]

if HAS_BAYESIAN:
    __all__.append("BayesianForecaster")

if HAS_ENSEMBLE:
    __all__.append("EnsembleForecaster")

