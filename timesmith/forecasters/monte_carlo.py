"""Monte Carlo ensemble forecasting with uncertainty quantification."""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from timesmith.core.base import BaseForecaster
from timesmith.core.tags import set_tags
from timesmith.results.forecast import Forecast
from timesmith.typing import SeriesLike

logger = logging.getLogger(__name__)


class MonteCarloForecaster(BaseForecaster):
    """Monte Carlo ensemble forecaster with uncertainty quantification.

    Wraps a base forecaster and generates ensemble forecasts by sampling
    from parameter distributions or adding noise to predictions.
    """

    def __init__(
        self,
        base_forecaster: BaseForecaster,
        n_samples: int = 1000,
        random_state: Optional[int] = None,
        parameter_uncertainty: Optional[Dict[str, tuple]] = None,
        noise_level: Optional[float] = None,
    ):
        """Initialize Monte Carlo forecaster.

        Args:
            base_forecaster: Base forecaster to use for predictions.
            n_samples: Number of Monte Carlo samples.
            random_state: Random state for reproducibility.
            parameter_uncertainty: Optional dict mapping parameter names to (mean, std) tuples.
            noise_level: Optional noise level to add to predictions (as fraction of std).
        """
        super().__init__()
        self.base_forecaster = base_forecaster
        self.n_samples = n_samples
        self.random_state = random_state
        self.parameter_uncertainty = parameter_uncertainty
        self.noise_level = noise_level

        if random_state is not None:
            np.random.seed(random_state)

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="ForecastLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "MonteCarloForecaster":
        """Fit the base forecaster.

        Args:
            y: Target time series.
            X: Optional exogenous data.
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self.base_forecaster.fit(y, X, **fit_params)

        # Store fitted data for uncertainty estimation
        if isinstance(y, pd.Series):
            self.y_ = y.values
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            self.y_ = y.iloc[:, 0].values
        else:
            self.y_ = np.asarray(y)

        self._is_fitted = True
        return self

    def predict(
        self, fh: Any, X: Optional[Any] = None, **predict_params: Any
    ) -> Forecast:
        """Generate ensemble forecast with uncertainty.

        Args:
            fh: Forecast horizon.
            X: Optional exogenous data for forecast horizon.
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast with mean prediction and uncertainty intervals.
        """
        self._check_is_fitted()

        # Generate ensemble forecasts
        forecasts = []

        for i in range(self.n_samples):
            # If parameter uncertainty is provided, we'd need to clone and modify
            # the base forecaster. For now, we'll use noise-based uncertainty.
            forecast = self.base_forecaster.predict(fh, X, **predict_params)

            if isinstance(forecast, Forecast):
                y_pred = forecast.y_pred
            elif isinstance(forecast, pd.Series):
                y_pred = forecast.values
            else:
                y_pred = np.asarray(forecast)

            # Add noise if specified
            if self.noise_level is not None:
                # Estimate noise from residuals if available
                if hasattr(self, "y_"):
                    residual_std = np.std(self.y_)
                else:
                    residual_std = np.std(y_pred) if len(y_pred) > 1 else 1.0

                noise = np.random.normal(0, self.noise_level * residual_std, size=len(y_pred))
                y_pred = y_pred + noise

            forecasts.append(y_pred)

        forecasts = np.array(forecasts)

        # Calculate statistics
        mean_forecast = np.mean(forecasts, axis=0)
        std_forecast = np.std(forecasts, axis=0)

        # Calculate quantiles for prediction intervals
        quantiles = {
            "p05": np.quantile(forecasts, 0.05, axis=0),
            "p25": np.quantile(forecasts, 0.25, axis=0),
            "p50": np.quantile(forecasts, 0.50, axis=0),
            "p75": np.quantile(forecasts, 0.75, axis=0),
            "p95": np.quantile(forecasts, 0.95, axis=0),
        }

        # Create prediction intervals DataFrame
        y_int = pd.DataFrame(
            {
                "lower": quantiles["p05"],
                "upper": quantiles["p95"],
            }
        )

        # Convert mean forecast to Series if fh is array-like
        if isinstance(fh, (list, np.ndarray, pd.Index)):
            y_pred_series = pd.Series(mean_forecast, index=fh)
        else:
            y_pred_series = pd.Series(mean_forecast)

        return Forecast(
            y_pred=y_pred_series,
            fh=fh,
            y_int=y_int,
            metadata={
                "n_samples": self.n_samples,
                "std": std_forecast.tolist() if isinstance(std_forecast, np.ndarray) else std_forecast,
                "quantiles": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in quantiles.items()},
                "forecasts": forecasts.tolist() if isinstance(forecasts, np.ndarray) else forecasts,
            },
        )

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
            X: Optional exogenous data for forecast horizon.
            coverage: Coverage level (e.g., 0.9 for 90%).
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast with prediction intervals.
        """
        # predict() already includes intervals, so just adjust coverage
        forecast = self.predict(fh, X, **predict_params)

        # Adjust intervals to requested coverage
        alpha = 1 - coverage
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        # Get forecasts from metadata
        if "forecasts" in forecast.metadata:
            forecasts = np.array(forecast.metadata["forecasts"])
            lower = np.quantile(forecasts, lower_quantile, axis=0)
            upper = np.quantile(forecasts, upper_quantile, axis=0)
        else:
            # Fallback: use existing quantiles if available
            if "quantiles" in forecast.metadata:
                # Approximate from existing quantiles
                lower = forecast.metadata["quantiles"].get("p05", forecast.y_pred.values * 0.9)
                upper = forecast.metadata["quantiles"].get("p95", forecast.y_pred.values * 1.1)
            else:
                # Re-generate if needed
                return self.predict(fh, X, **predict_params)

        forecast.y_int = pd.DataFrame({"lower": lower, "upper": upper})
        return forecast

