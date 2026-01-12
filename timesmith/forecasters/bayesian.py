"""Bayesian forecasting with uncertainty quantification using MCMC."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseForecaster
from timesmith.core.tags import set_tags
from timesmith.results.forecast import Forecast

logger = logging.getLogger(__name__)

# Optional PyMC for Bayesian inference
try:
    import arviz as az
    import pymc as pm

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    logger.warning(
        "PyMC not available. BayesianForecaster requires PyMC. "
        "Install with: pip install pymc arviz"
    )


class BayesianForecaster(BaseForecaster):
    """Bayesian forecaster with uncertainty quantification using MCMC.

    Uses Bayesian inference to estimate forecaster parameters and provides
    uncertainty quantification through posterior sampling. Works by fitting
    a probabilistic model to the time series and sampling from the posterior.

    This is a general-purpose Bayesian forecaster that can be used with
    any time series model. For specific models (like ARIMA), consider using
    model-specific Bayesian implementations.
    """

    def __init__(
        self,
        model_type: str = "linear_trend",
        n_samples: int = 2000,
        n_tune: int = 1000,
        random_state: Optional[int] = None,
        prior_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Bayesian forecaster.

        Args:
            model_type: Type of probabilistic model:
                - 'linear_trend': Linear trend with noise
                - 'exponential_trend': Exponential trend
                - 'ar1': AR(1) model
            n_samples: Number of MCMC samples.
            n_tune: Number of tuning samples.
            random_state: Random state for reproducibility.
            prior_params: Optional prior parameter distributions.
        """
        if not PYMC_AVAILABLE:
            raise ImportError(
                "PyMC is required for BayesianForecaster. "
                "Install with: pip install pymc arviz"
            )

        super().__init__()
        self.model_type = model_type
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.random_state = random_state
        self.prior_params = prior_params or {}

        self.trace_ = None
        self.model_ = None

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="ForecastLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "BayesianForecaster":
        """Fit Bayesian model using MCMC.

        Args:
            y: Target time series.
            X: Optional exogenous data (not yet supported).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if X is not None:
            logger.warning("Exogenous data X not yet supported in BayesianForecaster")

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

        if len(self.y_) < 3:
            raise ValueError("Need at least 3 valid data points")

        # Normalize time to start at 0
        if isinstance(self.index_, pd.DatetimeIndex):
            time_arr = (self.index_ - self.index_[0]).days
        else:
            time_arr = np.arange(len(self.y_))

        # Build and fit PyMC model
        self.trace_, self.model_ = self._fit_bayesian_model(time_arr, self.y_)

        self._is_fitted = True
        return self

    def _fit_bayesian_model(self, time: np.ndarray, y: np.ndarray) -> tuple:
        """Fit Bayesian model using PyMC.

        Args:
            time: Time array (normalized).
            y: Target values.

        Returns:
            Tuple of (trace, model).
        """
        with pm.Model() as model:
            if self.model_type == "linear_trend":
                # Linear trend model: y = a + b*t + noise
                a = pm.Normal(
                    "a",
                    mu=np.mean(y),
                    sigma=np.std(y),
                    **self.prior_params.get("a", {}),
                )
                b = pm.Normal(
                    "b",
                    mu=0.0,
                    sigma=np.std(y) / len(y),
                    **self.prior_params.get("b", {}),
                )
                sigma = pm.HalfNormal(
                    "sigma",
                    sigma=np.std(y),
                    **self.prior_params.get("sigma", {}),
                )

                # Likelihood
                mu = a + b * time
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

            elif self.model_type == "exponential_trend":
                # Exponential trend: y = a * exp(b*t) + noise
                a = pm.Lognormal(
                    "a",
                    mu=np.log(np.maximum(y[0], 0.1)),
                    sigma=0.5,
                    **self.prior_params.get("a", {}),
                )
                b = pm.Normal(
                    "b",
                    mu=0.0,
                    sigma=0.1,
                    **self.prior_params.get("b", {}),
                )
                sigma = pm.HalfNormal(
                    "sigma",
                    sigma=np.std(y),
                    **self.prior_params.get("sigma", {}),
                )

                # Likelihood
                mu = a * pm.math.exp(b * time)
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

            elif self.model_type == "ar1":
                # AR(1) model: y_t = phi * y_{t-1} + noise
                phi = pm.Normal(
                    "phi",
                    mu=0.0,
                    sigma=1.0,
                    **self.prior_params.get("phi", {}),
                )
                sigma = pm.HalfNormal(
                    "sigma",
                    sigma=np.std(y),
                    **self.prior_params.get("sigma", {}),
                )

                # Likelihood (AR(1))
                mu = phi * y[:-1]
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y[1:])

            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            # Sample
            trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                random_seed=self.random_state,
                return_inferencedata=True,
                progressbar=False,
            )

        logger.info(
            f"Completed Bayesian sampling: {len(trace.posterior.draw)} samples "
            f"(model={self.model_type})"
        )

        return trace, model

    def predict(
        self, fh: Any, X: Optional[Any] = None, **predict_params: Any
    ) -> Forecast:
        """Generate Bayesian forecast with uncertainty.

        Args:
            fh: Forecast horizon (integer or array-like).
            X: Optional exogenous data (not yet supported).
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast with mean prediction and uncertainty intervals.
        """
        self._check_is_fitted()

        if X is not None:
            logger.warning("Exogenous data X not yet supported in BayesianForecaster")

        # Convert fh to array
        if isinstance(fh, (int, np.integer)):
            fh_arr = np.arange(1, fh + 1)
        elif isinstance(fh, (list, np.ndarray, pd.Index)):
            fh_arr = np.asarray(fh)
        else:
            raise ValueError(f"Unsupported fh type: {type(fh)}")

        # Get forecast times (relative to last observed time)
        last_time = len(self.y_) - 1
        forecast_times = last_time + fh_arr

        # Sample from posterior and generate forecasts
        posterior = self.trace_.posterior
        forecasts = []

        # Limit samples for computational efficiency
        n_posterior_samples = min(100, len(posterior.draw))

        for i in range(n_posterior_samples):
            # Sample parameters from posterior
            if self.model_type == "linear_trend":
                a = float(posterior.a.values.flatten()[i])
                b = float(posterior.b.values.flatten()[i])
                y_pred = a + b * forecast_times
            elif self.model_type == "exponential_trend":
                a = float(posterior.a.values.flatten()[i])
                b = float(posterior.b.values.flatten()[i])
                y_pred = a * np.exp(b * forecast_times)
            elif self.model_type == "ar1":
                phi = float(posterior.phi.values.flatten()[i])
                # AR(1) forecast: y_t = phi * y_{t-1}
                y_pred = np.zeros(len(fh_arr))
                last_value = self.y_[-1]
                for j, t in enumerate(fh_arr):
                    last_value = phi * last_value
                    y_pred[j] = last_value
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            forecasts.append(y_pred)

        forecasts = np.array(forecasts)

        # Calculate statistics
        mean_forecast = np.mean(forecasts, axis=0)
        std_forecast = np.std(forecasts, axis=0)

        # Calculate quantiles
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

        # Convert mean forecast to Series
        if isinstance(fh, (list, np.ndarray, pd.Index)):
            y_pred_series = pd.Series(mean_forecast, index=fh)
        else:
            y_pred_series = pd.Series(mean_forecast)

        return Forecast(
            y_pred=y_pred_series,
            fh=fh,
            y_int=y_int,
            metadata={
                "n_samples": n_posterior_samples,
                "std": std_forecast.tolist()
                if isinstance(std_forecast, np.ndarray)
                else std_forecast,
                "quantiles": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in quantiles.items()
                },
                "model_type": self.model_type,
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
            X: Optional exogenous data (not yet supported).
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
        else:
            # Re-generate if needed (shouldn't happen)
            return self.predict(fh, X, **predict_params)

        lower = np.quantile(forecasts, lower_quantile, axis=0)
        upper = np.quantile(forecasts, upper_quantile, axis=0)

        forecast.y_int = pd.DataFrame({"lower": lower, "upper": upper})
        return forecast

    def get_posterior_summary(self) -> pd.DataFrame:
        """Get summary statistics of posterior distributions.

        Returns:
            DataFrame with posterior summary (mean, std, credible intervals).
        """
        self._check_is_fitted()
        return az.summary(self.trace_)
