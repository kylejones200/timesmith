"""Black-Scholes Monte Carlo forecaster for asset price forecasting."""

import logging
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from timesmith.core.base import BaseForecaster
from timesmith.core.tags import set_tags
from timesmith.results.forecast import Forecast
from timesmith.typing import SeriesLike
from timesmith.utils.monte_carlo import black_scholes_monte_carlo

logger = logging.getLogger(__name__)


class BlackScholesMonteCarloForecaster(BaseForecaster):
    """Black-Scholes Monte Carlo forecaster for asset prices.

    Uses geometric Brownian motion to simulate future price paths based on
    historical log returns. Estimates drift and volatility from historical data
    and generates ensemble forecasts with uncertainty quantification.

    This is specifically designed for financial time series (stock prices,
    commodity prices, etc.) and follows the Black-Scholes model assumptions:
    - Log returns are normally distributed
    - Market is efficient (random walk)
    - Geometric Brownian motion process

    Args:
        n_simulations: Number of Monte Carlo simulation paths (default: 1000).
        random_state: Random seed for reproducibility.
        use_log_returns: Whether to use log returns (default: True, recommended).

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
        >>> prices = pd.Series(100 + np.random.randn(100).cumsum(), index=dates)
        >>> forecaster = BlackScholesMonteCarloForecaster(n_simulations=1000)
        >>> forecaster.fit(prices)
        >>> forecast = forecaster.predict(fh=30)
        >>> print(forecast.predicted.mean())  # Mean forecast
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        random_state: Optional[int] = None,
        use_log_returns: bool = True,
    ):
        """Initialize Black-Scholes Monte Carlo forecaster.

        Args:
            n_simulations: Number of Monte Carlo simulation paths.
            random_state: Random seed for reproducibility.
            use_log_returns: Whether to use log returns (True) or simple returns (False).
        """
        super().__init__()
        self.n_simulations = n_simulations
        self.random_state = random_state
        self.use_log_returns = use_log_returns

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="ForecastLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "BlackScholesMonteCarloForecaster":
        """Fit the forecaster to historical data.

        Args:
            y: Historical price series (pandas Series with datetime index).
            X: Not used, present for API compatibility.
            **fit_params: Additional fit parameters (not used).

        Returns:
            Self for method chaining.
        """
        # Convert to Series if needed
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]
            else:
                raise ValueError("DataFrame must have exactly one column")

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Store historical data
        self.y_ = y.copy()
        self._is_fitted = True

        return self

    def predict(
        self, fh: Any, X: Optional[Any] = None, **predict_params: Any
    ) -> Forecast:
        """Generate forecast using Black-Scholes Monte Carlo simulation.

        Args:
            fh: Forecast horizon. Can be:
                - Integer: number of periods ahead
                - Array-like: specific periods to forecast
                - pd.Index: specific dates/indices to forecast
            X: Not used, present for API compatibility.
            **predict_params: Additional prediction parameters (not used).

        Returns:
            Forecast object with:
            - predicted: Mean forecast (pandas Series)
            - intervals: Prediction intervals (DataFrame with 'lower' and 'upper')
            - metadata: Additional information including all simulation paths
        """
        self._check_is_fitted()

        # Determine forecast horizon
        if isinstance(fh, (int, np.integer)):
            forecast_days = int(fh)
            # Generate future dates if we have a datetime index
            if isinstance(self.y_.index, pd.DatetimeIndex):
                last_date = self.y_.index[-1]
                freq = pd.infer_freq(self.y_.index) or 'D'
                forecast_index = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_days,
                    freq=freq
                )
            else:
                forecast_index = pd.RangeIndex(
                    start=len(self.y_),
                    stop=len(self.y_) + forecast_days
                )
        elif isinstance(fh, (list, np.ndarray, pd.Index)):
            forecast_days = len(fh)
            forecast_index = pd.Index(fh)
        else:
            raise ValueError(f"Unsupported forecast horizon type: {type(fh)}")

        # Run Black-Scholes Monte Carlo simulation
        price_paths = black_scholes_monte_carlo(
            historical_data=self.y_,
            forecast_days=forecast_days,
            n_simulations=self.n_simulations,
            random_state=self.random_state,
        )

        # Calculate statistics across simulations
        mean_forecast = np.mean(price_paths, axis=1)
        std_forecast = np.std(price_paths, axis=1)

        # Calculate quantiles for prediction intervals
        quantiles = {
            "p05": np.quantile(price_paths, 0.05, axis=1),
            "p25": np.quantile(price_paths, 0.25, axis=1),
            "p50": np.quantile(price_paths, 0.50, axis=1),
            "p75": np.quantile(price_paths, 0.75, axis=1),
            "p95": np.quantile(price_paths, 0.95, axis=1),
        }

        # Create prediction intervals DataFrame
        y_int = pd.DataFrame(
            {
                "lower": quantiles["p05"],
                "upper": quantiles["p95"],
            },
            index=forecast_index,
        )

        # Create mean forecast Series
        y_pred = pd.Series(mean_forecast, index=forecast_index)

        return Forecast(
            y_pred=y_pred,
            fh=fh,
            y_int=y_int,
            metadata={
                "n_simulations": self.n_simulations,
                "std": std_forecast.tolist(),
                "quantiles": {k: v.tolist() for k, v in quantiles.items()},
                "all_paths": price_paths.tolist(),  # Store all simulation paths
                "model": "Black-Scholes Monte Carlo",
            },
        )

    def predict_interval(
        self,
        fh: Any,
        X: Optional[Any] = None,
        coverage: float = 0.9,
        **predict_params: Any,
    ) -> Forecast:
        """Generate forecast with custom prediction intervals.

        Args:
            fh: Forecast horizon.
            X: Not used, present for API compatibility.
            coverage: Coverage level (e.g., 0.9 for 90% interval).
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast with custom prediction intervals.
        """
        forecast = self.predict(fh, X, **predict_params)

        # Adjust intervals to requested coverage
        alpha = 1 - coverage
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        # Get all paths from metadata
        if "all_paths" in forecast.metadata:
            price_paths = np.array(forecast.metadata["all_paths"])
            lower = np.quantile(price_paths, lower_quantile, axis=1)
            upper = np.quantile(price_paths, upper_quantile, axis=1)
        else:
            # Fallback: approximate from existing quantiles
            if "quantiles" in forecast.metadata:
                # Interpolate from existing quantiles
                lower = forecast.metadata["quantiles"].get("p05", forecast.y_pred.values * 0.9)
                upper = forecast.metadata["quantiles"].get("p95", forecast.y_pred.values * 1.1)
            else:
                # Re-generate if needed
                return self.predict(fh, X, **predict_params)

        forecast.y_int = pd.DataFrame(
            {"lower": lower, "upper": upper},
            index=forecast.y_pred.index,
        )

        return forecast

