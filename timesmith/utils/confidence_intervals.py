"""Bootstrap confidence intervals for time series forecasts."""

import logging
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd

from timesmith.typing import SeriesLike

logger = logging.getLogger(__name__)


def bootstrap_confidence_intervals(
    model_fit_func: Callable,
    data: SeriesLike,
    forecast_steps: int,
    n_bootstraps: int = 100,
    confidence: float = 0.95,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate bootstrap confidence intervals for forecasts.

    Args:
        model_fit_func: Function that takes data and returns a fitted model with
            a `forecast(steps)` or `predict()` method.
        data: Time series data.
        forecast_steps: Number of steps to forecast.
        n_bootstraps: Number of bootstrap iterations (default: 100).
        confidence: Confidence level (default: 0.95).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (mean_forecast, lower_bound, upper_bound) as numpy arrays.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Convert to Series if needed
    if isinstance(data, pd.DataFrame) and data.shape[1] == 1:
        data_series = data.iloc[:, 0]
    elif not isinstance(data, pd.Series):
        data_series = pd.Series(data)
    else:
        data_series = data

    forecasts = []
    successful_bootstraps = 0

    for i in range(n_bootstraps):
        try:
            # Bootstrap resample with replacement
            sample = data_series.sample(n=len(data_series), replace=True).sort_index()

            # Fit model on bootstrap sample
            model = model_fit_func(sample)

            # Generate forecast
            if hasattr(model, "forecast"):
                forecast = model.forecast(steps=forecast_steps)
            elif hasattr(model, "predict"):
                # For models that use predict instead of forecast
                if isinstance(data_series.index, pd.DatetimeIndex):
                    freq = pd.infer_freq(data_series.index) or "D"
                    future_index = pd.date_range(
                        start=data_series.index[-1] + pd.Timedelta(days=1),
                        periods=forecast_steps,
                        freq=freq,
                    )
                    forecast = model.predict(future_index)
                else:
                    # Numeric index
                    last_idx = data_series.index[-1]
                    future_index = np.arange(
                        last_idx + 1, last_idx + 1 + forecast_steps
                    )
                    forecast = model.predict(future_index)
            else:
                raise AttributeError("Model must have 'forecast' or 'predict' method")

            # Convert to numpy array if needed
            if isinstance(forecast, pd.Series):
                forecast = forecast.values
            elif isinstance(forecast, np.ndarray):
                pass
            else:
                forecast = np.array(forecast)

            if len(forecast) != forecast_steps:
                logger.warning(
                    f"Forecast length {len(forecast)} != {forecast_steps}, skipping"
                )
                continue

            forecasts.append(forecast)
            successful_bootstraps += 1

        except (ValueError, AttributeError, RuntimeError) as e:
            # Skip failed bootstrap iterations
            logger.debug(f"Bootstrap iteration {i} failed: {e}")
            continue

    if successful_bootstraps == 0:
        raise RuntimeError(
            "All bootstrap iterations failed. Check model_fit_func and data."
        )

    if successful_bootstraps < n_bootstraps * 0.5:
        logger.warning(
            f"Only {successful_bootstraps}/{n_bootstraps} bootstrap iterations succeeded. "
            "Results may be unreliable."
        )

    forecasts = np.array(forecasts)

    # Calculate percentiles
    alpha = (1 - confidence) / 2
    mean_forecast = np.mean(forecasts, axis=0)
    lower_bound = np.percentile(forecasts, alpha * 100, axis=0)
    upper_bound = np.percentile(forecasts, (1 - alpha) * 100, axis=0)

    return mean_forecast, lower_bound, upper_bound


def parametric_confidence_intervals(
    model: Any,
    forecast_steps: int,
    confidence: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate parametric confidence intervals from model.

    Args:
        model: Fitted model with get_forecast() method (e.g., statsmodels ARIMA).
        forecast_steps: Number of steps to forecast.
        confidence: Confidence level (default: 0.95).

    Returns:
        Tuple of (mean_forecast, lower_bound, upper_bound) as numpy arrays.
    """
    if not hasattr(model, "get_forecast"):
        raise AttributeError(
            "Model must have 'get_forecast' method for parametric CIs"
        )

    forecast_result = model.get_forecast(steps=forecast_steps)
    mean_forecast = forecast_result.predicted_mean.values
    conf_int = forecast_result.conf_int(alpha=1 - confidence)

    lower_bound = conf_int.iloc[:, 0].values
    upper_bound = conf_int.iloc[:, 1].values

    return mean_forecast, lower_bound, upper_bound

