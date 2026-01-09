"""Model comparison utilities for time series forecasting."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from timesmith.eval.metrics import mae, mape, rmse, r2_score
from timesmith.results.forecast import Forecast
from timesmith.typing import SeriesLike

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Container for a single model's forecast and metrics."""

    name: str
    forecast: SeriesLike
    metrics: Optional[Dict[str, float]] = None
    confidence_intervals: Optional[pd.DataFrame] = None
    model_params: Optional[Dict[str, Any]] = None


class ModelComparison:
    """Compare multiple forecasting models.

    Stores results from multiple models and provides comparison utilities.
    """

    def __init__(self):
        """Initialize model comparison."""
        self.results: List[ModelResult] = []

    def add_result(self, result: ModelResult) -> None:
        """Add a model result to comparison.

        Args:
            result: ModelResult to add.
        """
        self.results.append(result)

    def compare_metrics(self, actual: SeriesLike) -> pd.DataFrame:
        """Compare all models' metrics.

        Args:
            actual: Actual values to compare against.

        Returns:
            DataFrame with metrics for each model, sorted by RMSE.
        """
        # Convert to Series if needed
        if isinstance(actual, pd.DataFrame) and actual.shape[1] == 1:
            actual = actual.iloc[:, 0]
        elif not isinstance(actual, pd.Series):
            actual = pd.Series(actual)

        comparison_data = []

        for result in self.results:
            # Extract forecast values
            if isinstance(result.forecast, Forecast):
                forecast_values = result.forecast.y_pred
            elif isinstance(result.forecast, pd.Series):
                forecast_values = result.forecast
            elif isinstance(result.forecast, pd.DataFrame) and result.forecast.shape[1] == 1:
                forecast_values = result.forecast.iloc[:, 0]
            else:
                forecast_values = pd.Series(result.forecast)

            # Align indices
            common_idx = actual.index.intersection(forecast_values.index)
            if len(common_idx) == 0:
                logger.warning(
                    f"No common indices for {result.name}, skipping metrics calculation"
                )
                continue

            actual_aligned = actual.loc[common_idx]
            forecast_aligned = forecast_values.loc[common_idx]

            # Calculate metrics if not already done
            if result.metrics is None:
                result.metrics = {
                    "MAE": mae(actual_aligned, forecast_aligned),
                    "RMSE": rmse(actual_aligned, forecast_aligned),
                    "MAPE": mape(actual_aligned, forecast_aligned),
                    "R²": r2_score(actual_aligned, forecast_aligned),
                }

            metrics_dict = result.metrics.copy()
            metrics_dict["Model"] = result.name
            comparison_data.append(metrics_dict)

        if not comparison_data:
            logger.warning("No valid model results to compare")
            return pd.DataFrame()

        df = pd.DataFrame(comparison_data)
        df = df.set_index("Model")

        # Sort by RMSE (best first)
        if "RMSE" in df.columns:
            df = df.sort_values("RMSE")

        return df

    def get_best_model(self, metric: str = "RMSE") -> Optional[ModelResult]:
        """Get the best performing model based on a metric.

        Args:
            metric: Metric to use for ranking ('RMSE', 'MAE', 'MAPE', 'R²').

        Returns:
            Best model result or None if no results.
        """
        if not self.results:
            return None

        # Filter results with metrics
        results_with_metrics = [r for r in self.results if r.metrics is not None]
        if not results_with_metrics:
            return None

        # Determine if metric is higher-is-better or lower-is-better
        if metric == "R²":
            # Higher is better
            best_result = max(
                results_with_metrics,
                key=lambda r: r.metrics.get(metric, -np.inf),
            )
        else:
            # Lower is better (RMSE, MAE, MAPE)
            best_result = min(
                results_with_metrics,
                key=lambda r: r.metrics.get(metric, np.inf),
            )

        return best_result


def compare_models(
    actual: SeriesLike,
    forecasts: Dict[str, SeriesLike],
) -> pd.DataFrame:
    """Quick comparison function for multiple forecasts.

    Args:
        actual: Actual values.
        forecasts: Dictionary mapping model names to forecast Series or Forecast objects.

    Returns:
        DataFrame with comparison metrics for each model.
    """
    comparison = ModelComparison()

    # Convert actual to Series if needed
    if isinstance(actual, pd.DataFrame) and actual.shape[1] == 1:
        actual_series = actual.iloc[:, 0]
    elif not isinstance(actual, pd.Series):
        actual_series = pd.Series(actual)
    else:
        actual_series = actual

    for name, forecast in forecasts.items():
        # Extract forecast values
        if isinstance(forecast, Forecast):
            forecast_values = forecast.y_pred
        elif isinstance(forecast, pd.Series):
            forecast_values = forecast
        elif isinstance(forecast, pd.DataFrame) and forecast.shape[1] == 1:
            forecast_values = forecast.iloc[:, 0]
        else:
            forecast_values = pd.Series(forecast)

        # Align indices
        common_idx = actual_series.index.intersection(forecast_values.index)
        if len(common_idx) == 0:
            logger.warning(f"No common indices for {name}, skipping")
            continue

        actual_aligned = actual_series.loc[common_idx]
        forecast_aligned = forecast_values.loc[common_idx]

        # Calculate metrics
        metrics = {
            "MAE": mae(actual_aligned, forecast_aligned),
            "RMSE": rmse(actual_aligned, forecast_aligned),
            "MAPE": mape(actual_aligned, forecast_aligned),
            "R²": r2_score(actual_aligned, forecast_aligned),
        }

        result = ModelResult(name=name, forecast=forecast_values, metrics=metrics)
        comparison.add_result(result)

    return comparison.compare_metrics(actual_series)

