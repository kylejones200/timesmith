"""Backtest functionality for forecasters."""

import logging
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from timesmith.core.base import BaseForecaster
from timesmith.eval.metrics import mae, mape, rmse
from timesmith.eval.splitters import ExpandingWindowSplit
from timesmith.results.backtest import BacktestResult
from timesmith.tasks.forecast import ForecastTask

logger = logging.getLogger(__name__)

# Constants
DEFAULT_INITIAL_WINDOW_RATIO = 0.8  # 80% of data for initial training window


def backtest_forecaster(
    forecaster: BaseForecaster,
    task: ForecastTask,
    splitter: Optional[Any] = None,
    metrics: Optional[list] = None,
) -> BacktestResult:
    """Run backtest on a forecaster with a task.

    Args:
        forecaster: Forecaster or forecaster pipeline to test.
        task: ForecastTask with y, fh, and optional X.
        splitter: Optional splitter (defaults to ExpandingWindowSplit).
        metrics: Optional list of metric functions (defaults to [mae, rmse, mape]).

    Returns:
        BacktestResult with results table and summary.
    """
    if metrics is None:
        metrics = [mae, rmse, mape]

    if splitter is None:
        # Default: use expanding window with initial window = 80% of data
        n = len(task.y)
        initial_window = max(1, int(DEFAULT_INITIAL_WINDOW_RATIO * n))
        splitter = ExpandingWindowSplit(initial_window=initial_window, fh=task.fh)

    results_rows = []
    fold_id = 0

    for train_idx, test_idx, cutoff in splitter.split(task.y):
        # Get train/test splits
        y_train = (
            task.y.iloc[train_idx] if hasattr(task.y, "iloc") else task.y[train_idx]
        )
        y_test = task.y.iloc[test_idx] if hasattr(task.y, "iloc") else task.y[test_idx]

        X_train = None
        X_test = None
        if task.X is not None:
            X_train = (
                task.X.iloc[train_idx] if hasattr(task.X, "iloc") else task.X[train_idx]
            )
            X_test = (
                task.X.iloc[test_idx] if hasattr(task.X, "iloc") else task.X[test_idx]
            )

        # Fit forecaster
        logger.debug(f"Fitting forecaster for fold {fold_id}")
        forecaster.fit(y_train, X_train)

        # Predict
        logger.debug(f"Predicting for fold {fold_id}")
        forecast = forecaster.predict(task.fh, X_test)

        # Extract predictions
        if hasattr(forecast, "y_pred"):
            y_pred = forecast.y_pred
        elif isinstance(forecast, pd.Series):
            y_pred = forecast
        elif isinstance(forecast, pd.DataFrame):
            y_pred = forecast.iloc[:, 0] if forecast.shape[1] > 0 else forecast
        else:
            y_pred = forecast

        # Ensure y_pred and y_test are aligned
        y_pred = _align_predictions(y_pred, y_test)

        # Compute metrics
        metric_values = {}
        for metric_func in metrics:
            try:
                metric_name = metric_func.__name__
                metric_value = metric_func(y_test, y_pred)
                metric_values[metric_name] = metric_value
            except (ValueError, TypeError, AttributeError) as e:
                # Specific exceptions for common metric computation errors
                logger.warning(
                    f"Error computing {metric_func.__name__}: {e}. "
                    f"This may indicate a problem with the forecast alignment or data types."
                )
                metric_values[metric_name] = None
            except Exception as e:
                # Unexpected errors should be logged with full traceback
                logger.error(
                    f"Unexpected error computing {metric_func.__name__}: {e}",
                    exc_info=True,
                )
                metric_values[metric_name] = None

        # Store results
        results_rows.append(
            {
                "fold_id": fold_id,
                "cutoff": cutoff,
                "fh": task.fh,
                "y_true": y_test,
                "y_pred": y_pred,
                **metric_values,
            }
        )

        fold_id += 1

    # Create results DataFrame
    results_df = pd.DataFrame(results_rows)

    # Compute summary metrics
    summary = {}
    for metric_func in metrics:
        metric_name = metric_func.__name__
        if metric_name in results_df.columns:
            values = results_df[metric_name].dropna()
            if len(values) > 0:
                summary[f"mean_{metric_name}"] = float(values.mean())
                summary[f"std_{metric_name}"] = float(values.std())

    # Create per-fold metrics DataFrame
    metric_cols = [m.__name__ for m in metrics if m.__name__ in results_df.columns]
    per_fold_metrics = results_df[["fold_id", "cutoff"] + metric_cols].copy()

    return BacktestResult(
        results=results_df,
        summary=summary,
        per_fold_metrics=per_fold_metrics,
    )


def _align_predictions(
    y_pred: Union[pd.Series, pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """Align predictions with test data.

    Args:
        y_pred: Predictions.
        y_test: Test data.

    Returns:
        Aligned predictions as numpy array.

    Raises:
        ValueError: If prediction and test lengths are incompatible.
    """
    # Get test length (handle both Series/DataFrame and arrays)
    n_test = len(y_test)

    # Convert predictions to numpy array
    if isinstance(y_pred, pd.Series):
        y_pred_array = y_pred.values
    elif isinstance(y_pred, pd.DataFrame):
        # Take first column if DataFrame
        y_pred_array = (
            y_pred.iloc[:, 0].values if y_pred.shape[1] > 0 else y_pred.values
        )
    elif hasattr(y_pred, "values"):
        y_pred_array = y_pred.values
    elif hasattr(y_pred, "__array__"):
        y_pred_array = np.asarray(y_pred)
    else:
        y_pred_array = np.asarray(y_pred)

    n_pred = len(y_pred_array)

    # Handle length mismatches
    if n_pred == n_test:
        return y_pred_array
    elif n_pred > n_test:
        # Truncate if predictions are longer
        logger.warning(
            f"Prediction length ({n_pred}) exceeds test length ({n_test}). "
            f"Truncating predictions."
        )
        return y_pred_array[:n_test]
    else:
        # Raise error if predictions are shorter (don't pad silently)
        raise ValueError(
            f"Prediction length ({n_pred}) is shorter than test length ({n_test}). "
            f"This indicates a problem with the forecast horizon or model output. "
            f"Expected {n_test} predictions, got {n_pred}."
        )
