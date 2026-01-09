"""Backtest functionality for forecasters."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseForecaster
from timesmith.eval.metrics import mae, mape, rmse
from timesmith.eval.splitters import ExpandingWindowSplit, SlidingWindowSplit
from timesmith.results.backtest import BacktestResult
from timesmith.tasks.forecast import ForecastTask

logger = logging.getLogger(__name__)


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
        initial_window = max(1, int(0.8 * n))
        splitter = ExpandingWindowSplit(initial_window=initial_window, fh=task.fh)

    results_rows = []
    fold_id = 0

    for train_idx, test_idx, cutoff in splitter.split(task.y):
        # Get train/test splits
        y_train = task.y.iloc[train_idx] if hasattr(task.y, "iloc") else task.y[train_idx]
        y_test = task.y.iloc[test_idx] if hasattr(task.y, "iloc") else task.y[test_idx]

        X_train = None
        X_test = None
        if task.X is not None:
            X_train = task.X.iloc[train_idx] if hasattr(task.X, "iloc") else task.X[train_idx]
            X_test = task.X.iloc[test_idx] if hasattr(task.X, "iloc") else task.X[test_idx]

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
            except Exception as e:
                logger.warning(f"Error computing {metric_func.__name__}: {e}")
                metric_values[metric_func.__name__] = None

        # Store results
        results_rows.append({
            "fold_id": fold_id,
            "cutoff": cutoff,
            "fh": task.fh,
            "y_true": y_test,
            "y_pred": y_pred,
            **metric_values,
        })

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


def _align_predictions(y_pred: Any, y_test: Any) -> Any:
    """Align predictions with test data.

    Args:
        y_pred: Predictions.
        y_test: Test data.

    Returns:
        Aligned predictions.
    """
    import pandas as pd

    if isinstance(y_pred, pd.Series) and isinstance(y_test, pd.Series):
        # Try to align by index
        if len(y_pred) == len(y_test):
            return y_pred.values
        elif len(y_pred) > len(y_test):
            return y_pred.iloc[:len(y_test)].values
        else:
            # Pad with last value if needed
            last_val = y_pred.iloc[-1]
            padded = pd.Series([last_val] * (len(y_test) - len(y_pred)))
            return pd.concat([y_pred, padded]).values

    # Convert to array and take first len(y_test) values
    if hasattr(y_pred, "values"):
        y_pred = y_pred.values
    elif hasattr(y_pred, "__array__"):
        y_pred = y_pred.__array__()

    if isinstance(y_test, pd.Series):
        n = len(y_test)
    else:
        n = len(y_test)

    if len(y_pred) >= n:
        return y_pred[:n]
    else:
        # Pad with last value
        last_val = y_pred[-1]
        return np.concatenate([y_pred, [last_val] * (n - len(y_pred))])

