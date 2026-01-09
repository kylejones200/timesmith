"""Metrics for evaluating forecasts."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def mae(y_true: Any, y_pred: Any) -> float:
    """Mean Absolute Error.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Mean absolute error.
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got {len(y_true)} and {len(y_pred)}"
        )

    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: Any, y_pred: Any) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Root mean squared error.
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got {len(y_true)} and {len(y_pred)}"
        )

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: Any, y_pred: Any) -> float:
    """Mean Absolute Percentage Error with safe zero handling.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Mean absolute percentage error. Returns NaN if all true values are zero.
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got {len(y_true)} and {len(y_pred)}"
        )

    # Handle zeros: only compute MAPE where y_true != 0
    mask = y_true != 0
    if not np.any(mask):
        return float(np.nan)

    percentage_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
    return float(np.mean(percentage_errors))


def bias(y_true: Any, y_pred: Any) -> float:
    """Calculate bias (mean error).

    Bias measures the average difference between predicted and actual values.
    Positive bias indicates over-estimation, negative indicates under-estimation.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Bias value.
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got {len(y_true)} and {len(y_pred)}"
        )

    # Remove NaN and infinite values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return float(np.nan)

    return float(np.mean(y_pred - y_true))


def ubrmse(y_true: Any, y_pred: Any) -> float:
    """Calculate Unbiased Root Mean Square Error (ubRMSE).

    ubRMSE removes the impact of bias from RMSE, measuring only the random
    component of error. Useful when assessing precision separately from accuracy.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        ubRMSE value.
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got {len(y_true)} and {len(y_pred)}"
        )

    # Remove NaN and infinite values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return float(np.nan)

    # Calculate bias
    bias_val = np.mean(y_pred - y_true)

    # Remove bias from predictions
    y_pred_unbiased = y_pred - bias_val

    # Calculate RMSE of unbiased predictions
    return float(np.sqrt(np.mean((y_true - y_pred_unbiased) ** 2)))


def smape(y_true: Any, y_pred: Any) -> float:
    """Symmetric Mean Absolute Percentage Error.

    SMAPE is symmetric and handles zero values better than MAPE.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        SMAPE value (percentage).
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got {len(y_true)} and {len(y_pred)}"
        )

    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Handle division by zero
    mask = denominator > 0
    if not np.any(mask):
        return float(np.nan)

    return float(np.mean(numerator[mask] / denominator[mask]) * 100)


def r2_score(y_true: Any, y_pred: Any) -> float:
    """R-squared coefficient of determination.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        R² score.
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got {len(y_true)} and {len(y_pred)}"
        )

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Handle constant values case where ss_tot = 0
    if ss_tot == 0:
        # If actual values are constant and predictions match, R² = 1
        if ss_res == 0:
            return 1.0
        # If actual values are constant but predictions don't match, R² = 0
        else:
            return 0.0

    return float(1 - (ss_res / ss_tot))


def _to_array(data: Any) -> np.ndarray:
    """Convert data to numpy array.

    Args:
        data: Data to convert (Series, DataFrame, array, etc.).

    Returns:
        Numpy array.
    """
    if isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            return data.iloc[:, 0].values
        else:
            raise ValueError("DataFrame must have single column for metrics")
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)

