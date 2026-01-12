"""Data validation utilities for edge cases and data quality checks."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from timesmith.exceptions import DataError
from timesmith.typing import SeriesLike

logger = logging.getLogger(__name__)


def validate_data_quality(
    y: SeriesLike,
    min_length: int = 1,
    allow_all_nan: bool = False,
    allow_all_zero: bool = True,
    name: str = "data",
) -> None:
    """Validate data quality for time series operations.

    Args:
        y: Time series data to validate.
        min_length: Minimum required length.
        allow_all_nan: If False, raises error if all values are NaN.
        allow_all_zero: If False, raises error if all values are zero.
        name: Name of the variable for error messages.

    Raises:
        DataError: If data quality checks fail.
    """
    # Convert to array for checking
    if isinstance(y, pd.Series):
        values = y.values
        length = len(y)
    elif isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise DataError(
                f"{name} must be a single-column DataFrame or Series",
                context={"name": name, "shape": y.shape},
            )
        values = y.iloc[:, 0].values
        length = len(y)
    else:
        values = np.asarray(y)
        length = len(values)

    # Check minimum length
    if length < min_length:
        raise DataError(
            f"{name} has length {length}, but minimum required is {min_length}",
            context={"name": name, "length": length, "min_length": min_length},
        )

    # Check for all NaN
    if not allow_all_nan and np.all(np.isnan(values)):
        raise DataError(
            f"{name} contains only NaN values",
            context={"name": name, "length": length},
        )

    # Check for all zero
    if not allow_all_zero and np.all(values == 0):
        raise DataError(
            f"{name} contains only zero values",
            context={"name": name, "length": length},
        )

    # Check for infinite values
    if np.any(np.isinf(values)):
        logger.warning(f"{name} contains infinite values, which may cause issues")


def validate_forecast_horizon(
    fh: Any,
    max_horizon: Optional[int] = None,
    name: str = "forecast_horizon",
) -> np.ndarray:
    """Validate and normalize forecast horizon.

    Args:
        fh: Forecast horizon (int, list, or array).
        max_horizon: Optional maximum allowed horizon.
        name: Name of the variable for error messages.

    Returns:
        Normalized forecast horizon as numpy array.

    Raises:
        ValidationError: If forecast horizon is invalid.
    """
    from timesmith.exceptions import ValidationError

    # Convert to array
    if isinstance(fh, int):
        if fh <= 0:
            raise ValidationError(
                f"{name} must be positive, got {fh}",
                context={"name": name, "value": fh},
            )
        fh_array = np.array([fh])
    elif isinstance(fh, (list, tuple)):
        fh_array = np.array(fh)
    elif isinstance(fh, np.ndarray):
        fh_array = fh
    else:
        raise ValidationError(
            f"{name} must be int, list, or array, got {type(fh).__name__}",
            context={"name": name, "type": type(fh).__name__},
        )

    # Check for valid values
    if len(fh_array) == 0:
        raise ValidationError(
            f"{name} cannot be empty",
            context={"name": name},
        )

    if np.any(fh_array <= 0):
        raise ValidationError(
            f"{name} must contain only positive values",
            context={"name": name, "values": fh_array.tolist()},
        )

    if np.any(np.isnan(fh_array)) or np.any(np.isinf(fh_array)):
        raise ValidationError(
            f"{name} contains NaN or infinite values",
            context={"name": name},
        )

    # Check maximum horizon
    if max_horizon is not None:
        if np.any(fh_array > max_horizon):
            raise ValidationError(
                f"{name} contains values greater than maximum {max_horizon}",
                context={"name": name, "max_horizon": max_horizon, "values": fh_array.tolist()},
            )

    return fh_array


def check_data_alignment(
    y: SeriesLike,
    X: Optional[Any] = None,
    name_y: str = "y",
    name_x: str = "X",
) -> None:
    """Check that y and X are properly aligned.

    Args:
        y: Target time series.
        X: Optional exogenous/feature data.
        name_y: Name for y variable in error messages.
        name_x: Name for X variable in error messages.

    Raises:
        ValidationError: If data is not aligned.
    """
    from timesmith.exceptions import ValidationError

    if X is None:
        return

    # Get indices
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_index = y.index
    else:
        y_index = np.arange(len(y))

    if isinstance(X, (pd.Series, pd.DataFrame)):
        x_index = X.index
    else:
        x_index = np.arange(len(X))

    # Check length
    if len(y) != len(X):
        raise ValidationError(
            f"{name_y} and {name_x} must have the same length, "
            f"got {len(y)} and {len(X)}",
            context={"name_y": name_y, "name_x": name_x, "len_y": len(y), "len_x": len(X)},
        )

    # Check index alignment if both are pandas objects
    if isinstance(y, (pd.Series, pd.DataFrame)) and isinstance(X, (pd.Series, pd.DataFrame)):
        if not y_index.equals(x_index):
            raise ValidationError(
                f"{name_y} and {name_x} must have aligned indices",
                context={"name_y": name_y, "name_x": name_x},
            )

