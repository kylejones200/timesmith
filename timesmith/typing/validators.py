"""Runtime validators for time series data structures."""

import logging
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)


def is_series(data: object) -> bool:
    """Check if data is SeriesLike (pandas Series or single-column DataFrame with datetime/int index).

    Args:
        data: Data to check.

    Returns:
        True if data is SeriesLike, False otherwise.
    """
    if isinstance(data, pd.Series):
        return True

    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            index = data.index
            if isinstance(index, (pd.DatetimeIndex, pd.RangeIndex, pd.Index)):
                # Check if index is integer-like
                if isinstance(index, pd.Index) and not isinstance(
                    index, pd.DatetimeIndex
                ):
                    try:
                        pd.to_numeric(index)
                        return True
                    except (ValueError, TypeError):
                        return False
                return True

    return False


def is_panel(data: object) -> bool:
    """Check if data is PanelLike (DataFrame with entity key plus time index).

    Args:
        data: Data to check.

    Returns:
        True if data is PanelLike, False otherwise.
    """
    if not isinstance(data, pd.DataFrame):
        return False

    index = data.index

    # MultiIndex with (entity, time) structure
    if isinstance(index, pd.MultiIndex) and index.nlevels >= 2:
        return True

    # Regular DataFrame with entity column
    # This is a heuristic - we check if there's a common entity column name
    entity_cols = ["entity", "id", "group", "key"]
    if any(col in data.columns for col in entity_cols):
        return True

    # If index is DatetimeIndex and has multiple columns, assume panel
    if isinstance(index, pd.DatetimeIndex) and data.shape[1] > 1:
        return True

    return False


def is_table(data: object) -> bool:
    """Check if data is TableLike (DataFrame with row index aligned to time).

    Args:
        data: Data to check.

    Returns:
        True if data is TableLike, False otherwise.
    """
    if not isinstance(data, pd.DataFrame):
        return False

    index = data.index
    if isinstance(index, (pd.DatetimeIndex, pd.RangeIndex, pd.Index)):
        # Check if index is integer-like for time alignment
        if isinstance(index, pd.Index) and not isinstance(index, pd.DatetimeIndex):
            try:
                pd.to_numeric(index)
                return True
            except (ValueError, TypeError):
                return False
        return True

    return False


def assert_series(data: object, name: str = "data") -> None:
    """Assert that data is SeriesLike, raise clear error if not.

    Args:
        data: Data to validate.
        name: Name of the variable for error messages.

    Raises:
        TypeError: If data is not SeriesLike.
    """
    if not is_series(data):
        raise TypeError(
            f"{name} must be SeriesLike (pandas Series or single-column "
            f"DataFrame with datetime or integer index). "
            f"Got {type(data).__name__}."
        )


def assert_panel(data: object, name: str = "data") -> None:
    """Assert that data is PanelLike, raise clear error if not.

    Args:
        data: Data to validate.
        name: Name of the variable for error messages.

    Raises:
        TypeError: If data is not PanelLike.
    """
    if not is_panel(data):
        raise TypeError(
            f"{name} must be PanelLike (DataFrame with entity key plus "
            f"time index, or MultiIndex with entity then time). "
            f"Got {type(data).__name__}."
        )


def assert_table(data: object, name: str = "data") -> None:
    """Assert that data is TableLike, raise clear error if not.

    Args:
        data: Data to validate.
        name: Name of the variable for error messages.

    Raises:
        TypeError: If data is not TableLike.
    """
    if not is_table(data):
        raise TypeError(
            f"{name} must be TableLike (DataFrame with row index aligned "
            f"to time or window end times). Got {type(data).__name__}."
        )


# Aliases for consistency with naming convention
assert_series_like = assert_series
assert_panel_like = assert_panel

