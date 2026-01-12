"""Protocol definitions for time series data structures."""

from typing import Optional, Protocol, Union

import pandas as pd


class SeriesLike(Protocol):
    """Protocol for series-like data: pandas Series or single-column DataFrame.

    Must have a datetime or integer index.
    """

    index: Union[pd.DatetimeIndex, pd.Index]
    values: object


class PanelLike(Protocol):
    """Protocol for panel-like data: DataFrame with entity key plus time index.

    Can be a DataFrame with MultiIndex (entity, time) or a regular DataFrame
    with an entity column and time index.
    """

    index: Union[pd.DatetimeIndex, pd.MultiIndex, pd.Index]
    columns: pd.Index


class TableLike(Protocol):
    """Protocol for table-like data: DataFrame with row index aligned to time.

    Rows represent time points or window end times.
    """

    index: Union[pd.DatetimeIndex, pd.Index]
    columns: pd.Index


class ForecastLike(Protocol):
    """Protocol for forecast results.

    A small dataclass-like structure with:
    - y_pred: predicted values
    - y_int: optional prediction intervals
    - fh: forecast horizon
    - metadata: optional metadata dict
    """

    y_pred: object
    y_int: Optional[object]
    fh: object
    metadata: Optional[dict]
