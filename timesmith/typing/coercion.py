"""Coercion helpers for converting data to SeriesLike and PanelLike formats."""

from typing import Union

import pandas as pd

from timesmith.typing.validators import assert_series_like, assert_panel_like, is_series, is_panel


def to_series(data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """Coerce data to pandas Series if possible.
    
    Args:
        data: SeriesLike data (Series or single-column DataFrame).
        
    Returns:
        pandas Series.
        
    Raises:
        TypeError: If data cannot be coerced to Series.
    """
    assert_series_like(data, name="data")
    
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, pd.DataFrame) and data.shape[1] == 1:
        return data.iloc[:, 0]
    else:
        raise TypeError(f"Cannot coerce {type(data).__name__} to Series")


def to_dataframe(data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """Coerce data to pandas DataFrame.
    
    Args:
        data: SeriesLike or PanelLike data.
        
    Returns:
        pandas DataFrame.
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, pd.Series):
        return data.to_frame()
    else:
        raise TypeError(f"Cannot coerce {type(data).__name__} to DataFrame")


def ensure_series_like(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """Ensure data is SeriesLike, converting if necessary.
    
    Args:
        data: Data that should be SeriesLike.
        
    Returns:
        SeriesLike data (Series or single-column DataFrame).
        
    Raises:
        TypeError: If data cannot be made SeriesLike.
    """
    if is_series(data):
        return data
    
    # Try to convert DataFrame to Series if single column
    if isinstance(data, pd.DataFrame) and data.shape[1] == 1:
        return data
    
    raise TypeError(
        f"Cannot ensure SeriesLike format. Data must be Series or "
        f"single-column DataFrame. Got {type(data).__name__} with shape {getattr(data, 'shape', 'unknown')}"
    )


def ensure_panel_like(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure data is PanelLike.
    
    Args:
        data: Data that should be PanelLike.
        
    Returns:
        PanelLike DataFrame.
        
    Raises:
        TypeError: If data is not PanelLike.
    """
    assert_panel_like(data, name="data")
    return data

