"""Time series data handling utilities.

These are pure utility functions that don't fit into the estimator hierarchy.
They operate on data directly and don't maintain state.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


def load_ts_data(
    file_path: Union[str, Path],
    date_col: str = "date",
    value_col: str = "value",
    index_col: Optional[str] = None,
) -> pd.Series:
    """Load time series data from CSV file.

    Args:
        file_path: Path to CSV file.
        date_col: Name of date column.
        value_col: Name of value column.
        index_col: Column to use as index (if None, uses date_col).

    Returns:
        Time series with datetime index.
    """
    df = pd.read_csv(file_path)

    df[date_col] = pd.to_datetime(df[date_col])

    index_col = index_col or date_col
    df.set_index(index_col, inplace=True)

    return df[value_col]


def ensure_datetime_index(
    data: Union[pd.Series, pd.DataFrame],
) -> Union[pd.Series, pd.DataFrame]:
    """Ensure data has datetime index.

    Args:
        data: Series or DataFrame to check.

    Returns:
        Data with datetime index (converted if needed).
    """
    if isinstance(data.index, pd.DatetimeIndex):
        return data

    # Try to convert index to datetime
    try:
        data = data.copy()
        data.index = pd.to_datetime(data.index)
        return data
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert index to datetime: {e}")
        return data


def resample_ts(data: pd.Series, freq: str = "D", method: str = "mean") -> pd.Series:
    """Resample time series to different frequency.

    Args:
        data: Time series data.
        freq: Target frequency (e.g., 'D', 'W', 'M', 'H').
        method: Aggregation method ('mean', 'sum', 'last', 'first').

    Returns:
        Resampled time series.
    """
    data = ensure_datetime_index(data)

    method_map = {
        "mean": lambda d: d.resample(freq).mean(),
        "sum": lambda d: d.resample(freq).sum(),
        "last": lambda d: d.resample(freq).last(),
        "first": lambda d: d.resample(freq).first(),
    }

    return method_map.get(method, method_map["mean"])(data)


def split_ts(
    data: Union[pd.Series, pd.DataFrame],
    train_size: Union[float, int, None] = None,
    test_size: Union[float, int, None] = None,
    date_split: Optional[str] = None,
) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:
    """Split time series into train and test sets.

    Args:
        data: Time series data.
        train_size: If float: proportion of data for training.
            If int: number of observations for training.
            Default: 0.8 (if test_size not provided).
        test_size: If float: proportion of data for testing.
            If int: number of observations for testing.
            If provided, takes precedence over train_size.
        date_split: Date string to split on (e.g., '2023-01-01').

    Returns:
        Tuple of (train_data, test_data).
    """
    data = ensure_datetime_index(data)

    # Handle date_split first (takes precedence)
    if date_split is not None:
        split_date = pd.to_datetime(date_split)
        return (data[data.index < split_date], data[data.index >= split_date])

    # Convert test_size to train_size if provided
    if test_size is not None:
        if isinstance(test_size, float):
            train_size = 1.0 - test_size
        else:  # int
            train_size = len(data) - test_size
    elif train_size is None:
        train_size = 0.8  # Default

    # Perform split
    if isinstance(train_size, float):
        split_idx = int(len(data) * train_size)
        return (data.iloc[:split_idx], data.iloc[split_idx:])
    else:  # int
        return (data.iloc[:train_size], data.iloc[train_size:])


def train_test_split(
    data: Union[pd.Series, pd.DataFrame],
    test_size: Union[float, int] = 0.2,
    train_size: Optional[Union[float, int]] = None,
    method: str = "time",
) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:
    """Split time series into train and test sets (time-aware split).

    This is a convenience function for time series cross-validation that ensures
    temporal ordering is preserved (no shuffling).

    Args:
        data: Time series data (Series or DataFrame).
        test_size: If float: proportion of data for testing (0.0 to 1.0).
            If int: number of observations for testing.
        train_size: If float: proportion of data for training.
            If int: number of observations for training.
            If provided, test_size is ignored.
        method: Split method. Currently only 'time' is supported (temporal split).

    Returns:
        Tuple of (train_data, test_data).

    Example:
        >>> train, test = ts.train_test_split(data, test_size=0.2, method='time')
    """
    if method != "time":
        raise ValueError(
            f"Method '{method}' not supported. Only 'time' method is available "
            "for time series splitting."
        )

    # Use split_ts with test_size converted appropriately
    return split_ts(data, train_size=train_size, test_size=test_size)


def detect_frequency(data: pd.Series) -> str:
    """Detect the frequency of a time series.

    Args:
        data: Time series data.

    Returns:
        Detected frequency string (e.g., 'D', 'H', 'M').
    """
    data = ensure_datetime_index(data)

    inferred = pd.infer_freq(data.index)

    freq_map = {
        "H": "H",
        "D": "D",
        "W": "W",
        "M": "M",
        "Q": "Q",
        "Y": "Y",
    }

    return freq_map.get(inferred, "D") if inferred else "D"


def fill_missing_dates(data: pd.Series, method: str = "forward") -> pd.Series:
    """Fill missing dates in time series.

    Args:
        data: Time series data.
        method: Fill method ('forward', 'backward', 'interpolate').

    Returns:
        Time series with filled missing dates.
    """
    data = ensure_datetime_index(data)

    full_index = pd.date_range(
        start=data.index.min(), end=data.index.max(), freq=detect_frequency(data)
    )

    data = data.reindex(full_index)

    if method == "forward":
        return data.fillna(method="ffill")
    elif method == "backward":
        return data.fillna(method="bfill")
    elif method == "interpolate":
        return data.interpolate()
    else:
        logger.warning(f"Unknown method {method}, using forward fill")
        return data.fillna(method="ffill")


def remove_outliers_iqr(data: pd.Series, factor: float = 1.5) -> pd.Series:
    """Remove outliers using IQR method.

    Args:
        data: Time series data.
        factor: IQR factor for outlier detection.

    Returns:
        Time series with outliers removed.
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    return data[(data >= lower_bound) & (data <= upper_bound)]


def detect_anomalies_mad(
    series: pd.Series, threshold_std: float = 3.0, window: int = 3
) -> pd.Series:
    """Detect anomalous values using median absolute deviation (MAD).

    Flags values that are more than threshold_std MAD from the rolling median.
    Uses MAD instead of std for robustness to outliers.

    Args:
        series: Time series data.
        threshold_std: Number of MAD units for threshold (roughly equivalent to std).
        window: Window size for rolling median.

    Returns:
        Boolean series indicating anomalies.

    Example:
        >>> anomalies = detect_anomalies_mad(oil_series, threshold_std=3.0)
        >>> print(f"Found {anomalies.sum()} anomalies")
    """
    import numpy as np

    # Use rolling median and MAD (center=False to avoid future data leakage)
    rolling_median = series.rolling(window=window, center=False).median()

    # Calculate MAD (Median Absolute Deviation)
    mad = series.rolling(window=window, center=False).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )

    # Avoid division by zero
    mad = mad.replace(0, 1e-10)

    # Modified z-score using MAD (factor of 0.6745 makes it consistent with std)
    modified_z_score = 0.6745 * np.abs((series - rolling_median) / mad)
    anomalies = modified_z_score > threshold_std

    return anomalies.fillna(False)
