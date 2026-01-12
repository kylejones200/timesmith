"""Data loaders for external time series sources (FRED, Yahoo Finance, etc.)."""

import logging
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def load_fred(
    series_id: str,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    api_key: Optional[str] = None,
) -> pd.Series:
    """Load time series data from FRED (Federal Reserve Economic Data).

    Args:
        series_id: FRED series ID (e.g., 'UNRATE' for unemployment rate).
        start: Start date (string or Timestamp). Defaults to earliest available.
        end: End date (string or Timestamp). Defaults to latest available.
        api_key: Optional FRED API key. If not provided, uses pandas_datareader
            without key (may have rate limits).

    Returns:
        Time series with datetime index.

    Example:
        >>> df = ts.load_fred('UNRATE', start='2010-01-01', end='2024-01-01')
    """
    try:
        import pandas_datareader.data as web
    except ImportError:
        raise ImportError(
            "pandas_datareader is required for FRED data. "
            "Install with: pip install pandas-datareader"
        )

    try:
        df = web.DataReader(
            series_id, "fred", start=start, end=end, api_key=api_key
        )
        # Convert to Series if single column
        if isinstance(df, pd.DataFrame) and df.shape[1] == 1:
            series = df.iloc[:, 0]
            series.name = series_id
            return series
        elif isinstance(df, pd.Series):
            return df
        else:
            # Multiple columns - return as-is (user can select column)
            logger.warning(
                f"FRED series {series_id} returned multiple columns. "
                "Returning DataFrame. Use df[column_name] to select a series."
            )
            return df
    except Exception as e:
        raise ValueError(
            f"Failed to load FRED series {series_id}: {e}. "
            "Check that the series ID is valid and you have an internet connection."
        ) from e


def load_yahoo(
    symbol: str,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    column: str = "Close",
) -> pd.Series:
    """Load stock price data from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol (e.g., 'TSLA', 'AAPL').
        start: Start date (string or Timestamp). Defaults to 1 year ago.
        end: End date (string or Timestamp). Defaults to today.
        column: Column to extract ('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume').
            Defaults to 'Close'.

    Returns:
        Time series with datetime index.

    Example:
        >>> df = ts.load_yahoo('TSLA', start='2020-01-01')
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for Yahoo Finance data. "
            "Install with: pip install yfinance"
        )

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)

        if df.empty:
            raise ValueError(f"No data returned for symbol {symbol}")

        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found. Available columns: {list(df.columns)}"
            )

        series = df[column].copy()
        series.name = symbol
        return series
    except Exception as e:
        raise ValueError(
            f"Failed to load Yahoo Finance data for {symbol}: {e}. "
            "Check that the symbol is valid and you have an internet connection."
        ) from e

