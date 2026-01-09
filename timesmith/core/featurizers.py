"""Featurizer implementations for time series feature engineering."""

import logging
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseFeaturizer
from timesmith.core.tags import set_tags
from timesmith.utils.ts_utils import ensure_datetime_index

logger = logging.getLogger(__name__)


class LagFeaturizer(BaseFeaturizer):
    """Create lagged features from time series.

    Transforms SeriesLike to TableLike by creating lag features.
    """

    def __init__(self, lags: List[int] = [1, 2, 3, 7, 14]):
        """Initialize lag featurizer.

        Args:
            lags: List of lag periods to create.
        """
        self.lags = lags
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="TableLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "LagFeaturizer":
        """Fit the featurizer (no-op for lags).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.DataFrame:
        """Create lag features.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            TableLike DataFrame with lag features.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        df = pd.DataFrame({"value": series})

        for lag in self.lags:
            df[f"lag_{lag}"] = series.shift(lag)

        return df


class RollingFeaturizer(BaseFeaturizer):
    """Create rolling window features from time series.

    Transforms SeriesLike to TableLike by creating rolling statistics.
    """

    def __init__(
        self,
        windows: List[int] = [7, 14, 30],
        functions: List[str] = ["mean", "std"],
    ):
        """Initialize rolling featurizer.

        Args:
            windows: List of window sizes.
            functions: List of functions to apply ('mean', 'std', 'min', 'max', 'median').
        """
        self.windows = windows
        self.functions = functions
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="TableLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "RollingFeaturizer":
        """Fit the featurizer (no-op for rolling features).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.DataFrame:
        """Create rolling features.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            TableLike DataFrame with rolling features.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        df = pd.DataFrame({"value": series})

        function_map = {
            "mean": lambda d, w: d.rolling(w, min_periods=1).mean(),
            "std": lambda d, w: d.rolling(w, min_periods=1).std().fillna(0),
            "min": lambda d, w: d.rolling(w, min_periods=1).min(),
            "max": lambda d, w: d.rolling(w, min_periods=1).max(),
            "median": lambda d, w: d.rolling(w, min_periods=1).median(),
        }

        for window in self.windows:
            for func in self.functions:
                if func not in function_map:
                    logger.warning(f"Unknown function {func}, skipping")
                    continue
                df[f"rolling_{func}_{window}"] = function_map[func](series, window)

        return df


class TimeFeaturizer(BaseFeaturizer):
    """Create time-based features from datetime index.

    Transforms SeriesLike to TableLike by extracting time features.
    """

    def __init__(self):
        """Initialize time featurizer."""
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="TableLike",
            handles_missing=False,
            requires_sorted_index=False,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "TimeFeaturizer":
        """Fit the featurizer (no-op for time features).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.DataFrame:
        """Create time features.

        Args:
            y: SeriesLike data with datetime index.
            X: Optional exogenous data (ignored).

        Returns:
            TableLike DataFrame with time features.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            series = y
            index = y.index
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
            index = y.index
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        index = ensure_datetime_index(pd.Series(index=index)).index

        df = pd.DataFrame({"value": series}, index=index)

        df["year"] = index.year
        df["month"] = index.month
        df["day"] = index.day
        df["dayofweek"] = index.dayofweek
        df["dayofyear"] = index.dayofyear
        df["week"] = index.isocalendar().week
        df["quarter"] = index.quarter

        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["is_month_start"] = index.is_month_start.astype(int)
        df["is_month_end"] = index.is_month_end.astype(int)

        return df


class DifferencingFeaturizer(BaseFeaturizer):
    """Create differenced features from time series.

    Transforms SeriesLike to TableLike by creating differenced features.
    """

    def __init__(self, orders: List[int] = [1]):
        """Initialize differencing featurizer.

        Args:
            orders: List of differencing orders (e.g., [1] for first difference).
        """
        self.orders = orders
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="TableLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "DifferencingFeaturizer":
        """Fit the featurizer (no-op for differencing).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.DataFrame:
        """Create differenced features.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            TableLike DataFrame with differenced features.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        df = pd.DataFrame({"value": series})

        for order in self.orders:
            diff_series = series
            for _ in range(order):
                diff_series = diff_series.diff()
            df[f"diff_{order}"] = diff_series

        return df

