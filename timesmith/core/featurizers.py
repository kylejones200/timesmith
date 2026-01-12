"""Featurizer implementations for time series feature engineering."""

import logging
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseFeaturizer
from timesmith.core.tags import set_tags
from timesmith.utils.rolling import (
    rolling_max,
    rolling_mean,
    rolling_median,
    rolling_min,
    rolling_std,
)
from timesmith.utils.ts_utils import ensure_datetime_index

logger = logging.getLogger(__name__)


class LagFeaturizer(BaseFeaturizer):
    """Create lagged features from time series.

    Transforms SeriesLike to TableLike by creating lag features.
    Supports automatic lead prevention, differences, percentage changes,
    and seasonal lags.
    """

    def __init__(
        self,
        lags: List[int] = [1, 2, 3, 7, 14],
        include_diff: bool = False,
        include_pct_change: bool = False,
        seasonal_lags: Optional[List[int]] = None,
        prevent_leads: bool = True,
    ):
        """Initialize lag featurizer.

        Args:
            lags: List of lag periods to create.
            include_diff: If True, include differenced features (lag differences).
            include_pct_change: If True, include percentage change features.
            seasonal_lags: Optional list of seasonal lag periods
                (e.g., [12, 24] for monthly).
            prevent_leads: If True, ensures no future data leakage (only positive lags).
        """
        # Filter out negative lags if prevent_leads is True
        if prevent_leads:
            self.lags = [lag for lag in lags if lag > 0]
            if len(self.lags) < len(lags):
                logger.warning(f"Filtered out non-positive lags. Using: {self.lags}")
        else:
            self.lags = lags

        self.include_diff = include_diff
        self.include_pct_change = include_pct_change
        self.seasonal_lags = seasonal_lags or []
        self.prevent_leads = prevent_leads

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="TableLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "LagFeaturizer":
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
        """Create lag features using vectorized NumPy operations.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            TableLike DataFrame with lag features.
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

        # Convert to numpy array for vectorized operations
        values = np.asarray(series, dtype=np.float64)
        n = len(values)

        # Pre-allocate result dictionary for all features
        feature_dict = {"value": values}

        # Standard lag features - vectorized
        for lag in self.lags:
            lagged = np.full(n, np.nan, dtype=np.float64)
            lagged[lag:] = values[:-lag] if lag > 0 else values
            feature_dict[f"lag_{lag}"] = lagged

        # Difference features - vectorized
        if self.include_diff:
            for lag in self.lags:
                if lag > 0:
                    diff_values = np.full(n, np.nan, dtype=np.float64)
                    diff_values[lag:] = values[lag:] - values[:-lag]
                    feature_dict[f"diff_{lag}"] = diff_values

        # Percentage change features - vectorized
        if self.include_pct_change:
            for lag in self.lags:
                if lag > 0:
                    pct_values = np.full(n, np.nan, dtype=np.float64)
                    prev_values = values[:-lag]
                    curr_values = values[lag:]
                    # Avoid division by zero
                    mask = prev_values != 0
                    pct_values[lag:][mask] = (
                        curr_values[mask] - prev_values[mask]
                    ) / prev_values[mask]
                    feature_dict[f"pct_change_{lag}"] = pct_values

        # Seasonal lag features - vectorized
        for seasonal_lag in self.seasonal_lags:
            if seasonal_lag > 0 or not self.prevent_leads:
                lagged = np.full(n, np.nan, dtype=np.float64)
                if seasonal_lag > 0:
                    lagged[seasonal_lag:] = values[:-seasonal_lag]
                elif seasonal_lag < 0:
                    lagged[:seasonal_lag] = values[-seasonal_lag:]
                else:
                    lagged = values.copy()
                feature_dict[f"seasonal_lag_{seasonal_lag}"] = lagged

        # Create DataFrame from dictionary (faster than column-by-column)
        df = pd.DataFrame(feature_dict, index=index)
        return df


class RollingFeaturizer(BaseFeaturizer):
    """Create rolling window features from time series.

    Transforms SeriesLike to TableLike by creating rolling statistics.
    """

    def __init__(
        self,
        windows: List[int] = [7, 14, 30],
        functions: List[str] = ["mean", "std"],
        n_jobs: Optional[int] = None,
    ):
        """Initialize rolling featurizer.

        Args:
            windows: List of window sizes.
            functions: List of functions to apply
                ('mean', 'std', 'min', 'max', 'median').
            n_jobs: Number of parallel jobs for computing statistics.
                None uses all CPUs.
        """
        self.windows = windows
        self.functions = functions
        self.n_jobs = n_jobs
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="TableLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "RollingFeaturizer":
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
        """Create rolling features using optimized NumPy vectorized operations.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            TableLike DataFrame with rolling features.
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

        # Convert to numpy array for vectorized operations
        values = np.asarray(series, dtype=np.float64)

        # Pre-allocate result dictionary
        feature_dict = {"value": values}

        # Use parallelized rolling statistics if many windows/functions
        from timesmith.utils.rolling import rolling_statistics

        if len(self.windows) * len(self.functions) > 4:
            rolling_results = rolling_statistics(
                values, self.windows, self.functions, n_jobs=self.n_jobs
            )
            for key, result in rolling_results.items():
                # Fill NaN with 0 for std (matching pandas behavior)
                if key.startswith("rolling_std_"):
                    result = np.nan_to_num(result, nan=0.0)
                feature_dict[key] = result
        else:
            # Small number of operations - use direct calls
            function_map = {
                "mean": rolling_mean,
                "std": rolling_std,
                "min": rolling_min,
                "max": rolling_max,
                "median": rolling_median,
            }
            for window in self.windows:
                for func in self.functions:
                    if func not in function_map:
                        logger.warning(f"Unknown function {func}, skipping")
                        continue
                    rolling_func = function_map[func]
                    result = rolling_func(values, window, min_periods=1)
                    # Fill NaN with 0 for std (matching pandas behavior)
                    if func == "std":
                        result = np.nan_to_num(result, nan=0.0)
                    feature_dict[f"rolling_{func}_{window}"] = result

        # Create DataFrame from dictionary (faster than column-by-column)
        df = pd.DataFrame(feature_dict, index=index)
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

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "TimeFeaturizer":
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

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "DifferencingFeaturizer":
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


class SeasonalFeaturizer(BaseFeaturizer):
    """Create seasonal features using sine/cosine transformations.

    Transforms SeriesLike to TableLike by creating seasonal sine/cosine features.
    """

    def __init__(self, seasonal_periods: List[int]):
        """Initialize seasonal featurizer.

        Args:
            seasonal_periods: List of seasonal periods
                (e.g., [12, 365] for monthly/yearly).
            seasonal_periods: List of seasonal periods (e.g., [12, 365] for monthly/yearly).
        """
        self.seasonal_periods = seasonal_periods
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="TableLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "SeasonalFeaturizer":
        """Fit the featurizer (no-op for seasonal features).

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
        """Create seasonal features.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            TableLike DataFrame with seasonal features.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        df = pd.DataFrame({"value": series})
        n = len(series)
        t = np.arange(n)

        for period in self.seasonal_periods:
            df[f"seasonal_sin_{period}"] = np.sin(2 * np.pi * t / period)
            df[f"seasonal_cos_{period}"] = np.cos(2 * np.pi * t / period)

        return df


class DegradationRateFeaturizer(BaseFeaturizer):
    """Create degradation rate features (rate of change) from time series.

    Transforms SeriesLike to TableLike by creating percentage change features.
    """

    def __init__(self, periods: List[int] = [1, 3, 5]):
        """Initialize degradation rate featurizer.

        Args:
            periods: List of periods for rate of change calculation.
        """
        self.periods = periods
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="TableLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "DegradationRateFeaturizer":
        """Fit the featurizer (no-op for degradation rates).

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
        """Create degradation rate features.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            TableLike DataFrame with degradation rate features.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        df = pd.DataFrame({"value": series})

        for period in self.periods:
            # Calculate percentage change (rate of change)
            df[f"degradation_rate_{period}"] = series.pct_change(periods=period)

        return df
