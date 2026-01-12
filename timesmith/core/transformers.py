"""Transformer implementations for time series preprocessing."""

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

import pandas as pd

from timesmith.core.base import BaseTransformer
from timesmith.core.tags import set_tags
from timesmith.utils.ts_utils import detect_frequency, ensure_datetime_index

if TYPE_CHECKING:
    from timesmith.typing import SeriesLike, TableLike

logger = logging.getLogger(__name__)


class OutlierRemover(BaseTransformer):
    """Remove outliers using IQR method.

    Transforms SeriesLike by removing outliers.
    """

    def __init__(self, factor: float = 1.5):
        """Initialize outlier remover.

        Args:
            factor: IQR factor for outlier detection (default: 1.5).
        """
        self.factor = factor
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=True,
            requires_sorted_index=False,
        )

    def fit(
        self,
        y: Union["SeriesLike", Any],
        X: Optional[Union["TableLike", Any]] = None,
        **fit_params: Any,
    ) -> "OutlierRemover":
        """Fit the transformer (computes IQR bounds).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        self.lower_bound_ = q1 - self.factor * iqr
        self.upper_bound_ = q3 + self.factor * iqr

        self._is_fitted = True
        return self

    def transform(
        self, y: Union["SeriesLike", Any], X: Optional[Union["TableLike", Any]] = None
    ) -> Union["SeriesLike", Any]:
        """Remove outliers.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            SeriesLike data with outliers removed.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            mask = (y >= self.lower_bound_) & (y <= self.upper_bound_)
            return y[mask]
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
            mask = (series >= self.lower_bound_) & (series <= self.upper_bound_)
            return y[mask]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")


class MissingValueFiller(BaseTransformer):
    """Fill missing values in time series.

    Transforms SeriesLike by filling missing values.
    """

    def __init__(self, method: str = "forward"):
        """Initialize missing value filler.

        Args:
            method: Fill method ('forward', 'backward', 'interpolate').
        """
        self.method = method
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=True,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "MissingValueFiller":
        """Fit the transformer (no-op for filling).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(
        self, y: Union["SeriesLike", Any], X: Optional[Union["TableLike", Any]] = None
    ) -> Union["SeriesLike", Any]:
        """Fill missing values.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            SeriesLike data with missing values filled.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        if self.method == "forward":
            return series.fillna(method="ffill")
        elif self.method == "backward":
            return series.fillna(method="bfill")
        elif self.method == "interpolate":
            return series.interpolate()
        else:
            logger.warning(f"Unknown method {self.method}, using forward fill")
            return series.fillna(method="ffill")


class Resampler(BaseTransformer):
    """Resample time series to different frequency.

    Transforms SeriesLike by resampling to target frequency.
    """

    def __init__(self, freq: str = "D", method: str = "mean"):
        """Initialize resampler.

        Args:
            freq: Target frequency (e.g., 'D', 'W', 'M', 'H').
            method: Aggregation method ('mean', 'sum', 'last', 'first').
        """
        self.freq = freq
        self.method = method
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "Resampler":
        """Fit the transformer (no-op for resampling).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(
        self, y: Union["SeriesLike", Any], X: Optional[Union["TableLike", Any]] = None
    ) -> Union["SeriesLike", Any]:
        """Resample time series.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            Resampled SeriesLike data.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        series = ensure_datetime_index(series)

        method_map = {
            "mean": lambda d: d.resample(self.freq).mean(),
            "sum": lambda d: d.resample(self.freq).sum(),
            "last": lambda d: d.resample(self.freq).last(),
            "first": lambda d: d.resample(self.freq).first(),
        }

        resampled = method_map.get(self.method, method_map["mean"])(series)

        if isinstance(y, pd.Series):
            return resampled
        else:
            return resampled.to_frame()


class MissingDateFiller(BaseTransformer):
    """Fill missing dates in time series.

    Transforms SeriesLike by adding missing dates and filling values.
    """

    def __init__(self, method: str = "forward"):
        """Initialize missing date filler.

        Args:
            method: Fill method ('forward', 'backward', 'interpolate').
        """
        self.method = method
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=True,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "MissingDateFiller":
        """Fit the transformer (detects frequency).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        series = ensure_datetime_index(series)
        self.freq_ = detect_frequency(series)

        self._is_fitted = True
        return self

    def transform(
        self, y: Union["SeriesLike", Any], X: Optional[Union["TableLike", Any]] = None
    ) -> Union["SeriesLike", Any]:
        """Fill missing dates.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            SeriesLike data with missing dates filled.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            series = y
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            series = y.iloc[:, 0]
        else:
            raise ValueError("y must be SeriesLike (Series or single-column DataFrame)")

        series = ensure_datetime_index(series)

        full_index = pd.date_range(
            start=series.index.min(), end=series.index.max(), freq=self.freq_
        )

        series = series.reindex(full_index)

        if self.method == "forward":
            filled = series.fillna(method="ffill")
        elif self.method == "backward":
            filled = series.fillna(method="bfill")
        elif self.method == "interpolate":
            filled = series.interpolate()
        else:
            logger.warning(f"Unknown method {self.method}, using forward fill")
            filled = series.fillna(method="ffill")

        if isinstance(y, pd.Series):
            return filled
        else:
            return filled.to_frame()
