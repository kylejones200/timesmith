"""Base classes for time series estimators."""

import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import pandas as pd

from timesmith.exceptions import NotFittedError, UnsupportedOperationError

if TYPE_CHECKING:
    from timesmith.typing import ForecastLike, SeriesLike, TableLike

logger = logging.getLogger(__name__)


class BaseObject:
    """Base class for all time series objects with parameter management.

    Provides get_params, set_params, clone, and __repr__ methods.
    """

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this object.

        Args:
            deep: If True, will return the parameters of this object and
                contained subobjects that are estimators.

        Returns:
            Parameter names mapped to their values.
        """
        params = {}
        for key in self.__dict__:
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                params[key] = value.get_params(deep=True)
            else:
                params[key] = value
        return params

    def set_params(self, **params: Any) -> "BaseObject":
        """Set the parameters of this object.

        Args:
            **params: Parameter names mapped to their values.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If parameter name is invalid or is a private attribute.
        """
        valid_params = self.get_params(deep=False)
        for key, value in params.items():
            # Prevent setting private attributes (starting with _)
            if key.startswith("_"):
                raise ValueError(
                    f"Cannot set private attribute '{key}' for {self.__class__.__name__}. "
                    f"Private attributes (starting with '_') are not configurable parameters."
                )
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key} for {self.__class__.__name__}. "
                    f"Valid parameters are: {list(valid_params.keys())}"
                )
            setattr(self, key, value)
        return self

    def clone(self) -> "BaseObject":
        """Create a deep copy of this object.

        Returns:
            A new instance with the same parameters.
        """
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        """String representation of the object."""
        class_name = self.__class__.__name__
        params = self.get_params(deep=False)
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{class_name}({params_str})"


class BaseEstimator(BaseObject):
    """Base class for all estimators with fit capability.

    Adds fitted state management to BaseObject.
    """

    def __init__(self):
        """Initialize the estimator."""
        self._is_fitted = False

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "BaseEstimator":
        """Fit the estimator to data.

        Args:
            y: Target data.
            X: Optional exogenous/feature data.
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        """Check if the estimator has been fitted.

        Returns:
            True if fitted, False otherwise.
        """
        return self._is_fitted

    def _check_is_fitted(self) -> None:
        """Check if estimator is fitted, raise error if not."""
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)


class BaseTransformer(BaseEstimator):
    """Base class for transformers.

    Transformers modify data but don't predict future values.
    """

    def transform(
        self, y: Union["SeriesLike", Any], X: Optional[Union["TableLike", Any]] = None
    ) -> Union["SeriesLike", Any]:
        """Transform the data.

        Args:
            y: Target data to transform (SeriesLike).
            X: Optional exogenous/feature data (TableLike).

        Returns:
            Transformed data (SeriesLike).
        """
        self._check_is_fitted()
        raise NotImplementedError("Subclasses must implement transform")

    def inverse_transform(
        self, y: Union["SeriesLike", Any], X: Optional[Union["TableLike", Any]] = None
    ) -> Union["SeriesLike", Any]:
        """Inverse transform the data.

        Args:
            y: Transformed data to inverse transform (SeriesLike).
            X: Optional exogenous/feature data (TableLike).

        Returns:
            Inverse transformed data (SeriesLike).
        """
        self._check_is_fitted()
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse_transform"
        )

    def fit_transform(
        self,
        y: Union["SeriesLike", Any],
        X: Optional[Union["TableLike", Any]] = None,
        **fit_params: Any,
    ) -> Union["SeriesLike", Any]:
        """Fit the transformer and transform the data.

        Args:
            y: Target data.
            X: Optional exogenous/feature data.
            **fit_params: Additional fit parameters.

        Returns:
            Transformed data.
        """
        return self.fit(y, X, **fit_params).transform(y, X)


class BaseForecaster(BaseEstimator):
    """Base class for forecasters.

    Forecasters predict future values of time series.
    """

    def _parse_forecast_horizon(self, fh: Any) -> int:
        """Parse forecast horizon to integer number of periods.

        Args:
            fh: Forecast horizon (can be integer, array, or other format).

        Returns:
            Number of periods to forecast.
        """
        import numpy as np

        if isinstance(fh, (list, tuple)) or hasattr(fh, "__len__"):
            # Check if it's a numpy array or list-like
            if hasattr(fh, "shape"):  # numpy array
                return len(fh)
            elif isinstance(fh, (list, tuple)):
                return len(fh)
            else:
                return len(fh)
        elif isinstance(fh, int):
            return fh
        else:
            return int(fh)

    def _create_forecast_index(
        self, train_index: Any, n_periods: int
    ) -> pd.DatetimeIndex:
        """Create forecast index based on training data frequency.

        Args:
            train_index: Training data index (DatetimeIndex or similar).
            n_periods: Number of periods to forecast.

        Returns:
            Forecast index (DatetimeIndex).
        """
        from timesmith.utils.ts_utils import detect_frequency

        last_date = pd.Timestamp(train_index[-1])
        freq = detect_frequency(pd.Series(index=train_index))

        if isinstance(freq, str):
            next_date = last_date + pd.tseries.frequencies.to_offset(freq)
            forecast_index = pd.date_range(
                start=next_date, periods=n_periods, freq=freq
            )
        else:
            # Fallback: estimate from spacing
            if len(train_index) > 1:
                avg_delta = train_index[-1] - train_index[-2]
                next_date = last_date + avg_delta
                forecast_index = pd.date_range(
                    start=next_date, periods=n_periods, freq=avg_delta
                )
            else:
                # Default to daily if only one data point
                forecast_index = pd.date_range(
                    start=last_date, periods=n_periods + 1, freq="D"
                )[1:]

        return forecast_index

    def predict(self, fh: Any, X: Optional[Any] = None, **predict_params: Any) -> Any:
        """Make forecasts.

        Args:
            fh: Forecast horizon (can be integer, array, or other format).
            X: Optional exogenous/feature data for the forecast horizon.
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast results (ForecastLike).
        """
        self._check_is_fitted()
        raise NotImplementedError("Subclasses must implement predict")

    def predict_interval(
        self,
        fh: Any,
        X: Optional[Any] = None,
        coverage: float = 0.9,
        **predict_params: Any,
    ) -> Any:
        """Make forecasts with prediction intervals.

        Args:
            fh: Forecast horizon.
            X: Optional exogenous/feature data for the forecast horizon.
            coverage: Coverage level for prediction intervals (e.g., 0.9 for 90%).
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast results with intervals (ForecastLike with y_int).
        """
        self._check_is_fitted()
        raise UnsupportedOperationError("predict_interval", self.__class__.__name__)


class BaseDetector(BaseEstimator):
    """Base class for anomaly detectors.

    Detectors identify anomalies in time series.
    """

    def score(self, y: Any, X: Optional[Any] = None) -> Any:
        """Compute anomaly scores.

        Args:
            y: Target data.
            X: Optional exogenous/feature data.

        Returns:
            Anomaly scores.
        """
        self._check_is_fitted()
        raise NotImplementedError("Subclasses must implement score")

    def predict(
        self, y: Any, X: Optional[Any] = None, threshold: Optional[float] = None
    ) -> Any:
        """Predict anomaly flags.

        Args:
            y: Target data.
            X: Optional exogenous/feature data.
            threshold: Optional threshold for binary classification.

        Returns:
            Anomaly flags (boolean array or similar).
        """
        self._check_is_fitted()
        raise NotImplementedError("Subclasses must implement predict")


class BaseFeaturizer(BaseTransformer):
    """Base class for featurizers.

    Featurizers are transformers that output TableLike data.
    """

    def transform(
        self, y: Union["SeriesLike", Any], X: Optional[Union["TableLike", Any]] = None
    ) -> "TableLike":
        """Transform data to table format.

        Args:
            y: Target data to transform (SeriesLike).
            X: Optional exogenous/feature data (TableLike).

        Returns:
            TableLike data (DataFrame with time-aligned rows).
        """
        self._check_is_fitted()
        raise NotImplementedError("Subclasses must implement transform")
