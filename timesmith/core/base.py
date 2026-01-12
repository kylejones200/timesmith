"""Base classes for time series estimators."""

import copy
import logging
from typing import Any, Dict, Optional

from timesmith.exceptions import NotFittedError, UnsupportedOperationError

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
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(
                    f"Invalid parameter {key} for {self.__class__.__name__}. "
                    f"Valid parameters are: {list(self.get_params(deep=False).keys())}"
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

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "BaseEstimator":
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

    def transform(self, y: Any, X: Optional[Any] = None) -> Any:
        """Transform the data.

        Args:
            y: Target data to transform.
            X: Optional exogenous/feature data.

        Returns:
            Transformed data.
        """
        self._check_is_fitted()
        raise NotImplementedError("Subclasses must implement transform")

    def inverse_transform(self, y: Any, X: Optional[Any] = None) -> Any:
        """Inverse transform the data.

        Args:
            y: Transformed data to inverse transform.
            X: Optional exogenous/feature data.

        Returns:
            Inverse transformed data.
        """
        self._check_is_fitted()
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse_transform"
        )

    def fit_transform(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> Any:
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

    def predict(
        self, fh: Any, X: Optional[Any] = None, **predict_params: Any
    ) -> Any:
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
        self, fh: Any, X: Optional[Any] = None, coverage: float = 0.9, **predict_params: Any
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
        raise UnsupportedOperationError(
            "predict_interval", self.__class__.__name__
        )


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

    def predict(self, y: Any, X: Optional[Any] = None, threshold: Optional[float] = None) -> Any:
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

    def transform(self, y: Any, X: Optional[Any] = None) -> Any:
        """Transform data to table format.

        Args:
            y: Target data to transform.
            X: Optional exogenous/feature data.

        Returns:
            TableLike data (DataFrame with time-aligned rows).
        """
        self._check_is_fitted()
        raise NotImplementedError("Subclasses must implement transform")

