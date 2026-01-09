"""Pipeline for chaining transformers."""

import logging
from typing import Any, List, Optional

from timesmith.core.base import BaseEstimator, BaseTransformer

logger = logging.getLogger(__name__)


class Pipeline(BaseEstimator):
    """Pipeline that chains transformer steps.

    Supports scitype change across steps via adapters.

    Attributes:
        steps: List of (name, transformer) tuples.
    """

    def __init__(self, steps: List[tuple]):
        """Initialize pipeline.

        Args:
            steps: List of (name, transformer) tuples.
        """
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self) -> None:
        """Validate that all steps are transformers."""
        for name, step in self.steps:
            if not isinstance(step, BaseTransformer):
                raise TypeError(
                    f"Step '{name}' must be a BaseTransformer, "
                    f"got {type(step).__name__}"
                )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "Pipeline":
        """Fit all steps in the pipeline.

        Args:
            y: Target data.
            X: Optional exogenous/feature data.
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        Xt = X
        yt = y

        for name, step in self.steps:
            logger.debug(f"Fitting step: {name}")
            step.fit(yt, Xt, **fit_params)
            yt = step.transform(yt, Xt)
            # Xt might be transformed by adapters if needed

        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> Any:
        """Transform data through all steps.

        Args:
            y: Target data.
            X: Optional exogenous/feature data.

        Returns:
            Transformed data.
        """
        self._check_is_fitted()

        Xt = X
        yt = y

        for name, step in self.steps:
            logger.debug(f"Transforming step: {name}")
            yt = step.transform(yt, Xt)
            # Xt might be transformed by adapters if needed

        return yt

    def inverse_transform(self, y: Any, X: Optional[Any] = None) -> Any:
        """Inverse transform data through all steps in reverse.

        Args:
            y: Transformed data.
            X: Optional exogenous/feature data.

        Returns:
            Inverse transformed data.
        """
        self._check_is_fitted()

        Xt = X
        yt = y

        for name, step in reversed(self.steps):
            if hasattr(step, "inverse_transform"):
                logger.debug(f"Inverse transforming step: {name}")
                yt = step.inverse_transform(yt, Xt)

        return yt

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for all steps.

        Args:
            deep: If True, will return parameters of contained subobjects.

        Returns:
            Dictionary of parameters.
        """
        params = {}
        for name, step in self.steps:
            step_params = step.get_params(deep=deep)
            for key, value in step_params.items():
                params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params: Any) -> "Pipeline":
        """Set parameters for steps.

        Args:
            **params: Parameters in format 'step_name__param_name': value.

        Returns:
            Self for method chaining.
        """
        step_params = {}
        for key, value in params.items():
            if "__" in key:
                step_name, param_name = key.split("__", 1)
                if step_name not in step_params:
                    step_params[step_name] = {}
                step_params[step_name][param_name] = value
            else:
                # Global parameter
                setattr(self, key, value)

        for name, step in self.steps:
            if name in step_params:
                step.set_params(**step_params[name])

        return self


def make_pipeline(*steps: BaseTransformer) -> Pipeline:
    """Create a pipeline from transformers.

    Args:
        *steps: Transformer objects. Names will be auto-generated.

    Returns:
        Pipeline object.
    """
    named_steps = [(f"step{i}", step) for i, step in enumerate(steps)]
    return Pipeline(named_steps)

