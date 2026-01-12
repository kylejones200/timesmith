"""Pipeline for transformer then forecaster."""

import logging
from typing import Any, List, Optional

from timesmith.core.base import BaseForecaster, BaseTransformer

logger = logging.getLogger(__name__)


class ForecasterPipeline(BaseForecaster):
    """Pipeline that chains transformer(s) then a forecaster.

    Attributes:
        steps: List of (name, transformer) tuples for preprocessing.
        forecaster: Final forecaster step.
    """

    def __init__(self, steps: List[tuple], forecaster: BaseForecaster):
        """Initialize forecaster pipeline.

        Args:
            steps: List of (name, transformer) tuples.
            forecaster: Final forecaster step.
        """
        super().__init__()  # Initialize BaseForecaster (sets _is_fitted=False)
        self.steps = steps
        self.forecaster = forecaster
        self._validate_steps()

    def _validate_steps(self) -> None:
        """Validate that steps are transformers and forecaster is a forecaster."""
        for name, step in self.steps:
            if not isinstance(step, BaseTransformer):
                raise TypeError(
                    f"Step '{name}' must be a BaseTransformer, "
                    f"got {type(step).__name__}"
                )
        if not isinstance(self.forecaster, BaseForecaster):
            raise TypeError(
                f"Forecaster must be a BaseForecaster, "
                f"got {type(self.forecaster).__name__}"
            )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "ForecasterPipeline":
        """Fit all transformers then the forecaster.

        Args:
            y: Target data.
            X: Optional exogenous/feature data.
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        Xt = X
        yt = y

        # Fit and transform through preprocessing steps
        for name, step in self.steps:
            logger.debug(f"Fitting preprocessing step: {name}")
            step.fit(yt, Xt, **fit_params)
            yt = step.transform(yt, Xt)

        # Fit forecaster on transformed data
        logger.debug("Fitting forecaster")
        self.forecaster.fit(yt, Xt, **fit_params)

        self._is_fitted = True
        return self

    def predict(self, fh: Any, X: Optional[Any] = None, **predict_params: Any) -> Any:
        """Make forecasts through the pipeline.

        Args:
            fh: Forecast horizon.
            X: Optional exogenous/feature data for forecast horizon.
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast results.
        """
        self._check_is_fitted()

        # Transform X if provided and needed
        Xt = X
        if X is not None:
            # Note: In practice, X transformation might need special handling
            # for forecast horizon. This is a simplified version.
            for name, step in self.steps:
                if hasattr(step, "transform"):
                    # X transformation depends on adapter implementation
                    pass

        # Predict using forecaster
        return self.forecaster.predict(fh, Xt, **predict_params)

    def predict_interval(
        self,
        fh: Any,
        X: Optional[Any] = None,
        coverage: float = 0.9,
        **predict_params: Any,
    ) -> Any:
        """Make forecasts with intervals through the pipeline.

        Args:
            fh: Forecast horizon.
            X: Optional exogenous/feature data for forecast horizon.
            coverage: Coverage level for prediction intervals.
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast results with intervals.
        """
        self._check_is_fitted()

        Xt = X
        if hasattr(self.forecaster, "predict_interval"):
            return self.forecaster.predict_interval(fh, Xt, coverage, **predict_params)
        else:
            raise NotImplementedError(
                f"{self.forecaster.__class__.__name__} "
                "does not support predict_interval"
            )

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for all steps and forecaster.

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

        forecaster_params = self.forecaster.get_params(deep=deep)
        for key, value in forecaster_params.items():
            params[f"forecaster__{key}"] = value

        return params

    def set_params(self, **params: Any) -> "ForecasterPipeline":
        """Set parameters for steps and forecaster.

        Args:
            **params: Parameters in format 'step_name__param_name': value.

        Returns:
            Self for method chaining.
        """
        step_params = {}
        forecaster_params = {}

        for key, value in params.items():
            if key.startswith("forecaster__"):
                param_name = key.replace("forecaster__", "", 1)
                forecaster_params[param_name] = value
            elif "__" in key:
                step_name, param_name = key.split("__", 1)
                if step_name not in step_params:
                    step_params[step_name] = {}
                step_params[step_name][param_name] = value
            else:
                setattr(self, key, value)

        for name, step in self.steps:
            if name in step_params:
                step.set_params(**step_params[name])

        if forecaster_params:
            self.forecaster.set_params(**forecaster_params)

        return self


def make_forecaster_pipeline(
    *transformers: BaseTransformer, forecaster: BaseForecaster
) -> ForecasterPipeline:
    """Create a forecaster pipeline from transformers and a forecaster.

    Args:
        *transformers: Transformer objects. Names will be auto-generated.
        forecaster: Final forecaster step.

    Returns:
        ForecasterPipeline object.
    """
    named_steps = [(f"step{i}", step) for i, step in enumerate(transformers)]
    return ForecasterPipeline(named_steps, forecaster)
