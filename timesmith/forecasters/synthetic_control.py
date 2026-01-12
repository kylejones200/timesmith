"""Synthetic control forecaster for causal inference and counterfactual analysis."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from timesmith.core.base import BaseForecaster
from timesmith.core.tags import set_tags
from timesmith.results.forecast import Forecast

logger = logging.getLogger(__name__)


class SyntheticControlForecaster(BaseForecaster):
    """Synthetic control forecaster for counterfactual analysis.

    Creates a synthetic control unit as a weighted combination of control units
    to estimate what would have happened in the absence of treatment.
    """

    def __init__(
        self,
        treatment_start: Optional[int] = None,
        pre_period_min: int = 5,
    ):
        """Initialize synthetic control forecaster.

        Args:
            treatment_start: Index where treatment begins (None = end of data).
            pre_period_min: Minimum number of pre-treatment periods required.
        """
        super().__init__()
        self.treatment_start = treatment_start
        self.pre_period_min = pre_period_min
        self.weights_ = None
        self.control_indices_ = None

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="ForecastLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "SyntheticControlForecaster":
        """Fit synthetic control model.

        Args:
            y: Target time series (treated unit).
            X: Control units as DataFrame (each column is a control unit).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if X is None:
            raise ValueError(
                "X (control units) is required for synthetic control. "
                "Provide control units as DataFrame with each column as a control unit."
            )

        if isinstance(y, pd.Series):
            self.y_ = y.values
            self.index_ = y.index
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            self.y_ = y.iloc[:, 0].values
            self.index_ = y.index
        else:
            self.y_ = np.asarray(y, dtype=float)
            self.index_ = np.arange(len(self.y_))

        if isinstance(X, pd.DataFrame):
            self.X_ = X.values
            self.control_names_ = X.columns.tolist()
        else:
            self.X_ = np.asarray(X)
            if self.X_.ndim == 1:
                self.X_ = self.X_.reshape(-1, 1)
            self.control_names_ = [f"control_{i}" for i in range(self.X_.shape[1])]

        if len(self.y_) != len(self.X_):
            raise ValueError(
                f"y and X must have same length. Got {len(self.y_)} and {len(self.X_)}"
            )

        # Determine treatment start
        if self.treatment_start is None:
            self.treatment_start_ = len(self.y_)
        else:
            self.treatment_start_ = self.treatment_start

        if self.treatment_start_ < self.pre_period_min:
            raise ValueError(
                f"treatment_start ({self.treatment_start_}) must be >= "
                f"pre_period_min ({self.pre_period_min})"
            )

        # Split pre/post treatment
        treated_pre = self.y_[: self.treatment_start_]
        control_pre = self.X_[: self.treatment_start_, :]

        # Find optimal weights
        self.weights_ = self._find_weights(treated_pre, control_pre)

        # Calculate pre-treatment fit quality
        synthetic_pre = control_pre @ self.weights_
        self.pre_rmse_ = float(np.sqrt(np.mean((treated_pre - synthetic_pre) ** 2)))

        logger.info(
            f"Synthetic control fitted: {len(self.control_names_)} control units, "
            f"pre-treatment RMSE: {self.pre_rmse_:.6f}"
        )

        self._is_fitted = True
        return self

    def _find_weights(
        self, treated_pre: np.ndarray, control_pre: np.ndarray
    ) -> np.ndarray:
        """Find optimal weights for synthetic control.

        Args:
            treated_pre: Pre-treatment values of treated unit.
            control_pre: Pre-treatment values of control units (n_periods, n_controls).

        Returns:
            Optimal weights (n_controls,).
        """
        n_controls = control_pre.shape[1]

        def objective(weights: np.ndarray) -> float:
            synthetic = control_pre @ weights
            return float(np.sum((treated_pre - synthetic) ** 2))

        # Constraints: weights sum to 1, each weight between 0 and 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_controls)]
        initial = np.ones(n_controls) / n_controls

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        if not result.success:
            logger.warning(
                f"Weight optimization did not converge: {result.message}. "
                "Using uniform weights."
            )
            return initial

        return result.x

    def predict(
        self, fh: Any, X: Optional[Any] = None, **predict_params: Any
    ) -> Forecast:
        """Generate counterfactual forecast (what would have happened without treatment).

        Args:
            fh: Forecast horizon (ignored, uses post-treatment period).
            X: Optional control units for post-treatment (uses fit data if None).
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast results with counterfactual predictions.
        """
        self._check_is_fitted()

        # Use post-treatment period
        treated_post = self.y_[self.treatment_start_ :]
        control_post = self.X_[self.treatment_start_ :, :]

        # Generate synthetic control (counterfactual)
        synthetic_post = control_post @ self.weights_

        # Calculate treatment effect
        treatment_effect = treated_post - synthetic_post

        # Create forecast index
        if isinstance(self.index_, pd.DatetimeIndex):
            post_index = self.index_[self.treatment_start_ :]
        else:
            post_index = np.arange(
                self.treatment_start_, self.treatment_start_ + len(synthetic_post)
            )

        y_pred_series = pd.Series(synthetic_post, index=post_index)

        return Forecast(
            y_pred=y_pred_series,
            fh=len(synthetic_post),
            metadata={
                "method": "synthetic_control",
                "treatment_effect_mean": float(np.mean(treatment_effect)),
                "treatment_effect_std": float(np.std(treatment_effect)),
                "pre_rmse": self.pre_rmse_,
                "n_controls": len(self.control_names_),
                "top_controls": [
                    (name, float(weight))
                    for name, weight in zip(self.control_names_, self.weights_)
                    if weight > 0.01
                ][:5],
            },
        )

    def get_weights(self) -> Dict[str, float]:
        """Get synthetic control weights.

        Returns:
            Dictionary mapping control unit names to weights.
        """
        self._check_is_fitted()

        return {
            name: float(weight)
            for name, weight in zip(self.control_names_, self.weights_)
            if weight > 0.01  # Only return significant contributors
        }
