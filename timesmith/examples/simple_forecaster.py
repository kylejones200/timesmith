"""Simple forecaster example for demonstration."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseForecaster
from timesmith.core.tags import set_tags
from timesmith.results.forecast import Forecast

logger = logging.getLogger(__name__)


class NaiveForecaster(BaseForecaster):
    """Simple naive forecaster that predicts the last value.

    This is a demonstration forecaster that implements BaseForecaster.
    """

    def __init__(self):
        """Initialize naive forecaster."""
        super().__init__()
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
            supports_panel=False,
            requires_fh=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "NaiveForecaster":
        """Fit the forecaster (stores last value).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if isinstance(y, pd.Series):
            self.last_value_ = y.iloc[-1]
        elif isinstance(y, pd.DataFrame):
            self.last_value_ = y.iloc[-1, 0]
        else:
            self.last_value_ = y[-1]

        self._is_fitted = True
        return self

    def predict(
        self, fh: Any, X: Optional[Any] = None, **predict_params: Any
    ) -> Forecast:
        """Make predictions using last value.

        Args:
            fh: Forecast horizon (integer or array).
            X: Optional exogenous data (ignored).
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast object with predictions.
        """
        self._check_is_fitted()

        # Convert fh to array
        if isinstance(fh, int):
            fh_array = np.arange(1, fh + 1)
        elif isinstance(fh, (list, np.ndarray)):
            fh_array = np.array(fh)
        else:
            fh_array = np.array([fh])

        # Predict last value for all horizons
        y_pred = np.full(len(fh_array), self.last_value_)

        return Forecast(y_pred=y_pred, fh=fh_array)

