"""Simple transformer example for demonstration."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseTransformer
from timesmith.core.tags import set_tags

logger = logging.getLogger(__name__)


class LogTransformer(BaseTransformer):
    """Simple log transformer for demonstration.

    This is a demonstration transformer that implements BaseTransformer.
    """

    def __init__(self, offset: float = 1.0):
        """Initialize log transformer.

        Args:
            offset: Offset to add before taking log (to handle zeros/negatives).
        """
        super().__init__()
        self.offset = offset
        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=False,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "LogTransformer":
        """Fit the transformer (no-op for log transform).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> Any:
        """Apply log transformation.

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).

        Returns:
            Log-transformed series.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            return pd.Series(
                np.log(y.values + self.offset),
                index=y.index,
                name=y.name,
            )
        elif isinstance(y, pd.DataFrame):
            return pd.DataFrame(
                np.log(y.values + self.offset),
                index=y.index,
                columns=y.columns,
            )
        else:
            return np.log(np.array(y) + self.offset)

    def inverse_transform(self, y: Any, X: Optional[Any] = None) -> Any:
        """Apply inverse log transformation.

        Args:
            y: Log-transformed series.
            X: Optional exogenous data (ignored).

        Returns:
            Original scale series.
        """
        self._check_is_fitted()

        if isinstance(y, pd.Series):
            return pd.Series(
                np.exp(y.values) - self.offset,
                index=y.index,
                name=y.name,
            )
        elif isinstance(y, pd.DataFrame):
            return pd.DataFrame(
                np.exp(y.values) - self.offset,
                index=y.index,
                columns=y.columns,
            )
        else:
            return np.exp(np.array(y)) - self.offset

