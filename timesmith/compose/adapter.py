"""Adapter objects that convert between scitypes."""

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

import pandas as pd

from timesmith.core.base import BaseTransformer

if TYPE_CHECKING:
    from timesmith.typing import SeriesLike, TableLike

logger = logging.getLogger(__name__)


class Adapter(BaseTransformer):
    """Base class for adapters that convert between scitypes.

    Examples:
        - Series to Table via window features
        - Table to Series via aligned join
    """

    def transform(self, y: Any, X: Optional[Any] = None) -> Any:
        """Transform data to different scitype.

        Args:
            y: Target data to transform.
            X: Optional exogenous/feature data.

        Returns:
            Transformed data in different scitype.
        """
        self._check_is_fitted()
        raise NotImplementedError("Subclasses must implement transform")


# Example adapters (simplified implementations)


class SeriesToTableAdapter(Adapter):
    """Adapter that converts SeriesLike to TableLike via window features.

    This is a placeholder - full implementation would create window features.
    """

    def __init__(self, window_size: int = 10):
        """Initialize adapter.

        Args:
            window_size: Size of rolling window for features.
        """
        self.window_size = window_size

    def fit(
        self,
        y: Union["SeriesLike", Any],
        X: Optional[Union["TableLike", Any]] = None,
        **fit_params: Any,
    ) -> "SeriesToTableAdapter":
        """Fit the adapter (no-op for this simple version).

        Args:
            y: Target data.
            X: Optional exogenous/feature data.
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(
        self, y: Union["SeriesLike", Any], X: Optional[Union["TableLike", Any]] = None
    ) -> "TableLike":
        """Convert Series to Table using rolling window features.

        Args:
            y: SeriesLike data.
            X: Optional exogenous/feature data.

        Returns:
            TableLike data with window features.
        """
        self._check_is_fitted()

        # Simplified: just convert Series to DataFrame
        if isinstance(y, pd.Series):
            return y.to_frame()
        return y


class TableToSeriesAdapter(Adapter):
    """Adapter that converts TableLike to SeriesLike via aligned join.

    This is a placeholder - full implementation would handle alignment.
    """

    def fit(
        self,
        y: Union["SeriesLike", "TableLike", Any],
        X: Optional[Union["TableLike", Any]] = None,
        **fit_params: Any,
    ) -> "TableToSeriesAdapter":
        """Fit the adapter (no-op for this simple version).

        Args:
            y: Target data.
            X: Optional exogenous/feature data.
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(
        self, y: Union["TableLike", Any], X: Optional[Union["TableLike", Any]] = None
    ) -> Union["SeriesLike", Any]:
        """Convert Table to Series by selecting first column or aggregating.

        Args:
            y: TableLike data.
            X: Optional exogenous/feature data.

        Returns:
            SeriesLike data.
        """
        self._check_is_fitted()

        # Simplified: take first column if DataFrame
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                return y.iloc[:, 0]
            # Otherwise, take first column as default
            return y.iloc[:, 0]
        return y
