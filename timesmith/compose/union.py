"""FeatureUnion for running multiple featurizers and concatenating results."""

import logging
from typing import Any, List, Optional

import pandas as pd

from timesmith.core.base import BaseFeaturizer

logger = logging.getLogger(__name__)


class FeatureUnion(BaseFeaturizer):
    """Runs multiple featurizers then concatenates their table outputs.

    Attributes:
        featurizers: List of (name, featurizer) tuples.
    """

    def __init__(self, featurizers: List[tuple]):
        """Initialize feature union.

        Args:
            featurizers: List of (name, featurizer) tuples.
        """
        self.featurizers = featurizers
        self._validate_featurizers()

    def _validate_featurizers(self) -> None:
        """Validate that all featurizers are BaseFeaturizer instances."""
        for name, featurizer in self.featurizers:
            if not isinstance(featurizer, BaseFeaturizer):
                raise TypeError(
                    f"Featurizer '{name}' must be a BaseFeaturizer, "
                    f"got {type(featurizer).__name__}"
                )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "FeatureUnion":
        """Fit all featurizers.

        Args:
            y: Target data.
            X: Optional exogenous/feature data.
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        for name, featurizer in self.featurizers:
            logger.debug(f"Fitting featurizer: {name}")
            featurizer.fit(y, X, **fit_params)

        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.DataFrame:
        """Transform data through all featurizers and concatenate results.

        Args:
            y: Target data.
            X: Optional exogenous/feature data.

        Returns:
            Concatenated TableLike data.
        """
        self._check_is_fitted()

        tables = []
        for name, featurizer in self.featurizers:
            logger.debug(f"Transforming featurizer: {name}")
            table = featurizer.transform(y, X)
            # Prefix column names with featurizer name to avoid conflicts
            if isinstance(table, pd.DataFrame):
                table.columns = [f"{name}__{col}" for col in table.columns]
            tables.append(table)

        # Concatenate along columns
        result = pd.concat(tables, axis=1)
        return result

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for all featurizers.

        Args:
            deep: If True, will return parameters of contained subobjects.

        Returns:
            Dictionary of parameters.
        """
        params = {}
        for name, featurizer in self.featurizers:
            featurizer_params = featurizer.get_params(deep=deep)
            for key, value in featurizer_params.items():
                params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params: Any) -> "FeatureUnion":
        """Set parameters for featurizers.

        Args:
            **params: Parameters in format 'featurizer_name__param_name': value.

        Returns:
            Self for method chaining.
        """
        featurizer_params = {}
        for key, value in params.items():
            if "__" in key:
                featurizer_name, param_name = key.split("__", 1)
                if featurizer_name not in featurizer_params:
                    featurizer_params[featurizer_name] = {}
                featurizer_params[featurizer_name][param_name] = value
            else:
                setattr(self, key, value)

        for name, featurizer in self.featurizers:
            if name in featurizer_params:
                featurizer.set_params(**featurizer_params[name])

        return self

