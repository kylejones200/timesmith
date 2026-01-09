"""Core base classes, tags, and validation for time series estimators."""

import logging

from timesmith.core.base import (
    BaseDetector,
    BaseEstimator,
    BaseFeaturizer,
    BaseForecaster,
    BaseObject,
    BaseTransformer,
)
from timesmith.core.featurizers import (
    DifferencingFeaturizer,
    LagFeaturizer,
    RollingFeaturizer,
    TimeFeaturizer,
)
from timesmith.core.tags import get_tags, set_tags
from timesmith.core.transformers import (
    MissingDateFiller,
    MissingValueFiller,
    OutlierRemover,
    Resampler,
)
from timesmith.core.validate import validate_input

# Decomposition (always available - no optional deps)
try:
    from timesmith.core.decomposition import (
        DecomposeTransformer,
        DetrendTransformer,
        DeseasonalizeTransformer,
        detect_trend,
        detect_seasonality,
    )
    HAS_DECOMPOSITION = True
except ImportError as e:
    HAS_DECOMPOSITION = False
    logger.warning(f"Decomposition module not available: {e}")

# Change point detection (requires optional ruptures)
try:
    from timesmith.core.changepoint import (
        PELTDetector,
        BayesianChangePointDetector,
        preprocess_for_changepoint,
    )
    HAS_CHANGEPOINT = True
except ImportError:
    HAS_CHANGEPOINT = False

__all__ = [
    "BaseObject",
    "BaseEstimator",
    "BaseTransformer",
    "BaseForecaster",
    "BaseDetector",
    "BaseFeaturizer",
    "LagFeaturizer",
    "RollingFeaturizer",
    "TimeFeaturizer",
    "DifferencingFeaturizer",
    "OutlierRemover",
    "MissingValueFiller",
    "Resampler",
    "MissingDateFiller",
    "get_tags",
    "set_tags",
    "validate_input",
]

# Conditionally add decomposition exports
if HAS_DECOMPOSITION:
    __all__.extend([
        "DecomposeTransformer",
        "DetrendTransformer",
        "DeseasonalizeTransformer",
        "detect_trend",
        "detect_seasonality",
    ])

# Conditionally add change point detection exports
if HAS_CHANGEPOINT:
    __all__.extend([
        "PELTDetector",
        "BayesianChangePointDetector",
        "preprocess_for_changepoint",
    ])

