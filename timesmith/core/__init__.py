"""Core base classes, tags, and validation for time series estimators."""

import logging

logger = logging.getLogger(__name__)

from timesmith.core.base import (  # noqa: E402
    BaseDetector,
    BaseEstimator,
    BaseFeaturizer,
    BaseForecaster,
    BaseObject,
    BaseTransformer,
)
from timesmith.core.featurizers import (  # noqa: E402
    DegradationRateFeaturizer,
    DifferencingFeaturizer,
    LagFeaturizer,
    RollingFeaturizer,
    SeasonalFeaturizer,
    TimeFeaturizer,
)
from timesmith.core.tags import get_tags, set_tags  # noqa: E402
from timesmith.core.transformers import (  # noqa: E402
    MissingDateFiller,
    MissingValueFiller,
    OutlierRemover,
    Resampler,
)
from timesmith.core.validate import validate_input  # noqa: E402

# Decomposition (always available - no optional deps)
try:
    from timesmith.core.decomposition import (
        DecomposeTransformer,  # noqa: F401
        DeseasonalizeTransformer,  # noqa: F401
        DetrendTransformer,  # noqa: F401
        detect_seasonality,  # noqa: F401
        detect_trend,  # noqa: F401
    )

    HAS_DECOMPOSITION = True
except ImportError as e:
    HAS_DECOMPOSITION = False
    logger.warning(f"Decomposition module not available: {e}")

# Change point detection (requires optional ruptures)
try:
    from timesmith.core.changepoint import (
        BayesianChangePointDetector,  # noqa: F401
        CUSUMDetector,  # noqa: F401
        PELTDetector,  # noqa: F401
        preprocess_for_changepoint,  # noqa: F401
    )

    HAS_CHANGEPOINT = True
except ImportError:
    HAS_CHANGEPOINT = False

# Filters (requires optional scipy)
try:
    from timesmith.core.filters import (
        ButterworthFilter,  # noqa: F401
        SavitzkyGolayFilter,  # noqa: F401
    )

    HAS_FILTERS = True
except ImportError:
    HAS_FILTERS = False

# Advanced outlier detection (optional sklearn for IsolationForest)
try:
    from timesmith.core.outliers import (
        HampelOutlierRemover,  # noqa: F401
        IsolationForestOutlierRemover,  # noqa: F401
        ZScoreOutlierRemover,  # noqa: F401
    )

    HAS_ADVANCED_OUTLIERS = True
except ImportError:
    HAS_ADVANCED_OUTLIERS = False

# Wavelet methods (optional PyWavelets)
try:
    from timesmith.core.wavelet import (
        WaveletDenoiser,  # noqa: F401
        WaveletDetector,  # noqa: F401
    )

    HAS_WAVELET = True
except ImportError:
    HAS_WAVELET = False

# Seasonal baseline detection (always available)
from timesmith.core.seasonal import SeasonalBaselineDetector  # noqa: E402, F401

# Ensemble detection (always available)
from timesmith.core.ensemble_detector import VotingEnsembleDetector  # noqa: E402, F401

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
    "DegradationRateFeaturizer",
    "SeasonalFeaturizer",
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
    __all__.extend(
        [
            "DecomposeTransformer",
            "DetrendTransformer",
            "DeseasonalizeTransformer",
            "detect_trend",
            "detect_seasonality",
        ]
    )

# Conditionally add change point detection exports
if HAS_CHANGEPOINT:
    __all__.extend(
        [
            "PELTDetector",
            "BayesianChangePointDetector",
            "CUSUMDetector",
            "preprocess_for_changepoint",
        ]
    )

# Conditionally add filter exports
if HAS_FILTERS:
    __all__.extend(
        [
            "ButterworthFilter",
            "SavitzkyGolayFilter",
        ]
    )

# Conditionally add advanced outlier detection exports
if HAS_ADVANCED_OUTLIERS:
    __all__.extend(
        [
            "HampelOutlierRemover",
            "IsolationForestOutlierRemover",
            "ZScoreOutlierRemover",
        ]
    )

# Conditionally add wavelet exports
if HAS_WAVELET:
    __all__.extend(
        [
            "WaveletDenoiser",
            "WaveletDetector",
        ]
    )

# Seasonal baseline detector (always available)
__all__.append("SeasonalBaselineDetector")

# Ensemble detector (always available)
__all__.append("VotingEnsembleDetector")
