"""Core base classes, tags, and validation for time series estimators."""

import logging

logger = logging.getLogger(__name__)

from timesmith.core.base import (
    BaseDetector,
    BaseEstimator,
    BaseFeaturizer,
    BaseForecaster,
    BaseObject,
    BaseTransformer,
)
from timesmith.core.featurizers import (
    DegradationRateFeaturizer,
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
        CUSUMDetector,
        preprocess_for_changepoint,
    )
    HAS_CHANGEPOINT = True
except ImportError:
    HAS_CHANGEPOINT = False

# Filters (requires optional scipy)
try:
    from timesmith.core.filters import (
        ButterworthFilter,
        SavitzkyGolayFilter,
    )
    HAS_FILTERS = True
except ImportError:
    HAS_FILTERS = False

# Advanced outlier detection (optional sklearn for IsolationForest)
try:
    from timesmith.core.outliers import (
        HampelOutlierRemover,
        IsolationForestOutlierRemover,
        ZScoreOutlierRemover,
    )
    HAS_ADVANCED_OUTLIERS = True
except ImportError:
    HAS_ADVANCED_OUTLIERS = False

# Wavelet methods (optional PyWavelets)
try:
    from timesmith.core.wavelet import (
        WaveletDenoiser,
        WaveletDetector,
    )
    HAS_WAVELET = True
except ImportError:
    HAS_WAVELET = False

# Seasonal baseline detection (always available)
from timesmith.core.seasonal import SeasonalBaselineDetector

# Ensemble detection (always available)
from timesmith.core.ensemble_detector import VotingEnsembleDetector

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
    "DegradationRateFeaturizer",
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
        "CUSUMDetector",
        "preprocess_for_changepoint",
    ])

# Conditionally add filter exports
if HAS_FILTERS:
    __all__.extend([
        "ButterworthFilter",
        "SavitzkyGolayFilter",
    ])

# Conditionally add advanced outlier detection exports
if HAS_ADVANCED_OUTLIERS:
    __all__.extend([
        "HampelOutlierRemover",
        "IsolationForestOutlierRemover",
        "ZScoreOutlierRemover",
    ])

# Conditionally add wavelet exports
if HAS_WAVELET:
    __all__.extend([
        "WaveletDenoiser",
        "WaveletDetector",
    ])

# Seasonal baseline detector (always available)
__all__.append("SeasonalBaselineDetector")

# Ensemble detector (always available)
__all__.append("VotingEnsembleDetector")

