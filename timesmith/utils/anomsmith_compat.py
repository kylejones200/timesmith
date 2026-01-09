"""Compatibility utilities for AnomSmith integration.

NOTE: This module is NOT imported by default in timesmith.__init__ to avoid
circular imports. Timesmith (foundation) must never import downstream repos
by default. Users must explicitly import these utilities if needed:

    from timesmith.utils.anomsmith_compat import (
        convert_to_anomsmith_format,
        convert_from_anomsmith_format,
        get_anomsmith_detector,
        list_anomsmith_detectors,
    )

This ensures a clean dependency graph where timesmith does not depend on
downstream repos like anomsmith.
"""

import logging
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from timesmith.typing import SeriesLike

logger = logging.getLogger(__name__)

# Try to import anomsmith
try:
    import anomsmith as am
    HAS_ANOMSMITH = True
except ImportError:
    HAS_ANOMSMITH = False
    am = None
    logger.warning(
        "anomsmith not installed. AnomSmith compatibility functions will not be available. "
        "Install with: pip install anomsmith"
    )


def convert_to_anomsmith_format(data: SeriesLike) -> np.ndarray:
    """Convert TimeSmith SeriesLike to AnomSmith format.

    Args:
        data: TimeSmith SeriesLike data.

    Returns:
        NumPy array in format AnomSmith expects.
    """
    if isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            return data.iloc[:, 0].values
        else:
            raise ValueError("AnomSmith supports single series only")
    else:
        return np.asarray(data)


def convert_from_anomsmith_format(
    results: Any,
    index: Optional[pd.Index] = None,
    name: str = "result"
) -> pd.Series:
    """Convert AnomSmith results to TimeSmith SeriesLike format.

    Args:
        results: Results from AnomSmith detector.
        index: Optional index for the resulting Series.
        name: Name for the resulting Series.

    Returns:
        Pandas Series in TimeSmith format.
    """
    if isinstance(results, pd.Series):
        return results
    elif isinstance(results, (list, np.ndarray)):
        results_array = np.asarray(results)
        if index is None:
            index = pd.RangeIndex(len(results_array))
        return pd.Series(results_array, index=index, name=name)
    else:
        # Try to convert scalar or other types
        if index is None:
            index = pd.RangeIndex(1)
        return pd.Series([results], index=index, name=name)


def get_anomsmith_detector(detector_name: str, **params) -> Any:
    """Get an AnomSmith detector by name.

    Args:
        detector_name: Name of the AnomSmith detector.
        **params: Parameters to pass to the detector.

    Returns:
        AnomSmith detector instance.

    Raises:
        ImportError: If anomsmith is not installed.
        ValueError: If detector name is not found.
    """
    if not HAS_ANOMSMITH:
        raise ImportError(
            "anomsmith is required. Install with: pip install anomsmith"
        )

    if hasattr(am, detector_name):
        detector_class = getattr(am, detector_name)
        return detector_class(**params)
    else:
        available = [x for x in dir(am) if not x.startswith('_')]
        raise ValueError(
            f"AnomSmith detector '{detector_name}' not found. "
            f"Available detectors: {available}"
        )


def list_anomsmith_detectors() -> list:
    """List all available AnomSmith detectors.

    Returns:
        List of detector names.
    """
    if not HAS_ANOMSMITH:
        return []

    # Filter for detector-like classes (heuristic)
    detectors = []
    for name in dir(am):
        if name.startswith('_'):
            continue
        obj = getattr(am, name)
        if isinstance(obj, type):
            # Check if it looks like a detector (has predict, detect, or score)
            if any(hasattr(obj, method) for method in ['predict', 'detect', 'score', 'fit', 'train']):
                detectors.append(name)

    return sorted(detectors)

