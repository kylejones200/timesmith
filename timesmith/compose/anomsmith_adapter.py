"""Adapter to integrate AnomSmith detectors with TimeSmith."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseDetector
from timesmith.core.tags import set_tags
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
        "anomsmith not installed. AnomSmith adapters will not be available. "
        "Install with: pip install anomsmith"
    )


class AnomSmithAdapter(BaseDetector):
    """Adapter to use AnomSmith detectors within TimeSmith.

    This adapter allows you to use any AnomSmith detector as a TimeSmith detector,
    ensuring full compatibility between the two libraries.

    Args:
        anomsmith_detector: An AnomSmith detector instance or detector name.
        detector_params: Optional parameters to pass to AnomSmith detector.

    Example:
        >>> import timesmith as ts
        >>> import anomsmith as am
        >>> 
        >>> # Create AnomSmith detector
        >>> am_detector = am.SomeDetector(threshold=3.0)
        >>> 
        >>> # Wrap it for TimeSmith
        >>> detector = ts.AnomSmithAdapter(am_detector)
        >>> detector.fit(y)
        >>> anomalies = detector.predict(y)
    """

    def __init__(
        self,
        anomsmith_detector: Any = None,
        detector_name: Optional[str] = None,
        detector_params: Optional[dict] = None,
    ):
        """Initialize AnomSmith adapter.

        Args:
            anomsmith_detector: An AnomSmith detector instance.
            detector_name: Name of AnomSmith detector to create (if detector not provided).
            detector_params: Parameters for creating detector (if using detector_name).
        """
        super().__init__()
        
        if not HAS_ANOMSMITH:
            raise ImportError(
                "anomsmith is required for AnomSmithAdapter. "
                "Install with: pip install anomsmith"
            )

        if anomsmith_detector is not None:
            self.anomsmith_detector = anomsmith_detector
        elif detector_name is not None:
            # Try to create detector from name
            if hasattr(am, detector_name):
                detector_class = getattr(am, detector_name)
                params = detector_params or {}
                self.anomsmith_detector = detector_class(**params)
            else:
                raise ValueError(f"AnomSmith detector '{detector_name}' not found")
        else:
            raise ValueError("Must provide either anomsmith_detector or detector_name")

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "AnomSmithAdapter":
        """Fit the AnomSmith detector.

        Args:
            y: Time series data.
            X: Not used, present for API compatibility.
            **fit_params: Additional fit parameters passed to AnomSmith detector.

        Returns:
            Self for method chaining.
        """
        # Convert to format AnomSmith expects
        if isinstance(y, pd.Series):
            y_for_am = y.values
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y_for_am = y.iloc[:, 0].values
            else:
                raise ValueError("AnomSmith adapters support single series only")
        else:
            y_for_am = np.asarray(y)

        # Fit AnomSmith detector
        if hasattr(self.anomsmith_detector, 'fit'):
            self.anomsmith_detector.fit(y_for_am, **fit_params)
        elif hasattr(self.anomsmith_detector, 'train'):
            self.anomsmith_detector.train(y_for_am, **fit_params)

        # Store original data for compatibility
        self.y_ = y
        self._is_fitted = True
        return self

    def predict(self, y: Any, X: Optional[Any] = None) -> pd.Series:
        """Detect anomalies using AnomSmith detector.

        Args:
            y: Time series data to detect anomalies in.
            X: Not used, present for API compatibility.

        Returns:
            Boolean Series indicating anomalies (True = anomaly).
        """
        self._check_is_fitted()

        # Convert to format AnomSmith expects
        if isinstance(y, pd.Series):
            y_for_am = y.values
            index = y.index
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y_for_am = y.iloc[:, 0].values
                index = y.index
            else:
                raise ValueError("AnomSmith adapters support single series only")
        else:
            y_for_am = np.asarray(y)
            index = pd.RangeIndex(len(y_for_am))

        # Predict using AnomSmith detector
        if hasattr(self.anomsmith_detector, 'predict'):
            anomalies = self.anomsmith_detector.predict(y_for_am)
        elif hasattr(self.anomsmith_detector, 'detect'):
            anomalies = self.anomsmith_detector.detect(y_for_am)
        elif hasattr(self.anomsmith_detector, 'score'):
            # If only score is available, threshold it
            scores = self.anomsmith_detector.score(y_for_am)
            threshold = getattr(self.anomsmith_detector, 'threshold', 3.0)
            anomalies = scores > threshold
        else:
            raise AttributeError(
                "AnomSmith detector must have 'predict', 'detect', or 'score' method"
            )

        # Convert to boolean array if needed
        if isinstance(anomalies, (list, np.ndarray)):
            anomalies = np.asarray(anomalies, dtype=bool)
        else:
            anomalies = np.asarray([bool(x) for x in anomalies])

        return pd.Series(anomalies, index=index, name='anomaly')

    def score(self, y: Any, X: Optional[Any] = None) -> pd.Series:
        """Get anomaly scores from AnomSmith detector.

        Args:
            y: Time series data.
            X: Not used, present for API compatibility.

        Returns:
            Series with anomaly scores.
        """
        self._check_is_fitted()

        # Convert to format AnomSmith expects
        if isinstance(y, pd.Series):
            y_for_am = y.values
            index = y.index
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y_for_am = y.iloc[:, 0].values
                index = y.index
            else:
                raise ValueError("AnomSmith adapters support single series only")
        else:
            y_for_am = np.asarray(y)
            index = pd.RangeIndex(len(y_for_am))

        # Get scores from AnomSmith detector
        if hasattr(self.anomsmith_detector, 'score'):
            scores = self.anomsmith_detector.score(y_for_am)
        elif hasattr(self.anomsmith_detector, 'predict_proba'):
            # Use probability of anomaly class
            proba = self.anomsmith_detector.predict_proba(y_for_am)
            if proba.ndim > 1:
                scores = proba[:, -1]  # Last column is usually anomaly class
            else:
                scores = proba
        else:
            # Fallback: use predict and convert to scores
            anomalies = self.predict(y, X)
            scores = anomalies.astype(float)

        return pd.Series(scores, index=index, name='anomaly_score')

