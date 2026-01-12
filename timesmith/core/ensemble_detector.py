"""Ensemble anomaly detection methods."""

import logging
from typing import Any, List, Optional

import numpy as np

from timesmith.core.base import BaseDetector
from timesmith.core.tags import set_tags

logger = logging.getLogger(__name__)


class VotingEnsembleDetector(BaseDetector):
    """Voting ensemble for anomaly detection.

    Combines multiple anomaly detectors using majority voting.
    Flags a point as anomalous if at least `threshold` detectors agree.
    """

    def __init__(
        self,
        detectors: List[BaseDetector],
        threshold: int = 2,
    ):
        """Initialize voting ensemble detector.

        Args:
            detectors: List of BaseDetector instances to ensemble.
            threshold: Minimum number of detectors that must flag an anomaly.
        """
        super().__init__()
        self.detectors = detectors
        self.threshold = threshold

        if len(detectors) == 0:
            raise ValueError("Must provide at least one detector")

        if threshold < 1 or threshold > len(detectors):
            raise ValueError(
                f"threshold ({threshold}) must be between 1 and {len(detectors)}"
            )

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "VotingEnsembleDetector":
        """Fit all detectors in the ensemble.

        Args:
            y: Target time series.
            X: Optional exogenous data.
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        for detector in self.detectors:
            detector.fit(y, X, **fit_params)

        # Store data for scoring
        if isinstance(y, np.ndarray):
            self.y_ = y
        else:
            self.y_ = np.asarray(y)

        self._is_fitted = True
        return self

    def score(self, y: Any, X: Optional[Any] = None) -> np.ndarray:
        """Compute ensemble anomaly scores (number of detectors flagging each point).

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data.

        Returns:
            Array of scores (number of detectors flagging each point, 0 to n_detectors).
        """
        self._check_is_fitted()

        # Get predictions from all detectors
        predictions = []
        for detector in self.detectors:
            pred = detector.predict(y, X)
            predictions.append(pred)

        # Count votes (True = anomaly)
        votes = np.sum(predictions, axis=0)

        return votes.astype(float)

    def predict(
        self, y: Any, X: Optional[Any] = None, threshold: Optional[int] = None
    ) -> np.ndarray:
        """Predict anomaly flags using majority voting.

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data.
            threshold: Optional threshold (uses self.threshold if not provided).

        Returns:
            Boolean array with True at anomalies.
        """
        threshold = threshold or self.threshold
        scores = self.score(y, X)

        flags = scores >= threshold

        n_anomalies = flags.sum()
        logger.info(
            f"Voting ensemble detected {n_anomalies} anomalies "
            f"({n_anomalies / len(flags) * 100:.1f}%) "
            f"with threshold={threshold}/{len(self.detectors)}"
        )

        return flags
