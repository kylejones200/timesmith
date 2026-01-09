"""Causal inference tools for time series.

Transfer entropy and related causal measures.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseDetector
from timesmith.core.tags import set_tags

logger = logging.getLogger(__name__)


def transfer_entropy(
    x: np.ndarray,
    y: np.ndarray,
    lag: int = 1,
    bins: int = 10,
) -> float:
    """Compute transfer entropy from X to Y.

    Transfer entropy measures the amount of information transferred from X to Y,
    quantifying causal influence.

    Args:
        x: Source time series.
        y: Target time series (must have same length as x).
        lag: Time lag for past values.
        bins: Number of bins for discretization.

    Returns:
        Transfer entropy from X to Y (non-negative, bits).
    """
    if len(x) != len(y):
        raise ValueError(f"Series must have same length: {len(x)} != {len(y)}")

    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")

    # Discretize series
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)

    if x_min == x_max or y_min == y_max:
        return 0.0

    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)
    x_edges[-1] += 1e-10
    y_edges[-1] += 1e-10

    x_disc = np.digitize(x, x_edges) - 1
    y_disc = np.digitize(y, y_edges) - 1

    # Clip to valid range
    x_disc = np.clip(x_disc, 0, bins - 1)
    y_disc = np.clip(y_disc, 0, bins - 1)

    # Compute conditional entropies
    # H(Y_t | Y_{t-lag})
    n = len(y)
    if n <= lag:
        return 0.0

    y_t = y_disc[lag:]
    y_past = y_disc[: n - lag]

    # H(Y_t | Y_{t-lag})
    joint_counts = np.zeros((bins, bins), dtype=np.int32)
    for i in range(len(y_t)):
        if 0 <= y_t[i] < bins and 0 <= y_past[i] < bins:
            joint_counts[y_t[i], y_past[i]] += 1

    total = np.sum(joint_counts)
    if total == 0:
        return 0.0

    joint_probs = joint_counts / total
    joint_entropy = -np.sum(joint_probs[joint_probs > 0] * np.log2(joint_probs[joint_probs > 0]))

    y_past_counts = np.bincount(y_past[y_past >= 0], minlength=bins)
    y_past_probs = y_past_counts / (np.sum(y_past_counts) + 1e-10)
    y_past_entropy = -np.sum(y_past_probs[y_past_probs > 0] * np.log2(y_past_probs[y_past_probs > 0]))

    h_y_given_y_past = joint_entropy - y_past_entropy

    # H(Y_t | Y_{t-lag}, X_{t-lag})
    x_past = x_disc[: n - lag]
    joint_counts_3d = np.zeros((bins, bins, bins), dtype=np.int32)
    for i in range(len(y_t)):
        if (
            0 <= y_t[i] < bins
            and 0 <= y_past[i] < bins
            and 0 <= x_past[i] < bins
        ):
            joint_counts_3d[y_t[i], y_past[i], x_past[i]] += 1

    total_3d = np.sum(joint_counts_3d)
    if total_3d == 0:
        return 0.0

    joint_probs_3d = joint_counts_3d / total_3d
    joint_entropy_3d = -np.sum(
        joint_probs_3d[joint_probs_3d > 0] * np.log2(joint_probs_3d[joint_probs_3d > 0])
    )

    marginal_counts = np.zeros((bins, bins), dtype=np.int32)
    for i in range(len(y_past)):
        if 0 <= y_past[i] < bins and 0 <= x_past[i] < bins:
            marginal_counts[y_past[i], x_past[i]] += 1

    marginal_probs = marginal_counts / (np.sum(marginal_counts) + 1e-10)
    marginal_entropy = -np.sum(marginal_probs[marginal_probs > 0] * np.log2(marginal_probs[marginal_probs > 0]))

    h_y_given_y_past_x_past = joint_entropy_3d - marginal_entropy

    # Transfer entropy
    te = h_y_given_y_past - h_y_given_y_past_x_past
    return float(max(0.0, te))  # Ensure non-negative


class TransferEntropyDetector(BaseDetector):
    """Detector using transfer entropy for causal inference.

    Uses transfer entropy to detect causal relationships and anomalies.
    """

    def __init__(self, lag: int = 1, bins: int = 10, threshold: Optional[float] = None):
        """Initialize transfer entropy detector.

        Args:
            lag: Time lag for past values.
            bins: Number of bins for discretization.
            threshold: Optional threshold for binary classification.
        """
        super().__init__()
        self.lag = lag
        self.bins = bins
        self.threshold = threshold

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "TransferEntropyDetector":
        """Fit the detector (computes transfer entropy if X provided).

        Args:
            y: Target time series.
            X: Optional source time series for causal inference.
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if isinstance(y, pd.Series):
            self.y_ = y.values
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            self.y_ = y.iloc[:, 0].values
        else:
            self.y_ = np.asarray(y)

        if X is not None:
            if isinstance(X, pd.Series):
                self.X_ = X.values
            elif isinstance(X, pd.DataFrame) and X.shape[1] == 1:
                self.X_ = X.iloc[:, 0].values
            else:
                self.X_ = np.asarray(X)

            # Compute transfer entropy
            self.te_ = transfer_entropy(self.X_, self.y_, lag=self.lag, bins=self.bins)
        else:
            self.X_ = None
            self.te_ = None

        self._is_fitted = True
        return self

    def score(self, y: Any, X: Optional[Any] = None) -> Any:
        """Compute transfer entropy scores.

        Args:
            y: Target time series.
            X: Optional source time series.

        Returns:
            Transfer entropy score(s).
        """
        self._check_is_fitted()

        if X is None:
            if self.X_ is None:
                raise ValueError("X must be provided for scoring")
            X = self.X_

        if isinstance(y, pd.Series):
            y_vals = y.values
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y_vals = y.iloc[:, 0].values
        else:
            y_vals = np.asarray(y)

        if isinstance(X, pd.Series):
            x_vals = X.values
        elif isinstance(X, pd.DataFrame) and X.shape[1] == 1:
            x_vals = X.iloc[:, 0].values
        else:
            x_vals = np.asarray(X)

        te = transfer_entropy(x_vals, y_vals, lag=self.lag, bins=self.bins)
        return te

    def predict(self, y: Any, X: Optional[Any] = None, threshold: Optional[float] = None) -> Any:
        """Predict causal relationships based on transfer entropy.

        Args:
            y: Target time series.
            X: Optional source time series.
            threshold: Optional threshold for binary classification.

        Returns:
            Boolean array indicating causal relationship (if threshold provided),
            or transfer entropy value.
        """
        self._check_is_fitted()

        threshold = threshold or self.threshold
        score = self.score(y, X)

        if threshold is not None:
            return score >= threshold
        else:
            return score

