"""Change point detection for time series."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from timesmith.core.base import BaseDetector
from timesmith.core.tags import set_tags
from timesmith.typing import SeriesLike

logger = logging.getLogger(__name__)

# Optional scipy for median filtering
try:
    from scipy.ndimage import median_filter

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

    def median_filter(arr, size):
        """Simple median filter using numpy (fallback when scipy not available)."""
        if size <= 1:
            return arr
        result = np.zeros_like(arr)
        half = size // 2
        for i in range(len(arr)):
            start = max(0, i - half)
            end = min(len(arr), i + half + 1)
            result[i] = np.median(arr[start:end])
        return result


# Optional ruptures library for PELT algorithm
try:
    import ruptures as rpt

    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    logger.warning(
        "ruptures library not installed. PELT functionality will be limited. "
        "Install with: pip install ruptures"
    )

# Optional numba for acceleration
try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Dummy decorator when numba is not available."""

        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


def preprocess_for_changepoint(
    y: SeriesLike,
    median_window: int = 5,
    detrend_window: int = 100,
) -> np.ndarray:
    """Preprocess time series for change point detection.

    Applies median filtering to remove spikes and baseline removal to eliminate drift.

    Args:
        y: Time series values.
        median_window: Window size for spike removal.
        detrend_window: Window size for baseline removal (0 to skip).

    Returns:
        Preprocessed time series.
    """
    if isinstance(y, pd.Series):
        y_arr = y.values
    elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y_arr = y.iloc[:, 0].values
    else:
        y_arr = np.asarray(y)

    if len(y_arr) == 0:
        raise ValueError("y cannot be empty")

    # Median filter to remove spikes while preserving sharp edges
    y_filtered = median_filter(y_arr, size=median_window)

    # Optional detrending (remove long-wavelength drift)
    if detrend_window > 0 and len(y_arr) > detrend_window:
        # Compute baseline with large median filter
        baseline = median_filter(y_filtered, size=detrend_window)
        # Remove baseline and restore median to preserve absolute scale
        y_processed = y_filtered - baseline + np.median(y_filtered)
    else:
        y_processed = y_filtered

    return y_processed


@njit(cache=True)
def _bayesian_changepoint_kernel(
    y: np.ndarray,
    hazard: float,
    beta0: float,
) -> np.ndarray:
    """Numba-optimized kernel for Bayesian online change-point detection.

    Args:
        y: Time series values (1D array).
        hazard: Hazard rate (1 / expected_segment_length).
        beta0: Prior variance parameter.

    Returns:
        Change point probabilities at each time step.
    """
    n = len(y)

    # Initialize
    run_length_probs = np.zeros(n + 1, dtype=np.float64)
    run_length_probs[0] = 1.0

    change_point_probs = np.zeros(n, dtype=np.float64)

    # Track sufficient statistics
    sum_x = np.zeros(n + 1, dtype=np.float64)
    sum_x2 = np.zeros(n + 1, dtype=np.float64)
    count = np.zeros(n + 1, dtype=np.float64)

    for t in range(n):
        x = y[t]

        # Roll arrays manually (Numba doesn't support np.roll directly)
        for idx in range(n, 0, -1):
            sum_x[idx] = sum_x[idx - 1]
            sum_x2[idx] = sum_x2[idx - 1]
            count[idx] = count[idx - 1]

        sum_x[0] = 0.0
        sum_x2[0] = 0.0
        count[0] = 0.0

        # Update statistics
        for idx in range(1, n + 1):
            sum_x[idx] += x
            sum_x2[idx] += x * x
            count[idx] += 1.0

        # Compute predictive probabilities
        predictive_probs = np.ones(n + 1, dtype=np.float64) * 1e-10

        max_r = min(t + 1, n)
        for r in range(max_r):
            if count[r] > 0.0:
                n_r = count[r]
                mean_r = sum_x[r] / n_r
                var_r = (sum_x2[r] / n_r - mean_r * mean_r) + beta0

                # Gaussian log predictive
                diff = x - mean_r
                predictive_probs[r] = np.exp(-0.5 * (diff * diff) / (var_r + 1e-6))

        # Growth probabilities (no change)
        run_length_probs = run_length_probs * predictive_probs

        # Change point probability
        cp_prob = np.sum(run_length_probs) * hazard
        change_point_probs[t] = cp_prob / (cp_prob + 1e-10)

        # Update run lengths (shift and apply hazard)
        for idx in range(n, 0, -1):
            run_length_probs[idx] = run_length_probs[idx - 1] * (1.0 - hazard)

        run_length_probs[0] = cp_prob

        # Normalize
        total = np.sum(run_length_probs)
        if total > 0.0:
            run_length_probs = run_length_probs / total

    return change_point_probs


class PELTDetector(BaseDetector):
    """Change point detector using PELT (Pruned Exact Linear Time) algorithm."""

    def __init__(
        self,
        penalty: Optional[float] = None,
        model: str = "l2",
        min_size: int = 3,
        jump: int = 1,
        preprocess: bool = True,
    ):
        """Initialize PELT detector.

        Args:
            penalty: Penalty value (higher = fewer change points).
                Auto-tuned if None.
            model: Cost function model ('l2' for mean shift,
                'rbf' for distributional change).
            min_size: Minimum segment length.
            jump: Subsample (1 = no subsampling).
            preprocess: Whether to preprocess data before detection.
        """
        if not RUPTURES_AVAILABLE:
            raise ImportError(
                "ruptures library required for PELT detector. "
                "Install with: pip install ruptures"
            )

        super().__init__()
        self.penalty = penalty
        self.model = model
        self.min_size = min_size
        self.jump = jump
        self.preprocess = preprocess

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "PELTDetector":
        """Fit the change point detector.

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
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

        if len(self.y_) == 0:
            raise ValueError("y cannot be empty")

        # Preprocess if requested
        if self.preprocess:
            self.y_processed_ = preprocess_for_changepoint(self.y_)
        else:
            self.y_processed_ = self.y_

        # Auto-tune penalty if not provided
        if self.penalty is None:
            n = len(self.y_processed_)
            self.penalty_ = np.log(n) * np.var(self.y_processed_)
            logger.info(f"Auto-tuned penalty: {self.penalty_:.2f}")
        else:
            self.penalty_ = self.penalty

        self._is_fitted = True
        return self

    def score(self, y: Any, X: Optional[Any] = None) -> np.ndarray:
        """Compute change point scores (indices of detected change points).

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).

        Returns:
            Array of change point indices.
        """
        self._check_is_fitted()

        # Use processed data
        y_data = self.y_processed_

        # Create PELT model
        algo = rpt.Pelt(model=self.model, min_size=self.min_size, jump=self.jump)

        # Fit and predict
        algo.fit(y_data.reshape(-1, 1))
        change_points = algo.predict(pen=self.penalty_)

        # Remove the final point (always equals length of signal)
        change_points = np.array(change_points[:-1])

        logger.info(
            f"PELT detected {len(change_points)} change points (model={self.model})"
        )

        return change_points

    def predict(
        self, y: Any, X: Optional[Any] = None, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Predict change point flags (binary array).

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).
            threshold: Optional threshold (ignored for PELT, uses penalty instead).

        Returns:
            Boolean array with True at change points.
        """
        change_points = self.score(y, X)

        # Create boolean array
        flags = np.zeros(len(self.y_), dtype=bool)
        flags[change_points] = True

        return flags


class BayesianChangePointDetector(BaseDetector):
    """Change point detector using Bayesian online change-point detection."""

    def __init__(
        self,
        expected_segment_length: float = 100.0,
        threshold: float = 0.5,
        preprocess: bool = True,
    ):
        """Initialize Bayesian change point detector.

        Args:
            expected_segment_length: Expected length between change points.
            threshold: Probability threshold for flagging change points (0-1).
            preprocess: Whether to preprocess data before detection.
        """
        super().__init__()
        self.expected_segment_length = expected_segment_length
        self.threshold = threshold
        self.preprocess = preprocess

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "BayesianChangePointDetector":
        """Fit the change point detector.

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if isinstance(y, pd.Series):
            self.y_ = y.values
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            self.y_ = y.iloc[:, 0].values
        else:
            self.y_ = np.asarray(y, dtype=np.float64)

        if len(self.y_) == 0:
            raise ValueError("y cannot be empty")

        # Preprocess if requested
        if self.preprocess:
            self.y_processed_ = preprocess_for_changepoint(self.y_)
        else:
            self.y_processed_ = self.y_.astype(np.float64)

        self._is_fitted = True
        return self

    def score(self, y: Any, X: Optional[Any] = None) -> np.ndarray:
        """Compute change point probabilities.

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).

        Returns:
            Array of change point probabilities at each time step.
        """
        self._check_is_fitted()

        y_data = self.y_processed_
        hazard = 1.0 / self.expected_segment_length
        beta0 = np.var(y_data)

        # Call optimized kernel
        change_point_probs = _bayesian_changepoint_kernel(y_data, hazard, beta0)

        return change_point_probs

    def predict(
        self, y: Any, X: Optional[Any] = None, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Predict change point flags.

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).
            threshold: Optional threshold (uses self.threshold if not provided).

        Returns:
            Boolean array with True at change points.
        """
        threshold = threshold or self.threshold
        probs = self.score(y, X)

        # Extract change points above threshold
        change_points = np.where(probs > threshold)[0]

        logger.info(
            f"Bayesian detection found {len(change_points)} change points "
            f"(threshold={threshold})"
        )

        # Create boolean array
        flags = np.zeros(len(self.y_), dtype=bool)
        flags[change_points] = True

        return flags


class CUSUMDetector(BaseDetector):
    """Change point detector using CUSUM (Cumulative Sum) method.

    CUSUM detects changes by tracking cumulative deviations from a baseline.
    Detects change points by looking for significant shifts in the rate values.
    """

    def __init__(
        self,
        baseline_window: int = 10,
        threshold: float = 3.0,
    ):
        """Initialize CUSUM detector.

        Args:
            baseline_window: Window size for detecting changes.
            threshold: Z-score threshold for change detection (lower = more sensitive).
        """
        super().__init__()
        self.baseline_window = baseline_window
        self.threshold = threshold

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "CUSUMDetector":
        """Fit the change point detector.

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
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

        if len(self.y_) == 0:
            raise ValueError("y cannot be empty")

        if len(self.y_) < self.baseline_window * 2:
            raise ValueError(
                f"Need at least {self.baseline_window * 2} data points for CUSUM"
            )

        self._is_fitted = True
        return self

    def score(self, y: Any, X: Optional[Any] = None) -> np.ndarray:
        """Compute change point scores (indices of detected change points).

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).

        Returns:
            Array of change point indices.
        """
        self._check_is_fitted()

        rates = self.y_
        change_points = []
        skip_until = 0

        # Slide a window and compare statistics before/after each point
        for i in range(self.baseline_window, len(rates) - self.baseline_window):
            # Skip if we're in a cooldown period after detecting a change
            if i < skip_until:
                continue

            # Get windows before and after this point
            before = rates[max(0, i - self.baseline_window) : i]
            after = rates[i : min(len(rates), i + self.baseline_window)]

            if len(before) > 0 and len(after) > 0:
                # Calculate means
                mean_before = np.mean(before)
                mean_after = np.mean(after)

                # Calculate pooled standard deviation
                std_before = np.std(before)
                std_after = np.std(after)
                pooled_std = np.sqrt(
                    (std_before**2 + std_after**2) / 2 + 1e-10
                )  # Add small constant to avoid div by zero

                # Calculate z-score of difference in means
                z_score = abs(mean_before - mean_after) / pooled_std

                if z_score > self.threshold:
                    change_points.append(i)
                    # Skip ahead to avoid detecting the same change multiple times
                    skip_until = i + self.baseline_window

        change_points = np.array(change_points)

        logger.info(f"CUSUM detected {len(change_points)} change points")

        return change_points

    def predict(
        self, y: Any, X: Optional[Any] = None, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Predict change point flags (binary array).

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).
            threshold: Optional threshold (ignored for CUSUM, uses self.threshold).

        Returns:
            Boolean array with True at change points.
        """
        change_points = self.score(y, X)

        # Create boolean array
        flags = np.zeros(len(self.y_), dtype=bool)
        flags[change_points] = True

        return flags
