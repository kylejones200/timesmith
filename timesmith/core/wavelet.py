"""Wavelet-based transformers and detectors for time series."""

import logging
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from timesmith.core.base import BaseDetector, BaseTransformer
from timesmith.core.tags import set_tags

logger = logging.getLogger(__name__)

# Optional PyWavelets for wavelet transforms
try:
    import pywt

    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logger.warning(
        "PyWavelets not available. Wavelet functionality will be unavailable. "
        "Install with: pip install PyWavelets"
    )


class WaveletDenoiser(BaseTransformer):
    """Wavelet-based signal denoising transformer.

    Uses wavelet thresholding to remove noise from time series signals.
    """

    def __init__(
        self,
        wavelet: str = "db4",
        threshold_mode: str = "soft",
        level: int = 5,
    ):
        """Initialize wavelet denoiser.

        Args:
            wavelet: Wavelet type (e.g., 'db4', 'haar', 'bior2.2').
            threshold_mode: Thresholding mode ('soft' or 'hard').
            level: Decomposition level.
        """
        if not PYWT_AVAILABLE:
            raise ImportError(
                "PyWavelets is required for WaveletDenoiser. "
                "Install with: pip install PyWavelets"
            )

        super().__init__()
        self.wavelet = wavelet
        self.threshold_mode = threshold_mode
        self.level = level

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "WaveletDenoiser":
        """Fit the transformer (no-op, but required by interface).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if isinstance(y, pd.Series):
            self.y_ = y.values
            self.index_ = y.index
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            self.y_ = y.iloc[:, 0].values
            self.index_ = y.index
        else:
            self.y_ = np.asarray(y, dtype=float)
            self.index_ = np.arange(len(self.y_))

        # Remove invalid values
        valid_mask = np.isfinite(self.y_)
        self.y_ = self.y_[valid_mask]
        self.index_ = self.index_[valid_mask]

        if len(self.y_) < 2**self.level:
            raise ValueError(
                f"Need at least {2**self.level} data points for level {self.level} decomposition"
            )

        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.Series:
        """Denoise time series using wavelet thresholding.

        Args:
            y: SeriesLike data (should match fit data).
            X: Optional exogenous data (ignored).

        Returns:
            Denoised SeriesLike data.
        """
        self._check_is_fitted()

        # Perform wavelet decomposition
        coeffs = pywt.wavedec(self.y_, self.wavelet, level=self.level)

        # Calculate threshold using universal threshold (Donoho & Johnstone)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Median absolute deviation
        threshold = sigma * np.sqrt(2 * np.log(len(self.y_)))

        # Apply threshold to detail coefficients (keep approximation)
        coeffs_thresh = [coeffs[0]]  # Keep approximation
        for c in coeffs[1:]:  # Threshold detail coefficients
            coeffs_thresh.append(pywt.threshold(c, threshold, mode=self.threshold_mode))

        # Reconstruct
        denoised = pywt.waverec(coeffs_thresh, self.wavelet)
        denoised = denoised[: len(self.y_)]  # Trim to original length

        return pd.Series(denoised, index=self.index_)


class WaveletDetector(BaseDetector):
    """Wavelet-based anomaly detector for time series.

    Detects anomalies by identifying large coefficients in wavelet
    detail levels, which indicate sudden changes or anomalies.
    """

    def __init__(
        self,
        wavelet: str = "db4",
        threshold_factor: float = 3.0,
        level: int = 5,
    ):
        """Initialize wavelet detector.

        Args:
            wavelet: Wavelet type (e.g., 'db4', 'haar', 'bior2.2').
            threshold_factor: Threshold factor for anomaly detection (in terms of MAD).
            level: Decomposition level.
        """
        if not PYWT_AVAILABLE:
            raise ImportError(
                "PyWavelets is required for WaveletDetector. "
                "Install with: pip install PyWavelets"
            )

        super().__init__()
        self.wavelet = wavelet
        self.threshold_factor = threshold_factor
        self.level = level

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="SeriesLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self, y: Any, X: Optional[Any] = None, **fit_params: Any
    ) -> "WaveletDetector":
        """Fit the detector.

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if isinstance(y, pd.Series):
            self.y_ = y.values
            self.index_ = y.index
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            self.y_ = y.iloc[:, 0].values
            self.index_ = y.index
        else:
            self.y_ = np.asarray(y, dtype=float)
            self.index_ = np.arange(len(self.y_))

        # Remove invalid values
        valid_mask = np.isfinite(self.y_)
        self.y_ = self.y_[valid_mask]
        self.index_ = self.index_[valid_mask]

        if len(self.y_) < 2**self.level:
            raise ValueError(
                f"Need at least {2**self.level} data points for level {self.level} decomposition"
            )

        self._is_fitted = True
        return self

    def score(self, y: Any, X: Optional[Any] = None) -> np.ndarray:
        """Compute anomaly scores using wavelet decomposition.

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).

        Returns:
            Array of anomaly scores.
        """
        self._check_is_fitted()

        # Perform wavelet decomposition
        coeffs = pywt.wavedec(self.y_, self.wavelet, level=self.level)

        # Focus on detail coefficients (high-frequency anomalies)
        detail_coeffs = coeffs[1:]

        # Calculate threshold for each detail level
        anomaly_scores = np.zeros(len(self.y_))

        for detail in detail_coeffs:
            if len(detail) == 0:
                continue

            # Use robust statistics (median, MAD)
            detail_abs = np.abs(detail)
            median_detail = np.median(detail_abs)
            mad = np.median(np.abs(detail_abs - median_detail))
            threshold = median_detail + self.threshold_factor * (mad / 0.6745)

            # Find anomalies in this detail level
            anomaly_mask = detail_abs > threshold

            if not np.any(anomaly_mask):
                continue

            # Map back to original time indices
            scale_factor = len(self.y_) // len(detail)
            anomaly_indices = np.where(anomaly_mask)[0]

            # Add scores
            for idx in anomaly_indices:
                start_idx = idx * scale_factor
                end_idx = min((idx + 1) * scale_factor, len(self.y_))
                anomaly_scores[start_idx:end_idx] += detail_abs[idx]

        return anomaly_scores

    def predict(
        self, y: Any, X: Optional[Any] = None, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Predict anomaly flags.

        Args:
            y: Target time series (should match fit data).
            X: Optional exogenous data (ignored).
            threshold: Optional threshold (uses percentile if not provided).

        Returns:
            Boolean array with True at anomalies.
        """
        scores = self.score(y, X)

        # Threshold based on percentile if not provided
        if threshold is None:
            threshold = (
                np.percentile(scores[scores > 0], 95) if np.any(scores > 0) else 0
            )

        flags = np.zeros(len(self.y_), dtype=bool)
        flags[scores > threshold] = True

        logger.info(f"Wavelet detector found {flags.sum()} anomalies")

        return flags

    def get_wavelet_coefficients(self, y: Any) -> Tuple[np.ndarray, list]:
        """Get wavelet decomposition coefficients.

        Args:
            y: Time series data (should match fit data).

        Returns:
            Tuple of (approximation, details).
        """
        self._check_is_fitted()

        coeffs = pywt.wavedec(self.y_, self.wavelet, level=self.level)
        approximation = coeffs[0]
        details = coeffs[1:]

        return approximation, details
