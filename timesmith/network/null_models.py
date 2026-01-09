"""Null models and significance testing for network metrics.

Provides surrogate data generation and statistical significance testing
for time series network analysis.
"""

import logging
from typing import Callable, Dict, Literal, Optional
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SurrogateMethod = Literal["shuffle", "phase", "circular", "iaaft", "block_bootstrap"]


@dataclass
class NetworkSignificanceResult:
    """Results from network significance testing.

    Attributes:
        metric_name: Name of the metric being tested.
        observed_value: Observed value of the metric.
        null_mean: Mean of the null distribution.
        null_std: Standard deviation of the null distribution.
        z_score: Z-score: (observed - mean) / std.
        p_value: Two-tailed p-value (approximate, based on normal distribution).
        n_surrogates: Number of surrogates used.
        surrogate_method: Method used to generate surrogates.
        significant: Whether the result is significant at alpha=0.05 (two-tailed).
        confidence_interval: 95% confidence interval for the metric under the null.
    """

    metric_name: str
    observed_value: float
    null_mean: float
    null_std: float
    z_score: float
    p_value: float
    n_surrogates: int
    surrogate_method: str
    significant: bool
    confidence_interval: tuple[float, float]
    alpha: float = 0.05

    def __str__(self) -> str:
        """String representation of the result."""
        sig_str = "significant" if self.significant else "not significant"
        return (
            f"{self.metric_name}: {self.observed_value:.4f} "
            f"(z={self.z_score:.3f}, p={self.p_value:.4f}, {sig_str})"
        )


def generate_surrogate(
    x: np.ndarray,
    method: SurrogateMethod = "shuffle",
    rng: Optional[np.random.Generator] = None,
    **kwargs
) -> np.ndarray:
    """Generate a surrogate time series using the specified method.

    Args:
        x: Original time series.
        method: Surrogate generation method.
        rng: Random number generator.
        **kwargs: Additional arguments for surrogate generation.

    Returns:
        Surrogate time series.
    """
    if rng is None:
        rng = np.random.default_rng()

    if method == "shuffle":
        return rng.permutation(x)

    elif method == "phase":
        # Phase randomization (preserves power spectrum)
        fft = np.fft.fft(x)
        phases = np.angle(fft)
        random_phases = rng.uniform(0, 2 * np.pi, len(phases))
        # Keep DC and Nyquist components real
        if len(phases) > 0:
            random_phases[0] = 0
            if len(phases) % 2 == 0:
                random_phases[len(phases) // 2] = 0
        fft_surrogate = np.abs(fft) * np.exp(1j * random_phases)
        return np.real(np.fft.ifft(fft_surrogate))

    elif method == "circular":
        # Circular shift
        shift = rng.integers(0, len(x))
        return np.roll(x, shift)

    elif method == "iaaft":
        # Iterative Amplitude Adjusted Fourier Transform (simplified)
        # Full IAAFT is more complex, this is a basic implementation
        target_amplitude = np.abs(np.fft.fft(x))
        surrogate = rng.permutation(x)
        for _ in range(kwargs.get('max_iter', 100)):
            fft_surrogate = np.fft.fft(surrogate)
            phases = np.angle(fft_surrogate)
            fft_surrogate = target_amplitude * np.exp(1j * phases)
            surrogate = np.real(np.fft.ifft(fft_surrogate))
            # Rank-order match to original
            sorted_indices = np.argsort(surrogate)
            original_sorted = np.sort(x)
            surrogate = original_sorted[np.argsort(sorted_indices)]
        return surrogate

    elif method == "block_bootstrap":
        # Block bootstrap (preserves local structure)
        block_size = kwargs.get('block_size', 10)
        n_blocks = (len(x) + block_size - 1) // block_size
        blocks = [x[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
        sampled_blocks = [blocks[i] for i in rng.integers(0, len(blocks), size=n_blocks)]
        return np.concatenate(sampled_blocks)[:len(x)]

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_network_metric_significance(
    x: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    method: SurrogateMethod = "shuffle",
    n_surrogates: int = 200,
    alpha: float = 0.05,
    metric_name: str = "metric",
    rng: Optional[np.random.Generator] = None,
    **kwargs
) -> NetworkSignificanceResult:
    """Test significance of a network metric against null distribution.

    Args:
        x: Original time series.
        metric_fn: Function that takes a time series and returns a metric value.
        method: Surrogate generation method.
        n_surrogates: Number of surrogate series to generate.
        alpha: Significance level (two-tailed).
        metric_name: Name of the metric for reporting.
        rng: Random number generator.
        **kwargs: Additional arguments for surrogate generation.

    Returns:
        NetworkSignificanceResult with test results.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Compute observed metric
    observed = metric_fn(x)

    # Generate surrogates and compute metrics
    null_values = []
    for _ in range(n_surrogates):
        surrogate = generate_surrogate(x, method=method, rng=rng, **kwargs)
        try:
            null_value = metric_fn(surrogate)
            null_values.append(null_value)
        except Exception as e:
            logger.warning(f"Failed to compute metric for surrogate: {e}")
            continue

    if len(null_values) == 0:
        raise RuntimeError("All surrogate metric computations failed")

    null_values = np.array(null_values)
    null_mean = float(np.mean(null_values))
    null_std = float(np.std(null_values))

    # Compute z-score and p-value
    if null_std > 0:
        z_score = float((observed - null_mean) / null_std)
        # Two-tailed p-value (approximate, using normal distribution)
        from scipy import stats
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_score))))
    else:
        z_score = 0.0
        p_value = 1.0

    # Confidence interval (95%)
    ci_lower = float(np.percentile(null_values, 2.5))
    ci_upper = float(np.percentile(null_values, 97.5))
    confidence_interval = (ci_lower, ci_upper)

    significant = p_value < alpha

    return NetworkSignificanceResult(
        metric_name=metric_name,
        observed_value=observed,
        null_mean=null_mean,
        null_std=null_std,
        z_score=z_score,
        p_value=p_value,
        n_surrogates=len(null_values),
        surrogate_method=method,
        significant=significant,
        confidence_interval=confidence_interval,
        alpha=alpha,
    )

