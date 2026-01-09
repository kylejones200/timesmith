"""Multiscale graph analysis for time series.

Coarse-grains time series at multiple scales and computes graph features
at each scale to create a scale signature for detection stability.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from timesmith.network._constructors import build_hvg, build_nvg, build_recurrence_network, build_transition_network
from timesmith.network.graph import Graph

logger = logging.getLogger(__name__)


def coarse_grain(x: np.ndarray, scale: int, method: str = "mean") -> np.ndarray:
    """Coarse-grain a time series by aggregating points at a given scale.

    Args:
        x: Input time series (1D array).
        scale: Coarse-graining scale (number of points to aggregate).
        method: Aggregation method: "mean", "median", "max", "min", "std".

    Returns:
        Coarse-grained time series.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    if scale >= len(x):
        raise ValueError(f"scale ({scale}) must be less than series length ({len(x)})")

    n = len(x)
    n_coarse = n // scale

    # Truncate to multiple of scale
    x_truncated = x[:n_coarse * scale]
    x_reshaped = x_truncated.reshape(n_coarse, scale)

    if method == "mean":
        return np.mean(x_reshaped, axis=1)
    elif method == "median":
        return np.median(x_reshaped, axis=1)
    elif method == "max":
        return np.max(x_reshaped, axis=1)
    elif method == "min":
        return np.min(x_reshaped, axis=1)
    elif method == "std":
        return np.std(x_reshaped, axis=1, ddof=1)
    else:
        raise ValueError(f"Unknown method: {method}. Must be one of: mean, median, max, min, std")


class MultiscaleGraphs:
    """Multiscale graph analysis for time series.

    Analyzes time series at multiple temporal scales by coarse-graining
    and computing graph features at each scale. Creates a scale signature
    (feature vector across scales) useful for detection stability.
    """

    def __init__(
        self,
        method: str = "hvg",
        scales: Optional[List[int]] = None,
        coarse_method: str = "mean",
        **method_kwargs
    ):
        """Initialize multiscale graph analyzer.

        Args:
            method: Network method: 'hvg', 'nvg', 'recurrence', 'transition'.
            scales: List of coarse-graining scales. If None, uses [1, 2, 4, 8, 16].
            coarse_method: Coarse-graining aggregation: "mean", "median", "max", "min", "std".
            **method_kwargs: Additional parameters for the network builder.
        """
        self.method = method.lower()
        if scales is None:
            # Default scales: powers of 2 up to reasonable limit
            self.scales = [1, 2, 4, 8, 16]
        else:
            self.scales = sorted(scales)  # Sort for consistency
        self.coarse_method = coarse_method
        self.method_kwargs = method_kwargs

        self.x_ = None
        self.scale_stats_ = None

    def fit(self, x: np.ndarray) -> "MultiscaleGraphs":
        """Fit the multiscale analyzer to a time series.

        Args:
            x: Input time series (1D array).

        Returns:
            Self for method chaining.
        """
        x = np.asarray(x, dtype=np.float64).squeeze()
        if x.ndim != 1:
            raise ValueError("Input must be a 1D array")

        if len(x) < max(self.scales):
            raise ValueError(
                f"Series length ({len(x)}) must be >= max scale ({max(self.scales)})"
            )

        self.x_ = x
        self.scale_stats_ = {}

        # Compute graph features at each scale
        for scale in self.scales:
            try:
                # Coarse-grain the series
                if scale == 1:
                    x_coarse = self.x_
                else:
                    x_coarse = coarse_grain(self.x_, scale=scale, method=self.coarse_method)

                # Skip if coarse-grained series is too short
                if len(x_coarse) < 10:
                    self.scale_stats_[scale] = None
                    continue

                # Build graph at this scale (using dispatch dictionary)
                method_builders = {
                    "hvg": lambda x: build_hvg(
                        x,
                        weighted=self.method_kwargs.get('weighted', False),
                        limit=self.method_kwargs.get('limit'),
                        directed=self.method_kwargs.get('directed', False),
                    ),
                    "nvg": lambda x: build_nvg(
                        x,
                        weighted=self.method_kwargs.get('weighted', False),
                        limit=self.method_kwargs.get('limit'),
                        directed=self.method_kwargs.get('directed', False),
                    ),
                    "recurrence": lambda x: build_recurrence_network(
                        x,
                        threshold=self.method_kwargs.get('threshold'),
                        embedding_dimension=self.method_kwargs.get('embedding_dimension'),
                        time_delay=self.method_kwargs.get('time_delay', 1),
                        metric=self.method_kwargs.get('metric', 'euclidean'),
                        rule=self.method_kwargs.get('rule'),
                        k=self.method_kwargs.get('k'),
                    ),
                    "transition": lambda x: build_transition_network(
                        x,
                        n_bins=self.method_kwargs.get('n_bins', 10),
                        order=self.method_kwargs.get('order', 1),
                        symbolizer=self.method_kwargs.get('symbolizer'),
                    ),
                }

                builder = method_builders.get(self.method)
                if builder is None:
                    raise ValueError(f"Unknown method: {self.method}. Must be one of {list(method_builders.keys())}")

                G_nx = builder(x_coarse)
                # Handle methods that return (graph, matrix) vs just graph
                if isinstance(G_nx, tuple):
                    G_nx, A = G_nx

                # Convert to Graph object and get stats
                edges = list(G_nx.edges())
                graph = Graph(
                    edges=edges,
                    n_nodes=G_nx.number_of_nodes(),
                    directed=G_nx.is_directed(),
                    weighted=False,
                )

                stats = graph.summary()
                self.scale_stats_[scale] = stats

            except Exception as e:
                # Handle errors gracefully (e.g., constant series, too short)
                logger.warning(f"Failed to compute graph at scale {scale}: {e}")
                self.scale_stats_[scale] = None

        return self

    def scale_signature(self, features: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Get scale signature (feature values across scales).

        Args:
            features: List of feature names to include. If None, uses common features:
                ['n_nodes', 'n_edges', 'avg_degree', 'std_degree', 'density'].

        Returns:
            Dictionary mapping feature names to arrays of length n_scales.
        """
        if self.scale_stats_ is None:
            raise ValueError("Must call fit() first")

        if features is None:
            features = ['n_nodes', 'n_edges', 'avg_degree', 'std_degree', 'density']

        # Vectorized feature extraction
        signature = {
            feature: np.array([
                self.scale_stats_.get(scale, {}).get(feature, np.nan)
                for scale in self.scales
            ], dtype=np.float64)
            for feature in features
        }

        return signature

    def stats(self) -> Dict[int, Dict]:
        """Get full statistics at each scale.

        Returns:
            Dictionary mapping scale to statistics dictionary.
        """
        if self.scale_stats_ is None:
            raise ValueError("Must call fit() first")

        return self.scale_stats_.copy()

    def fit_transform(self, x: np.ndarray, features: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Fit and return scale signature in one step.

        Args:
            x: Input time series (1D array).
            features: Features to include in signature.

        Returns:
            Scale signature dictionary.
        """
        return self.fit(x).scale_signature(features=features)

