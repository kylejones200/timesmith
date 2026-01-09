"""Windowed network construction for large time series.

Provides high-level API for building graph statistics per window,
storing only time series of stats (not full graphs).
"""

import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from timesmith.network._constructors import build_hvg, build_nvg, build_recurrence_network, build_transition_network
from timesmith.network.graph import Graph
from timesmith.network.metrics import graph_summary

logger = logging.getLogger(__name__)


def ts_to_windows(x: np.ndarray, width: int, by: int = 1, start: int = 0, end: Optional[int] = None) -> np.ndarray:
    """Extract sliding windows from a time series.

    Args:
        x: Input time series (1D array).
        width: Window width (number of time points per window).
        by: Step size between consecutive windows.
        start: Starting index (0-based).
        end: Ending index (exclusive). If None, use len(x).

    Returns:
        Array of shape (n_windows, width) where each row is a window.
    """
    if x.ndim != 1:
        raise ValueError(f"x must be 1D array, got shape {x.shape}")

    n = len(x)

    if end is None:
        end = n

    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")

    if by <= 0:
        raise ValueError(f"by must be positive, got {by}")

    if start < 0 or start >= n:
        raise ValueError(f"start must be in [0, {n-1}], got {start}")

    if end <= start or end > n:
        raise ValueError(f"end must be in ({start}, {n}], got {end}")

    if width > (end - start):
        raise ValueError(f"width ({width}) cannot exceed series length ({end - start})")

    # Calculate number of windows
    n_windows = (end - start - width) // by + 1

    if n_windows <= 0:
        raise ValueError(f"No windows possible with width={width}, by={by}, start={start}, end={end}")

    # Extract windows
    windows = np.zeros((n_windows, width))

    for i in range(n_windows):
        window_start = start + i * by
        window_end = window_start + width
        windows[i] = x[window_start:window_end]

    return windows


def build_windows(
    x: np.ndarray,
    window: int,
    step: int = 1,
    method: str = "hvg",
    output: str = "stats",
    **method_kwargs
) -> Dict[str, np.ndarray]:
    """Build graph statistics per window (memory efficient for large series).

    For meter data with millions of points, this computes graph stats per window
    and returns only the time series of stats, not full graphs.

    Args:
        x: Input time series (1D array).
        window: Window width (number of time points per window).
        step: Step size between consecutive windows.
        method: Network method: 'hvg', 'nvg', 'recurrence', 'transition'.
        output: Output mode: 'stats' (recommended), 'degrees', or 'edges'.
        **method_kwargs: Additional parameters for the network builder.

    Returns:
        Dictionary mapping stat names to arrays of length n_windows.
    """
    # Extract windows
    windows = ts_to_windows(x, width=window, by=step)
    n_windows = windows.shape[0]

    # Initialize result storage
    result = {
        'n_nodes': np.zeros(n_windows, dtype=np.int64),
        'n_edges': np.zeros(n_windows, dtype=np.int64),
        'avg_degree': np.zeros(n_windows, dtype=np.float64),
        'std_degree': np.zeros(n_windows, dtype=np.float64),
        'density': np.zeros(n_windows, dtype=np.float64),
    }

    # Build network for each window
    for i, window_data in enumerate(windows):
        try:
            if method == "hvg":
                G_nx, A = build_hvg(
                    window_data,
                    weighted=method_kwargs.get('weighted', False),
                    limit=method_kwargs.get('limit'),
                    directed=method_kwargs.get('directed', False),
                )
            elif method == "nvg":
                G_nx, A = build_nvg(
                    window_data,
                    weighted=method_kwargs.get('weighted', False),
                    limit=method_kwargs.get('limit'),
                    directed=method_kwargs.get('directed', False),
                )
            elif method == "recurrence":
                G_nx = build_recurrence_network(
                    window_data,
                    threshold=method_kwargs.get('threshold'),
                    embedding_dimension=method_kwargs.get('embedding_dimension'),
                    time_delay=method_kwargs.get('time_delay', 1),
                    metric=method_kwargs.get('metric', 'euclidean'),
                    rule=method_kwargs.get('rule'),
                    k=method_kwargs.get('k'),
                )
            elif method == "transition":
                G_nx = build_transition_network(
                    window_data,
                    n_bins=method_kwargs.get('n_bins', 10),
                    order=method_kwargs.get('order', 1),
                    symbolizer=method_kwargs.get('symbolizer'),
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # Convert to Graph object and get stats
            edges = list(G_nx.edges())
            graph = Graph(
                edges=edges,
                n_nodes=G_nx.number_of_nodes(),
                directed=G_nx.is_directed(),
                weighted=False,
            )

            stats = graph.summary()

            result['n_nodes'][i] = stats['n_nodes']
            result['n_edges'][i] = stats['n_edges']
            result['avg_degree'][i] = stats['avg_degree']
            result['std_degree'][i] = stats.get('std_degree', 0.0)
            result['density'][i] = stats['density']

        except Exception as e:
            # Handle errors gracefully (e.g., constant windows)
            logger.warning(f"Failed to compute graph for window {i}: {e}")
            result['n_nodes'][i] = 0
            result['n_edges'][i] = 0
            result['avg_degree'][i] = np.nan
            result['std_degree'][i] = np.nan
            result['density'][i] = np.nan

    return result

