"""Native network construction algorithms (replacing ts2net dependency)."""

import logging
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def build_hvg(
    series: np.ndarray,
    weighted: bool = False,
    limit: Optional[int] = None,
    directed: bool = False,
) -> Tuple[nx.Graph, np.ndarray]:
    """Build Horizontal Visibility Graph (HVG).

    Two nodes i and j are connected if all intermediate values are below
    the line connecting (i, y[i]) and (j, y[j]).

    Args:
        series: Time series values.
        weighted: If True, edges are weighted by distance.
        limit: Maximum temporal distance for edges.
        directed: If True, create directed graph.

    Returns:
        Tuple of (NetworkX graph, adjacency matrix).
    """
    n = len(series)
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))

    # Build edges
    for i in range(n):
        for j in range(i + 1, n):
            if limit is not None and (j - i) > limit:
                continue

            # Check horizontal visibility: all intermediate values must be below the line
            visible = True
            for k in range(i + 1, j):
                # Line equation: y = y[i] + (y[j] - y[i]) * (k - i) / (j - i)
                line_value = series[i] + (series[j] - series[i]) * (k - i) / (j - i)
                if series[k] >= line_value:
                    visible = False
                    break

            if visible:
                if weighted:
                    weight = abs(series[j] - series[i])
                    G.add_edge(i, j, weight=weight)
                else:
                    G.add_edge(i, j)

    # Convert to undirected if needed
    if not directed:
        G = G.to_undirected()

    # Build adjacency matrix
    A = nx.adjacency_matrix(G, weight="weight" if weighted else None).toarray()

    return G, A


def build_nvg(
    series: np.ndarray,
    weighted: bool = False,
    limit: Optional[int] = None,
    directed: bool = False,
) -> Tuple[nx.Graph, np.ndarray]:
    """Build Natural Visibility Graph (NVG).

    Two nodes i and j are connected if all intermediate values are below
    the line connecting (i, y[i]) and (j, y[j]) in the natural visibility sense.

    Args:
        series: Time series values.
        weighted: If True, edges are weighted by distance.
        limit: Maximum temporal distance for edges.
        directed: If True, create directed graph.

    Returns:
        Tuple of (NetworkX graph, adjacency matrix).
    """
    n = len(series)
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))

    # Build edges
    for i in range(n):
        for j in range(i + 1, n):
            if limit is not None and (j - i) > limit:
                continue

            # Check natural visibility: all intermediate points must be below the line
            visible = True
            for k in range(i + 1, j):
                # Line connecting (i, series[i]) and (j, series[j])
                # y = series[i] + (series[j] - series[i]) * (k - i) / (j - i)
                line_value = series[i] + (series[j] - series[i]) * (k - i) / (j - i)
                if series[k] >= line_value:
                    visible = False
                    break

            if visible:
                if weighted:
                    # Weight by Euclidean distance in (time, value) space
                    weight = np.sqrt((j - i) ** 2 + (series[j] - series[i]) ** 2)
                    G.add_edge(i, j, weight=weight)
                else:
                    G.add_edge(i, j)

    # Convert to undirected if needed
    if not directed:
        G = G.to_undirected()

    # Build adjacency matrix
    A = nx.adjacency_matrix(G, weight="weight" if weighted else None).toarray()

    return G, A


def build_recurrence_network(
    series: np.ndarray,
    threshold: Optional[float] = None,
    embedding_dimension: Optional[int] = None,
    time_delay: int = 1,
    metric: str = "euclidean",
    rule: Optional[str] = None,
    k: Optional[int] = None,
) -> nx.Graph:
    """Build Recurrence Network.

    Args:
        series: Time series values.
        threshold: Distance threshold for recurrence (epsilon rule).
        embedding_dimension: Embedding dimension (m).
        time_delay: Time delay (tau).
        metric: Distance metric ('euclidean' or 'manhattan').
        rule: Threshold rule ('epsilon' or 'knn').
        k: Number of neighbors for k-NN rule.

    Returns:
        NetworkX graph.
    """
    n = len(series)

    # Default embedding dimension
    if embedding_dimension is None:
        embedding_dimension = 1

    # Create phase space vectors
    if embedding_dimension > 1:
        # Time-delay embedding (vectorized)
        max_idx = n - (embedding_dimension - 1) * time_delay
        indices = np.arange(max_idx)[:, None] + np.arange(embedding_dimension) * time_delay
        vectors = series[indices]
    else:
        vectors = series.reshape(-1, 1)

    # Compute distance matrix (import at top level would be better, but scipy is optional)
    from scipy.spatial.distance import pdist, squareform

    metric_map = {
        "euclidean": "euclidean",
        "manhattan": "cityblock",
    }
    scipy_metric = metric_map.get(metric)
    if scipy_metric is None:
        raise ValueError(f"Unsupported metric: {metric}. Must be one of {list(metric_map.keys())}")

    distances = squareform(pdist(vectors, metric=scipy_metric))

    # Determine threshold
    if rule == "knn" or (rule is None and k is not None):
        # k-NN rule
        k = k if k is not None else 5
        threshold = np.partition(distances, k, axis=1)[:, k]

    elif threshold is None:
        # Default: use 10th percentile of distances
        threshold = np.percentile(distances[distances > 0], 10)

    # Build graph (vectorized edge creation)
    G = nx.Graph()
    G.add_nodes_from(range(len(vectors)))

    if rule == "knn":
        # For k-NN, use per-node threshold (vectorized)
        # Create mask: edge exists if distance <= threshold[i] OR distance <= threshold[j]
        threshold_matrix = np.minimum(threshold[:, None], threshold[None, :])
        mask = (distances <= threshold_matrix) & (np.triu(np.ones_like(distances, dtype=bool), k=1))
    else:
        # Epsilon rule (vectorized)
        mask = (distances <= threshold) & (np.triu(np.ones_like(distances, dtype=bool), k=1))

    # Add edges from mask
    edges = np.argwhere(mask)
    G.add_edges_from(edges)

    return G


def build_transition_network(
    series: np.ndarray,
    n_bins: int = 10,
    order: int = 1,
    symbolizer: Optional[str] = None,
) -> nx.DiGraph:
    """Build Transition Network.

    Args:
        series: Time series values.
        n_bins: Number of bins for symbolization.
        order: Order of transition patterns.
        symbolizer: Symbolization method ('equal_width' or 'ordinal').

    Returns:
        Directed NetworkX graph.
    """
    n = len(series)

    if symbolizer is None:
        symbolizer = "equal_width"

    # Symbolize time series
    if symbolizer == "ordinal":
        # Ordinal pattern encoding (vectorized)
        n_patterns = n - order + 1
        if n_patterns <= 0:
            raise ValueError(f"Series too short for order {order}")
        # Create sliding window view
        indices = np.arange(n_patterns)[:, None] + np.arange(order)
        patterns = np.argsort(series[indices], axis=1)
        # Convert to tuples for hashing
        pattern_tuples = [tuple(p) for p in patterns]
        unique_patterns = list(dict.fromkeys(pattern_tuples))  # Preserves order
        pattern_to_idx = {pattern: idx for idx, pattern in enumerate(unique_patterns)}
        symbol_series = np.array([pattern_to_idx[pattern] for pattern in pattern_tuples])
    else:
        # Equal-width binning
        bins = np.linspace(series.min(), series.max(), n_bins + 1)[1:-1]
        symbol_series = np.digitize(series, bins)

    # Build transition network
    n_states = len(np.unique(symbol_series))
    G = nx.DiGraph()
    G.add_nodes_from(range(n_states))

    # Count transitions (vectorized)
    if len(symbol_series) > order:
        from_states = symbol_series[:-order]
        to_states = symbol_series[order:]
        # Use Counter-like approach with unique pairs
        transitions = np.column_stack([from_states, to_states])
        # Count unique transitions
        unique_transitions, counts = np.unique(transitions, axis=0, return_counts=True)
        # Add edges with weights
        for (from_state, to_state), count in zip(unique_transitions, counts):
            G.add_edge(int(from_state), int(to_state), weight=int(count))

    return G

