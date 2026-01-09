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
        # Time-delay embedding
        vectors = []
        max_idx = n - (embedding_dimension - 1) * time_delay
        for i in range(max_idx):
            vec = [series[i + j * time_delay] for j in range(embedding_dimension)]
            vectors.append(vec)
        vectors = np.array(vectors)
    else:
        vectors = series.reshape(-1, 1)

    # Compute distance matrix
    if metric == "euclidean":
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(vectors, metric="euclidean"))
    elif metric == "manhattan":
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(vectors, metric="cityblock"))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Determine threshold
    if rule == "knn" or (rule is None and k is not None):
        # k-NN rule
        k = k if k is not None else 5
        threshold = np.partition(distances, k, axis=1)[:, k]

    elif threshold is None:
        # Default: use 10th percentile of distances
        threshold = np.percentile(distances[distances > 0], 10)

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(len(vectors)))

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if rule == "knn":
                # For k-NN, use per-node threshold
                if distances[i, j] <= threshold[i] or distances[i, j] <= threshold[j]:
                    G.add_edge(i, j)
            else:
                # Epsilon rule
                if distances[i, j] <= threshold:
                    G.add_edge(i, j)

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
        # Ordinal pattern encoding
        symbols = []
        for i in range(n - order + 1):
            subseq = series[i : i + order]
            pattern = tuple(np.argsort(subseq))
            symbols.append(pattern)
        unique_patterns = list(set(symbols))
        pattern_to_idx = {pattern: idx for idx, pattern in enumerate(unique_patterns)}
        symbol_series = np.array([pattern_to_idx[pattern] for pattern in symbols])
    else:
        # Equal-width binning
        bins = np.linspace(series.min(), series.max(), n_bins + 1)[1:-1]
        symbol_series = np.digitize(series, bins)

    # Build transition network
    G = nx.DiGraph()
    G.add_nodes_from(range(len(np.unique(symbol_series))))

    # Count transitions
    for i in range(len(symbol_series) - order):
        from_state = symbol_series[i]
        to_state = symbol_series[i + order]
        if G.has_edge(from_state, to_state):
            G[from_state][to_state]["weight"] += 1
        else:
            G.add_edge(from_state, to_state, weight=1)

    return G

