"""Multivariate network construction from distance matrices.

Implements k-NN, ε-NN, and weighted network builders from distance matrices.
"""

from typing import Tuple, Optional

import networkx as nx
import numpy as np


def net_knn(
    D: np.ndarray,
    k: int,
    mutual: bool = False,
    weighted: bool = False,
    directed: bool = False
) -> Tuple[nx.Graph, np.ndarray]:
    """k-Nearest Neighbors network from distance matrix.

    Each node is connected to its k nearest neighbors.

    Args:
        D: Distance matrix (n, n) - smaller = more similar.
        k: Number of nearest neighbors per node.
        mutual: If True, require mutual k-NN (i in kNN(j) AND j in kNN(i)).
        weighted: If True, edge weights = distances.
        directed: If True, create directed graph (i → j if j in kNN(i)).

    Returns:
        Tuple of (NetworkX graph, adjacency matrix).
    """
    n = D.shape[0]

    if D.shape != (n, n):
        raise ValueError(f"D must be square, got shape {D.shape}")

    if k <= 0 or k >= n:
        raise ValueError(f"k must be in range [1, {n-1}], got {k}")

    # Create adjacency matrix
    A = np.zeros((n, n))

    for i in range(n):
        # Find k nearest neighbors (excluding self)
        distances = D[i].copy()
        distances[i] = np.inf  # Exclude self
        neighbors = np.argpartition(distances, k - 1)[:k]

        for j in neighbors:
            weight = D[i, j] if weighted else 1.0
            A[i, j] = weight

    # Apply mutual k-NN constraint
    if mutual:
        if directed:
            A = A * A.T  # Both i→j and j→i must exist
        else:
            A = np.minimum(A, A.T)  # Symmetric mutual k-NN

    # Make symmetric for undirected
    if not directed:
        A = (A + A.T) / 2

    # Build NetworkX graph
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if A[i, j] > 0:
                if weighted:
                    G.add_edge(i, j, weight=float(A[i, j]))
                else:
                    G.add_edge(i, j)

    return G, A


def net_enn(
    D: np.ndarray,
    epsilon: Optional[float] = None,
    percentile: Optional[float] = None,
    weighted: bool = False,
    directed: bool = False
) -> Tuple[nx.Graph, np.ndarray]:
    """ε-Nearest Neighbors network from distance matrix.

    Each node is connected to all nodes within distance epsilon.

    Args:
        D: Distance matrix (n, n) - smaller = more similar.
        epsilon: Distance threshold. If None, use percentile.
        percentile: Percentile of distances to use as threshold (0-100).
        weighted: If True, edge weights = distances.
        directed: If True, create directed graph.

    Returns:
        Tuple of (NetworkX graph, adjacency matrix).
    """
    n = D.shape[0]

    if D.shape != (n, n):
        raise ValueError(f"D must be square, got shape {D.shape}")

    # Determine threshold
    if epsilon is None and percentile is None:
        raise ValueError("Either epsilon or percentile must be provided")

    if percentile is not None:
        # Use percentile of all distances (excluding diagonal)
        distances_flat = D[np.triu_indices(n, k=1)]
        epsilon = np.percentile(distances_flat, percentile)

    # Create adjacency matrix
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j and D[i, j] <= epsilon:
                weight = D[i, j] if weighted else 1.0
                A[i, j] = weight

    # Make symmetric for undirected
    if not directed:
        A = (A + A.T) / 2

    # Build NetworkX graph
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if A[i, j] > 0:
                if weighted:
                    G.add_edge(i, j, weight=float(A[i, j]))
                else:
                    G.add_edge(i, j)

    return G, A

