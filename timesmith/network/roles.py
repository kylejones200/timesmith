"""Node role analysis for time series networks."""

from typing import Dict, Tuple

import networkx as nx
import numpy as np


def node_roles(
    G: nx.Graph,
    n_roles: int = 4,
    features: str = "basic",
    n_init: int = 10,
    seed: int = 3363,
) -> Tuple[Dict, np.ndarray]:
    """Compute node roles using clustering of network features.

    Args:
        G: NetworkX graph (will be converted to undirected).
        n_roles: Number of roles to identify.
        features: Feature set to use ("basic" only currently).
        n_init: Number of K-means initializations.
        seed: Random seed.

    Returns:
        Tuple of (node_to_role dictionary, cluster centers array).
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError(
            "scikit-learn is required for node roles. Install with: pip install scikit-learn"
        )

    if features != "basic":
        raise ValueError("Only 'basic' features supported.")

    H = G.to_undirected()
    nodes = list(H.nodes())

    # Basic features: degree, clustering, pagerank, eigenvector centrality, core number, betweenness, closeness
    # Use dict comprehensions and vectorized operations where possible
    deg = np.array([H.degree(n) for n in nodes], dtype=float)
    clustering_dict = nx.clustering(H)
    cc = np.array([clustering_dict[n] for n in nodes], dtype=float)
    pagerank_dict = nx.pagerank(H)
    pr = np.array([pagerank_dict[n] for n in nodes], dtype=float)

    try:
        ev = nx.eigenvector_centrality_numpy(H)
        ev = np.array([ev[n] for n in nodes], dtype=float)
    except Exception:
        ev = np.zeros_like(deg)

    core = nx.core_number(H)
    core = np.array([core[n] for n in nodes], dtype=float)
    btw = np.array(
        list(nx.betweenness_centrality(H, normalized=True).values()), dtype=float
    )
    clo = np.array(list(nx.closeness_centrality(H).values()), dtype=float)

    # Stack features and normalize
    X = np.vstack([deg, cc, pr, ev, core, btw, clo]).T
    X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)

    # Cluster
    km = KMeans(n_clusters=int(n_roles), n_init=n_init, random_state=seed)
    lab = km.fit_predict(X)

    assign = {n: int(c) for n, c in zip(nodes, lab)}
    return assign, km.cluster_centers_
