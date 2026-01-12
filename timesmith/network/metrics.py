"""Network metrics for time series networks.

Provides efficient computation of clustering, path lengths, and modularity.
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    from networkx.algorithms import community

    HAS_NETWORKX = True
    HAS_COMMUNITY = True
except ImportError:
    HAS_NETWORKX = False
    HAS_COMMUNITY = False
    nx = None
    community = None
    logger.warning(
        "NetworkX not installed. Network metrics will not work. "
        "Install with: pip install networkx or pip install timesmith[network]"
    )


def compute_clustering(
    G: Any,
    method: str = "average",
    sample_size: Optional[int] = None,
) -> Dict[str, float]:
    """Compute clustering coefficient metrics.

    Args:
        G: NetworkX graph or timesmith.network.Graph object.
        method: Method: "average" (average clustering), "global" (transitivity),
            or "local" (returns per-node clustering).
        sample_size: For large graphs, sample nodes for local clustering computation.

    Returns:
        Dictionary with clustering metrics.
    """
    if not HAS_NETWORKX:
        raise ImportError(
            "NetworkX is required for compute_clustering. Install with: pip install networkx"
        )

    # Convert Graph to NetworkX if needed
    if hasattr(G, "as_networkx"):
        G = G.as_networkx()

    if G.number_of_nodes() == 0:
        return {
            "avg_clustering": np.nan,
            "transitivity": np.nan,
        }

    # Convert to undirected for clustering computation
    if G.is_directed():
        G_und = G.to_undirected()
    else:
        G_und = G

    results = {}

    # Average clustering
    if method in ("average", "local"):
        if sample_size and G_und.number_of_nodes() > sample_size:
            # Sample nodes for large graphs
            nodes = list(G_und.nodes())
            sampled = np.random.choice(nodes, size=sample_size, replace=False)
            local_clustering = nx.clustering(G_und, nodes=sampled)
            clustering_values = list(local_clustering.values())
            results["avg_clustering"] = float(np.mean(clustering_values))
            if method == "local":
                results["clustering_std"] = float(np.std(clustering_values))
        else:
            results["avg_clustering"] = float(nx.average_clustering(G_und))
            if method == "local":
                local_clustering = nx.clustering(G_und)
                clustering_values = list(local_clustering.values())
                results["clustering_std"] = float(np.std(clustering_values))
    else:
        results["avg_clustering"] = float(nx.average_clustering(G_und))

    # Global transitivity
    results["transitivity"] = float(nx.transitivity(G_und))

    return results


def compute_path_lengths(
    G: Any,
    method: str = "average",
    sample_size: Optional[int] = None,
    weight: Optional[str] = None,
) -> Dict[str, float]:
    """Compute shortest path length metrics.

    Args:
        G: NetworkX graph or timesmith.network.Graph object.
        method: Method: "average" (average path length), "diameter" (longest shortest path),
            or "eccentricity" (returns per-node eccentricity).
        sample_size: For large graphs, sample node pairs for path computation.
        weight: Edge attribute to use as weight (default: unweighted).

    Returns:
        Dictionary with path length metrics.
    """
    if not HAS_NETWORKX:
        raise ImportError(
            "NetworkX is required for compute_path_lengths. Install with: pip install networkx"
        )

    # Convert Graph to NetworkX if needed
    if hasattr(G, "as_networkx"):
        G = G.as_networkx()

    if G.number_of_nodes() == 0:
        return {
            "avg_path_length": np.nan,
            "diameter": np.nan,
            "radius": np.nan,
        }

    # Convert to undirected for path computation
    if G.is_directed():
        G_und = G.to_undirected()
    else:
        G_und = G

    # Check connectivity
    if not nx.is_connected(G_und):
        # Use largest connected component
        largest_cc = max(nx.connected_components(G_und), key=len)
        G_und = G_und.subgraph(largest_cc).copy()
        if G_und.number_of_nodes() <= 1:
            return {
                "avg_path_length": np.nan,
                "diameter": np.nan,
                "radius": np.nan,
            }

    results = {}

    # Average shortest path length
    if sample_size and G_und.number_of_nodes() > sample_size:
        # Sample node pairs for large graphs
        nodes = list(G_und.nodes())
        sampled = np.random.choice(
            nodes, size=min(sample_size, len(nodes)), replace=False
        )
        path_lengths = []
        for i, u in enumerate(sampled):
            for v in sampled[i + 1 :]:
                try:
                    if weight:
                        length = nx.shortest_path_length(G_und, u, v, weight=weight)
                    else:
                        length = nx.shortest_path_length(G_und, u, v)
                    path_lengths.append(length)
                except nx.NetworkXNoPath:
                    continue
        results["avg_path_length"] = (
            float(np.mean(path_lengths)) if path_lengths else np.nan
        )
    else:
        try:
            if weight:
                results["avg_path_length"] = float(
                    nx.average_shortest_path_length(G_und, weight=weight)
                )
            else:
                results["avg_path_length"] = float(
                    nx.average_shortest_path_length(G_und)
                )
        except (nx.NetworkXError, nx.NetworkXNoPath):
            results["avg_path_length"] = np.nan

    # Diameter and radius
    try:
        if method in ("diameter", "eccentricity"):
            eccentricity = nx.eccentricity(G_und)
            ecc_values = list(eccentricity.values())
            results["diameter"] = float(max(ecc_values))
            results["radius"] = float(min(ecc_values))
            if method == "eccentricity":
                results["eccentricity_std"] = float(np.std(ecc_values))
        else:
            # Compute diameter only if needed
            results["diameter"] = float(nx.diameter(G_und))
            results["radius"] = float(nx.radius(G_und))
    except (nx.NetworkXError, nx.NetworkXNoPath):
        results["diameter"] = np.nan
        results["radius"] = np.nan

    return results


def compute_modularity(
    G: Any,
    method: str = "louvain",
    weight: Optional[str] = None,
    resolution: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[str, Union[float, dict]]:
    """Compute modularity and community structure.

    Args:
        G: NetworkX graph or timesmith.network.Graph object.
        method: Community detection method: "louvain", "leiden", "greedy", or "label_propagation".
        weight: Edge attribute to use as weight (default: unweighted).
        resolution: Resolution parameter for modularity (higher = more communities).
        seed: Random seed for community detection.

    Returns:
        Dictionary with modularity metrics.
    """
    if not HAS_NETWORKX or not HAS_COMMUNITY:
        raise ImportError(
            "NetworkX with community algorithms is required for compute_modularity. "
            "Install with: pip install networkx"
        )

    # Convert Graph to NetworkX if needed
    if hasattr(G, "as_networkx"):
        G = G.as_networkx()

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return {
            "modularity": np.nan,
            "n_communities": 0,
        }

    # Convert to undirected for modularity computation
    if G.is_directed():
        G_und = G.to_undirected()
    else:
        G_und = G

    results = {}

    # Detect communities
    if method == "louvain":
        communities_generator = community.louvain_communities(
            G_und, weight=weight, resolution=resolution, seed=seed
        )
        communities = list(communities_generator)
    elif method == "leiden":
        try:
            communities_generator = community.leiden_communities(
                G_und, weight=weight, resolution_parameter=resolution, seed=seed
            )
            communities = list(communities_generator)
        except AttributeError:
            # Fallback to louvain if leiden not available
            communities_generator = community.louvain_communities(
                G_und, weight=weight, resolution=resolution, seed=seed
            )
            communities = list(communities_generator)
    elif method == "greedy":
        try:
            communities_generator = community.greedy_modularity_communities(
                G_und, weight=weight, resolution=resolution
            )
        except TypeError:
            # Fallback for older NetworkX versions
            communities_generator = community.greedy_modularity_communities(
                G_und, weight=weight
            )
        communities = list(communities_generator)
    elif method == "label_propagation":
        communities_dict = community.label_propagation_communities(G_und)
        communities = [set(com) for com in communities_dict]
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'louvain', 'leiden', 'greedy', or 'label_propagation'"
        )

    # Compute modularity
    if weight:
        modularity = community.modularity(G_und, communities, weight=weight)
    else:
        modularity = community.modularity(G_und, communities)

    results["modularity"] = float(modularity)
    results["n_communities"] = len(communities)

    return results


def network_metrics(
    G: Any,
    include: Optional[list] = None,
    sample_size: Optional[int] = None,
    **kwargs,
) -> Dict[str, Union[float, dict]]:
    """Compute comprehensive network metrics.

    Args:
        G: NetworkX graph or timesmith.network.Graph object.
        include: Metrics to include: ["clustering", "path_lengths", "modularity"].
            If None, includes all metrics.
        sample_size: For large graphs, sample nodes/pairs for expensive computations.
        **kwargs: Additional arguments passed to metric functions.

    Returns:
        Dictionary with all requested network metrics.
    """
    if include is None:
        include = ["clustering", "path_lengths", "modularity"]

    results = {}

    if "clustering" in include:
        clustering_kwargs = {
            k: v for k, v in kwargs.items() if k in ["method", "sample_size"]
        }
        if sample_size:
            clustering_kwargs["sample_size"] = sample_size
        results.update(compute_clustering(G, **clustering_kwargs))

    if "path_lengths" in include:
        path_kwargs = {
            k: v for k, v in kwargs.items() if k in ["method", "sample_size", "weight"]
        }
        if sample_size:
            path_kwargs["sample_size"] = sample_size
        results.update(compute_path_lengths(G, **path_kwargs))

    if "modularity" in include:
        modularity_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["method", "weight", "resolution", "seed"]
        }
        results.update(compute_modularity(G, **modularity_kwargs))

    return results


def graph_summary(
    G: Any,
    motifs: Optional[str] = None,
    motif_samples: Optional[int] = None,
    seed: int = 3363,
) -> dict:
    """Compute a comprehensive summary of graph properties.

    Args:
        G: NetworkX graph or timesmith.network.Graph object.
        motifs: Type of motifs to compute ('3node', '4node', or None).
        motif_samples: Maximum number of samples for motif counting.
        seed: Random seed for sampling.

    Returns:
        Dictionary of graph properties.
    """
    if not HAS_NETWORKX:
        raise ImportError(
            "NetworkX is required for graph_summary. Install with: pip install networkx"
        )

    # Convert Graph to NetworkX if needed
    if hasattr(G, "as_networkx"):
        G = G.as_networkx()

    und = G.to_undirected() if G.is_directed() else G
    deg = dict(und.degree())
    deg_vals = np.array(list(deg.values()), dtype=float)

    out = {
        "n": und.number_of_nodes(),
        "m": und.number_of_edges(),
        "density": nx.density(und),
        "deg_mean": float(deg_vals.mean()) if deg_vals.size else 0.0,
        "deg_std": float(deg_vals.std(ddof=1)) if deg_vals.size > 1 else 0.0,
        "avg_degree": float(deg_vals.mean()) if deg_vals.size else 0.0,
        "assortativity": (
            nx.degree_assortativity_coefficient(und)
            if und.number_of_edges()
            else np.nan
        ),
        "avg_clustering": (
            nx.average_clustering(und) if und.number_of_nodes() else np.nan
        ),
    }

    # Add small world properties
    try:
        if und.number_of_nodes() > 1 and und.number_of_edges() > 0:
            L = nx.average_shortest_path_length(und)
            C = nx.average_clustering(und)
            k_bar = 2.0 * und.number_of_edges() / und.number_of_nodes()
            p = (2.0 * und.number_of_edges()) / (
                und.number_of_nodes() * (und.number_of_nodes() - 1)
            )
            C_er = p
            if k_bar > 1.0:
                import math

                L_er = math.log(und.number_of_nodes()) / math.log(k_bar)
            else:
                L_er = np.nan
            num = (C / C_er) if C_er > 0 else np.nan
            den = (L / L_er) if (L_er is not None and not np.isnan(L_er)) else np.nan
            sigma = (
                num / den
                if (not np.isnan(num) and not np.isnan(den) and den != 0)
                else np.nan
            )
            out.update({"C": C, "L": L, "C_er": C_er, "L_er": L_er, "sigma": sigma})
        else:
            out.update(
                {
                    "C": np.nan,
                    "L": np.nan,
                    "C_er": np.nan,
                    "L_er": np.nan,
                    "sigma": np.nan,
                }
            )
    except (nx.NetworkXError, nx.NetworkXNoPath):
        out.update(
            {"C": np.nan, "L": np.nan, "C_er": np.nan, "L_er": np.nan, "sigma": np.nan}
        )

    # Add motif counts if requested
    if motifs in ("directed3", "all") and G.is_directed():
        # Simplified motif counting (full implementation would be more complex)
        out["motifs_directed_3"] = {}  # Placeholder

    if motifs in ("undirected4", "all"):
        # Simplified motif counting
        out["motifs_undirected_4"] = {}  # Placeholder

    return out
