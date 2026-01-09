"""Network analysis for time series.

This module provides tools for converting time series to networks
and analyzing network properties.
"""

from timesmith.network.graph import Graph
from timesmith.network.constructors import (
    HVGFeaturizer,
    NVGFeaturizer,
    RecurrenceNetworkFeaturizer,
    TransitionNetworkFeaturizer,
)
from timesmith.network.metrics import (
    graph_summary,
    network_metrics,
    compute_clustering,
    compute_path_lengths,
    compute_modularity,
)
from timesmith.network.causal import (
    transfer_entropy,
    conditional_transfer_entropy,
    transfer_entropy_network,
    TransferEntropyDetector,
)
from timesmith.network.windows import build_windows, ts_to_windows
from timesmith.network.multiscale import MultiscaleGraphs, coarse_grain
from timesmith.network.motifs import directed_3node_motifs, undirected_4node_motifs
from timesmith.network.roles import node_roles
from timesmith.network.multivariate import net_knn, net_enn
from timesmith.network.null_models import (
    generate_surrogate,
    compute_network_metric_significance,
    NetworkSignificanceResult,
)

__all__ = [
    "Graph",
    "HVGFeaturizer",
    "NVGFeaturizer",
    "RecurrenceNetworkFeaturizer",
    "TransitionNetworkFeaturizer",
    "graph_summary",
    "network_metrics",
    "compute_clustering",
    "compute_path_lengths",
    "compute_modularity",
    "transfer_entropy",
    "conditional_transfer_entropy",
    "transfer_entropy_network",
    "TransferEntropyDetector",
    "build_windows",
    "ts_to_windows",
    "MultiscaleGraphs",
    "coarse_grain",
    "directed_3node_motifs",
    "undirected_4node_motifs",
    "node_roles",
    "net_knn",
    "net_enn",
    "generate_surrogate",
    "compute_network_metric_significance",
    "NetworkSignificanceResult",
]
