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
    TransferEntropyDetector,
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
    "TransferEntropyDetector",
]
