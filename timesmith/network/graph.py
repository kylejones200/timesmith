"""Lightweight graph representation for time series networks.

Primary storage is edges + optional adjacency matrix.
NetworkX conversion is lazy and optional.
"""

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Import for type hints only
try:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        import networkx as nx
except ImportError:
    pass


class Graph:
    """Lightweight graph representation.

    Primary storage is edges + optional adjacency matrix.
    NetworkX conversion is lazy and optional.

    Attributes:
        edges: List of (int, int) or (int, int, float) tuples (edge list).
        n_nodes: Number of nodes.
        directed: Whether graph is directed.
        weighted: Whether edges have weights.
    """

    def __init__(
        self,
        edges: List[Tuple],
        n_nodes: int,
        directed: bool = False,
        weighted: bool = False,
    ):
        """Initialize graph.

        Args:
            edges: List of edge tuples (unweighted or weighted).
            n_nodes: Number of nodes.
            directed: Whether graph is directed.
            weighted: Whether edges have weights.
        """
        self.edges = edges
        self.n_nodes = n_nodes
        self.directed = directed
        self.weighted = weighted
        self._adjacency: Optional = None  # scipy sparse matrix, not dense
        self._degrees: Optional[NDArray] = None
        self._in_degrees: Optional[NDArray] = None
        self._out_degrees: Optional[NDArray] = None

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        if hasattr(self, "_n_edges_cached"):
            return self._n_edges_cached
        return len(self.edges)

    def degree_sequence(self) -> NDArray[np.int64]:
        """Degree sequence (cached).

        For undirected graphs, returns total degree.
        For directed graphs, returns out-degree.

        Returns:
            Array of degrees for each node.
        """
        if self.directed:
            return self.out_degree_sequence()
        else:
            if self._degrees is None:
                degrees = np.zeros(self.n_nodes, dtype=np.int64)
                for edge in self.edges:
                    i, j = edge[0], edge[1]
                    degrees[i] += 1
                    if i != j:
                        degrees[j] += 1
                self._degrees = degrees
            return self._degrees

    def in_degree_sequence(self) -> NDArray[np.int64]:
        """In-degree sequence for directed graphs (cached).

        Returns:
            Array of in-degrees for each node.
        """
        if not self.directed:
            raise ValueError("in_degree_sequence() only valid for directed graphs")
        if self._in_degrees is None:
            in_degrees = np.zeros(self.n_nodes, dtype=np.int64)
            for edge in self.edges:
                j = edge[1]  # Destination node
                in_degrees[j] += 1
            self._in_degrees = in_degrees
        return self._in_degrees

    def out_degree_sequence(self) -> NDArray[np.int64]:
        """Out-degree sequence for directed graphs (cached).

        Returns:
            Array of out-degrees for each node.
        """
        if not self.directed:
            raise ValueError("out_degree_sequence() only valid for directed graphs")
        if self._out_degrees is None:
            out_degrees = np.zeros(self.n_nodes, dtype=np.int64)
            for edge in self.edges:
                i = edge[0]  # Source node
                out_degrees[i] += 1
            self._out_degrees = out_degrees
        return self._out_degrees

    def adjacency_matrix(self, format: str = "sparse"):
        """Adjacency matrix (lazy, sparse by default).

        Args:
            format: Output format: "sparse" (CSR), "dense", or "coo".

        Returns:
            Adjacency matrix. Sparse by default to avoid memory blowup.

        Raises:
            ValueError: If format="dense" and n_nodes > 50_000.
        """
        from scipy import sparse as sp

        # Safety guardrail: refuse dense for large graphs
        if format == "dense" and self.n_nodes > 50_000:
            raise ValueError(
                f"Refusing to build dense adjacency matrix for n={self.n_nodes} nodes. "
                f"This would require ~{self.n_nodes**2 * 8 / 1e9:.1f} GB of memory. "
                f"Use format='sparse' or format='coo' instead."
            )

        # Build sparse COO from edges (memory efficient)
        if self._adjacency is None:
            if len(self.edges) == 0:
                # Empty graph
                self._adjacency = sp.coo_matrix((self.n_nodes, self.n_nodes))
            else:
                # Extract edge data
                if self.weighted:
                    rows = [e[0] for e in self.edges]
                    cols = [e[1] for e in self.edges]
                    data = [e[2] for e in self.edges]
                else:
                    rows = [e[0] for e in self.edges]
                    cols = [e[1] for e in self.edges]
                    data = [1.0] * len(self.edges)

                # Add reverse edges for undirected graphs
                if not self.directed:
                    reverse_rows = cols.copy()
                    reverse_cols = rows.copy()
                    rows = rows + reverse_rows
                    cols = cols + reverse_cols
                    data = data + data

                self._adjacency = sp.coo_matrix(
                    (data, (rows, cols)), shape=(self.n_nodes, self.n_nodes)
                )

        # Convert to requested format
        if format == "dense":
            return self._adjacency.toarray()
        elif format == "coo":
            return self._adjacency.tocoo()
        else:  # sparse (default)
            return self._adjacency.tocsr()

    def as_networkx(self, force: bool = False):
        """Convert to NetworkX graph (optional dependency).

        Args:
            force: If False, refuse conversion for n > 200_000 nodes.

        Returns:
            NetworkX graph object.

        Raises:
            ImportError: If NetworkX is not installed.
            ValueError: If n_nodes > 200_000 and force=False.
        """
        # Safety guardrail for large graphs
        if not force and self.n_nodes > 200_000:
            raise ValueError(
                f"Refusing NetworkX conversion for n={self.n_nodes} nodes. "
                f"NetworkX is not designed for graphs this large. "
                f"Use force=True to override, or work with edges_coo() / "
                f"adjacency_matrix(format='sparse') instead."
            )

        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is required for as_networkx(). "
                "Install with: pip install networkx"
            )

        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        G.add_nodes_from(range(self.n_nodes))

        if self.weighted:
            G.add_weighted_edges_from(self.edges)
        else:
            G.add_edges_from(self.edges)

        return G

    def summary(self, include_triangles: bool = False) -> dict:
        """Graph summary statistics (computed from edges/degrees, no dense matrix).

        Args:
            include_triangles: If True, compute triangle count (requires edge list, slower).

        Returns:
            Dictionary with graph statistics.
        """
        degrees = self.degree_sequence()
        max_edges = self.n_nodes * (self.n_nodes - 1)
        if not self.directed:
            max_edges //= 2

        stats = {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "avg_degree": float(np.mean(degrees)),
            "std_degree": float(np.std(degrees)) if len(degrees) > 1 else 0.0,
            "min_degree": int(np.min(degrees)) if len(degrees) > 0 else 0,
            "max_degree": int(np.max(degrees)) if len(degrees) > 0 else 0,
            "density": self.n_edges / max_edges if max_edges > 0 else 0.0,
        }

        # For directed graphs, add in/out degree statistics
        if self.directed:
            in_degrees = self.in_degree_sequence()
            out_degrees = self.out_degree_sequence()
            total_degrees = in_degrees + out_degrees

            stats["avg_in_degree"] = float(np.mean(in_degrees))
            stats["std_in_degree"] = (
                float(np.std(in_degrees)) if len(in_degrees) > 1 else 0.0
            )
            stats["avg_out_degree"] = float(np.mean(out_degrees))
            stats["std_out_degree"] = (
                float(np.std(out_degrees)) if len(out_degrees) > 1 else 0.0
            )

            # Irreversibility score
            irreversibility = np.zeros(self.n_nodes, dtype=np.float64)
            mask = total_degrees > 0
            irreversibility[mask] = (
                np.abs(in_degrees[mask] - out_degrees[mask]) / total_degrees[mask]
            )
            stats["irreversibility_score"] = float(np.mean(irreversibility))

        if include_triangles and len(self.edges) > 0:
            triangles = self._count_triangles()
            stats["triangles"] = triangles

        return stats

    def _count_triangles(self) -> int:
        """Count triangles from edge list (memory efficient)."""
        if len(self.edges) == 0:
            return 0

        # Build neighbor sets (sparse representation)
        neighbors = {i: set() for i in range(self.n_nodes)}
        for edge in self.edges:
            i, j = edge[0], edge[1]
            neighbors[i].add(j)
            if not self.directed:
                neighbors[j].add(i)

        # Count triangles
        triangles = 0
        for i in range(self.n_nodes):
            for j in neighbors[i]:
                if j > i:  # Avoid double counting
                    # Count common neighbors
                    common = neighbors[i] & neighbors[j]
                    triangles += len(common)

        # Each triangle counted 3 times (once per edge)
        return triangles // 3 if not self.directed else triangles

    def __repr__(self) -> str:
        return f"Graph(n_nodes={self.n_nodes}, n_edges={self.n_edges}, directed={self.directed})"
