"""Network construction featurizers for time series.

These featurizers convert time series to networks and extract network features
as TableLike output (degree sequences, metrics, etc.).
"""

import logging
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from timesmith.core.base import BaseFeaturizer
from timesmith.core.tags import set_tags
from timesmith.network.graph import Graph
from timesmith.network._constructors import (
    build_hvg,
    build_nvg,
    build_recurrence_network,
    build_transition_network,
)

logger = logging.getLogger(__name__)


def _validate_series(y: Any) -> np.ndarray:
    """Validate and convert to numpy array."""
    if isinstance(y, pd.Series):
        series = y.values
    elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        series = y.iloc[:, 0].values
    else:
        series = np.asarray(y)

    if series.ndim != 1:
        raise ValueError("y must be 1D (SeriesLike)")

    # Convert to float64 and handle non-finite values
    series = series.astype(np.float64)
    series = np.where(np.isfinite(series), series, np.nan)
    series = series[~np.isnan(series)]

    if len(series) == 0:
        raise ValueError("No valid numeric values in input series")

    return series


def _extract_network_features(graph: Graph, include_degrees: bool = True) -> pd.DataFrame:
    """Extract network features from Graph object.

    Args:
        graph: Graph object.
        include_degrees: Whether to include degree sequence.

    Returns:
        DataFrame with network features.
    """
    features = {}

    # Basic statistics
    features["n_nodes"] = graph.n_nodes
    features["n_edges"] = graph.n_edges
    features["density"] = graph.summary()["density"]

    # Degree statistics
    degrees = graph.degree_sequence()
    features["avg_degree"] = float(np.mean(degrees))
    features["std_degree"] = float(np.std(degrees)) if len(degrees) > 1 else 0.0
    features["min_degree"] = int(np.min(degrees)) if len(degrees) > 0 else 0
    features["max_degree"] = int(np.max(degrees)) if len(degrees) > 0 else 0

    # For directed graphs, add in/out degree stats
    if graph.directed:
        in_degrees = graph.in_degree_sequence()
        out_degrees = graph.out_degree_sequence()
        features["avg_in_degree"] = float(np.mean(in_degrees))
        features["avg_out_degree"] = float(np.mean(out_degrees))
        features["irreversibility_score"] = graph.summary().get("irreversibility_score", 0.0)

    # Create DataFrame with single row
    df = pd.DataFrame([features])

    # Optionally add degree sequence as columns (vectorized to avoid fragmentation)
    if include_degrees and graph.n_nodes <= 1000:  # Only for small graphs
        degree_cols = {f"degree_{i}": int(degrees[i]) for i in range(len(degrees))}
        # Use pd.concat to avoid fragmentation warning
        if degree_cols:
            degree_df = pd.DataFrame([degree_cols])
            df = pd.concat([df, degree_df], axis=1)

    return df


class HVGFeaturizer(BaseFeaturizer):
    """Horizontal Visibility Graph featurizer.

    Converts SeriesLike to TableLike by building HVG and extracting features.
    """

    def __init__(
        self,
        weighted: bool = False,
        limit: Optional[int] = None,
        directed: bool = False,
        include_degrees: bool = True,
    ):
        """Initialize HVG featurizer.

        Args:
            weighted: If True, edges will be weighted.
            limit: Maximum temporal distance for edges.
            directed: If True, create directed graph.
            include_degrees: Whether to include degree sequence in output.
        """
        super().__init__()
        self.weighted = weighted
        self.limit = limit
        self.directed = directed
        self.include_degrees = include_degrees

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="TableLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "HVGFeaturizer":
        """Fit the featurizer (no-op for HVG).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.DataFrame:
        """Build HVG and extract features.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            TableLike DataFrame with network features.
        """
        self._check_is_fitted()

        series = _validate_series(y)

        # Build HVG using native implementation
        G_nx, A = build_hvg(
            series,
            weighted=self.weighted,
            limit=self.limit,
            directed=self.directed,
        )

        # Convert to Graph object
        edges = list(G_nx.edges(data="weight" if self.weighted else False))
        if self.weighted:
            edges = [(u, v, w) for u, v, w in edges]
        else:
            edges = [(u, v) for u, v in edges]

        graph = Graph(
            edges=edges,
            n_nodes=len(series),
            directed=self.directed,
            weighted=self.weighted,
        )

        return _extract_network_features(graph, include_degrees=self.include_degrees)


class NVGFeaturizer(BaseFeaturizer):
    """Natural Visibility Graph featurizer.

    Converts SeriesLike to TableLike by building NVG and extracting features.
    """

    def __init__(
        self,
        weighted: bool = False,
        limit: Optional[int] = None,
        directed: bool = False,
        include_degrees: bool = True,
    ):
        """Initialize NVG featurizer.

        Args:
            weighted: If True, edges will be weighted.
            limit: Maximum temporal distance for edges.
            directed: If True, create directed graph.
            include_degrees: Whether to include degree sequence in output.
        """
        super().__init__()
        self.weighted = weighted
        self.limit = limit
        self.directed = directed
        self.include_degrees = include_degrees

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="TableLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "NVGFeaturizer":
        """Fit the featurizer (no-op for NVG).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.DataFrame:
        """Build NVG and extract features.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            TableLike DataFrame with network features.
        """
        self._check_is_fitted()

        series = _validate_series(y)

        # Build NVG using native implementation
        G_nx, A = build_nvg(
            series,
            weighted=self.weighted,
            limit=self.limit,
            directed=self.directed,
        )

        # Convert to Graph object
        edges = list(G_nx.edges(data="weight" if self.weighted else False))
        if self.weighted:
            edges = [(u, v, w) for u, v, w in edges]
        else:
            edges = [(u, v) for u, v in edges]

        graph = Graph(
            edges=edges,
            n_nodes=len(series),
            directed=self.directed,
            weighted=self.weighted,
        )

        return _extract_network_features(graph, include_degrees=self.include_degrees)


class RecurrenceNetworkFeaturizer(BaseFeaturizer):
    """Recurrence Network featurizer.

    Converts SeriesLike to TableLike by building recurrence network and extracting features.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        embedding_dimension: Optional[int] = None,
        time_delay: int = 1,
        metric: str = "euclidean",
        rule: Optional[str] = None,
        k: Optional[int] = None,
        include_degrees: bool = True,
    ):
        """Initialize recurrence network featurizer.

        Args:
            threshold: Distance threshold for recurrence (epsilon rule).
            embedding_dimension: Embedding dimension (m).
            time_delay: Time delay (tau).
            metric: Distance metric.
            rule: Threshold rule ('epsilon' or 'knn').
            k: Number of neighbors for k-NN rule.
            include_degrees: Whether to include degree sequence in output.
        """
        super().__init__()
        self.threshold = threshold
        self.embedding_dimension = embedding_dimension
        self.time_delay = time_delay
        self.metric = metric
        self.rule = rule
        self.k = k
        self.include_degrees = include_degrees

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="TableLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "RecurrenceNetworkFeaturizer":
        """Fit the featurizer (no-op for recurrence network).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.DataFrame:
        """Build recurrence network and extract features.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            TableLike DataFrame with network features.
        """
        self._check_is_fitted()

        series = _validate_series(y)

        # Build recurrence network using native implementation
        G_nx = build_recurrence_network(
            series,
            threshold=self.threshold,
            embedding_dimension=self.embedding_dimension,
            time_delay=self.time_delay,
            metric=self.metric,
            rule=self.rule,
            k=self.k,
        )

        # Convert to Graph object
        edges = list(G_nx.edges())
        graph = Graph(
            edges=edges,
            n_nodes=G_nx.number_of_nodes(),
            directed=G_nx.is_directed(),
            weighted=False,
        )

        return _extract_network_features(graph, include_degrees=self.include_degrees)


class TransitionNetworkFeaturizer(BaseFeaturizer):
    """Transition Network featurizer.

    Converts SeriesLike to TableLike by building transition network and extracting features.
    """

    def __init__(
        self,
        n_bins: int = 10,
        order: int = 1,
        symbolizer: Optional[str] = None,
        include_degrees: bool = True,
    ):
        """Initialize transition network featurizer.

        Args:
            n_bins: Number of bins for symbolization.
            order: Order of transition patterns.
            symbolizer: Symbolization method ('equal_width' or 'ordinal').
            include_degrees: Whether to include degree sequence in output.
        """
        super().__init__()
        self.n_bins = n_bins
        self.order = order
        self.symbolizer = symbolizer
        self.include_degrees = include_degrees

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="TableLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(self, y: Any, X: Optional[Any] = None, **fit_params: Any) -> "TransitionNetworkFeaturizer":
        """Fit the featurizer (no-op for transition network).

        Args:
            y: Target time series.
            X: Optional exogenous data (ignored).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(self, y: Any, X: Optional[Any] = None) -> pd.DataFrame:
        """Build transition network and extract features.

        Args:
            y: SeriesLike data.
            X: Optional exogenous data (ignored).

        Returns:
            TableLike DataFrame with network features.
        """
        self._check_is_fitted()

        series = _validate_series(y)

        # Build transition network using native implementation
        G_nx = build_transition_network(
            series,
            n_bins=self.n_bins,
            order=self.order,
            symbolizer=self.symbolizer,
        )

        # Convert to Graph object
        edges = list(G_nx.edges())
        graph = Graph(
            edges=edges,
            n_nodes=G_nx.number_of_nodes(),
            directed=G_nx.is_directed(),
            weighted=False,
        )

        return _extract_network_features(graph, include_degrees=self.include_degrees)
