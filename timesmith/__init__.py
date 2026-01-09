"""TimeSmith: A time series machine learning library with strict layer boundaries."""

__version__ = "0.0.1"

# Typing
from timesmith.typing import (
    ForecastLike,
    PanelLike,
    SeriesLike,
    TableLike,
    assert_panel,
    assert_series,
    assert_table,
    is_panel,
    is_series,
    is_table,
)

# Core
from timesmith.core import (
    BaseDetector,
    BaseEstimator,
    BaseFeaturizer,
    BaseForecaster,
    BaseObject,
    BaseTransformer,
    get_tags,
    set_tags,
    validate_input,
)

# Compose
from timesmith.compose import (
    Adapter,
    FeatureUnion,
    ForecasterPipeline,
    Pipeline,
    make_forecaster_pipeline,
    make_pipeline,
)

# Tasks
from timesmith.tasks import DetectTask, ForecastTask

# Eval
from timesmith.eval import (
    ExpandingWindowSplit,
    SlidingWindowSplit,
    backtest_forecaster,
    mae,
    mape,
    rmse,
    summarize_backtest,
)

# Results
from timesmith.results import BacktestResult, Forecast

# Utils
from timesmith.utils import (
    correlation_distance,
    cross_correlation_distance,
    detect_frequency,
    dtw_distance,
    ensure_datetime_index,
    euclidean_distance,
    fill_missing_dates,
    load_ts_data,
    manhattan_distance,
    monte_carlo_simulation,
    plot_monte_carlo,
    remove_outliers_iqr,
    resample_ts,
    split_ts,
)

# Network
from timesmith.network import (
    Graph,
    HVGFeaturizer,
    NVGFeaturizer,
    RecurrenceNetworkFeaturizer,
    TransitionNetworkFeaturizer,
    compute_clustering,
    compute_modularity,
    compute_path_lengths,
    graph_summary,
    network_metrics,
    transfer_entropy,
    TransferEntropyDetector,
)

# Core Featurizers and Transformers
from timesmith.core import (
    DifferencingFeaturizer,
    LagFeaturizer,
    MissingDateFiller,
    MissingValueFiller,
    OutlierRemover,
    Resampler,
    RollingFeaturizer,
    TimeFeaturizer,
)

# Forecasters
from timesmith.forecasters import (
    ARIMAForecaster,
    ExponentialMovingAverageForecaster,
    ExponentialSmoothingForecaster,
    SimpleMovingAverageForecaster,
    WeightedMovingAverageForecaster,
)

__all__ = [
    # Typing
    "SeriesLike",
    "PanelLike",
    "TableLike",
    "ForecastLike",
    "is_series",
    "is_panel",
    "is_table",
    "assert_series",
    "assert_panel",
    "assert_table",
    # Core
    "BaseObject",
    "BaseEstimator",
    "BaseTransformer",
    "BaseForecaster",
    "BaseDetector",
    "BaseFeaturizer",
    "get_tags",
    "set_tags",
    "validate_input",
    # Compose
    "Pipeline",
    "ForecasterPipeline",
    "Adapter",
    "FeatureUnion",
    "make_pipeline",
    "make_forecaster_pipeline",
    # Tasks
    "ForecastTask",
    "DetectTask",
    # Eval
    "ExpandingWindowSplit",
    "SlidingWindowSplit",
    "mae",
    "rmse",
    "mape",
    "backtest_forecaster",
    "summarize_backtest",
    # Results
    "Forecast",
    "BacktestResult",
    # Utils
    "load_ts_data",
    "ensure_datetime_index",
    "resample_ts",
    "split_ts",
    "detect_frequency",
    "fill_missing_dates",
    "remove_outliers_iqr",
    "monte_carlo_simulation",
    "plot_monte_carlo",
    # Featurizers
    "LagFeaturizer",
    "RollingFeaturizer",
    "TimeFeaturizer",
    "DifferencingFeaturizer",
    # Transformers
    "OutlierRemover",
    "MissingValueFiller",
    "Resampler",
    "MissingDateFiller",
    # Forecasters
    "ARIMAForecaster",
    "SimpleMovingAverageForecaster",
    "ExponentialMovingAverageForecaster",
    "WeightedMovingAverageForecaster",
    "ExponentialSmoothingForecaster",
    # Network
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
    # Distance metrics
    "correlation_distance",
    "cross_correlation_distance",
    "dtw_distance",
    "euclidean_distance",
    "manhattan_distance",
]

