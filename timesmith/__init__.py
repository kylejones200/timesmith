"""TimeSmith: A time series machine learning library with strict layer boundaries."""

__version__ = "0.1.1"

# Exceptions
# Compose
from timesmith.compose import (
    Adapter,
    FeatureUnion,
    ForecasterPipeline,
    Pipeline,
    make_forecaster_pipeline,
    make_pipeline,
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

# Eval
from timesmith.eval import (
    ExpandingWindowSplit,
    ModelComparison,
    ModelResult,
    SlidingWindowSplit,
    backtest_forecaster,
    bias,
    compare_models,
    mae,
    mape,
    rmse,
    summarize_backtest,
    ubrmse,
)
from timesmith.exceptions import (
    ConfigurationError,
    DataError,
    ForecastError,
    NotFittedError,
    PipelineError,
    TimeSmithError,
    TransformError,
    UnsupportedOperationError,
    ValidationError,
)

# Results
from timesmith.results import BacktestResult, Forecast

# Serialization
from timesmith.serialization import load_model, save_model

# Note: AnomSmith adapter and compatibility utilities are NOT imported by default
# to avoid circular imports. Timesmith (foundation) must never import downstream repos.
# Users can explicitly import if needed:
#   from timesmith.compose.anomsmith_adapter import AnomSmithAdapter
#   from timesmith.utils.anomsmith_compat import convert_to_anomsmith_format, ...
# Tasks
from timesmith.tasks import DetectTask, ForecastTask

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

# Utils
from timesmith.utils import (
    autocorrelation,
    autocorrelation_plot_data,
    black_scholes_monte_carlo,
    bootstrap_confidence_intervals,
    correlation_distance,
    create_sequences,
    create_sequences_with_exog,
    cross_correlation_distance,
    detect_frequency,
    dtw_distance,
    ensure_datetime_index,
    euclidean_distance,
    fill_missing_dates,
    load_ts_data,
    manhattan_distance,
    monte_carlo_simulation,
    parametric_confidence_intervals,
    partial_autocorrelation,
    plot_monte_carlo,
    remove_outliers_iqr,
    resample_ts,
    split_ts,
)

# Optional stationarity tests
try:
    from timesmith.utils.stationarity import is_stationary, test_stationarity  # noqa: F401

    HAS_STATIONARITY = True
except ImportError:
    HAS_STATIONARITY = False

# Optional data loaders
try:
    from timesmith.datasets import load_fred, load_yahoo  # noqa: F401

    HAS_DATA_LOADERS = True
except ImportError:
    HAS_DATA_LOADERS = False

# Optional plotting utilities (requires plotsmith)
HAS_PLOTTING = False
try:
    from timesmith.utils.plotting import (
        HAS_PLOTSMITH,
        plot_autocorrelation,  # noqa: F401
        plot_forecast,  # noqa: F401
        plot_monte_carlo_paths,  # noqa: F401
        plot_multiple_series,  # noqa: F401
        plot_residuals,  # noqa: F401
        plot_timeseries,  # noqa: F401
    )

    HAS_PLOTTING = HAS_PLOTSMITH
except ImportError:
    pass

# Climatology utilities
from timesmith.utils.climatology import (  # noqa: E402
    compute_climatology,  # noqa: F401
    compute_anomalies,  # noqa: F401
    detect_extreme_events,  # noqa: F401
)

# Core Featurizers and Transformers
from timesmith.core import (  # noqa: E402
    BayesianChangePointDetector,
    CUSUMDetector,  # noqa: F401
    DecomposeTransformer,
    DeseasonalizeTransformer,
    DetrendTransformer,
    DifferencingFeaturizer,
    LagFeaturizer,
    MissingDateFiller,
    MissingValueFiller,
    OutlierRemover,
    PELTDetector,
    Resampler,
    RollingFeaturizer,
    SeasonalFeaturizer,
    TimeFeaturizer,
    VotingEnsembleDetector,  # noqa: F401
    detect_seasonality,
    detect_trend,
    preprocess_for_changepoint,
)

# Network
from timesmith.network import (  # noqa: E402
    Graph,
    HVGFeaturizer,
    MultiscaleGraphs,
    NetworkSignificanceResult,
    NVGFeaturizer,
    RecurrenceNetworkFeaturizer,
    TransferEntropyDetector,
    TransitionNetworkFeaturizer,
    build_windows,
    coarse_grain,
    compute_clustering,
    compute_modularity,
    compute_network_metric_significance,
    compute_path_lengths,
    conditional_transfer_entropy,
    directed_3node_motifs,
    generate_surrogate,
    graph_summary,
    net_enn,
    net_knn,
    network_metrics,
    node_roles,
    transfer_entropy,
    transfer_entropy_network,
    ts_to_windows,
    undirected_4node_motifs,
)

# Optional filters
try:
    from timesmith.core.filters import ButterworthFilter, SavitzkyGolayFilter  # noqa: F401

    HAS_FILTERS = True
except ImportError:
    HAS_FILTERS = False

# Forecasters
from timesmith.forecasters import (  # noqa: E402
    ARIMAForecaster,
    BlackScholesMonteCarloForecaster,
    ExponentialMovingAverageForecaster,
    ExponentialSmoothingForecaster,
    MonteCarloForecaster,
    SimpleMovingAverageForecaster,
    WeightedMovingAverageForecaster,
)

# Optional forecasters
try:
    from timesmith.forecasters.prophet import ProphetForecaster  # noqa: F401

    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    from timesmith.forecasters.var import VARForecaster  # noqa: F401

    HAS_VAR = True
except ImportError:
    HAS_VAR = False

try:
    from timesmith.forecasters.lstm import LSTMForecaster  # noqa: F401

    HAS_LSTM = True
except ImportError:
    HAS_LSTM = False

try:
    from timesmith.forecasters.kalman import KalmanFilterForecaster  # noqa: F401

    HAS_KALMAN = True
except ImportError:
    HAS_KALMAN = False

# Optional Bayesian forecaster
try:
    from timesmith.forecasters.bayesian import BayesianForecaster  # noqa: F401

    HAS_BAYESIAN = True
except ImportError:
    HAS_BAYESIAN = False

# Optional Ensemble forecaster
try:
    from timesmith.forecasters.ensemble import EnsembleForecaster  # noqa: F401

    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False

__all__ = [
    # Exceptions
    "TimeSmithError",
    "ValidationError",
    "DataError",
    "ForecastError",
    "TransformError",
    "PipelineError",
    "ConfigurationError",
    "NotFittedError",
    "UnsupportedOperationError",
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
    "bias",
    "ubrmse",
    "backtest_forecaster",
    "summarize_backtest",
    "ModelComparison",
    "ModelResult",
    "compare_models",
    # Results
    "Forecast",
    "BacktestResult",
    # Utils
    "load_ts_data",
    "ensure_datetime_index",
    "resample_ts",
    "split_ts",
    "train_test_split",
    "detect_frequency",
    "fill_missing_dates",
    "remove_outliers_iqr",
    "monte_carlo_simulation",
    "black_scholes_monte_carlo",
    "plot_monte_carlo",
    "create_sequences",
    "create_sequences_with_exog",
    # Plotting utilities (if plotsmith available)
    "autocorrelation",
    "partial_autocorrelation",
    "autocorrelation_plot_data",
    "bootstrap_confidence_intervals",
    "parametric_confidence_intervals",
    # Featurizers
    "LagFeaturizer",
    "RollingFeaturizer",
    "TimeFeaturizer",
    "DifferencingFeaturizer",
    "SeasonalFeaturizer",
    # Transformers
    "OutlierRemover",
    "MissingValueFiller",
    "Resampler",
    "MissingDateFiller",
    "DecomposeTransformer",
    "DetrendTransformer",
    "DeseasonalizeTransformer",
    # Change Point Detection
    "PELTDetector",
    "BayesianChangePointDetector",
    "preprocess_for_changepoint",
    "detect_trend",
    "detect_seasonality",
    # Forecasters
    "ARIMAForecaster",
    "SimpleMovingAverageForecaster",
    "ExponentialMovingAverageForecaster",
    "WeightedMovingAverageForecaster",
    "ExponentialSmoothingForecaster",
    "MonteCarloForecaster",
    "BlackScholesMonteCarloForecaster",
    "LinearTrendForecaster",
    "SyntheticControlForecaster",
    # Filters (conditionally exported)
    # Stationarity tests (conditionally exported)
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
    # Distance metrics
    "correlation_distance",
    "cross_correlation_distance",
    "dtw_distance",
    "euclidean_distance",
    "manhattan_distance",
    # Serialization
    "save_model",
    "load_model",
]

# Conditionally add plotting functions
if HAS_PLOTTING:
    __all__.extend(
        [
            "plot_timeseries",
            "plot_forecast",
            "plot_residuals",
            "plot_multiple_series",
            "plot_autocorrelation",
            "plot_monte_carlo_paths",
        ]
    )

# Conditionally add optional components
if HAS_BAYESIAN:
    __all__.append("BayesianForecaster")

if HAS_ENSEMBLE:
    __all__.append("EnsembleForecaster")

if HAS_PROPHET:
    __all__.append("ProphetForecaster")

if HAS_VAR:
    __all__.append("VARForecaster")

if HAS_LSTM:
    __all__.append("LSTMForecaster")

if HAS_KALMAN:
    __all__.append("KalmanFilterForecaster")

if HAS_FILTERS:
    __all__.extend(["ButterworthFilter", "SavitzkyGolayFilter"])

if HAS_STATIONARITY:
    __all__.extend(["test_stationarity", "is_stationary"])

if HAS_DATA_LOADERS:
    __all__.extend(["load_fred", "load_yahoo"])
