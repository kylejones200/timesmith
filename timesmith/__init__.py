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
    bias,
    mae,
    mape,
    rmse,
    ubrmse,
    summarize_backtest,
)

# Results
from timesmith.results import BacktestResult, Forecast

# Utils
from timesmith.utils import (
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
    plot_monte_carlo,
    remove_outliers_iqr,
    resample_ts,
    split_ts,
)

# Optional stationarity tests
try:
    from timesmith.utils.stationarity import test_stationarity
    HAS_STATIONARITY = True
except ImportError:
    HAS_STATIONARITY = False

# Climatology utilities
from timesmith.utils.climatology import (
    compute_climatology,
    compute_anomalies,
    detect_extreme_events,
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
    DecomposeTransformer,
    DeseasonalizeTransformer,
    DetrendTransformer,
    DifferencingFeaturizer,
    LagFeaturizer,
    MissingDateFiller,
    MissingValueFiller,
    OutlierRemover,
    PELTDetector,
    BayesianChangePointDetector,
    CUSUMDetector,
    Resampler,
    VotingEnsembleDetector,
    RollingFeaturizer,
    TimeFeaturizer,
    detect_seasonality,
    detect_trend,
    preprocess_for_changepoint,
)

# Optional filters
try:
    from timesmith.core.filters import ButterworthFilter, SavitzkyGolayFilter
    HAS_FILTERS = True
except ImportError:
    HAS_FILTERS = False

# Forecasters
from timesmith.forecasters import (
    ARIMAForecaster,
    ExponentialMovingAverageForecaster,
    ExponentialSmoothingForecaster,
    MonteCarloForecaster,
    SimpleMovingAverageForecaster,
    WeightedMovingAverageForecaster,
)

# Optional Bayesian forecaster
try:
    from timesmith.forecasters.bayesian import BayesianForecaster
    HAS_BAYESIAN = True
except ImportError:
    HAS_BAYESIAN = False

# Optional Ensemble forecaster
try:
    from timesmith.forecasters.ensemble import EnsembleForecaster
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False

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
    "bias",
    "ubrmse",
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
    "TransferEntropyDetector",
    # Distance metrics
    "correlation_distance",
    "cross_correlation_distance",
    "dtw_distance",
    "euclidean_distance",
    "manhattan_distance",
]

# Conditionally add optional components
if HAS_BAYESIAN:
    __all__.append("BayesianForecaster")

if HAS_ENSEMBLE:
    __all__.append("EnsembleForecaster")

if HAS_FILTERS:
    __all__.extend(["ButterworthFilter", "SavitzkyGolayFilter"])

if HAS_STATIONARITY:
    __all__.append("test_stationarity")

