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
]

