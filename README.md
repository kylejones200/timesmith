# TimeSmith

[![PyPI version](https://badge.fury.io/py/timesmith.svg)](https://badge.fury.io/py/timesmith)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/kylejones200/timesmith/workflows/Tests/badge.svg)](https://github.com/kylejones200/timesmith/actions)
[![Documentation](https://readthedocs.org/projects/timesmith/badge/?version=latest)](https://timesmith.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

TimeSmith is a time series machine learning library with strict layer boundaries and a clean architecture.

## Architecture

TimeSmith uses a four-layer architecture with strict boundaries:

### Layer 1: Typing (`timesmith/typing`)
Scientific types and runtime validators for time series data structures:
- `SeriesLike`: pandas Series or single-column DataFrame with datetime/int index
- `PanelLike`: DataFrame with entity key plus time index
- `TableLike`: DataFrame with row index aligned to time
- `ForecastLike`: Forecast results with predictions and optional intervals

### Layer 2: Core (`timesmith/core`)
Base classes, parameter handling, tags, and input validation:
- `BaseObject`: Parameter management (`get_params`, `set_params`, `clone`)
- `BaseEstimator`: Base class with `fit` capability
- `BaseTransformer`: Transformers with `transform` and optional `inverse_transform`
- `BaseForecaster`: Forecasters with `predict` and optional `predict_interval`
- `BaseDetector`: Anomaly detectors with `score` and `predict`
- `BaseFeaturizer`: Transformers that output `TableLike` data

### Layer 3: Compose (`timesmith/compose`)
Pipeline and adapter objects for composition:
- `Pipeline`: Chains transformer steps with scitype change support
- `ForecasterPipeline`: Transformer(s) then forecaster
- `Adapter`: Converts between scitypes (Series ↔ Table)
- `FeatureUnion`: Runs multiple featurizers and concatenates results

### Layer 4: Tasks & Eval (`timesmith/tasks`, `timesmith/eval`)
Task objects and evaluation tools:
- `ForecastTask`: Binds data, horizon, and target semantics
- `DetectTask`: Anomaly detection task definition
- `ExpandingWindowSplit`, `SlidingWindowSplit`: Cross-validation splitters
- `backtest_forecaster`: Run backtests with metrics
- `summarize_backtest`: Aggregate and per-fold metrics

## Quick Example

```python
import pandas as pd
from timesmith import ForecastTask, backtest_forecaster, make_forecaster_pipeline
from timesmith.examples import NaiveForecaster, LogTransformer

# Load data
y = pd.Series([...], index=pd.date_range("2020-01-01", periods=100))

# Create forecast task
task = ForecastTask(y=y, fh=5, frequency="D")

# Build pipeline
transformer = LogTransformer(offset=1.0)
forecaster = NaiveForecaster()
pipeline = make_forecaster_pipeline(transformer, forecaster=forecaster)

# Run backtest
result = backtest_forecaster(pipeline, task)

# Summarize results
summary = summarize_backtest(result)
print(f"Mean MAE: {summary['aggregate_metrics']['mean_mae']:.4f}")
```

## Running the Example

```bash
python examples/basic_forecast.py
```

This will:
1. Load/generate example data
2. Create a forecast task
3. Build a pipeline with transformer and forecaster
4. Run backtest
5. Print summary metrics

## Installation

```bash
pip install timesmith
```

## Development

```bash
# Install dependencies (requires uv: https://github.com/astral-sh/uv)
uv sync --group dev

# Run tests
uv run pytest tests/

# Format code
uv run black timesmith tests

# Run linting
uv run ruff check timesmith tests
```

## Features

- **Model Serialization**: Save and load fitted models
- **Comprehensive Error Handling**: Custom exceptions with context
- **Logging Configuration**: Centralized logging with environment variable support
- **Data Validation**: Edge case handling and data quality checks
- **Pipeline Composition**: Flexible transformer and forecaster pipelines
- **Backtesting**: Time series cross-validation with multiple metrics
- **Network Analysis**: Graph-based time series analysis
- **Multiple Forecasters**: ARIMA, LSTM, Bayesian, and more

## Model Serialization


```python
from timesmith import SimpleMovingAverageForecaster, save_model, load_model

# Fit and save
forecaster = SimpleMovingAverageForecaster(window=5)
forecaster.fit(y)
save_model(forecaster, "model.pkl")

# Load later
loaded = load_model("model.pkl")
forecast = loaded.predict(fh=10)
```

## Error Handling

TimeSmith provides custom exceptions for better error handling:

```python
from timesmith import NotFittedError, DataError, ValidationError

try:
    forecaster.predict(fh=5)
except NotFittedError as e:
    print(f"Error: {e}")
    print(f"Context: {e.context}")
```

## Logging

Configure logging via environment variables:

```bash
export TIMESMITH_LOG_LEVEL=DEBUG
export TIMESMITH_LOG_FORMAT=detailed
```

Or programmatically:

```python
from timesmith.logging_config import configure_logging
configure_logging(level="INFO")
```

## Design Principles

1. **Strict Layer Boundaries**: Core cannot import eval. Typing cannot import anything.
2. **Validation at Boundaries**: Validate once at public API boundaries only.
3. **Task Semantics**: Tasks hold semantics. Estimators only store params and fitted state.
4. **Composition**: Use pipelines and adapters for flexible workflows.
5. **Type Safety**: Scientific types with runtime validators for data structures.

## Requirements

**Core dependencies (minimal):**
- Python 3.12 or higher
- pandas >= 1.5.0
- numpy >= 1.20.0

**Optional dependencies:**
- `scipy` - For filters, changepoint detection, distances (install: `pip install timesmith[scipy]`)
- `networkx` - For network analysis features (install: `pip install timesmith[network]`)
- See `pyproject.toml` for all optional dependency groups

See `pyproject.toml` for optional dependencies for specific forecasters.