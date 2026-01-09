"""Basic forecast example demonstrating TimeSmith architecture.

This script demonstrates:
1. Loading data (synthetic for this example)
2. Creating a forecast task
3. Building a pipeline with transformer and forecaster
4. Running backtest
5. Printing summary
"""

import logging
import sys

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, "..")

from timesmith import (
    ForecastTask,
    backtest_forecaster,
    make_forecaster_pipeline,
    summarize_backtest,
)

# Import example implementations
try:
    from timesmith.examples.simple_forecaster import NaiveForecaster
    from timesmith.examples.simple_transformer import LogTransformer
except ImportError:
    # Fallback if examples not in package
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from timesmith.examples.simple_forecaster import NaiveForecaster
    from timesmith.examples.simple_transformer import LogTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_example_data():
    """Load or generate example time series data.

    Returns:
        pandas Series with time index.
    """
    # Generate synthetic data
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    trend = np.linspace(10, 20, n)
    noise = np.random.normal(0, 1, n)
    y = pd.Series(trend + noise, index=dates, name="value")
    return y


def main():
    """Run basic forecast example."""
    logger.info("Loading example data...")
    y = load_example_data()
    logger.info(f"Loaded data with {len(y)} observations")

    # Create forecast task
    logger.info("Creating forecast task...")
    task = ForecastTask(y=y, fh=5, frequency="D")
    logger.info(f"Forecast task: fh={task.fh}, frequency={task.frequency}")

    # Create pipeline: log transform -> naive forecaster
    logger.info("Building forecaster pipeline...")
    transformer = LogTransformer(offset=1.0)
    forecaster = NaiveForecaster()
    pipeline = make_forecaster_pipeline(transformer, forecaster=forecaster)
    logger.info("Pipeline created: LogTransformer -> NaiveForecaster")

    # Run backtest
    logger.info("Running backtest...")
    result = backtest_forecaster(pipeline, task)
    logger.info(f"Backtest completed with {len(result.results)} folds")

    # Summarize results
    logger.info("Summarizing results...")
    summary = summarize_backtest(result)

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"\nNumber of folds: {summary['n_folds']}")
    print("\nAggregate Metrics:")
    for metric, value in summary["aggregate_metrics"].items():
        print(f"  {metric}: {value:.4f}")

    if "per_fold_metrics" in summary:
        print("\nPer-Fold Metrics:")
        print(summary["per_fold_metrics"].to_string(index=False))

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

