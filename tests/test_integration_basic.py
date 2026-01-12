"""Basic integration tests for TimeSmith core functionality.

These tests verify that the main components work together correctly.
"""

import numpy as np
import pandas as pd
import pytest

from timesmith import (
    ForecastTask,
    NotFittedError,
    SimpleMovingAverageForecaster,
    backtest_forecaster,
    make_forecaster_pipeline,
    summarize_backtest,
)
from timesmith.examples import LogTransformer, NaiveForecaster


class TestBasicForecastWorkflow:
    """Test basic forecasting workflow."""

    def test_simple_forecast(self):
        """Test a simple forecast without backtesting."""
        # Create simple data
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y = pd.Series(np.random.randn(20).cumsum(), index=dates)

        # Create forecaster
        forecaster = SimpleMovingAverageForecaster(window=3)

        # Fit and predict
        forecaster.fit(y)
        forecast = forecaster.predict(fh=5)

        # Check results
        assert forecast is not None
        assert hasattr(forecast, "y_pred")
        assert len(forecast.y_pred) == 5

    def test_forecast_without_fit_raises_error(self):
        """Test that predicting without fitting raises NotFittedError."""
        forecaster = SimpleMovingAverageForecaster(window=3)
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y = pd.Series(np.random.randn(20).cumsum(), index=dates)

        with pytest.raises(NotFittedError):
            forecaster.predict(fh=5)

    def test_pipeline_workflow(self):
        """Test pipeline with transformer and forecaster."""
        # Create data
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        y = pd.Series(np.abs(np.random.randn(30).cumsum()) + 1, index=dates)

        # Create pipeline
        transformer = LogTransformer(offset=1.0)
        forecaster = NaiveForecaster()
        pipeline = make_forecaster_pipeline(transformer, forecaster=forecaster)

        # Fit and predict
        pipeline.fit(y)
        forecast = pipeline.predict(fh=5)

        # Check results
        assert forecast is not None
        assert len(forecast.y_pred) == 5

    def test_backtest_workflow(self):
        """Test backtesting workflow."""
        # Create data
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        y = pd.Series(np.random.randn(50).cumsum(), index=dates)

        # Create task
        task = ForecastTask(y=y, fh=5, frequency="D")

        # Create forecaster
        forecaster = SimpleMovingAverageForecaster(window=5)

        # Run backtest
        result = backtest_forecaster(forecaster, task)

        # Check results
        assert result is not None
        assert hasattr(result, "results")
        assert len(result.results) > 0

        # Summarize
        summary = summarize_backtest(result)
        assert "aggregate_metrics" in summary
        assert "per_fold_metrics" in summary

