"""Tests for pipelines and composition."""

import numpy as np
import pandas as pd
import pytest

from timesmith import (
    ForecasterPipeline,
    NotFittedError,
    SimpleMovingAverageForecaster,
    make_forecaster_pipeline,
)
from timesmith.examples import LogTransformer, NaiveForecaster


class TestForecasterPipeline:
    """Tests for ForecasterPipeline."""

    def test_create_pipeline(self):
        """Test creating a forecaster pipeline."""
        transformer = LogTransformer(offset=1.0)
        forecaster = NaiveForecaster()
        pipeline = make_forecaster_pipeline(transformer, forecaster=forecaster)

        assert isinstance(pipeline, ForecasterPipeline)
        assert len(pipeline.steps) == 1
        assert pipeline.forecaster == forecaster

    def test_pipeline_fit_and_predict(self):
        """Test fitting and predicting with pipeline."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        y = pd.Series(np.abs(np.random.randn(30).cumsum()) + 1, index=dates)

        transformer = LogTransformer(offset=1.0)
        forecaster = NaiveForecaster()
        pipeline = make_forecaster_pipeline(transformer, forecaster=forecaster)

        pipeline.fit(y)
        assert pipeline.is_fitted

        forecast = pipeline.predict(fh=5)
        assert forecast is not None
        assert len(forecast.y_pred) == 5

    def test_pipeline_predict_without_fit(self):
        """Test that pipeline predict raises error if not fitted."""
        transformer = LogTransformer(offset=1.0)
        forecaster = NaiveForecaster()
        pipeline = make_forecaster_pipeline(transformer, forecaster=forecaster)

        with pytest.raises(NotFittedError):
            pipeline.predict(fh=5)

    def test_pipeline_with_multiple_transformers(self):
        """Test pipeline with multiple transformers."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        y = pd.Series(np.abs(np.random.randn(30).cumsum()) + 1, index=dates)

        transformer1 = LogTransformer(offset=1.0)
        transformer2 = LogTransformer(offset=0.5)
        forecaster = NaiveForecaster()

        # Create pipeline manually
        pipeline = ForecasterPipeline(
            steps=[("log1", transformer1), ("log2", transformer2)],
            forecaster=forecaster,
        )

        pipeline.fit(y)
        forecast = pipeline.predict(fh=5)

        assert forecast is not None
        assert len(forecast.y_pred) == 5

    def test_pipeline_get_params(self):
        """Test that pipeline get_params works."""
        transformer = LogTransformer(offset=1.0)
        forecaster = NaiveForecaster()
        pipeline = make_forecaster_pipeline(transformer, forecaster=forecaster)

        params = pipeline.get_params()
        # Check for forecaster-related params (they use double underscore notation)
        forecaster_params = [k for k in params.keys() if k.startswith("forecaster__")]
        assert len(forecaster_params) > 0, "Should have forecaster parameters"
        assert len(params) > 0

    def test_pipeline_set_params(self):
        """Test that pipeline set_params works."""
        transformer = LogTransformer(offset=1.0)
        forecaster = SimpleMovingAverageForecaster(window=5)
        pipeline = make_forecaster_pipeline(transformer, forecaster=forecaster)

        pipeline.set_params(forecaster__window=10)
        assert pipeline.forecaster.window == 10
