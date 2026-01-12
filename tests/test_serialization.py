"""Tests for model serialization."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timesmith import SimpleMovingAverageForecaster
from timesmith.exceptions import DataError, NotFittedError
from timesmith.serialization import load_model, save_model


class TestModelSerialization:
    """Tests for model serialization."""

    def test_save_and_load_pickle(self):
        """Test saving and loading model with pickle."""
        # Create and fit a model
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y = pd.Series(np.random.randn(20).cumsum(), index=dates)

        forecaster = SimpleMovingAverageForecaster(window=5)
        forecaster.fit(y)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"
            save_model(forecaster, filepath, format="pickle")

            loaded = load_model(filepath, format="pickle")

            # Verify loaded model
            assert loaded.is_fitted
            assert loaded.window == 5
            assert loaded.__class__ == forecaster.__class__

            # Verify predictions match
            original_pred = forecaster.predict(fh=3)
            loaded_pred = loaded.predict(fh=3)
            np.testing.assert_array_almost_equal(
                original_pred.y_pred, loaded_pred.y_pred
            )

    def test_save_unfitted_raises_error(self):
        """Test that saving unfitted model raises error."""
        forecaster = SimpleMovingAverageForecaster(window=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"
            with pytest.raises(NotFittedError):
                save_model(forecaster, filepath)

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nonexistent.pkl"
            with pytest.raises(DataError, match="not found"):
                load_model(filepath)

    def test_save_with_metadata(self):
        """Test saving model with metadata."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y = pd.Series(np.random.randn(20).cumsum(), index=dates)

        forecaster = SimpleMovingAverageForecaster(window=5)
        forecaster.fit(y)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"
            save_model(forecaster, filepath, include_metadata=True)

            # Check metadata file exists
            metadata_path = filepath.with_suffix(filepath.suffix + ".metadata.json")
            assert metadata_path.exists()

            # Load and verify metadata
            from timesmith.serialization import get_model_metadata

            metadata = get_model_metadata(filepath)
            assert metadata["class_name"] == "SimpleMovingAverageForecaster"
            assert metadata["is_fitted"] is True

    def test_auto_format_detection(self):
        """Test automatic format detection."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y = pd.Series(np.random.randn(20).cumsum(), index=dates)

        forecaster = SimpleMovingAverageForecaster(window=5)
        forecaster.fit(y)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with .pkl extension
            filepath = Path(tmpdir) / "model.pkl"
            save_model(forecaster, filepath, format="auto")
            loaded = load_model(filepath, format="auto")
            assert loaded.is_fitted

    def test_save_load_consistency(self):
        """Test that saved and loaded models produce same predictions."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y = pd.Series(np.random.randn(20).cumsum(), index=dates)

        forecaster = SimpleMovingAverageForecaster(window=5)
        forecaster.fit(y)
        original_pred = forecaster.predict(fh=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"
            save_model(forecaster, filepath)
            loaded = load_model(filepath)
            loaded_pred = loaded.predict(fh=5)

            np.testing.assert_array_almost_equal(
                original_pred.y_pred, loaded_pred.y_pred
            )

