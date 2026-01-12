"""Tests for forecasters."""

import numpy as np
import pandas as pd
import pytest

from timesmith import SimpleMovingAverageForecaster
from timesmith.exceptions import NotFittedError


class TestSimpleMovingAverageForecaster:
    """Tests for SimpleMovingAverageForecaster."""

    def test_init(self):
        """Test forecaster initialization."""
        forecaster = SimpleMovingAverageForecaster(window=5)
        assert forecaster.window == 5
        assert not forecaster.is_fitted

    def test_fit(self):
        """Test fitting the forecaster."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y = pd.Series(np.random.randn(20).cumsum(), index=dates)

        forecaster = SimpleMovingAverageForecaster(window=3)
        forecaster.fit(y)

        assert forecaster.is_fitted
        assert hasattr(forecaster, "train_index_")
        assert hasattr(forecaster, "last_ma_value_")

    def test_predict_without_fit(self):
        """Test that predict raises error if not fitted."""
        forecaster = SimpleMovingAverageForecaster(window=3)
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y = pd.Series(np.random.randn(20).cumsum(), index=dates)

        with pytest.raises(NotFittedError):
            forecaster.predict(fh=5)

    def test_predict(self):
        """Test prediction."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y = pd.Series(np.random.randn(20).cumsum(), index=dates)

        forecaster = SimpleMovingAverageForecaster(window=5)
        forecaster.fit(y)
        forecast = forecaster.predict(fh=3)

        assert forecast is not None
        assert hasattr(forecast, "y_pred")
        assert len(forecast.y_pred) == 3
        assert all(np.isfinite(forecast.y_pred))

    def test_predict_with_array_fh(self):
        """Test prediction with array forecast horizon."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y = pd.Series(np.random.randn(20).cumsum(), index=dates)

        forecaster = SimpleMovingAverageForecaster(window=5)
        forecaster.fit(y)
        forecast = forecaster.predict(fh=[1, 3, 5])

        assert len(forecast.y_pred) == 3

    def test_predict_consistency(self):
        """Test that predictions are consistent for same data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y = pd.Series(np.random.randn(20).cumsum(), index=dates)

        forecaster1 = SimpleMovingAverageForecaster(window=5)
        forecaster1.fit(y)
        forecast1 = forecaster1.predict(fh=3)

        forecaster2 = SimpleMovingAverageForecaster(window=5)
        forecaster2.fit(y)
        forecast2 = forecaster2.predict(fh=3)

        np.testing.assert_array_almost_equal(forecast1.y_pred, forecast2.y_pred)

    def test_fit_with_dataframe(self):
        """Test fitting with single-column DataFrame."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        df = pd.DataFrame({"value": np.random.randn(20).cumsum()}, index=dates)

        forecaster = SimpleMovingAverageForecaster(window=5)
        forecaster.fit(df)

        assert forecaster.is_fitted

    def test_predict_with_small_window(self):
        """Test prediction with small window size."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        y = pd.Series(np.random.randn(10).cumsum(), index=dates)

        forecaster = SimpleMovingAverageForecaster(window=2)
        forecaster.fit(y)
        forecast = forecaster.predict(fh=2)

        assert len(forecast.y_pred) == 2

    def test_window_larger_than_data(self):
        """Test behavior when window is larger than data."""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        y = pd.Series(np.random.randn(5).cumsum(), index=dates)

        forecaster = SimpleMovingAverageForecaster(window=10)
        forecaster.fit(y)
        forecast = forecaster.predict(fh=2)

        # When window > data length, rolling mean may return NaN
        # This is expected behavior - use smaller window or more data
        assert len(forecast.y_pred) == 2
        # Check that we get a result (may be NaN if insufficient data)
        assert forecast.y_pred is not None

