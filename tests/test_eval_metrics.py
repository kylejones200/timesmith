"""Tests for evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from timesmith.eval.metrics import mae, mape, rmse


class TestMetrics:
    """Tests for forecast metrics."""

    def test_mae(self):
        """Test mean absolute error."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.5, 2.5, 2.5, 4.5, 5.5]
        result = mae(y_true, y_pred)
        # MAE = mean(|1-1.5|, |2-2.5|, |3-2.5|, |4-4.5|, |5-5.5|)
        #      = mean(0.5, 0.5, 0.5, 0.5, 0.5) = 0.5
        assert result == 0.5

    def test_rmse(self):
        """Test root mean squared error."""
        y_true = [1, 2, 3]
        y_pred = [1.5, 2.5, 2.5]
        result = rmse(y_true, y_pred)
        expected = np.sqrt(np.mean([0.5**2, 0.5**2, 0.5**2]))
        assert abs(result - expected) < 1e-10

    def test_mape_with_zeros(self):
        """Test MAPE handles zeros safely."""
        y_true = [0, 2, 3]
        y_pred = [0.5, 2.5, 3.5]
        result = mape(y_true, y_pred)
        # Should compute MAPE only for non-zero values
        assert not np.isnan(result)

    def test_mae_with_series(self):
        """Test MAE with pandas Series."""
        y_true = pd.Series([1, 2, 3])
        y_pred = pd.Series([1.5, 2.5, 2.5])
        result = mae(y_true, y_pred)
        # MAE = mean(|1-1.5|, |2-2.5|, |3-2.5|) = mean(0.5, 0.5, 0.5) = 0.5
        assert result == pytest.approx(0.5, abs=0.01)
