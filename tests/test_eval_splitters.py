"""Tests for evaluation splitters."""

import pytest
import pandas as pd

from timesmith.eval.splitters import ExpandingWindowSplit, SlidingWindowSplit


class TestExpandingWindowSplit:
    """Tests for ExpandingWindowSplit."""

    def test_basic_split(self):
        """Test basic expanding window split."""
        y = pd.Series(range(10), index=pd.date_range("2020-01-01", periods=10))
        splitter = ExpandingWindowSplit(initial_window=3, step_size=1, fh=2)

        splits = list(splitter.split(y))
        assert len(splits) > 0

        # Check first split
        train_idx, test_idx, cutoff = splits[0]
        assert cutoff == 3
        assert len(y.iloc[train_idx]) == 3
        assert len(y.iloc[test_idx]) == 2

    def test_insufficient_data(self):
        """Test splitter raises error with insufficient data."""
        y = pd.Series(range(3))
        splitter = ExpandingWindowSplit(initial_window=3, fh=2)

        with pytest.raises(ValueError, match="must be >="):
            list(splitter.split(y))


class TestSlidingWindowSplit:
    """Tests for SlidingWindowSplit."""

    def test_basic_split(self):
        """Test basic sliding window split."""
        y = pd.Series(range(10), index=pd.date_range("2020-01-01", periods=10))
        splitter = SlidingWindowSplit(window_size=3, step_size=1, fh=2)

        splits = list(splitter.split(y))
        assert len(splits) > 0

        # Check first split
        train_idx, test_idx, cutoff = splits[0]
        assert len(y.iloc[train_idx]) == 3
        assert len(y.iloc[test_idx]) == 2

