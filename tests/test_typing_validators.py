"""Tests for typing validators."""

import pandas as pd
import pytest

from timesmith.typing.validators import (
    assert_panel,
    assert_panel_like,
    assert_series,
    assert_series_like,
    assert_table,
    is_panel,
    is_series,
    is_table,
)


class TestSeriesValidators:
    """Tests for SeriesLike validators."""

    def test_is_series_with_series(self):
        """Test is_series with pandas Series."""
        s = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        assert is_series(s) is True

    def test_is_series_with_single_column_dataframe(self):
        """Test is_series with single-column DataFrame."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3)
        )
        assert is_series(df) is True

    def test_is_series_with_multi_column_dataframe(self):
        """Test is_series with multi-column DataFrame."""
        df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            index=pd.date_range("2020-01-01", periods=3),
        )
        assert is_series(df) is False

    def test_is_series_with_integer_index(self):
        """Test is_series with integer index."""
        s = pd.Series([1, 2, 3], index=[0, 1, 2])
        assert is_series(s) is True

    def test_assert_series_valid(self):
        """Test assert_series with valid series."""
        s = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        assert_series(s)  # Should not raise

    def test_assert_series_invalid(self):
        """Test assert_series with invalid data."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(TypeError, match="must be SeriesLike"):
            assert_series(df)

    def test_assert_series_like_alias(self):
        """Test that assert_series_like is an alias for assert_series."""
        s = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        assert_series_like(s)  # Should not raise

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(TypeError, match="must be SeriesLike"):
            assert_series_like(df)


class TestPanelValidators:
    """Tests for PanelLike validators."""

    def test_is_panel_with_multiindex(self):
        """Test is_panel with MultiIndex DataFrame."""
        index = pd.MultiIndex.from_tuples(
            [("A", "2020-01-01"), ("A", "2020-01-02"), ("B", "2020-01-01")],
            names=["entity", "time"],
        )
        df = pd.DataFrame({"value": [1, 2, 3]}, index=index)
        assert is_panel(df) is True

    def test_is_panel_with_entity_column(self):
        """Test is_panel with entity column."""
        df = pd.DataFrame(
            {
                "entity": ["A", "A", "B"],
                "value": [1, 2, 3],
            },
            index=pd.date_range("2020-01-01", periods=3),
        )
        assert is_panel(df) is True

    def test_assert_panel_valid(self):
        """Test assert_panel with valid panel."""
        index = pd.MultiIndex.from_tuples(
            [("A", "2020-01-01"), ("B", "2020-01-02")], names=["entity", "time"]
        )
        df = pd.DataFrame({"value": [1, 2]}, index=index)
        assert_panel(df)  # Should not raise

    def test_assert_panel_like_alias(self):
        """Test that assert_panel_like is an alias for assert_panel."""
        index = pd.MultiIndex.from_tuples(
            [("A", "2020-01-01"), ("B", "2020-01-02")], names=["entity", "time"]
        )
        df = pd.DataFrame({"value": [1, 2]}, index=index)
        assert_panel_like(df)  # Should not raise


class TestTableValidators:
    """Tests for TableLike validators."""

    def test_is_table_with_dataframe(self):
        """Test is_table with DataFrame."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6]},
            index=pd.date_range("2020-01-01", periods=3),
        )
        assert is_table(df) is True

    def test_assert_table_valid(self):
        """Test assert_table with valid table."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3)
        )
        assert_table(df)  # Should not raise
