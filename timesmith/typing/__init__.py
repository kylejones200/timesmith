"""Type definitions and validators for time series data structures.

This is the single source of truth for SeriesLike, PanelLike, and validators.
Downstream repos must import from here:

    from timesmith.typing import SeriesLike, PanelLike
    from timesmith.typing.validators import assert_series_like, assert_panel_like
"""

from timesmith.typing.protocols import (
    ForecastLike,
    PanelLike,
    SeriesLike,
    TableLike,
)
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

__all__ = [
    "SeriesLike",
    "PanelLike",
    "TableLike",
    "ForecastLike",
    "is_series",
    "is_panel",
    "is_table",
    "assert_series",
    "assert_series_like",
    "assert_panel",
    "assert_panel_like",
    "assert_table",
]
