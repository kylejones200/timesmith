"""Type definitions and validators for time series data structures."""

from timesmith.typing.protocols import (
    ForecastLike,
    PanelLike,
    SeriesLike,
    TableLike,
)
from timesmith.typing.validators import (
    assert_panel,
    assert_series,
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
    "assert_panel",
    "assert_table",
]

