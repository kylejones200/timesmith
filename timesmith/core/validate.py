"""Input validation for time series estimators.

Validation should happen once at public API boundaries only.
Do not validate inside inner loops.
"""

import logging
from typing import Any, Optional

from timesmith.exceptions import ValidationError
from timesmith.typing.validators import assert_panel, assert_series, assert_table

logger = logging.getLogger(__name__)


def validate_input(
    data: Any,
    scitype: str,
    name: str = "data",
    allow_none: bool = False,
) -> None:
    """Validate input data matches expected scitype.

    Args:
        data: Data to validate.
        scitype: Expected scitype ("SeriesLike", "PanelLike", or "TableLike").
        name: Name of the variable for error messages.
        allow_none: If True, None values are allowed.

    Raises:
        TypeError: If data doesn't match expected scitype.
        ValueError: If data is None and allow_none is False.
    """
    if data is None:
        if not allow_none:
            raise ValidationError(
                f"{name} cannot be None",
                context={"name": name, "scitype": scitype, "allow_none": allow_none},
            )
        return

    scitype = scitype.lower()
    try:
        if scitype == "serieslike":
            assert_series(data, name=name)
        elif scitype == "panellike":
            assert_panel(data, name=name)
        elif scitype == "tablelike":
            assert_table(data, name=name)
        else:
            raise ValidationError(
                f"Unknown scitype: {scitype}. Must be one of "
                "'SeriesLike', 'PanelLike', 'TableLike'",
                context={"name": name, "scitype": scitype},
            )
    except (TypeError, ValueError) as e:
        # Wrap validation errors from validators with context
        raise ValidationError(
            str(e),
            context={"name": name, "scitype": scitype, "original_error": type(e).__name__},
        ) from e

