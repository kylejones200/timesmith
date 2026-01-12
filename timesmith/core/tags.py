"""Tag system for time series estimators."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_tags(obj: Any, tag: Optional[str] = None) -> Dict[str, Any]:
    """Get tags from an object.

    Args:
        obj: Object to get tags from.
        tag: Optional specific tag name to retrieve. If None, returns all tags.

    Returns:
        Dictionary of tags, or single tag value if tag name specified.
    """
    if not hasattr(obj, "_tags"):
        return {}

    tags = obj._tags
    if tag is not None:
        return tags.get(tag, None)

    return tags.copy()


def set_tags(obj: Any, **tags: Any) -> None:
    """Set tags on an object.

    Args:
        obj: Object to set tags on.
        **tags: Tag names and values to set.
    """
    if not hasattr(obj, "_tags"):
        obj._tags = {}

    obj._tags.update(tags)


# Common tag names
SCITYPE_INPUT = "scitype_input"  # SeriesLike, PanelLike, TableLike
SCITYPE_OUTPUT = "scitype_output"  # SeriesLike, PanelLike, TableLike
HANDLES_MISSING = "handles_missing"  # bool
REQUIRES_SORTED_INDEX = "requires_sorted_index"  # bool
SUPPORTS_PANEL = "supports_panel"  # bool
REQUIRES_FH = "requires_fh"  # bool
