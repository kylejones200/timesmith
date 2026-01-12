"""Task objects that bind data, horizon, and target semantics."""

from timesmith.tasks.detect import DetectTask
from timesmith.tasks.forecast import ForecastTask

__all__ = [
    "ForecastTask",
    "DetectTask",
]
