"""Task objects that bind data, horizon, and target semantics."""

from timesmith.tasks.forecast import ForecastTask
from timesmith.tasks.detect import DetectTask

__all__ = [
    "ForecastTask",
    "DetectTask",
]

