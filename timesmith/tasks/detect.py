"""Anomaly detection task definition."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DetectTask:
    """Anomaly detection task that binds data and target semantics.

    Attributes:
        y: Target time series data.
        X: Optional exogenous/feature data.
        labels: Optional ground truth anomaly labels.
        horizon: Optional detection horizon.
    """

    def __init__(
        self,
        y: Any,
        X: Optional[Any] = None,
        labels: Optional[Any] = None,
        horizon: Optional[Any] = None,
    ):
        """Initialize detection task.

        Args:
            y: Target time series data.
            X: Optional exogenous/feature data.
            labels: Optional ground truth anomaly labels.
            horizon: Optional detection horizon.
        """
        self.y = y
        self.X = X
        self.labels = labels
        self.horizon = horizon

    def __repr__(self) -> str:
        """String representation of the task."""
        return f"DetectTask(horizon={self.horizon})"

