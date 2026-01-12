"""Forecast task definition."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ForecastTask:
    """Forecast task that binds data, horizon, and target semantics.

    Tasks hold semantics. Estimators do not store global config beyond
    params and fitted state.

    Attributes:
        y: Target time series data.
        X: Optional exogenous/feature data.
        fh: Forecast horizon (can be integer, array, or other format).
        cutoff: Optional cutoff time (last time point used for training).
        frequency: Optional frequency string (e.g., 'D', 'H', 'M').
    """

    def __init__(
        self,
        y: Any,
        fh: Any,
        X: Optional[Any] = None,
        cutoff: Optional[Any] = None,
        frequency: Optional[str] = None,
    ):
        """Initialize forecast task.

        Args:
            y: Target time series data.
            fh: Forecast horizon.
            X: Optional exogenous/feature data.
            cutoff: Optional cutoff time.
            frequency: Optional frequency string.
        """
        self.y = y
        self.fh = fh
        self.X = X
        self.cutoff = cutoff
        self.frequency = frequency

    def __repr__(self) -> str:
        """String representation of the task."""
        return (
            f"ForecastTask(fh={self.fh}, cutoff={self.cutoff}, "
            f"frequency={self.frequency})"
        )
