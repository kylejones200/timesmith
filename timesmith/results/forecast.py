"""Forecast result dataclass."""

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass
class Forecast:
    """Forecast result with predictions and optional intervals.

    Attributes:
        y_pred: Predicted values (Series or array-like).
        fh: Forecast horizon (can be integer, array, or other format).
        y_int: Optional prediction intervals (DataFrame with lower/upper columns).
        metadata: Optional metadata dictionary.
    """

    y_pred: Any
    fh: Any
    y_int: Optional[Any] = None
    metadata: Optional[dict] = None

    def __post_init__(self):
        """Validate forecast structure."""
        if self.y_pred is None:
            raise ValueError("y_pred cannot be None")

        if self.fh is None:
            raise ValueError("fh cannot be None")

