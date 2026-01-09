"""Backtest result dataclass."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class BacktestResult:
    """Backtest result with fold information and metrics.

    Attributes:
        results: DataFrame with columns: fold_id, cutoff, fh, y_true, y_pred, metrics.
        summary: Dictionary with aggregate metrics.
        per_fold_metrics: Optional DataFrame with per-fold metrics.
    """

    results: pd.DataFrame
    summary: Dict[str, float]
    per_fold_metrics: Optional[pd.DataFrame] = None

    def __post_init__(self):
        """Validate backtest result structure."""
        required_cols = ["fold_id", "cutoff", "fh", "y_true", "y_pred"]
        if not all(col in self.results.columns for col in required_cols):
            raise ValueError(
                f"results DataFrame must have columns: {required_cols}. "
                f"Got: {list(self.results.columns)}"
            )

