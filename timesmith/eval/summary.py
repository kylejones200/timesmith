"""Summary functions for backtest results."""

import logging
from typing import Dict

from timesmith.results.backtest import BacktestResult

logger = logging.getLogger(__name__)


def summarize_backtest(result: BacktestResult) -> Dict:
    """Summarize backtest results with aggregate and per-fold metrics.

    Args:
        result: BacktestResult from backtest_forecaster.

    Returns:
        Dictionary with aggregate metrics and per-fold metrics DataFrame.
    """
    summary = {
        "aggregate_metrics": result.summary.copy(),
        "n_folds": len(result.results),
    }

    if result.per_fold_metrics is not None:
        summary["per_fold_metrics"] = result.per_fold_metrics.copy()

    return summary
