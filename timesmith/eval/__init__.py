"""Evaluation tools: splitters, metrics, backtests, and summaries."""

from timesmith.eval.splitters import ExpandingWindowSplit, SlidingWindowSplit
from timesmith.eval.metrics import mae, mape, rmse
from timesmith.eval.backtest import backtest_forecaster
from timesmith.eval.summary import summarize_backtest

__all__ = [
    "ExpandingWindowSplit",
    "SlidingWindowSplit",
    "mae",
    "rmse",
    "mape",
    "backtest_forecaster",
    "summarize_backtest",
]

