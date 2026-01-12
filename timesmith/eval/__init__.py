"""Evaluation tools: splitters, metrics, backtests, and summaries."""

from timesmith.eval.backtest import backtest_forecaster
from timesmith.eval.comparison import ModelComparison, ModelResult, compare_models
from timesmith.eval.metrics import bias, mae, mape, r2_score, rmse, smape, ubrmse
from timesmith.eval.splitters import ExpandingWindowSplit, SlidingWindowSplit
from timesmith.eval.summary import summarize_backtest

__all__ = [
    "ExpandingWindowSplit",
    "SlidingWindowSplit",
    "mae",
    "rmse",
    "mape",
    "smape",
    "bias",
    "ubrmse",
    "r2_score",
    "backtest_forecaster",
    "summarize_backtest",
    "ModelComparison",
    "ModelResult",
    "compare_models",
]
