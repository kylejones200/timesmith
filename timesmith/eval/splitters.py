"""Time series cross-validation splitters."""

import logging
from typing import Any, Iterator, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class ExpandingWindowSplit:
    """Expanding window cross-validation splitter.

    Each fold uses all data up to the cutoff point for training,
    and tests on the next window.

    Attributes:
        initial_window: Initial training window size.
        step_size: Step size between folds.
        fh: Forecast horizon for each fold.
    """

    def __init__(
        self,
        initial_window: int,
        step_size: int = 1,
        fh: int = 1,
    ):
        """Initialize expanding window splitter.

        Args:
            initial_window: Initial training window size.
            step_size: Step size between folds.
            fh: Forecast horizon for each fold.
        """
        self.initial_window = initial_window
        self.step_size = step_size
        self.fh = fh

    def split(self, y: Any) -> Iterator[Tuple[Any, Any, Any]]:
        """Generate train/test splits.

        Args:
            y: Time series data (Series or DataFrame with time index).

        Yields:
            Tuples of (train_indices, test_indices, cutoff).
        """
        if isinstance(y, pd.Series):
            n = len(y)
        elif isinstance(y, pd.DataFrame):
            n = len(y)
        else:
            raise TypeError(f"y must be Series or DataFrame, got {type(y).__name__}")

        if n < self.initial_window + self.fh:
            raise ValueError(
                f"Data length ({n}) must be >= initial_window ({self.initial_window}) "
                f"+ fh ({self.fh})"
            )

        cutoff = self.initial_window
        fold_id = 0

        while cutoff + self.fh <= n:
            train_end = cutoff
            test_start = cutoff
            test_end = min(cutoff + self.fh, n)

            train_indices = slice(0, train_end)
            test_indices = slice(test_start, test_end)

            logger.debug(
                f"Fold {fold_id}: train=[0:{train_end}], test=[{test_start}:{test_end}], "
                f"cutoff={cutoff}"
            )

            yield train_indices, test_indices, cutoff

            cutoff += self.step_size
            fold_id += 1


class SlidingWindowSplit:
    """Sliding window cross-validation splitter.

    Each fold uses a fixed-size window for training,
    and tests on the next window.

    Attributes:
        window_size: Training window size.
        step_size: Step size between folds.
        fh: Forecast horizon for each fold.
    """

    def __init__(
        self,
        window_size: int,
        step_size: int = 1,
        fh: int = 1,
    ):
        """Initialize sliding window splitter.

        Args:
            window_size: Training window size.
            step_size: Step size between folds.
            fh: Forecast horizon for each fold.
        """
        self.window_size = window_size
        self.step_size = step_size
        self.fh = fh

    def split(self, y: Any) -> Iterator[Tuple[Any, Any, Any]]:
        """Generate train/test splits.

        Args:
            y: Time series data (Series or DataFrame with time index).

        Yields:
            Tuples of (train_indices, test_indices, cutoff).
        """
        if isinstance(y, pd.Series):
            n = len(y)
        elif isinstance(y, pd.DataFrame):
            n = len(y)
        else:
            raise TypeError(f"y must be Series or DataFrame, got {type(y).__name__}")

        if n < self.window_size + self.fh:
            raise ValueError(
                f"Data length ({n}) must be >= window_size ({self.window_size}) "
                f"+ fh ({self.fh})"
            )

        train_start = 0
        fold_id = 0

        while train_start + self.window_size + self.fh <= n:
            train_end = train_start + self.window_size
            test_start = train_end
            test_end = min(test_start + self.fh, n)
            cutoff = train_end

            train_indices = slice(train_start, train_end)
            test_indices = slice(test_start, test_end)

            logger.debug(
                f"Fold {fold_id}: train=[{train_start}:{train_end}], "
                f"test=[{test_start}:{test_end}], cutoff={cutoff}"
            )

            yield train_indices, test_indices, cutoff

            train_start += self.step_size
            fold_id += 1

