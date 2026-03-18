"""Walk-forward cross-validation with purge and embargo periods."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


@dataclass
class WalkForwardFold:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    fold_index: int

    @property
    def purge_start(self) -> datetime:
        return self.train_end

    @property
    def embargo_end(self) -> datetime:
        return self.test_start

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_index}: "
            f"Train[{self.train_start:%Y-%m-%d} → {self.train_end:%Y-%m-%d}] "
            f"Test[{self.test_start:%Y-%m-%d} → {self.test_end:%Y-%m-%d}]"
        )


class WalkForwardSplitter:
    """
    Walk-forward validation with purge gap and embargo period.

    Layout per fold:
    [------- train_window -------][purge][embargo][--- test_window ---]

    - purge: prevents label leakage (labels that look ahead from train into test)
    - embargo: additional safety gap at start of test period
    - window slides by step_days each iteration
    """

    def __init__(
        self,
        train_window_days: int = 756,
        test_window_days: int = 63,
        purge_days: int = 5,
        embargo_days: int = 2,
        step_days: Optional[int] = None,
    ):
        self.train_window = timedelta(days=train_window_days)
        self.test_window = timedelta(days=test_window_days)
        self.purge = timedelta(days=purge_days)
        self.embargo = timedelta(days=embargo_days)
        self.step = timedelta(days=step_days or test_window_days)

    def split(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[WalkForwardFold]:
        folds = []
        fold_idx = 0
        train_start = start_date

        while True:
            train_end = train_start + self.train_window
            test_start = train_end + self.purge + self.embargo
            test_end = test_start + self.test_window

            if test_end > end_date:
                break

            folds.append(WalkForwardFold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                fold_index=fold_idx,
            ))

            train_start += self.step
            fold_idx += 1

        return folds

    def split_dataframe(
        self,
        df: pd.DataFrame,
        date_column: Optional[str] = None,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, WalkForwardFold]]:
        """
        Split a DataFrame into (train, test, fold) tuples.
        Uses the DataFrame's index if date_column is None.
        """
        if date_column:
            dates = pd.to_datetime(df[date_column])
        else:
            dates = pd.to_datetime(df.index)

        start_date = dates.min().to_pydatetime()
        end_date = dates.max().to_pydatetime()

        folds = self.split(start_date, end_date)
        result = []

        for fold in folds:
            if date_column:
                train_mask = (dates >= fold.train_start) & (dates < fold.train_end)
                test_mask = (dates >= fold.test_start) & (dates < fold.test_end)
            else:
                train_mask = (dates >= fold.train_start) & (dates < fold.train_end)
                test_mask = (dates >= fold.test_start) & (dates < fold.test_end)

            train_df = df[train_mask]
            test_df = df[test_mask]

            if not train_df.empty and not test_df.empty:
                result.append((train_df, test_df, fold))

        return result

    def get_fold_count(self, start_date: datetime, end_date: datetime) -> int:
        return len(self.split(start_date, end_date))
