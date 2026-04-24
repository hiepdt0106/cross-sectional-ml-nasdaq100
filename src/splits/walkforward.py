"""
src/splits/walkforward.py
─────────────────────────
Purged Walk-Forward by Year — López de Prado (2018)

  Fold 1: train 2014→2019 (purge H tail) | embargo H | test 2020
  Fold 2: train 2014→2020 (purge H tail) | embargo H | test 2021
  ...

Two-sided anti-leakage:
  1. PURGE train tail: drop the last H days of training because their
     labels look into the test period (t1 > train_end).
  2. EMBARGO test start: drop the first H days of test because the
     labels of late-train samples may overlap into the test window.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class FoldSplit:
    """Result of one walk-forward fold."""
    fold:       int
    test_year:  int
    train_idx:  pd.MultiIndex
    test_idx:   pd.MultiIndex
    train_end:  pd.Timestamp
    purge_n:    int
    embargo_n:  int

    @property
    def train_size(self) -> int:
        return len(self.train_idx)

    @property
    def test_size(self) -> int:
        return len(self.test_idx)

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold}: train→{self.train_end.date()} "
            f"({self.train_size:,}) | purge {self.purge_n} + embargo {self.embargo_n} | "
            f"test {self.test_year} ({self.test_size:,})"
        )


def make_expanding_splits(
    df: pd.DataFrame,
    first_test_year: int = 2020,
    horizon: int = 10,
    max_train_years: int | None = None,
) -> list[FoldSplit]:
    """
    Purged walk-forward by year — López de Prado (2018).

    Expanding (default): training uses ALL data before the test year.
    Sliding (max_train_years > 0): uses only the last N years.

    Anti-leakage:
      - Purge: drop train samples whose t1 > test_start (labels look
        into the test window). If t1 is absent, fall back to cutting
        the last H training days.
      - Embargo: drop the first H test days (labels of late-train
        samples may overlap there).
    """
    dates = df.index.get_level_values("date")
    years = sorted(dates.year.unique())
    test_years = [y for y in years if y >= first_test_year]

    has_t1 = "t1" in df.columns

    splits: list[FoldSplit] = []

    for i, test_year in enumerate(test_years):
        if max_train_years is not None:
            min_train_year = test_year - max_train_years
            train_mask = (dates.year < test_year) & (dates.year >= min_train_year)
        else:
            train_mask = dates.year < test_year

        test_mask = dates.year == test_year

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        train_all = df.index[train_mask]
        test_all = df.index[test_mask]

        # ── Train end + Test start dates ──
        train_end = df.loc[train_all].index.get_level_values("date").max()
        test_dates = df.loc[test_all].index.get_level_values("date").unique().sort_values()

        if len(test_dates) <= horizon:
            continue

        test_start = test_dates[0]

        # ── PURGE train: drop samples whose labels look into test ──
        if has_t1:
            # Conservative purge: drop train samples with t1 >= test_start.
            # A sample with t1 == test_start is still using that day's data
            # to determine its label → must be purged.
            train_df = df.loc[train_all]
            t1_values = train_df["t1"]
            purge_mask = t1_values.notna() & (t1_values >= test_start)
            train_idx = train_all[~purge_mask.values]
            purge_n = int(purge_mask.sum())
        else:
            # Fallback: cut the last H training days (conservative proxy)
            log.warning(f"  Fold {i+1}: no t1 column → fallback purge of last H={horizon} days")
            train_dates = df.loc[train_all].index.get_level_values("date").unique().sort_values()
            if len(train_dates) <= horizon:
                continue
            purge_dates = train_dates[-horizon:]
            purge_mask_idx = df.loc[train_all].index.get_level_values("date").isin(purge_dates)
            train_idx = train_all[~purge_mask_idx]
            purge_n = int(purge_mask_idx.sum())

        # ── EMBARGO: drop the first H test days ──
        embargo_dates = test_dates[:horizon]
        embargo_mask = df.loc[test_all].index.get_level_values("date").isin(embargo_dates)
        test_idx = test_all[~embargo_mask]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        fold = FoldSplit(
            fold=i + 1,
            test_year=test_year,
            train_idx=train_idx,
            test_idx=test_idx,
            train_end=train_end,
            purge_n=purge_n,
            embargo_n=int(embargo_mask.sum()),
        )
        splits.append(fold)
        log.info(f"  {fold}")

    mode = f"sliding {max_train_years}y" if max_train_years else "expanding"
    log.info(f"Total: {len(splits)} folds ({mode}), purge={'t1-based' if has_t1 else 'H-day proxy'}")
    return splits
