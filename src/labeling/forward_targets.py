"""Forward-return targets aligned with the rebalancing backtest."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def add_forward_rebalance_targets(
    df: pd.DataFrame,
    horizon: int = 10,
    top_k: int = 10,
    return_col: str = "alpha_ret",
    label_col: str = "alpha_label",
    extreme_label_col: str = "alpha_ext_label",
    extreme_frac: float = 0.20,
) -> pd.DataFrame:
    """Add fixed-horizon forward returns and aligned classification labels.

    The target matches the research backtest:
    signal formed on day ``t`` close, entry at ``t+1`` open, exit at ``t+h`` close.
    ``label_col`` marks the exact top-K names used by the portfolio.
    ``extreme_label_col`` keeps only the top / bottom tails of each day and drops the
    middle bucket, which is a cleaner classification target for weak cross-sectional
    models.
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not (0.0 < extreme_frac < 0.5):
        raise ValueError("extreme_frac must be in (0, 0.5)")
    if "adj_close" not in df.columns:
        raise ValueError("df must contain 'adj_close'")

    use_open = "adj_open" in df.columns
    out = df.copy()

    def _per_ticker(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_index().copy()
        close = grp["adj_close"].to_numpy(dtype=float)
        if use_open:
            open_ = grp["adj_open"].to_numpy(dtype=float)
        else:
            open_ = close.copy()
        n = len(grp)
        rets = np.full(n, np.nan, dtype=float)

        for i in range(n):
            entry_idx = i + 1
            exit_idx = i + horizon
            if entry_idx >= n or exit_idx >= n:
                continue
            p0 = open_[entry_idx]
            p1 = close[exit_idx]
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and p1 > 0:
                rets[i] = float(np.log(p1 / p0))

        grp[return_col] = rets
        return grp

    out = out.groupby(level="ticker", group_keys=False).apply(_per_ticker)

    def _label_one_day(x: pd.Series) -> pd.Series:
        label = pd.Series(np.nan, index=x.index, dtype=float)
        valid = x.dropna()
        if len(valid) <= top_k:
            return label
        winners = valid.nlargest(top_k)
        label.loc[valid.index] = 0.0
        label.loc[winners.index] = 1.0
        return label

    out[label_col] = (
        out[return_col]
        .groupby(level="date", group_keys=False)
        .apply(_label_one_day)
    )

    def _label_extremes_one_day(x: pd.Series) -> pd.Series:
        label = pd.Series(np.nan, index=x.index, dtype=float)
        valid = x.dropna().sort_values(kind="mergesort")
        n = len(valid)
        if n < 6:
            return label

        tail_n = int(np.floor(n * extreme_frac))
        tail_n = max(1, tail_n)
        if tail_n * 2 >= n:
            return label

        losers = valid.iloc[:tail_n]
        winners = valid.iloc[-tail_n:]
        label.loc[losers.index] = 0.0
        label.loc[winners.index] = 1.0
        return label

    out[extreme_label_col] = (
        out[return_col]
        .groupby(level="date", group_keys=False)
        .apply(_label_extremes_one_day)
    )

    valid_rows = int(out[return_col].notna().sum())
    valid_labels = out[label_col].dropna()
    valid_extreme = out[extreme_label_col].dropna()
    if len(valid_labels) > 0:
        pos_rate = float((valid_labels == 1).mean())
        ext_pos_rate = float((valid_extreme == 1).mean()) if len(valid_extreme) else float("nan")
        log.info(
            "Forward rebalance targets: %s valid returns | %s top-K labels | %s extreme labels | top-K pos_rate=%.2f%% | extreme pos_rate=%.2f%%",
            f"{valid_rows:,}",
            f"{len(valid_labels):,}",
            f"{len(valid_extreme):,}",
            pos_rate * 100,
            ext_pos_rate * 100 if np.isfinite(ext_pos_rate) else float("nan"),
        )
    else:
        log.warning("Forward rebalance targets produced no valid classification labels")

    return out
