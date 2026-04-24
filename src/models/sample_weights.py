"""
Sample weights for overlapping labels (López de Prado, AFML Ch 4).

avg_uniqueness(i) = mean over t in [t0_i, t1_i] of 1/concurrency(t),
where concurrency(t) = number of labels whose span covers t.

Computed per-ticker because cross-sectional samples on different tickers
do not overlap in the label-information sense.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def avg_uniqueness(df: pd.DataFrame, t1_col: str = "t1") -> pd.Series:
    """Return Series (aligned to df.index) of avg-uniqueness weights in (0, 1]."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if t1_col not in df.columns:
        return out

    for _, grp in df.groupby(level="ticker", sort=False):
        dates = grp.index.get_level_values("date").to_numpy()
        t1 = pd.to_datetime(grp[t1_col]).to_numpy()
        n = len(grp)
        if n == 0:
            continue

        uniq = np.sort(np.unique(dates))
        d2i = {d: i for i, d in enumerate(uniq)}
        m = len(uniq)
        conc = np.zeros(m, dtype=np.int32)

        spans = np.full((n, 2), -1, dtype=np.int64)
        for i in range(n):
            if pd.isna(t1[i]):
                continue
            s = d2i.get(dates[i], -1)
            if s < 0:
                continue
            e = d2i.get(t1[i], -1)
            if e < 0:
                e = int(np.searchsorted(uniq, t1[i], side="right")) - 1
            if e < s:
                e = s
            if e >= m:
                e = m - 1
            spans[i, 0] = s
            spans[i, 1] = e
            conc[s : e + 1] += 1

        w = np.full(n, np.nan)
        inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1), 0.0)
        for i in range(n):
            s, e = spans[i]
            if s < 0:
                continue
            w[i] = float(inv[s : e + 1].mean())

        out.loc[grp.index] = w

    return out
