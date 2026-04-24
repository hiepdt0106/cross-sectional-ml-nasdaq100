"""
src/features/volatility.py
──────────────────────────
Group 2 — Intrinsic volatility (Chapter 3.5.2)

Features (10 columns):
─────────────────────────────────────────────────────────
  Realized vol:    vol_5d, vol_10d, vol_21d, vol_63d
  ATR:             atr_14d, atr_norm_14d
  Garman-Klass:    gk_vol_20d
  Vol dynamics:    vol_change_21d, vol_skew_21d
  Return shape:    skew_21d
─────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils import log_return as _log_return

log = logging.getLogger(__name__)


def _garman_klass_vol(high, low, close, open_, window):
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return gk.rolling(window, min_periods=max(window // 2, 5)).mean().apply(
        lambda x: np.sqrt(max(x, 0)))


def _atr(high, low, close, window):
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window).mean()


def add_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """10 intrinsic-volatility features."""
    log.info("Group 2 — Intrinsic volatility ...")
    df = df.copy()

    def _per_ticker(grp):
        c = grp["adj_close"]
        h, l, o = grp["adj_high"], grp["adj_low"], grp["adj_open"]

        if "ret_1d" not in grp.columns:
            grp["ret_1d"] = _log_return(c, 1)
        ret = grp["ret_1d"]

        # ── Realized vol multi-timeframe (4) ──
        for n in [5, 10, 21, 63]:
            grp[f"vol_{n}d"] = ret.rolling(n, min_periods=max(n // 2, 3)).std()

        # ── ATR (1) ── atr_14d pruned; keep only normalized ──
        atr_raw = _atr(h, l, c, window=14)
        grp["atr_norm_14d"] = atr_raw / c.replace(0, np.nan)

        # ── Garman-Klass (1) ──
        grp["gk_vol_20d"] = _garman_klass_vol(h, l, c, o, window=20)

        # ── Vol dynamics (1) — vol_change_21d pruned as redundant ──
        # Vol skew: neg_vol / pos_vol
        # min_periods=3: only need 3 up/down days within the 21-day window
        neg_vol = ret.where(ret < 0).rolling(21, min_periods=3).std()
        pos_vol = ret.where(ret > 0).rolling(21, min_periods=3).std()
        grp["vol_skew_21d"] = neg_vol / pos_vol.replace(0, np.nan)

        # ── Return shape (1) ──
        grp["skew_21d"] = ret.rolling(21, min_periods=15).skew()

        return grp

    df = df.groupby(level="ticker", group_keys=False).apply(_per_ticker)

    n_added = len([c for c in df.columns if c.startswith((
        "vol_5", "vol_10", "vol_21", "vol_63", "vol_change_21", "vol_skew",
        "atr_", "gk_", "skew_",
    ))])
    log.info(f"  → {n_added} features")
    return df