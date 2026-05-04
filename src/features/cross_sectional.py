"""Cross-sectional features computed within each date."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)



def _cross_sectional_zscore(
    df: pd.DataFrame,
    col: str,
    new_col: str,
    clip_val: float = 5.0,
) -> pd.DataFrame:
    """Add a per-date cross-sectional z-score column."""
    mean_cs = df.groupby(level="date")[col].transform("mean")
    std_cs = df.groupby(level="date")[col].transform("std").replace(0, np.nan)
    df[new_col] = ((df[col] - mean_cs) / std_cs).clip(-clip_val, clip_val)
    return df



def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional transforms and market-breadth context."""
    log.info("Cross-sectional features ...")
    df = df.copy()

    n_tickers = df.index.get_level_values("ticker").nunique()
    log.info("  Universe: %s tickers", n_tickers)

    if "ret_1d" in df.columns:
        df = _cross_sectional_zscore(df, "ret_1d", "cs_ret_zscore_1d")
    else:
        log.warning("  Missing ret_1d -> skip cs_ret_zscore_1d")

    if "vol_21d" in df.columns:
        df = _cross_sectional_zscore(df, "vol_21d", "cs_vol_zscore_21d")
    else:
        log.warning("  Missing vol_21d -> skip cs_vol_zscore_21d")

    if "mom_63d" in df.columns:
        df = _cross_sectional_zscore(df, "mom_63d", "cs_mom_zscore_63d")
    else:
        log.warning("  Missing mom_63d -> skip cs_mom_zscore_63d")

    if "adj_open" in df.columns and "adj_close" in df.columns:
        prev_close = df.groupby(level="ticker")["adj_close"].shift(1)
        ratio = df["adj_open"] / prev_close.replace(0, np.nan)
        df["overnight_gap"] = np.log(ratio.where(ratio > 0))
    else:
        log.warning("  Missing adj_open/adj_close -> skip overnight_gap")

    if all(c in df.columns for c in ["adj_close", "adj_high", "adj_low"]):
        hl_range = df["adj_high"] - df["adj_low"]
        df["close_position"] = (
            (df["adj_close"] - df["adj_low"]) / hl_range.replace(0, np.nan)
        )
    else:
        log.warning("  Missing adj_high/adj_low -> skip close_position")

    if "price_sma200" in df.columns:
        # Bug #10 NOTE (2026-04-27): briefly masked NaN before the comparison
        # so warmup days produce NaN breadth instead of 0.0. The change
        # perturbed early-fold breadth values and cascaded through the
        # adaptive ensemble. Reverted to v1 behaviour (NaN > 0 = False
        # during the SMA-200 warmup). See docs/notes/post_audit_fixes.md §10.
        breadth = (df["price_sma200"] > 0).groupby(level="date").mean().astype(float)
        df["market_breadth_200d"] = df.index.get_level_values("date").map(breadth.to_dict())
    else:
        log.warning("  Missing price_sma200 -> skip market_breadth_200d")

    added = [
        c for c in [
            "cs_ret_zscore_1d",
            "cs_vol_zscore_21d",
            "cs_mom_zscore_63d",
            "overnight_gap",
            "close_position",
            "market_breadth_200d",
        ]
        if c in df.columns
    ]
    log.info("  -> %s cross-sectional features", len(added))
    return df
