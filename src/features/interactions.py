"""Interaction features built from existing signals."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)



def _safe_interaction(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    new_col: str,
    transform_b: str = "identity",
) -> pd.DataFrame:
    """Multiply two columns when both are available."""
    if col_a not in df.columns:
        log.info("  %s: skip (%s not found)", new_col, col_a)
        return df
    if col_b not in df.columns:
        log.info("  %s: skip (%s not found)", new_col, col_b)
        return df

    a = df[col_a]
    b = df[col_b]
    if transform_b == "abs":
        b = b.abs()

    df[new_col] = a * b
    return df



def _safe_ratio(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    new_col: str,
) -> pd.DataFrame:
    """Add a ratio column with zero-safe division."""
    if numerator not in df.columns:
        log.info("  %s: skip (%s not found)", new_col, numerator)
        return df
    if denominator not in df.columns:
        log.info("  %s: skip (%s not found)", new_col, denominator)
        return df

    denom = df[denominator].replace(0, np.nan)
    df[new_col] = df[numerator] / denom
    return df



def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features that combine stock-level and market context."""
    log.info("Interaction features ...")
    df = df.copy()

    df = _safe_interaction(df, "mom_63d", "adx_14d", "mom_quality")
    df = _safe_interaction(df, "ret_5d", "vxn_zscore", "stress_reversal")
    df = _safe_interaction(df, "rolling_beta_63d", "p_high_vol", "beta_regime")
    df = _safe_interaction(df, "zspread", "mom_63d", "spread_momentum")
    df = _safe_ratio(df, "vol_21d", "market_dispersion_21d", "dispersion_adjusted_vol")
    df = _safe_interaction(df, "cs_mom_zscore_63d", "yield_spread_zscore", "cs_mom_x_yield")

    added = [
        c for c in [
            "mom_quality",
            "stress_reversal",
            "beta_regime",
            "spread_momentum",
            "dispersion_adjusted_vol",
            "cs_mom_x_yield",
        ]
        if c in df.columns
    ]
    log.info("  -> %s interaction features", len(added))
    return df
