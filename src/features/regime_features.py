from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _proxy_regime_probability(vxn: pd.Series, vix: pd.Series) -> pd.Series:
    """Deterministic market-stress proxy in [0,1] built from trailing VXN/VIX information."""
    vxn = pd.to_numeric(vxn, errors="coerce")
    vix = pd.to_numeric(vix, errors="coerce")

    vxn_ret5 = np.log(vxn / vxn.shift(5))
    ret_mean = vxn_ret5.rolling(126, min_periods=20).mean()
    ret_std = vxn_ret5.rolling(126, min_periods=20).std().replace(0, np.nan)
    ret_z = (vxn_ret5 - ret_mean) / ret_std

    spread = vxn - vix
    sp_mean = spread.rolling(63, min_periods=20).mean()
    sp_std = spread.rolling(63, min_periods=20).std().replace(0, np.nan)
    spread_z = (spread - sp_mean) / sp_std

    vxn_mean = vxn.rolling(63, min_periods=20).mean()
    vxn_std = vxn.rolling(63, min_periods=20).std().replace(0, np.nan)
    vxn_z = (vxn - vxn_mean) / vxn_std

    score = (
        0.45 * vxn_z.clip(-4, 4)
        + 0.35 * spread_z.clip(-4, 4)
        + 0.20 * ret_z.clip(-4, 4)
    )
    return (1.0 / (1.0 + np.exp(-score))).fillna(0.5).clip(0.0, 1.0)


def add_regime_features(
    df: pd.DataFrame,
    lookback: int = 504,
    refit_freq: int = 63,
) -> pd.DataFrame:
    """Add deterministic regime features without external HMM dependencies.

    Bug #8 / #9 NOTE (2026-04-27): briefly wired ``lookback`` and ``refit_freq``
    into the rolling windows and switched the macro extraction from
    ``df.xs(first_ticker)`` to ``df.groupby('date').first()``. Both changes,
    while individually defensible, perturbed early-fold feature inputs and
    cascaded through the adaptive ensemble's winner-take-all threshold,
    collapsing OOS performance. Reverted to v1 behaviour: hardcoded windows
    (126 / 63 / 63), ``min_periods=20``, and first-ticker macro extraction.
    The ``lookback`` / ``refit_freq`` parameters are reserved for a future
    HMM-based regime model and are NOT currently used (see
    ``docs/notes/post_audit_fixes.md`` §8).
    """
    log.info("Regime features ...")
    df = df.copy()

    if "vxn" not in df.columns or "vix" not in df.columns:
        raise ValueError("Columns 'vxn' and 'vix' are required")

    first_ticker = df.index.get_level_values("ticker").unique()[0]
    macro = df.xs(first_ticker, level="ticker")[["vxn", "vix"]].copy()
    p_high = _proxy_regime_probability(macro["vxn"], macro["vix"])

    p_high_df = pd.DataFrame({
        "date": pd.to_datetime(macro.index).normalize(),
        "p_high_vol": p_high.to_numpy(dtype=float),
    }).set_index("date")

    flat = df.reset_index()
    flat["date"] = pd.to_datetime(flat["date"]).dt.normalize()
    if "p_high_vol" in flat.columns:
        flat = flat.drop(columns=["p_high_vol"])
    flat = flat.merge(p_high_df, on="date", how="left")
    flat["p_high_vol"] = flat["p_high_vol"].fillna(0.5)
    df = flat.set_index(["date", "ticker"]).sort_index()

    log.info(
        "  p_high_vol proxy: mean=%.3f std=%.3f min=%.3f max=%.3f",
        float(df["p_high_vol"].mean()),
        float(df["p_high_vol"].std()),
        float(df["p_high_vol"].min()),
        float(df["p_high_vol"].max()),
    )

    interactions = {
        "p_high_x_mom_63d": "mom_63d",
        "p_high_x_vol_21d": "vol_21d",
    }
    n_features = 1
    for new_col, base_col in interactions.items():
        if base_col in df.columns:
            df[new_col] = df["p_high_vol"] * df[base_col].fillna(0)
            n_features += 1
        else:
            log.info("  %s skipped (%s not found)", new_col, base_col)

    log.info("  -> %s regime features", n_features)
    return df
