from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils import log_return as _log_return

log = logging.getLogger(__name__)


def _looks_like_treasury_proxy(ten: pd.Series, two: pd.Series) -> bool:
    """Detect synthetic 2Y proxy series such as us2y = us10y - 1.0."""
    valid = pd.concat([ten.rename("ten"), two.rename("two")], axis=1).dropna()
    if len(valid) < 60:
        return False
    spread = valid["ten"] - valid["two"]
    spread_std = float(spread.std())
    move_corr = float(valid["ten"].corr(valid["two"])) if len(valid) > 1 else np.nan
    return bool(np.isfinite(spread_std) and spread_std < 1e-4 and np.isfinite(move_corr) and move_corr > 0.999)


def add_macro_features(
    df: pd.DataFrame,
    vxn_window: int = 252,
) -> pd.DataFrame:
    """Add market-volatility and yield-curve features."""
    log.info("Macro features ...")
    df = df.copy()

    if "vix" not in df.columns or "vxn" not in df.columns:
        raise ValueError("Columns 'vix' and 'vxn' are required")

    first_ticker = df.index.get_level_values("ticker").unique()[0]
    macro = df.xs(first_ticker, level="ticker")[["vix", "vxn"]].copy()

    macro["vix_ret_1d"] = _log_return(macro["vix"], 1)
    macro["vxn_ret_1d"] = _log_return(macro["vxn"], 1)
    macro["vxn_ret_5d"] = _log_return(macro["vxn"], 5)

    vxn_mean = macro["vxn"].rolling(vxn_window, min_periods=60).mean()
    vxn_std = macro["vxn"].rolling(vxn_window, min_periods=60).std().replace(0, np.nan)
    macro["vxn_zscore"] = (macro["vxn"] - vxn_mean) / vxn_std

    vxn_ma5 = macro["vxn"].rolling(5, min_periods=3).mean()
    vxn_ma21 = macro["vxn"].rolling(21, min_periods=10).mean()
    macro["vxn_ma5_ma21"] = vxn_ma5 / vxn_ma21.replace(0, np.nan) - 1
    macro["vix_vxn_spread"] = macro["vxn"] - macro["vix"]

    has_treasury = "treasury_10y" in df.columns and "treasury_2y" in df.columns
    if has_treasury:
        log.info("Yield-curve features ...")
        treasury = df.xs(first_ticker, level="ticker")[["treasury_10y", "treasury_2y"]].copy()
        if _looks_like_treasury_proxy(treasury["treasury_10y"], treasury["treasury_2y"]):
            log.warning(
                "Detected synthetic / broken 2Y Treasury series (spread nearly constant). Skipping yield-curve features to avoid duplicate noise."
            )
        else:
            macro["yield_spread_10y2y"] = treasury["treasury_10y"] - treasury["treasury_2y"]
            macro["yield_spread_change_5d"] = macro["yield_spread_10y2y"].diff(5)
            ys_mean = macro["yield_spread_10y2y"].rolling(252, min_periods=60).mean()
            ys_std = macro["yield_spread_10y2y"].rolling(252, min_periods=60).std().replace(0, np.nan)
            macro["yield_spread_zscore"] = ((macro["yield_spread_10y2y"] - ys_mean) / ys_std).clip(-5, 5)
    else:
        log.info("  treasury_10y/treasury_2y not available -> skipping yield-curve features")

    macro_features = [c for c in macro.columns if c not in ["vix", "vxn"]]
    panel_flat = df.reset_index()
    overlap = [c for c in macro_features if c in panel_flat.columns]
    if overlap:
        panel_flat = panel_flat.drop(columns=overlap)
    panel_flat = panel_flat.merge(macro[macro_features].reset_index(), on="date", how="left")
    df = panel_flat.set_index(["date", "ticker"]).sort_index()

    if "vol_21d" in df.columns:
        vxn_daily = df["vxn"] / (100 * np.sqrt(252))
        df["zspread"] = df["vol_21d"] - vxn_daily
        df["zspread_ma5"] = df.groupby(level="ticker")["zspread"].transform(lambda x: x.rolling(5, min_periods=3).mean())
        df["zspread_change_5d"] = df.groupby(level="ticker")["zspread"].transform(lambda x: x - x.shift(5))
    else:
        log.warning("vol_21d not available -> skipping zspread")

    added = [
        c for c in df.columns
        if c.startswith(("vix_ret", "vxn_ret", "vxn_z", "vxn_ma5", "vxn_accel", "vix_vxn", "zspread", "yield_spread"))
    ]
    log.info("  -> %s features (macro + yield curve)", len(added))
    return df
