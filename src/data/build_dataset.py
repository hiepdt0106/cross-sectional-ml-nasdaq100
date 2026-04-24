from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def build_dataset(
    stocks: pd.DataFrame,
    macro: pd.DataFrame,
    bench: pd.Series | None = None,
    treasury: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge stock, macro, benchmark, and Treasury data into one panel."""
    if stocks.index.names != ["date", "ticker"]:
        raise ValueError("stocks.index must be a MultiIndex ['date', 'ticker']")

    macro = macro.copy()
    macro.index = pd.to_datetime(macro.index).normalize()
    if macro.index.tz is not None:
        macro.index = macro.index.tz_localize(None)

    if not macro.index.is_unique:
        raise ValueError("macro.index has duplicate dates")
    required_cols = {"vix", "vxn"}
    if not required_cols.issubset(macro.columns):
        raise ValueError(f"macro must contain columns {sorted(required_cols)}")

    stocks_flat = stocks.reset_index()
    stocks_flat["date"] = pd.to_datetime(stocks_flat["date"]).dt.normalize()

    macro_flat = macro.reset_index()
    if "index" in macro_flat.columns:
        macro_flat = macro_flat.rename(columns={"index": "date"})

    merged = stocks_flat.merge(macro_flat, on="date", how="left", validate="many_to_one")

    if bench is not None:
        bench = bench.copy()
        bench.index = pd.to_datetime(bench.index).normalize()
        if bench.index.tz is not None:
            bench.index = bench.index.tz_localize(None)
        bench_df = bench.to_frame("bench_close")
        bench_flat = bench_df.reset_index()
        if "index" in bench_flat.columns:
            bench_flat = bench_flat.rename(columns={"index": "date"})
        merged = merged.merge(bench_flat, on="date", how="left", validate="many_to_one")

    if treasury is not None:
        treasury = treasury.copy()
        treasury.index = pd.to_datetime(treasury.index).normalize()
        if treasury.index.tz is not None:
            treasury.index = treasury.index.tz_localize(None)

        rename_map = {}
        if "us10y" in treasury.columns:
            rename_map["us10y"] = "treasury_10y"
        if "us2y" in treasury.columns:
            rename_map["us2y"] = "treasury_2y"
        if rename_map:
            treasury = treasury.rename(columns=rename_map)

        treasury_flat = treasury.reset_index()
        if "index" in treasury_flat.columns:
            treasury_flat = treasury_flat.rename(columns={"index": "date"})

        merged = merged.merge(treasury_flat, on="date", how="left", validate="many_to_one")

        for col in ["treasury_10y", "treasury_2y"]:
            if col in merged.columns:
                n_na = int(merged[col].isna().sum())
                if n_na:
                    merged[col] = merged[col].ffill()
                    n_na_after = int(merged[col].isna().sum())
                    log.info("  %s: ffill %s -> %s NaN", col, n_na, n_na_after)

    merged = merged.set_index(["date", "ticker"]).sort_index()

    macro_cols = [c for c in macro.columns if c in merged.columns]
    remaining_na = int(merged[macro_cols].isna().sum().sum())
    if remaining_na:
        raise ValueError(
            "Macro still has NaN after merge. Align/fill macro via align_panel() before calling build_dataset()."
        )

    if "bench_close" in merged.columns:
        bench_na = int(merged["bench_close"].isna().sum())
        if bench_na:
            log.warning("bench_close has %s NaN -> ffill", bench_na)
            merged["bench_close"] = merged.groupby(level="ticker")["bench_close"].ffill()

    log.info("build_dataset: %s rows | columns=%s", f"{len(merged):,}", list(merged.columns))
    return merged
