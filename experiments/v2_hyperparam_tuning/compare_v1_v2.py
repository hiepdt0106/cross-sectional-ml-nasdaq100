"""
Side-by-side comparison: v1 baseline (outputs/metrics/walkforward_full.csv) vs
v2 tuned (results/walkforward_v2.csv). Prints per-fold deltas and applies the
promote criterion.

Promote criterion (decided 2026-04-22 before tuning ran):
  Mean OOS daily-AUC on 6 full-year folds (2020–2025) improves by ≥ +50bps
  AND improves on ≥ 4 of 6 folds individually.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V1_METRICS = ROOT / "outputs" / "metrics" / "walkforward_full.csv"
V2_METRICS = Path(__file__).resolve().parent / "results" / "walkforward_v2.csv"

PROMOTE_DELTA_BPS = 50.0
PROMOTE_MIN_WIN_FOLDS = 4
FULL_YEAR_FOLDS = list(range(1, 7))  # folds 1..6 (test 2020..2025)


def main():
    if not V1_METRICS.exists():
        sys.exit(f"Missing v1 metrics at {V1_METRICS}")
    if not V2_METRICS.exists():
        sys.exit(f"Missing v2 metrics at {V2_METRICS}. Run tune_walkforward.py first.")

    v1 = pd.read_csv(V1_METRICS)
    v2 = pd.read_csv(V2_METRICS)

    metrics = ["daily_auc", "global_auc", "top_k_ret", "rank_corr"]
    cols = ["fold", "test_year", "model"] + metrics

    merged = v1[cols].merge(
        v2[cols], on=["fold", "test_year", "model"], suffixes=("_v1", "_v2")
    )
    if merged.empty:
        sys.exit("No overlapping (fold, model) rows between v1 and v2.")

    for m in metrics:
        merged[f"{m}_delta_bps"] = (merged[f"{m}_v2"] - merged[f"{m}_v1"]) * 1e4

    print("=" * 100)
    print("PER-FOLD daily_auc — v1 vs v2  (delta in bps; +50bps≈+0.005 AUC)")
    print("=" * 100)
    print(merged.pivot_table(
        index=["fold", "test_year"], columns="model",
        values=["daily_auc_v1", "daily_auc_v2", "daily_auc_delta_bps"],
        aggfunc="first",
    ).round(4).to_string())

    print()
    print("=" * 100)
    print("MEAN across folds")
    print("=" * 100)

    summary_all = merged.groupby("model").agg(
        n_folds=("fold", "count"),
        v1_daily_auc=("daily_auc_v1", "mean"),
        v2_daily_auc=("daily_auc_v2", "mean"),
        delta_bps=("daily_auc_delta_bps", "mean"),
    ).round(4)
    print("\nALL folds (incl. partial 2026):")
    print(summary_all.to_string())

    full = merged[merged["fold"].isin(FULL_YEAR_FOLDS)]
    summary_full = full.groupby("model").agg(
        n_folds=("fold", "count"),
        v1_daily_auc=("daily_auc_v1", "mean"),
        v2_daily_auc=("daily_auc_v2", "mean"),
        delta_bps=("daily_auc_delta_bps", "mean"),
    ).round(4)
    print(f"\nFULL-year folds only ({FULL_YEAR_FOLDS[0]}..{FULL_YEAR_FOLDS[-1]}):")
    print(summary_full.to_string())

    print()
    print("=" * 100)
    print(f"PROMOTE check  (Δ ≥ +{PROMOTE_DELTA_BPS:.0f}bps mean AND wins ≥ {PROMOTE_MIN_WIN_FOLDS}/{len(FULL_YEAR_FOLDS)})")
    print("=" * 100)
    for model, grp in full.groupby("model"):
        wins = int((grp["daily_auc_delta_bps"] > 0).sum())
        delta = float(grp["daily_auc_delta_bps"].mean())
        promote = (delta >= PROMOTE_DELTA_BPS) and (wins >= PROMOTE_MIN_WIN_FOLDS)
        verdict = "✅ PROMOTE" if promote else "❌ KEEP v1"
        print(f"  {model:5s}: mean Δ = {delta:+7.1f} bps | wins = {wins}/{len(FULL_YEAR_FOLDS)} → {verdict}")

    print()
    out_csv = Path(__file__).resolve().parent / "results" / "comparison.csv"
    merged.to_csv(out_csv, index=False)
    print(f"Wrote merged comparison to {out_csv}")


if __name__ == "__main__":
    main()
