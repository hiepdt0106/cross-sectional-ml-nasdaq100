"""
scripts/run_features.py
────────────────────────
Feature engineering for the full research feature set.

All features are still generated (~80 columns), but get_feature_cols()
automatically drops REMOVED_FEATURES (20 noisy/redundant columns).

Order matters:
  1. Price/Momentum (Group 1)
  2. Volatility (Group 2)
  3. Macro + Yield Curve (Group 3 + 8)
  4. Relative/Residual (Group 4)
  5. Regime (Group 5)
  6. Cross-sectional (Group 6)
  7. Interactions (Group 7) — BEFORE ranking
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import logging

import pandas as pd

from scripts._common import add_common_args, setup_logging
from src.config import load_config, get_feature_cols, prepare_feature_sets, REMOVED_FEATURES
from src.features import (
    add_price_features,
    add_vol_features,
    add_macro_features,
    add_relative_features,
    add_regime_features,
    add_cross_sectional_features,
    add_interaction_features,
)
from src.utils.io import load, save

log = logging.getLogger(__name__)


def run_features(config_path: str | Path | None = None) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    cfg = load_config(config_path) if config_path else load_config()

    # ── Load dataset ──
    dataset_path = cfg.dir_processed / "dataset.parquet"
    log.info(f"Loading dataset: {dataset_path}")
    df = load(dataset_path)
    log.info(f"Dataset: {df.shape[0]:,} rows, {df.shape[1]} cols")
    log.info(f"Universe: {df.index.get_level_values('ticker').nunique()} tickers")

    # ── Feature engineering (generate full ~80 columns) ──
    df = add_price_features(df)
    df = add_vol_features(df)
    df = add_macro_features(df)
    df = add_relative_features(df)
    df = add_regime_features(
        df,
        lookback=cfg.regime.lookback,
        refit_freq=cfg.regime.refit_frequency,
    )
    df = add_cross_sectional_features(df)
    df = add_interaction_features(df)

    # ── Feature selection (get_feature_cols drops REMOVED_FEATURES) ──
    feature_cols = get_feature_cols(list(df.columns))
    feature_sets = prepare_feature_sets(df, feature_cols)
    feature_cols = feature_sets["full_cols"]
    base_cols = feature_sets["base_cols"]
    macro_cols = feature_sets["macro_cols"]
    raw_date_cols = feature_sets["raw_date_cols"]
    dropped_redundant = feature_sets.get("dropped_redundant", [])

    # Log pruning info
    all_possible = [c for c in df.columns if c not in {
        "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume",
        "vix", "vxn", "treasury_10y", "treasury_2y",
        "tb_label", "tb_barrier", "tb_return", "daily_vol", "t1", "holding_td",
        "bench_close",
    }]
    actually_removed = [c for c in all_possible if c in REMOVED_FEATURES]

    log.info(f"\n{'=' * 60}")
    log.info("Feature engineering complete!")
    log.info(f"  Generated: {len(all_possible)} features")
    log.info(f"  Pruned:    {len(actually_removed)} (REMOVED_FEATURES)")
    log.info(f"  Final:     {len(feature_cols)} selector features "
             f"({len(base_cols)} base + {len(macro_cols)} macro-interaction)")
    log.info(f"  Dropped raw date-level selector features: {len(raw_date_cols)}")
    log.info(f"  Dropped redundant selector features: {len(dropped_redundant)}")
    if dropped_redundant:
        log.info("  Redundant features removed: %s", dropped_redundant)
    log.info(f"  Total rows: {df.shape[0]:,}")
    log.info(f"  NaN rate:  {df[feature_cols].isna().mean().mean():.2%}")
    log.info(f"{'=' * 60}")

    # ── Save ──
    out_path = cfg.dir_processed / "dataset_featured.parquet"
    save(df, out_path)
    log.info(f"Saved: {out_path}")

    # Minimal committed panel for API-free backtest reruns. This avoids
    # shipping the full feature matrix while preserving the price/benchmark
    # fields required by scripts/run_backtest.py.
    backtest_cols = [
        "adj_open",
        "adj_close",
        "bench_close",
        "p_high_vol",
        "market_breadth_200d",
        "vxn_zscore",
        "yield_spread_zscore",
        *cfg.strategy.signal_anchor_features,
    ]
    backtest_cols = [c for c in dict.fromkeys(backtest_cols) if c in df.columns]
    backtest_panel_path = cfg.dir_processed / "backtest_panel.parquet"
    save(df[backtest_cols], backtest_panel_path)
    log.info(f"Backtest panel ({len(backtest_cols)} cols): {backtest_panel_path}")

    # Save feature list with is_macro flag
    feat_info = pd.DataFrame({
        "feature": feature_cols + raw_date_cols + dropped_redundant,
        "bucket": [
            "macro_interaction" if c in macro_cols else "base"
            for c in feature_cols
        ] + ["raw_date_context"] * len(raw_date_cols) + ["dropped_redundant"] * len(dropped_redundant),
    })
    feat_path = cfg.dir_outputs / "metrics" / "feature_columns.csv"
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    feat_info.to_csv(feat_path, index=False)
    log.info(f"Feature list ({len(feature_cols)} cols): {feat_path}")

    return df


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_features(args.config)


if __name__ == "__main__":
    main()
