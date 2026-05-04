"""Run walk-forward training and build the ensemble outputs for ML_Full."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import logging

from scripts._common import add_common_args, setup_logging
from src.config import load_config, get_feature_cols, prepare_feature_sets
from src.splits.walkforward import make_expanding_splits
from src.models.train import walk_forward_train, build_ensemble
from src.utils.io import load, save

log = logging.getLogger(__name__)


def run_models(config_path: str | Path | None = None):
    """Run walk-forward training for the full feature set and build ensembles."""
    cfg = load_config(config_path) if config_path else load_config()

    # ── Load labeled dataset ──
    labeled_path = cfg.dir_processed / "dataset_labeled.parquet"
    featured_path = cfg.dir_processed / "dataset_featured.parquet"

    if labeled_path.exists():
        df_path = labeled_path
    elif featured_path.exists():
        df_path = featured_path
    else:
        raise FileNotFoundError(
            "Need dataset_labeled.parquet or dataset_featured.parquet. "
            "Run run_labeling.py first."
        )

    log.info(f"Loading dataset: {df_path}")
    df = load(df_path)

    if all(col not in df.columns for col in ["tb_label", "alpha_label", "alpha_ext_label"]):
        raise RuntimeError(
            "Dataset is missing labels. Run run_labeling.py before run_models.py."
        )

    # ── Feature columns ──
    feature_cols = get_feature_cols(list(df.columns))
    forbidden_features = {
        "tb_label",
        "tb_barrier",
        "tb_return",
        "daily_vol",
        "t1",
        "holding_td",
        "alpha_ret",
        "alpha_label",
        "alpha_ext_label",
    }
    leaked = sorted(c for c in feature_cols if c in forbidden_features)
    if leaked:
        raise RuntimeError(f"Target leakage detected in feature columns: {leaked}")

    feature_sets = prepare_feature_sets(df, feature_cols)
    feature_cols_full = feature_sets["full_cols"]
    macro_cols = feature_sets["macro_cols"]
    raw_date_cols = feature_sets["raw_date_cols"]
    dropped_redundant = feature_sets.get("dropped_redundant", [])

    log.info(
        "Selector features: %s full (incl. %s macro-interaction) | dropped %s raw date-level cols",
        len(feature_cols_full),
        len(macro_cols),
        len(raw_date_cols),
    )
    if raw_date_cols:
        log.info("Dropped raw date-level selector features: %s", raw_date_cols)
    if dropped_redundant:
        log.info("Dropped redundant selector features: %s", dropped_redundant)

    if "alpha_ret" not in df.columns or all(col not in df.columns for col in ["alpha_ext_label", "alpha_label"]):
        raise RuntimeError(
            "dataset_labeled.parquet is stale: missing alpha_* targets. "
            "Run scripts/run_labeling.py from the current project before training."
        )

    # ── Walk-forward splits ──
    splits = make_expanding_splits(
        df,
        first_test_year=cfg.walkforward.first_test_year,
        horizon=cfg.labeling.horizon,
        max_train_years=cfg.walkforward.max_train_years,
    )

    if not splits:
        raise RuntimeError("No folds could be created!")
    target_col = "alpha_ext_label" if "alpha_ext_label" in df.columns else "alpha_label"
    return_col = "alpha_ret"
    log.info("Training target: %s | ranking return: %s", target_col, return_col)

    # ── Walk-forward training: ML_Full only ──
    log.info("━" * 60)
    log.info("▶ Training ML FULL (with macro features)")
    log.info("━" * 60)
    results_full, pred_full = walk_forward_train(
        df,
        splits,
        feature_cols_full,
        target=target_col,
        return_col=return_col,
        top_k=cfg.strategy.top_k,
        n_optuna_trials=35,
        inner_purge_days=cfg.labeling.horizon,
    )

    # ── Save walk-forward results (parquet + csv) ──
    metrics_dir = cfg.dir_outputs / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    save(results_full, metrics_dir / "walkforward_full.parquet")
    results_full.to_csv(metrics_dir / "walkforward_full.csv", index=False)

    # Ensemble outputs.
    log.info("\n" + "═" * 60)
    log.info("▶ Building ensembles from out-of-sample fold performance")
    log.info("═" * 60)

    ens_full = build_ensemble(
        pred_full,
        results_df=results_full,
        method="adaptive",
        top_k=cfg.strategy.top_k,
        context_col=cfg.backtest.regime_col,
    )
    ens_full_stacked = build_ensemble(
        pred_full,
        results_df=results_full,
        method="stacked",
        top_k=cfg.strategy.top_k,
        context_col=cfg.backtest.regime_col,
    )

    save(ens_full, cfg.dir_processed / "predictions_ens_full.parquet")
    save(ens_full_stacked, cfg.dir_processed / "predictions_ens_full_stacked.parquet")

    # Also save individual model predictions
    save(pred_full, cfg.dir_processed / "predictions_all_full.parquet")

    log.info("\n✓ Models complete!")
    log.info(f"  Ensemble Full (adaptive): {len(ens_full):,} predictions")


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_models(args.config)


if __name__ == "__main__":
    main()
