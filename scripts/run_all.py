"""Run the full end-to-end pipeline from data build to analysis."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import logging

from scripts._common import add_common_args, setup_logging

log = logging.getLogger(__name__)


def run_all(config_path: str | Path | None = None) -> None:
    log.info("═" * 60)
    log.info("ML Trading Strategy NASDAQ-100")
    log.info("═" * 60)

    # ── Step 1: Data Pipeline ──
    log.info("\n[1/8] Data Pipeline ...")
    from src.data.pipeline import run_data_pipeline
    run_data_pipeline(config_path)

    # ── Step 2: Feature Engineering ──
    log.info("\n[2/8] Feature Engineering ...")
    from scripts.run_features import run_features
    run_features(config_path)

    # ── Step 3: Labeling ──
    log.info("\n[3/8] Labeling ...")
    from scripts.run_labeling import run_labeling
    run_labeling(config_path)

    # ── Step 4-5: Walk-forward + Ensemble ──
    log.info("\n[4-5/8] Walk-forward Training + Ensemble ...")
    from scripts.run_models import run_models
    run_models(config_path)

    # ── Step 6: Backtest ──
    log.info("\n[6/8] Backtest ...")
    from scripts.run_backtest import run_all_backtests
    run_all_backtests(config_path)

    # ── Step 7: Analysis ──
    log.info("\n[7/8] Analysis ...")
    from scripts.run_analysis import run_analysis
    run_analysis(config_path)

    # ── Step 8: Reporting mart ──
    log.info("\n[8/8] Reporting export ...")
    from scripts.export_reporting import export_reporting
    export_reporting(config_path)

    log.info("\n" + "═" * 60)
    log.info("✓ Pipeline complete!")
    log.info("═" * 60)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_all(args.config)


if __name__ == "__main__":
    main()
