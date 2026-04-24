"""Apply triple-barrier labeling to feature dataset."""
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
from src.config import load_config
from src.labeling import add_forward_rebalance_targets, label as triple_barrier_label
from src.utils.io import load, save

log = logging.getLogger(__name__)


def run_labeling(config_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path) if config_path else load_config()
    in_path = cfg.dir_processed / "dataset_featured.parquet"
    out_path = cfg.dir_processed / "dataset_labeled.parquet"

    df = load(in_path)
    log.info("Loaded features: %s", df.shape)

    labeled = triple_barrier_label(
        df,
        horizon=cfg.labeling.horizon,
        pt_sl_mult=cfg.labeling.pt_sl_mult,
        vol_window=cfg.labeling.vol_window,
    )
    labeled = add_forward_rebalance_targets(
        labeled,
        horizon=cfg.strategy.rebalance_days,
        top_k=cfg.strategy.top_k,
        return_col="alpha_ret",
        label_col="alpha_label",
        extreme_label_col="alpha_ext_label",
    )

    keep_mask = labeled["tb_label"].notna()
    if "alpha_ret" in labeled.columns:
        keep_mask = keep_mask | labeled["alpha_ret"].notna()
    labeled = labeled.loc[keep_mask].copy()

    if "tb_label" in labeled.columns:
        labeled.loc[labeled["tb_label"].notna(), "tb_label"] = (
            labeled.loc[labeled["tb_label"].notna(), "tb_label"].astype(int)
        )
    if "alpha_label" in labeled.columns:
        labeled.loc[labeled["alpha_label"].notna(), "alpha_label"] = (
            labeled.loc[labeled["alpha_label"].notna(), "alpha_label"].astype(int)
        )
    if "alpha_ext_label" in labeled.columns:
        labeled.loc[labeled["alpha_ext_label"].notna(), "alpha_ext_label"] = (
            labeled.loc[labeled["alpha_ext_label"].notna(), "alpha_ext_label"].astype(int)
        )

    save(labeled, out_path)

    flat = labeled.reset_index()
    flat["year"] = flat["date"].dt.year
    label_summary = pd.DataFrame(
        {
            "metric": [
                "n_rows",
                "n_tickers",
                "label_pos",
                "label_neg",
                "pos_rate",
                "avg_holding_td",
                "alpha_label_pos_rate",
                "alpha_ext_label_pos_rate",
                "alpha_ext_label_coverage",
                "alpha_ret_mean",
            ],
            "value": [
                len(labeled),
                labeled.index.get_level_values("ticker").nunique(),
                int((labeled["tb_label"] == 1).sum()),
                int((labeled["tb_label"] == 0).sum()),
                float((labeled["tb_label"] == 1).mean()),
                float(labeled["holding_td"].mean()),
                float((labeled["alpha_label"] == 1).mean()) if "alpha_label" in labeled.columns else float("nan"),
                float((labeled["alpha_ext_label"] == 1).mean()) if "alpha_ext_label" in labeled.columns else float("nan"),
                float(labeled["alpha_ext_label"].notna().mean()) if "alpha_ext_label" in labeled.columns else float("nan"),
                float(labeled["alpha_ret"].mean()) if "alpha_ret" in labeled.columns else float("nan"),
            ],
        }
    )
    barrier_by_year = flat.groupby(["year", "tb_barrier"]).size().unstack(fill_value=0)

    label_summary.to_csv(cfg.dir_outputs / "metrics" / "label_summary.csv", index=False)
    barrier_by_year.to_csv(cfg.dir_outputs / "metrics" / "barrier_by_year.csv")

    log.info("Saved labeled dataset: %s", out_path)
    return labeled


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_labeling(args.config)


if __name__ == "__main__":
    main()
