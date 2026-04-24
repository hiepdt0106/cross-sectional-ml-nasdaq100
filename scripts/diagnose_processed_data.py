"""Inspect processed datasets for stale targets, dead regime features, and synthetic macro inputs."""
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
from src.features.macro_features import _looks_like_treasury_proxy
from src.utils.io import load

log = logging.getLogger(__name__)


def _series_stats(df: pd.DataFrame, col: str) -> dict[str, object]:
    if col not in df.columns:
        return {"exists": False}
    s = pd.to_numeric(df[col], errors="coerce")
    return {
        "exists": True,
        "non_na": float(s.notna().mean()),
        "min": float(s.min()) if s.notna().any() else None,
        "max": float(s.max()) if s.notna().any() else None,
        "std": float(s.std()) if s.notna().any() else None,
        "nunique": int(s.nunique(dropna=True)),
    }


def run_diagnostics(config_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path) if config_path else load_config()

    processed = cfg.dir_processed
    rows: list[dict[str, object]] = []

    dataset_path = processed / 'dataset.parquet'
    feat_path = processed / 'dataset_featured.parquet'
    lab_path = processed / 'dataset_labeled.parquet'

    if dataset_path.exists():
        df = load(dataset_path)
        first_ticker = df.index.get_level_values('ticker').unique()[0]
        daily = df.xs(first_ticker, level='ticker')
        if {'treasury_10y', 'treasury_2y'}.issubset(daily.columns):
            rows.append({
                'check': 'treasury_proxy_detected',
                'value': bool(_looks_like_treasury_proxy(daily['treasury_10y'], daily['treasury_2y'])),
            })

    if feat_path.exists():
        feat = load(feat_path)
        for col in ['p_high_vol', 'market_breadth_200d', 'yield_spread_10y2y', 'yield_spread_zscore']:
            stats = _series_stats(feat, col)
            rows.append({'check': f'feature::{col}', 'value': stats})

    if lab_path.exists():
        lab = load(lab_path)
        for col in ['alpha_ret', 'alpha_label', 'alpha_ext_label', 'tb_label', 'tb_return']:
            stats = _series_stats(lab, col)
            rows.append({'check': f'label::{col}', 'value': stats})


    alerts: list[dict[str, object]] = []
    for row in rows:
        check = row.get('check')
        value = row.get('value')
        if isinstance(value, dict) and value.get('exists'):
            if check == 'feature::p_high_vol' and value.get('nunique', 0) <= 1:
                alerts.append({'check': 'alert::p_high_vol_constant', 'value': True})
            if check == 'feature::yield_spread_zscore' and value.get('nunique', 0) <= 1:
                alerts.append({'check': 'alert::yield_spread_zscore_constant', 'value': True})
        if isinstance(value, dict) and check and check.startswith('label::alpha_'):
            if not value.get('exists') or value.get('non_na', 0.0) == 0.0:
                alerts.append({'check': f'alert::{check}_missing_or_empty', 'value': True})
    rows.extend(alerts)

    out = pd.DataFrame(rows)
    out_path = cfg.dir_outputs / 'metrics' / 'processed_data_diagnostics.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    log.info('Saved diagnostics -> %s', out_path)
    return out


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_diagnostics(args.config)


if __name__ == '__main__':
    main()
