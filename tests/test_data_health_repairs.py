from __future__ import annotations

import numpy as np
import pandas as pd


def test_treasury_proxy_detection_flags_synthetic_offset_series() -> None:
    from src.features.macro_features import _looks_like_treasury_proxy

    dates = pd.date_range('2020-01-01', periods=200, freq='B')
    ten = pd.Series(np.linspace(1.0, 4.0, len(dates)), index=dates)
    two = ten - 1.0
    assert _looks_like_treasury_proxy(ten, two) is True

    two_realistic = ten - np.sin(np.linspace(0, 8, len(dates)))
    assert _looks_like_treasury_proxy(ten, two_realistic) is False


def test_prepare_feature_sets_drops_perfect_duplicate_macro_columns() -> None:
    from src.config import prepare_feature_sets

    dates = pd.date_range('2024-01-01', periods=20, freq='B')
    tickers = ['AAA', 'BBB', 'CCC']
    idx = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])

    base = np.tile(np.array([0.1, 0.2, 0.3]), len(dates))
    df = pd.DataFrame(
        {
            'mom_63d': base,
            'vol_21d': base * 2,
            'rolling_beta_63d': base * 3,
            'cs_mom_zscore_63d': base * 4,
            'p_high_x_mom_63d': base,
            'p_high_x_vol_21d': base * 2,
            'beta_regime': base * 3,
            'cs_mom_x_yield': base * 4,
        },
        index=idx,
    )

    feature_cols = list(df.columns)
    out = prepare_feature_sets(df, feature_cols)
    assert 'p_high_x_mom_63d' not in out['full_cols']
    assert 'p_high_x_vol_21d' not in out['full_cols']
    assert 'beta_regime' not in out['full_cols']
    assert 'cs_mom_x_yield' not in out['full_cols']
    assert set(out['dropped_redundant']) >= {
        'p_high_x_mom_63d', 'p_high_x_vol_21d', 'beta_regime', 'cs_mom_x_yield'
    }


def test_run_models_requires_aligned_alpha_targets(tmp_path) -> None:
    import shutil
    import subprocess
    import sys

    from pathlib import Path

    source_root = Path(__file__).resolve().parents[1]
    root = tmp_path / 'proj'
    shutil.copytree(source_root, root)
    (root / 'data' / 'processed').mkdir(parents=True, exist_ok=True)

    dates = pd.date_range('2024-01-01', periods=4, freq='B')
    idx = pd.MultiIndex.from_product([dates, ['AAA', 'BBB']], names=['date', 'ticker'])
    df = pd.DataFrame(
        {
            'mom_63d': np.random.randn(len(idx)),
            'vol_21d': np.random.rand(len(idx)),
            'tb_label': [1, 0] * 4,
            'tb_return': np.random.randn(len(idx)) * 0.01,
            'adj_close': np.linspace(10, 20, len(idx)),
        },
        index=idx,
    )
    df.to_parquet(root / 'data' / 'processed' / 'dataset_labeled.parquet')

    result = subprocess.run(
        [sys.executable, str(root / 'scripts' / 'run_models.py'), '--config', str(root / 'configs' / 'base.yaml')],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert 'missing alpha_* targets' in (result.stderr + result.stdout)
