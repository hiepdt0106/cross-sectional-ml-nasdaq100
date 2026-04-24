"""Feature and config regression tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _make_sample_df(n_dates: int = 50, n_tickers: int = 10) -> pd.DataFrame:
    """Build a sample MultiIndex (date, ticker) DataFrame with OHLCV + macro."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    rows = []
    for tkr in tickers:
        base_price = 100 + rng.randn() * 20
        for i, d in enumerate(dates):
            ret = rng.randn() * 0.02
            c = base_price * np.exp(ret * (i + 1) * 0.01)
            rows.append({
                "date": d,
                "ticker": tkr,
                "adj_open": c * (1 + rng.randn() * 0.005),
                "adj_high": c * (1 + abs(rng.randn() * 0.01)),
                "adj_low": c * (1 - abs(rng.randn() * 0.01)),
                "adj_close": c,
                "adj_volume": rng.randint(1_000_000, 50_000_000),
                "vix": 20 + rng.randn() * 5,
                "vxn": 22 + rng.randn() * 5,
                "bench_close": 300 + rng.randn() * 10,
                "treasury_10y": 2.5 + rng.randn() * 0.5,
                "treasury_2y": 1.5 + rng.randn() * 0.5,
            })

    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return df


class TestCrossSectionalFeatures:
    """Tests cho src/features/cross_sectional.py."""

    def test_adds_5_features(self):
        from src.features.price import add_price_features
        from src.features.volatility import add_vol_features
        from src.features.cross_sectional import add_cross_sectional_features

        df = _make_sample_df(100, 10)
        df = add_price_features(df)
        df = add_vol_features(df)
        df = add_cross_sectional_features(df)

        expected = {"cs_ret_zscore_1d", "cs_vol_zscore_21d", "cs_mom_zscore_63d",
                    "overnight_gap", "close_position"}
        for col in expected:
            assert col in df.columns, f"Missing: {col}"

    def test_zscore_clipped(self):
        from src.features.price import add_price_features
        from src.features.cross_sectional import add_cross_sectional_features

        df = _make_sample_df(100, 10)
        df = add_price_features(df)
        df = add_cross_sectional_features(df)

        col = "cs_ret_zscore_1d"
        valid = df[col].dropna()
        assert valid.min() >= -5.0
        assert valid.max() <= 5.0

    def test_close_position_range(self):
        from src.features.cross_sectional import add_cross_sectional_features

        df = _make_sample_df(50, 5)
        df = add_cross_sectional_features(df)

        valid = df["close_position"].dropna()
        assert valid.min() >= 0.0 - 1e-6
        assert valid.max() <= 1.0 + 1e-6


class TestInteractionFeatures:
    """Tests cho src/features/interactions.py."""

    def test_creates_features(self):
        from src.features.price import add_price_features
        from src.features.volatility import add_vol_features
        from src.features.interactions import add_interaction_features

        df = _make_sample_df(100, 10)
        df = add_price_features(df)
        df = add_vol_features(df)
        # Add mock p_high_vol and vxn_zscore
        df["p_high_vol"] = 0.5
        df["vxn_zscore"] = np.random.randn(len(df))
        df["zspread"] = np.random.randn(len(df)) * 0.01

        df = add_interaction_features(df)

        for col in ["mom_quality", "stress_reversal"]:
            assert col in df.columns, f"Missing: {col}"

    def test_missing_input_skips_gracefully(self):
        from src.features.interactions import add_interaction_features

        df = _make_sample_df(50, 5)
        # No feature columns → should skip gracefully
        df = add_interaction_features(df)
        # Should not crash


class TestYieldCurveFeatures:
    """Tests cho yield curve features trong macro_features.py."""

    def test_yield_features_created(self):
        from src.features.price import add_price_features
        from src.features.volatility import add_vol_features
        from src.features.macro_features import add_macro_features

        df = _make_sample_df(100, 5)
        df = add_price_features(df)
        df = add_vol_features(df)
        df = add_macro_features(df)

        for col in ["yield_spread_10y2y", "yield_spread_change_5d", "yield_spread_zscore"]:
            assert col in df.columns, f"Missing: {col}"

    def test_yield_spread_sign(self):
        from src.features.price import add_price_features
        from src.features.volatility import add_vol_features
        from src.features.macro_features import add_macro_features

        df = _make_sample_df(100, 5)
        df = add_price_features(df)
        df = add_vol_features(df)
        df = add_macro_features(df)

        spread = df["yield_spread_10y2y"].dropna()
        # 10Y > 2Y on average in our mock data (2.5 vs 1.5)
        assert spread.mean() > 0


class TestBlockBootstrap:
    """Tests cho block_bootstrap_alpha."""

    def test_bootstrap_runs(self):
        from src.backtest.engine import block_bootstrap_alpha

        n = 200
        dates = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.RandomState(42)

        ml_eq = pd.DataFrame({
            "equity": np.cumsum(rng.randn(n) * 0.01 + 0.001) + 100,
            "daily_ret": rng.randn(n) * 0.01 + 0.001,
        }, index=dates)

        bh_eq = pd.DataFrame({
            "equity": np.cumsum(rng.randn(n) * 0.01) + 100,
            "daily_ret": rng.randn(n) * 0.01,
        }, index=dates)

        result = block_bootstrap_alpha(ml_eq, bh_eq, block_size=10, n_bootstrap=100)
        assert "bootstrap_p_value" in result
        assert "bootstrap_ci_95_lower" in result
        assert 0 <= result["bootstrap_p_value"] <= 1


class TestConfigFeatures:
    """Tests for config-driven feature settings."""

    def test_base_config_loads(self):
        from src.config import load_config, ROOT
        cfg = load_config(ROOT / "configs" / "base.yaml")
        assert cfg.data.start_date == "2016-01-01"
        assert cfg.treasury.ffill_limit == 5
        assert len(cfg.benchmark_mcap.tickers) == 10

    def test_legacy_config_alias_loads(self):
        from src.config import load_config, ROOT
        cfg = load_config(ROOT / "configs" / "base_v2.yaml")
        assert cfg.data.start_date == "2016-01-01"
        assert cfg.treasury.ffill_limit == 5
        assert cfg.strategy.max_weight_cap == 0.25

    def test_skip_rank_includes_yield(self):
        from src.config import SKIP_RANK_FEATURES
        assert "yield_spread_10y2y" in SKIP_RANK_FEATURES
        assert "yield_spread_change_5d" in SKIP_RANK_FEATURES

    def test_macro_prefixes_include_yield(self):
        from src.config import MACRO_FEATURE_PREFIXES
        assert any("yield" in p for p in MACRO_FEATURE_PREFIXES)


class TestComputeWeights:
    """Tests cho confidence-weighted."""

    def test_equal_weight(self):
        from src.backtest.engine import _compute_weights

        pool = pd.Series([0.8, 0.7, 0.6, 0.5, 0.4],
                         index=["A", "B", "C", "D", "E"])
        w = _compute_weights(pool, top_k=3, confidence_weighted=False)
        assert len(w) == 3
        assert abs(sum(w.values()) - 1.0) < 1e-6
        assert all(abs(v - 1 / 3) < 1e-6 for v in w.values())

    def test_confidence_weighted(self):
        from src.backtest.engine import _compute_weights

        pool = pd.Series([0.9, 0.7, 0.6], index=["A", "B", "C"])
        w = _compute_weights(pool, top_k=3, confidence_weighted=True, max_weight_cap=0.5)
        assert len(w) == 3
        assert abs(sum(w.values()) - 1.0) < 1e-6
        # A should have highest weight
        assert w["A"] > w["B"] > w["C"]
        # Cap check
        assert all(v <= 0.5 + 1e-6 for v in w.values())

    def test_max_weight_cap_enforced(self):
        from src.backtest.engine import _compute_weights

        pool = pd.Series([0.99, 0.51, 0.51], index=["A", "B", "C"])
        w = _compute_weights(pool, top_k=3, confidence_weighted=True, max_weight_cap=0.25)
        assert all(v <= 0.26 for v in w.values())  # small tolerance
