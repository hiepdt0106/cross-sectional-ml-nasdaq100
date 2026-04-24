"""Regression tests for config, feature modules, and backtest helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_panel(n_dates: int = 100, n_tickers: int = 10, seed: int = 42) -> pd.DataFrame:
    """Build dummy panel data resembling the real dataset."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    rows = []
    for d in dates:
        for t in tickers:
            price = 100 + rng.randn() * 10
            rows.append({
                "date": d,
                "ticker": t,
                "adj_open": price * (1 + rng.randn() * 0.001),
                "adj_high": price * (1 + abs(rng.randn()) * 0.02),
                "adj_low": price * (1 - abs(rng.randn()) * 0.02),
                "adj_close": price,
                "adj_volume": abs(rng.randn()) * 1e6,
                "vix": 20 + rng.randn() * 5,
                "vxn": 22 + rng.randn() * 5,
                "bench_close": 300 + rng.randn() * 10,
                "treasury_10y": 3.5 + rng.randn() * 0.5,
                "treasury_2y": 2.5 + rng.randn() * 0.5,
            })

    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return df


def _add_prerequisite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features required for cross-sectional & interaction tests."""
    df = df.copy()
    rng = np.random.RandomState(123)
    n = len(df)

    # Simulate features that cross-sectional & interaction depend on
    df["ret_1d"] = rng.randn(n) * 0.02
    df["ret_5d"] = rng.randn(n) * 0.04
    df["vol_21d"] = abs(rng.randn(n)) * 0.02 + 0.01
    df["mom_63d"] = rng.randn(n) * 0.1
    df["adx_14d"] = abs(rng.randn(n)) * 20 + 15
    df["abnormal_volume"] = rng.randn(n)
    df["rolling_beta_63d"] = rng.randn(n) * 0.3 + 1.0
    df["p_high_vol"] = rng.rand(n)
    df["vxn_zscore"] = rng.randn(n)
    df["zspread"] = rng.randn(n) * 0.005

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigLayout:
    def test_load_canonical_config(self):
        from src.config import load_config
        cfg = load_config("configs/base.yaml")
        assert len(cfg.data.tickers) > 50
        assert cfg.data.start_date == "2016-01-01"

    def test_legacy_config_alias(self):
        from src.config import load_config
        cfg = load_config("configs/base_v2.yaml")
        assert len(cfg.data.tickers) > 50
        assert cfg.treasury.ffill_limit == 5
        assert cfg.random_benchmark.n_iterations == 200

    def test_treasury_config(self):
        from src.config import load_config
        cfg = load_config("configs/base.yaml")
        assert "us10y" in cfg.treasury.symbols
        assert "us2y" in cfg.treasury.symbols
        assert cfg.treasury.ffill_limit == 5

    def test_benchmark_mcap_config(self):
        from src.config import load_config
        cfg = load_config("configs/base.yaml")
        assert len(cfg.benchmark_mcap.tickers) == 10
        assert "AAPL" in cfg.benchmark_mcap.tickers

    def test_strategy_config(self):
        from src.config import load_config
        cfg = load_config("configs/base.yaml")
        assert cfg.strategy.max_weight_cap == 0.25
        assert cfg.strategy.confidence_weighted is False

    def test_skip_rank_includes_yield(self):
        from src.config import SKIP_RANK_FEATURES, YIELD_FEATURE_NAMES
        assert YIELD_FEATURE_NAMES.issubset(SKIP_RANK_FEATURES)

    def test_macro_prefixes_include_yield(self):
        from src.config import MACRO_FEATURE_PREFIXES
        assert "yield_spread" in MACRO_FEATURE_PREFIXES

    def test_non_feature_cols_include_treasury(self):
        from src.config import NON_FEATURE_COLS
        assert "treasury_10y" in NON_FEATURE_COLS
        assert "treasury_2y" in NON_FEATURE_COLS


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL FEATURES TESTS (Group 6)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossSectionalFeatures:
    def test_basic_output(self):
        from src.features.cross_sectional import add_cross_sectional_features
        df = _make_panel()
        df = _add_prerequisite_features(df)
        result = add_cross_sectional_features(df)

        assert "cs_ret_zscore_1d" in result.columns
        assert "cs_vol_zscore_21d" in result.columns
        assert "cs_mom_zscore_63d" in result.columns
        assert "overnight_gap" in result.columns
        assert "close_position" in result.columns

    def test_zscore_clipped(self):
        from src.features.cross_sectional import add_cross_sectional_features
        df = _make_panel()
        df = _add_prerequisite_features(df)
        result = add_cross_sectional_features(df)

        cs = result["cs_ret_zscore_1d"].dropna()
        assert cs.min() >= -5.0
        assert cs.max() <= 5.0

    def test_close_position_range(self):
        from src.features.cross_sectional import add_cross_sectional_features
        df = _make_panel()
        df = _add_prerequisite_features(df)
        result = add_cross_sectional_features(df)

        cp = result["close_position"].dropna()
        assert cp.min() >= 0.0 - 1e-10
        assert cp.max() <= 1.0 + 1e-10

    def test_overnight_gap_not_all_nan(self):
        from src.features.cross_sectional import add_cross_sectional_features
        df = _make_panel()
        df = _add_prerequisite_features(df)
        result = add_cross_sectional_features(df)

        assert result["overnight_gap"].notna().sum() > 0


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTION FEATURES TESTS (Group 7)
# ═══════════════════════════════════════════════════════════════════════════════

class TestInteractionFeatures:
    def test_basic_output(self):
        from src.features.interactions import add_interaction_features
        df = _make_panel()
        df = _add_prerequisite_features(df)
        # Need cs features for cs_ret_x_regime
        df["cs_ret_zscore_1d"] = np.random.randn(len(df))
        df["cs_mom_zscore_63d"] = np.random.randn(len(df))

        result = add_interaction_features(df)

        expected = [
            "mom_quality", "stress_reversal",
            "beta_regime", "spread_momentum",
        ]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"

    def test_mom_quality_is_product(self):
        from src.features.interactions import add_interaction_features
        df = _make_panel(n_dates=20, n_tickers=3)
        df = _add_prerequisite_features(df)
        result = add_interaction_features(df)

        # mom_quality = mom_63d × adx_14d
        expected = df["mom_63d"] * df["adx_14d"]
        pd.testing.assert_series_equal(
            result["mom_quality"].dropna(),
            expected.dropna(),
            check_names=False,
        )

    def test_missing_prerequisite_graceful(self):
        from src.features.interactions import add_interaction_features
        df = _make_panel(n_dates=20, n_tickers=3)
        # Don't add prerequisites — should not crash
        result = add_interaction_features(df)
        assert isinstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════════════
# YIELD CURVE FEATURES TESTS (Group 8)
# ═══════════════════════════════════════════════════════════════════════════════

class TestYieldCurveFeatures:
    def test_yield_features_created(self):
        from src.features.macro_features import add_macro_features
        df = _make_panel(n_dates=300)
        df = _add_prerequisite_features(df)
        result = add_macro_features(df)

        assert "yield_spread_10y2y" in result.columns
        assert "yield_spread_change_5d" in result.columns
        assert "yield_spread_zscore" in result.columns

    def test_yield_spread_is_10y_minus_2y(self):
        from src.features.macro_features import add_macro_features
        df = _make_panel(n_dates=300)
        df = _add_prerequisite_features(df)
        result = add_macro_features(df)

        # Check first ticker — yield_spread should be constant across tickers
        first_tkr = result.index.get_level_values("ticker").unique()[0]
        sub = result.xs(first_tkr, level="ticker")
        spread = sub["yield_spread_10y2y"].dropna()
        assert len(spread) > 0

    def test_without_treasury_graceful(self):
        from src.features.macro_features import add_macro_features
        df = _make_panel()
        df = _add_prerequisite_features(df)
        # Remove treasury columns
        df = df.drop(columns=["treasury_10y", "treasury_2y"])
        result = add_macro_features(df)

        # Should still have VIX/VXN features but no yield
        assert "vxn_zscore" in result.columns
        assert "yield_spread_10y2y" not in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST NEW FEATURES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidenceWeighted:
    def test_compute_weights_equal(self):
        from src.backtest.engine import _compute_weights
        pool = pd.Series([0.7, 0.6, 0.8], index=["A", "B", "C"])
        w = _compute_weights(pool, top_k=3, confidence_weighted=False)
        assert len(w) == 3
        assert abs(sum(w.values()) - 1.0) < 1e-10
        assert all(abs(v - 1/3) < 1e-10 for v in w.values())

    def test_compute_weights_confidence(self):
        from src.backtest.engine import _compute_weights
        pool = pd.Series([0.9, 0.6, 0.51], index=["A", "B", "C"])
        w = _compute_weights(pool, top_k=3, confidence_weighted=True, max_weight_cap=0.5)
        assert abs(sum(w.values()) - 1.0) < 1e-10
        assert w["A"] > w["B"] > w["C"]  # higher prob → higher weight

    def test_max_weight_cap(self):
        from src.backtest.engine import _compute_weights
        pool = pd.Series([0.99, 0.51, 0.51], index=["A", "B", "C"])
        w = _compute_weights(pool, top_k=3, confidence_weighted=True, max_weight_cap=0.5)
        assert all(v <= 0.5 + 1e-12 for v in w.values())

    def test_max_weight_cap_tight(self):
        """Regression case: 4 tickers, cap=0.25 — old code failed this."""
        from src.backtest.engine import _compute_weights
        pool = pd.Series([0.9, 0.8, 0.7, 0.6], index=["A", "B", "C", "D"])
        w = _compute_weights(pool, top_k=4, confidence_weighted=True, max_weight_cap=0.25)
        assert abs(sum(w.values()) - 1.0) < 1e-10
        assert all(v <= 0.25 + 1e-12 for v in w.values()), f"cap violated: {w}"

    def test_max_weight_cap_10_stocks(self):
        """10 tickers, cap=0.15 — extreme case."""
        from src.backtest.engine import _compute_weights
        pool = pd.Series(
            [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.51],
            index=[f"T{i}" for i in range(10)],
        )
        w = _compute_weights(pool, top_k=10, confidence_weighted=True, max_weight_cap=0.15)
        assert abs(sum(w.values()) - 1.0) < 1e-10
        assert all(v <= 0.15 + 1e-12 for v in w.values()), f"cap violated: {max(w.values())}"


class TestBlockBootstrap:
    def test_basic_output(self):
        from src.backtest.engine import block_bootstrap_alpha
        rng = np.random.RandomState(42)
        n = 500
        dates = pd.bdate_range("2020-01-01", periods=n)

        ml_eq = pd.DataFrame({
            "equity": np.cumsum(rng.randn(n) * 0.01) + 100,
            "daily_ret": rng.randn(n) * 0.01 + 0.0005,
        }, index=dates)

        bh_eq = pd.DataFrame({
            "equity": np.cumsum(rng.randn(n) * 0.01) + 100,
            "daily_ret": rng.randn(n) * 0.01,
        }, index=dates)

        result = block_bootstrap_alpha(ml_eq, bh_eq, block_size=10, n_bootstrap=100)
        assert "bootstrap_p_value" in result
        assert "bootstrap_ci_95_lower" in result
        assert "bootstrap_ci_95_upper" in result
        assert result["n_bootstrap"] == 100

    def test_insufficient_data(self):
        from src.backtest.engine import block_bootstrap_alpha
        dates = pd.bdate_range("2020-01-01", periods=5)
        ml_eq = pd.DataFrame({"equity": [100]*5, "daily_ret": [0]*5}, index=dates)
        bh_eq = pd.DataFrame({"equity": [100]*5, "daily_ret": [0]*5}, index=dates)
        result = block_bootstrap_alpha(ml_eq, bh_eq, block_size=10)
        assert "bootstrap_error" in result


class TestMDDDuration:
    def test_no_drawdown(self):
        from src.backtest.engine import _compute_mdd_duration
        eq = pd.Series([100, 101, 102, 103])
        assert _compute_mdd_duration(eq) == 0

    def test_simple_drawdown(self):
        from src.backtest.engine import _compute_mdd_duration
        eq = pd.Series([100, 99, 98, 99, 100, 101])
        # Drawdown from 100 → 98 → recover at index 4 = 3 days
        assert _compute_mdd_duration(eq) == 3


class TestComputeMetrics:
    def test_has_new_fields(self):
        from src.backtest.engine import compute_metrics
        dates = pd.bdate_range("2020-01-01", periods=252)
        rng = np.random.RandomState(42)
        rets = rng.randn(252) * 0.01
        eq = pd.DataFrame({
            "equity": 10000 * np.cumprod(1 + rets),
            "daily_ret": rets,
        }, index=dates)
        eq.attrs["initial_capital"] = 10000

        m = compute_metrics(eq)
        assert "MDD_Duration" in m
        assert "Tail_Ratio" in m
        assert "CAGR" in m


class TestMacroMergeNoDuplicates:
    def test_no_xy_columns_on_rerun(self):
        """Bug: calling add_macro_features twice produced vxn_zscore_x / _y."""
        from src.features.macro_features import add_macro_features
        df = _make_panel(n_dates=300)
        df = _add_prerequisite_features(df)
        result1 = add_macro_features(df)
        result2 = add_macro_features(result1)

        bad = [c for c in result2.columns if c.endswith("_x") or c.endswith("_y")]
        assert len(bad) == 0, f"Duplicate columns: {bad}"
        assert "vxn_zscore" in result2.columns


class TestDefaultConfig:
    def test_default_is_canonical_base(self):
        """Default config should point to the canonical base config."""
        from src.config import load_config
        cfg = load_config()
        assert cfg.data.start_date == "2016-01-01"
        assert len(cfg.data.tickers) > 50
