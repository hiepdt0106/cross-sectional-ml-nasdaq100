from __future__ import annotations

import numpy as np
import pandas as pd



def _make_panel(n_dates: int = 260, n_tickers: int = 6) -> pd.DataFrame:
    rng = np.random.RandomState(123)
    dates = pd.bdate_range("2019-01-01", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows = []
    for t_idx, ticker in enumerate(tickers):
        price = 80 + t_idx * 5
        for i, d in enumerate(dates):
            drift = 0.0005 * (t_idx + 1)
            shock = rng.randn() * 0.01
            price *= np.exp(drift + shock)
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "adj_open": price * (1 + rng.randn() * 0.002),
                    "adj_high": price * (1 + abs(rng.randn() * 0.01)),
                    "adj_low": price * (1 - abs(rng.randn() * 0.01)),
                    "adj_close": price,
                    "adj_volume": rng.randint(1_000_000, 5_000_000),
                    "vix": 18 + np.sin(i / 20) * 3 + rng.randn() * 0.2,
                    "vxn": 20 + np.cos(i / 18) * 4 + rng.randn() * 0.3,
                    "bench_close": 250 + i * 0.2 + rng.randn() * 0.5,
                    "treasury_10y": 2.0 + np.sin(i / 25) * 0.15,
                    "treasury_2y": 1.4 + np.cos(i / 22) * 0.10,
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()



def test_market_breadth_feature_created_and_bounded():
    from src.features.price import add_price_features
    from src.features.cross_sectional import add_cross_sectional_features

    df = _make_panel()
    df = add_price_features(df)
    df = add_cross_sectional_features(df)

    assert "market_breadth_200d" in df.columns
    valid = df["market_breadth_200d"].dropna()
    assert len(valid) > 0
    assert valid.between(0.0, 1.0).all()



def test_new_interaction_features_created():
    from src.features.price import add_price_features
    from src.features.volatility import add_vol_features
    from src.features.relative import add_relative_features
    from src.features.cross_sectional import add_cross_sectional_features
    from src.features.macro_features import add_macro_features
    from src.features.interactions import add_interaction_features

    df = _make_panel()
    df = add_price_features(df)
    df = add_vol_features(df)
    df = add_relative_features(df)
    df = add_cross_sectional_features(df)
    df = add_macro_features(df)
    df["p_high_vol"] = 0.5
    df = add_interaction_features(df)

    for col in ["dispersion_adjusted_vol", "cs_mom_x_yield"]:
        assert col in df.columns
        assert df[col].notna().sum() > 0



def test_skip_rank_includes_new_absolute_scale_features():
    from src.config import SKIP_RANK_FEATURES

    assert "market_breadth_200d" in SKIP_RANK_FEATURES
    assert "dispersion_adjusted_vol" in SKIP_RANK_FEATURES
    assert "cs_mom_x_yield" in SKIP_RANK_FEATURES



def test_regime_overlay_scales_exposure_in_backtest():
    from src.backtest.engine import BacktestEngineConfig, run_backtest

    dates = pd.bdate_range("2020-01-01", periods=3)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])
    df = pd.DataFrame(
        {
            "adj_open": [100, 100, 101, 101, 102, 102],
            "adj_close": [101, 101, 102, 102, 103, 103],
            "p_high_vol": [1.0, 1.0, 0.2, 0.2, 0.2, 0.2],
        },
        index=idx,
    )
    pred_idx = pd.MultiIndex.from_product([dates[:1], ["A", "B"]], names=["date", "ticker"])
    pred_df = pd.DataFrame({"y_prob": [0.9, 0.8], "model": "ENS"}, index=pred_idx)

    cfg = BacktestEngineConfig(
        top_k=2,
        rebalance_days=1,
        regime_targeting=True,
        regime_sensitivity=0.40,
        regime_min_exposure=0.60,
        regime_max_exposure=1.00,
    )
    eq, trades = run_backtest(df, pred_df, cfg)

    assert len(eq) > 0
    assert abs(eq["regime_scale"].iloc[0] - 0.60) < 1e-10
    assert abs(eq["exposure"].iloc[0] - 0.60) < 1e-10
    assert abs(trades["regime_scale"].iloc[0] - 0.60) < 1e-10



def test_vol_target_scale_reduces_exposure_for_high_vol_basket():
    from src.backtest.engine import BacktestEngineConfig, _compute_vol_target_scale

    dates = pd.bdate_range("2020-01-01", periods=80)
    prices = []
    for ticker in ["A", "B"]:
        px = 100.0
        for i, d in enumerate(dates):
            px *= 1.08 if i % 2 == 0 else 0.92
            prices.append({"date": d, "ticker": ticker, "adj_close": px})
    df = pd.DataFrame(prices).set_index(["date", "ticker"]).sort_index()
    weights = {"A": 0.5, "B": 0.5}

    cfg = BacktestEngineConfig(
        vol_targeting=True,
        target_vol=0.10,
        vol_lookback_days=63,
        vol_min_scale=0.50,
        vol_max_scale=1.50,
    )
    scale, est_vol = _compute_vol_target_scale(df, dates[-1], weights, cfg)

    assert est_vol is not None and est_vol > 0.10
    assert 0.50 <= scale < 1.0



def test_stacked_ensemble_activates_when_history_is_large_enough():
    from src.models.train import build_ensemble

    rng = np.random.RandomState(7)
    models = ["LR", "RF", "LGBM"]
    folds = [1, 2, 3, 4, 5]
    rows = []
    results_rows = []
    for fold in folds:
        dates = pd.bdate_range(f"2020-0{fold}-01", periods=5)
        tickers = [f"T{i}" for i in range(5)]
        for d in dates:
            for i, ticker in enumerate(tickers):
                latent = 0.2 * i + rng.randn() * 0.1
                y_true = int(latent > 0.4)
                for m_idx, model in enumerate(models):
                    score = latent + 0.15 * m_idx + rng.randn() * 0.05
                    rows.append(
                        {
                            "date": d,
                            "ticker": ticker,
                            "y_true": y_true,
                            "adj_close": 100 + i,
                            "y_prob": score,
                            "model": model,
                            "fold": fold,
                        }
                    )
        for model in models:
            results_rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "daily_auc": 0.55 + 0.01 * fold,
                    "rank_corr": 0.03 + 0.01 * fold,
                }
            )

    pred_df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    results_df = pd.DataFrame(results_rows)
    ens = build_ensemble(pred_df, results_df=results_df, method="stacked")

    assert len(ens) == pred_df.reset_index()[["date", "ticker"]].drop_duplicates().shape[0]
    assert np.isfinite(ens["y_prob"]).all()
    assert "ensemble_method" in ens.columns
    assert "stacked" in set(ens["ensemble_method"])
