from __future__ import annotations

import pandas as pd


def test_backtest_engine_default_is_pure_ml_without_anchor_overlay():
    from src.backtest.engine import BacktestEngineConfig, _blend_signal_with_anchor

    dates = pd.bdate_range("2024-01-02", periods=1)
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "ticker"])
    df = pd.DataFrame(
        {
            "mom_63d": [0.1, 0.5, 0.2],
            "rel_strength_63d": [0.2, 0.6, 0.1],
            "trend_strength_21d": [0.3, 0.7, 0.4],
            "price_sma200": [0.05, 0.20, 0.01],
        },
        index=idx,
    )
    signal_pool = pd.Series({"A": 0.95, "B": 0.40, "C": 0.60})

    blended, used = _blend_signal_with_anchor(df, dates[0], signal_pool, BacktestEngineConfig())

    assert used is False
    assert blended.equals(signal_pool)



def test_regime_aware_ensemble_routes_to_different_models_by_context_bucket():
    from src.models.train import build_ensemble

    low_hist_dates = pd.bdate_range("2020-01-02", periods=20)
    high_hist_dates = pd.bdate_range("2020-02-03", periods=20)
    test_dates = pd.bdate_range("2021-01-04", periods=2)
    tickers = ["A", "B"]

    rows = []

    def add_rows(date, fold, regime, winner, lr_good, rf_good):
        for ticker in tickers:
            target_return = 0.04 if ticker == winner else -0.01
            for model, good in [("LR", lr_good), ("RF", rf_good)]:
                high_score = 0.90 if good else 0.10
                low_score = 0.10 if good else 0.90
                score = high_score if ticker == winner else low_score
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "adj_close": 100.0,
                        "target_return": target_return,
                        "y_prob": score,
                        "model": model,
                        "fold": fold,
                        "p_high_vol": regime,
                    }
                )

    for d in low_hist_dates:
        add_rows(d, fold=1, regime=0.10, winner="A", lr_good=True, rf_good=False)
    for d in high_hist_dates:
        add_rows(d, fold=1, regime=0.90, winner="A", lr_good=False, rf_good=True)

    add_rows(test_dates[0], fold=2, regime=0.10, winner="A", lr_good=True, rf_good=False)
    add_rows(test_dates[1], fold=2, regime=0.90, winner="A", lr_good=False, rf_good=True)

    pred_df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    ensemble = build_ensemble(pred_df, method="adaptive_regime", top_k=1, context_col="p_high_vol")

    low_day = ensemble.xs(test_dates[0], level="date")
    high_day = ensemble.xs(test_dates[1], level="date")

    assert low_day.loc["A", "y_prob"] > low_day.loc["B", "y_prob"]
    assert high_day.loc["A", "y_prob"] > high_day.loc["B", "y_prob"]
    assert ensemble.loc[pd.IndexSlice[test_dates[0], :], "ensemble_method"].iloc[0] == "adaptive_regime"
    assert ensemble.loc[pd.IndexSlice[test_dates[1], :], "ensemble_method"].iloc[0] == "adaptive_regime"
