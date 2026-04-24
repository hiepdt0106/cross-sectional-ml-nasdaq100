from __future__ import annotations

import numpy as np
import pandas as pd


def test_forward_targets_create_extreme_labels_and_keep_middle_nan():
    from src.labeling.forward_targets import add_forward_rebalance_targets

    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    tickers = ["A", "B", "C", "D", "E", "F"]

    rows = []
    for d in dates:
        for i, ticker in enumerate(tickers):
            base = 10 + i
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "adj_open": base,
                    "adj_close": base + i,
                }
            )
    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

    out = add_forward_rebalance_targets(df, horizon=1, top_k=2, extreme_frac=0.25)
    day0 = out.xs(dates[0], level="date")

    assert "alpha_ext_label" in out.columns
    assert int((day0["alpha_ext_label"] == 1).sum()) == 1
    assert int((day0["alpha_ext_label"] == 0).sum()) == 1
    assert int(day0["alpha_ext_label"].isna().sum()) == 4



def test_walk_forward_train_scores_full_test_rows_with_sparse_classifier_target():
    from src.models.train import walk_forward_train
    from src.splits.walkforward import FoldSplit

    dates = pd.bdate_range("2019-01-02", periods=320)
    tickers = ["A", "B", "C", "D", "E", "F"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    rng = np.random.RandomState(7)
    rank = np.tile(np.arange(len(tickers)), len(dates))
    feat1 = rank + rng.randn(len(idx)) * 0.1
    feat2 = rng.randn(len(idx))
    alpha_ret = 0.02 * rank + rng.randn(len(idx)) * 0.005

    df = pd.DataFrame(
        {
            "feat1": feat1,
            "feat2": feat2,
            "adj_close": 100 + feat1,
            "alpha_ret": alpha_ret,
            "alpha_ext_label": np.nan,
        },
        index=idx,
    )

    for d in dates:
        day_idx = pd.IndexSlice[d, :]
        day_ret = df.loc[day_idx, "alpha_ret"].sort_values()
        df.loc[day_ret.index[:1], "alpha_ext_label"] = 0.0
        df.loc[day_ret.index[-1:], "alpha_ext_label"] = 1.0

    train_mask = df.index.get_level_values("date").year == 2019
    test_mask = df.index.get_level_values("date").year == 2020
    fold = FoldSplit(
        fold=1,
        test_year=2020,
        train_idx=df.index[train_mask],
        test_idx=df.index[test_mask],
        train_end=pd.Timestamp("2019-12-31"),
        purge_n=0,
        embargo_n=0,
    )

    results, preds = walk_forward_train(
        df,
        [fold],
        feature_cols=["feat1", "feat2"],
        target="alpha_ext_label",
        return_col="alpha_ret",
        top_k=2,
        n_optuna_trials=2,
    )

    assert len(results) > 0
    assert len(preds) == int(test_mask.sum()) * 3
    assert preds.groupby("model").size().nunique() == 1



def test_hold_buffer_keeps_existing_name_when_still_near_top():
    from src.backtest.engine import _select_with_buffer

    signal_pool = pd.Series({"A": 0.90, "B": 0.89, "C": 0.88, "D": 0.87, "E": 0.86})
    selected = _select_with_buffer(signal_pool, top_k=2, prev_holdings={"C"}, hold_buffer=2)
    assert selected[0] == "C"
    assert len(selected) == 2



def test_signal_anchor_blends_model_score_with_factor_anchor():
    from src.backtest.engine import BacktestEngineConfig, _blend_signal_with_anchor

    dates = pd.bdate_range("2024-01-01", periods=1)
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
    cfg = BacktestEngineConfig(signal_anchor_weight=0.50)

    blended, used = _blend_signal_with_anchor(df, dates[0], signal_pool, cfg)

    assert used is True
    assert blended["B"] > blended["C"]
    assert blended["B"] > signal_pool.rank(method="average", pct=True)["B"]
