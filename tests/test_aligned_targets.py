"""Regression tests for forward-aligned labels and target-aware training helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_add_forward_rebalance_targets_aligns_with_t_plus_1_open_to_h_close():
    from src.labeling.forward_targets import add_forward_rebalance_targets

    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "ticker"])
    df = pd.DataFrame(
        {
            "adj_open": [10.0, 20.0, 11.0, 19.5, 12.0, 21.0, 13.0, 22.0],
            "adj_close": [10.5, 19.8, 11.5, 20.5, 12.5, 21.5, 13.5, 22.5],
        },
        index=idx,
    )

    out = add_forward_rebalance_targets(df, horizon=2, top_k=1)

    # 2024-01-01 signal enters on 2024-01-02 open and exits on 2024-01-03 close.
    aaa_ret = float(np.log(12.5 / 11.0))
    bbb_ret = float(np.log(21.5 / 19.5))
    assert np.isclose(out.loc[(dates[0], "AAA"), "alpha_ret"], aaa_ret)
    assert np.isclose(out.loc[(dates[0], "BBB"), "alpha_ret"], bbb_ret)
    assert out.loc[(dates[0], "AAA"), "alpha_label"] == 1
    assert out.loc[(dates[0], "BBB"), "alpha_label"] == 0


def test_lgbm_payload_prefers_aligned_return_column_when_available():
    from src.models.train import _build_lgbm_training_payload

    dates = pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3 + ["2024-01-03"] * 3)
    tickers = ["A", "B", "C"] * 3
    idx = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
    train_df = pd.DataFrame(
        {
            "alpha_ret": [0.03, 0.02, 0.01, -0.01, 0.04, 0.00, 0.02, -0.03, 0.01],
            "tb_return": [np.nan] * 9,
        },
        index=idx,
    )
    X = np.arange(len(train_df) * 2, dtype=float).reshape(len(train_df), 2)
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1], dtype=int)
    val_mask = np.array([False, False, False, False, False, False, True, True, True])

    payload = _build_lgbm_training_payload(
        train_df=train_df,
        X_train=X,
        y_binary=y,
        val_mask=val_mask,
        n_grades=5,
        return_col="alpha_ret",
    )

    assert payload["use_ranker"] is True
    assert payload["X_fit"].shape[0] == 6
    assert payload["X_val"].shape[0] == 3
    assert np.isfinite(payload["val_returns"]).all()
