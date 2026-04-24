from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.train import (
    _build_lgbm_training_payload,
    _return_to_grade_by_date,
    daily_rank_corr,
    top_k_return,
)


def _make_train_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=3)
    tickers = ["A", "B", "C", "D", "E"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    tb_return = pd.Series(
        [
            0.10, np.nan, 0.03, -0.02, 0.05,
            -0.10, 0.07, np.nan, 0.02, 0.01,
            0.08, np.nan, -0.01, 0.04, 0.02,
        ],
        index=idx,
        name="tb_return",
    )
    tb_label = (tb_return.fillna(0.0) > 0).astype(int)
    return pd.DataFrame({"tb_return": tb_return, "tb_label": tb_label}, index=idx)


def test_return_to_grade_by_date_handles_partial_nan() -> None:
    df = _make_train_frame()
    grades = _return_to_grade_by_date(df["tb_return"], n_grades=5)

    assert grades.index.equals(df.index)
    assert grades.dtype == np.int16
    valid = df["tb_return"].notna()
    assert not grades.loc[valid].isna().any()
    assert set(np.unique(grades.loc[valid])) <= {0, 1, 2, 3, 4}



def test_build_lgbm_training_payload_filters_nan_returns() -> None:
    df = _make_train_frame()
    X = np.arange(len(df) * 3, dtype=float).reshape(len(df), 3)
    y = df["tb_label"].to_numpy(dtype=int)
    val_mask = df.index.get_level_values("date") >= pd.Timestamp("2020-01-03")

    payload = _build_lgbm_training_payload(df, X, y, np.asarray(val_mask, dtype=bool), n_grades=5)

    valid_return = df["tb_return"].notna().to_numpy()
    fit_mask = (~np.asarray(val_mask, dtype=bool)) & valid_return
    val_rank_mask = np.asarray(val_mask, dtype=bool) & valid_return

    assert payload["use_ranker"] is True
    assert payload["X_full_train"].shape[0] == int(valid_return.sum())
    assert payload["X_fit"].shape[0] == int(fit_mask.sum())
    assert payload["X_val"].shape[0] == int(val_rank_mask.sum())
    assert not np.isnan(payload["val_returns"]).any()



def test_rank_metrics_ignore_nan_returns() -> None:
    df = _make_train_frame()
    scores = np.linspace(0.1, 0.9, len(df))

    rank_corr = daily_rank_corr(df, scores)
    topk = top_k_return(df, scores, k=3)

    assert np.isfinite(rank_corr)
    assert np.isfinite(topk)
