from __future__ import annotations

import pandas as pd


def test_get_feature_cols_excludes_aligned_label_columns():
    from src.config import get_feature_cols

    cols = [
        "mom_63d",
        "alpha_ret",
        "alpha_label",
        "alpha_ext_label",
        "tb_label",
        "p_high_vol",
    ]
    feature_cols = get_feature_cols(cols)

    assert "mom_63d" in feature_cols
    assert "p_high_vol" in feature_cols
    assert "alpha_ret" not in feature_cols
    assert "alpha_label" not in feature_cols
    assert "alpha_ext_label" not in feature_cols
    assert "tb_label" not in feature_cols



def test_build_ensemble_frame_keeps_raw_score_scale_for_flat_models():
    """When all model scores are inside [0,1], _build_ensemble_frame keeps the
    raw probability scale (no rank-normalisation) so adaptive ensemble weights
    fit on the raw blend at training time remain consistent with inference.
    Per-model rank-normalisation kicks in only when scores escape [0,1].
    See `docs/notes/post_audit_fixes.md` §11 for the design rationale.
    """
    from src.models.train import _build_ensemble_frame

    date = pd.Timestamp("2024-01-05")
    idx = pd.MultiIndex.from_tuples(
        [
            (date, "A"),
            (date, "B"),
            (date, "C"),
            (date, "A"),
            (date, "B"),
            (date, "C"),
        ],
        names=["date", "ticker"],
    )
    pred_df = pd.DataFrame(
        {
            "y_prob": [0.500, 0.501, 0.502, 0.700, 0.710, 0.720],
            "model": ["LR", "LR", "LR", "RF", "RF", "RF"],
            "fold": [1, 1, 1, 1, 1, 1],
        },
        index=idx,
    )

    frame = _build_ensemble_frame(pred_df)

    assert abs(frame.loc[(date, "A"), "LR"] - 0.500) < 1e-9
    assert abs(frame.loc[(date, "C"), "LR"] - 0.502) < 1e-9
    assert (frame["LR"].max() - frame["LR"].min()) < 0.01



def test_select_with_buffer_keeps_incumbent_when_scores_are_tiedish():
    from src.backtest.engine import _select_with_buffer

    signal_pool = pd.Series({"A": 0.610, "B": 0.605, "C": 0.598, "D": 0.590})
    selected = _select_with_buffer(
        signal_pool,
        top_k=2,
        prev_holdings={"C"},
        hold_buffer=0,
        hold_score_tolerance=0.01,
    )

    assert "C" in selected
    assert len(selected) == 2
