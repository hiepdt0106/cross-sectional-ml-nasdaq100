from __future__ import annotations

import pandas as pd


def _make_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=4)
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "ticker"])

    df = pd.DataFrame(
        {
            "mom_63d": [1.0, 2.0, 3.0] * len(dates),
            "vxn_zscore": [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, -0.3, -0.3, -0.3, 0.0, 0.0, 0.0],
            "market_dispersion_21d": [0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6],
            "p_high_x_mom_63d": [0.4, 0.8, 1.2] * len(dates),
            "stress_reversal": [0.1, 0.3, -0.2, -0.2, 0.1, 0.2, 0.5, 0.6, 0.7, -0.3, 0.2, 0.4],
            "dispersion_adjusted_vol": [0.5, 0.8, 1.1] * len(dates),
        },
        index=idx,
    )
    return df


def test_split_feature_cols_treats_macro_interactions_as_macro_dependent():
    from src.config import split_feature_cols

    features = [
        "mom_63d",
        "stress_reversal",
        "dispersion_adjusted_vol",
        "p_high_x_mom_63d",
        "vxn_zscore",
    ]
    base_cols, macro_cols = split_feature_cols(features)

    assert "mom_63d" in base_cols
    assert "stress_reversal" not in base_cols
    assert "dispersion_adjusted_vol" not in base_cols
    assert "p_high_x_mom_63d" in macro_cols
    assert "vxn_zscore" in macro_cols



def test_prepare_feature_sets_drops_raw_date_level_context_from_stock_selector():
    from src.config import prepare_feature_sets

    df = _make_panel()
    feature_cols = list(df.columns)
    sets = prepare_feature_sets(df, feature_cols)

    assert "mom_63d" in sets["base_cols"]
    assert "p_high_x_mom_63d" in sets["macro_cols"]
    assert "stress_reversal" in sets["macro_cols"]
    assert "dispersion_adjusted_vol" in sets["macro_cols"]

    assert "vxn_zscore" in sets["raw_date_cols"]
    assert "market_dispersion_21d" in sets["raw_date_cols"]

    assert "vxn_zscore" not in sets["full_cols"]
    assert "market_dispersion_21d" not in sets["full_cols"]
    assert "vxn_zscore" not in sets["base_cols"]
