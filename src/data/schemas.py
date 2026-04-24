"""Pandera validation schemas for pipeline data quality gates."""
from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Index, Check

# ── Raw OHLCV per-ticker (after fetch, before alignment) ───────────
RawOHLCVSchema = DataFrameSchema(
    columns={
        "adj_open": Column(float, Check.gt(0), nullable=True),
        "adj_high": Column(float, Check.gt(0), nullable=True),
        "adj_low": Column(float, Check.gt(0), nullable=True),
        "adj_close": Column(float, Check.gt(0), nullable=True),
        "adj_volume": Column(float, Check.ge(0), nullable=True),
    },
    index=Index(pa.DateTime, name="date"),
    strict=False,
    coerce=False,
)

# ── Macro data (VIX / VXN) ─────────────────────────────────────────
MacroSchema = DataFrameSchema(
    columns={
        "vix": Column(float, Check.ge(0), nullable=True),
        "vxn": Column(float, Check.ge(0), nullable=True),
    },
    index=Index(pa.DateTime),
    strict=False,
    coerce=False,
)

# ── Aligned stock panel (MultiIndex: date × ticker) ────────────────
StockPanelSchema = DataFrameSchema(
    columns={
        "adj_open": Column(float, Check.gt(0), nullable=True),
        "adj_high": Column(float, Check.gt(0), nullable=True),
        "adj_low": Column(float, Check.gt(0), nullable=True),
        "adj_close": Column(float, Check.gt(0), nullable=True),
        "adj_volume": Column(float, Check.ge(0), nullable=True),
    },
    strict=False,
    coerce=False,
)

# ── Final dataset (after feature engineering) ───────────────────────
DatasetSchema = DataFrameSchema(
    columns={
        "adj_close": Column(float, Check.gt(0), nullable=True),
        "adj_volume": Column(float, Check.ge(0), nullable=True),
        "vix": Column(float, Check.ge(0), nullable=True),
        "vxn": Column(float, Check.ge(0), nullable=True),
    },
    strict=False,
    coerce=False,
)

# ── Predictions (model output) ─────────────────────────────────────
PredictionSchema = DataFrameSchema(
    columns={
        "y_prob": Column(float, [Check.ge(0.0), Check.le(1.0)], nullable=False),
    },
    strict=False,
    coerce=False,
)


def validate_ohlcv(df, *, lazy: bool = True):
    """Validate raw OHLCV data. Returns validated DataFrame."""
    return RawOHLCVSchema.validate(df, lazy=lazy)


def validate_macro(df, *, lazy: bool = True):
    """Validate macro data."""
    return MacroSchema.validate(df, lazy=lazy)


def validate_stock_panel(df, *, lazy: bool = True):
    """Validate aligned stock panel."""
    return StockPanelSchema.validate(df, lazy=lazy)


def validate_dataset(df, *, lazy: bool = True):
    """Validate final assembled dataset."""
    return DatasetSchema.validate(df, lazy=lazy)


def validate_predictions(df, *, lazy: bool = True):
    """Validate model predictions."""
    return PredictionSchema.validate(df, lazy=lazy)
