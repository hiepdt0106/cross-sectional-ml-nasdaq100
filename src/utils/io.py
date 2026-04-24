"""
src/utils/io.py
───────────────
Save / load parquet, preserving MultiIndex across round-trips.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame to parquet; create parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load(path: str | Path) -> pd.DataFrame:
    """Load parquet, normalize datetime index, assert no duplicates."""
    df = pd.read_parquet(Path(path))

    # Normalize datetime in the index
    if isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
        idx = df.index.to_frame(index=False)
        idx["date"] = pd.to_datetime(idx["date"]).dt.normalize()
        df.index = pd.MultiIndex.from_frame(idx)
    elif df.index.name == "date":
        df.index = pd.to_datetime(df.index).normalize()

    # Files that stack multiple models on the same (date, ticker) index
    # legitimately contain duplicate index entries — discriminated by a
    # "model" column. Skip the uniqueness check in that case.
    if "model" not in df.columns:
        assert not df.index.duplicated().any(), \
            f"File {path} has duplicate index after load!"

    return df.sort_index()


def log_return(series: pd.Series, n: int = 1) -> pd.Series:
    """n-period log return: ln(P_t / P_{t-n}). Used for price, vol, macro."""
    return np.log(series / series.shift(n))
