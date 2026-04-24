"""
Purged time-ordered K-Fold CV for inner hyperparameter tuning.

Outer walk-forward (src/splits/walkforward.py) handles train/test isolation across
years. This module provides INNER CV inside a single fold's training data, used
during Optuna search to score hyperparameters without overfitting one validation
window.

Key idea: split sorted unique dates into K time-ordered chunks. For fold k,
val = chunk k. Fit = all other chunks MINUS a purge buffer of `purge_days`
trading dates immediately before AND after val (to chop overlapping triple-barrier
labels at the boundary).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def purged_kfold_split(
    date_index,
    n_splits: int = 3,
    purge_days: int = 10,
):
    """Yield (fit_idx, val_idx) integer-position arrays for purged time-ordered K-fold.

    `date_index` is the per-row date array (e.g., df.index.get_level_values("date")).
    Rows are kept in their original order; integer positions are returned.
    """
    idx = pd.Index(pd.to_datetime(np.asarray(date_index)))
    unique_dates = idx.unique().sort_values()
    n_dates = len(unique_dates)

    if n_dates < n_splits + 2 * purge_days + n_splits:
        raise ValueError(
            f"purged_kfold_split: need >= {n_splits + 2 * purge_days + n_splits} unique dates, got {n_dates}"
        )

    chunk_size = n_dates // n_splits

    for k in range(n_splits):
        val_start = k * chunk_size
        val_end = (k + 1) * chunk_size if k < n_splits - 1 else n_dates

        val_dates = unique_dates[val_start:val_end]

        purge_lo = max(0, val_start - purge_days)
        purge_hi = min(n_dates, val_end + purge_days)
        purged_dates = unique_dates[purge_lo:purge_hi]

        val_mask = np.asarray(idx.isin(val_dates))
        fit_mask = ~np.asarray(idx.isin(purged_dates))

        if fit_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        yield np.where(fit_mask)[0], np.where(val_mask)[0]


def daily_auc_score(y_true, y_score, dates) -> float:
    """Mean cross-sectional AUC by date. Returns 0.5 if no valid days."""
    df = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(y_score), "d": pd.to_datetime(np.asarray(dates))})
    aucs = []
    for _, grp in df.groupby("d", sort=False):
        grp = grp.dropna(subset=["y", "s"])
        if grp["y"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(grp["y"].values, grp["s"].values))
    return float(np.mean(aucs)) if aucs else 0.5


def cv_score_classifier(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    dates,
    sample_weight: np.ndarray | None = None,
    n_splits: int = 3,
    purge_days: int = 10,
) -> float:
    """Mean inner-CV daily-AUC for a probabilistic classifier.

    `model_factory` is a zero-arg callable that returns a fresh unfitted estimator.
    """
    scores = []
    for fit_idx, val_idx in purged_kfold_split(dates, n_splits=n_splits, purge_days=purge_days):
        m = model_factory()
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight[fit_idx]
        m.fit(X[fit_idx], y[fit_idx], **fit_kwargs)
        proba = m.predict_proba(X[val_idx])[:, 1]
        scores.append(daily_auc_score(y[val_idx], proba, np.asarray(dates)[val_idx]))
    return float(np.mean(scores)) if scores else 0.5
