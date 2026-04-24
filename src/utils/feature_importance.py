"""SHAP-based feature importance across walk-forward folds.

Computes per-fold SHAP values from tree models (RF / LGBM) and aggregates
them into stability-ranked importance tables for thesis reporting.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_tree_importance(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    *,
    max_samples: int = 2000,
    seed: int = 42,
) -> pd.Series:
    """Compute mean |SHAP| values for a tree model on a sample of X.

    Falls back to built-in feature_importances_ if shap is not installed.
    """
    if len(X) == 0:
        return pd.Series(dtype=float, index=feature_names)

    rng = np.random.RandomState(seed)
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # Binary classifiers return list of 2 arrays – take class-1
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        importance = np.abs(shap_values).mean(axis=0)
    except Exception:
        log.debug("SHAP unavailable or failed — using built-in importances")
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=float)
        else:
            return pd.Series(0.0, index=feature_names)

    if len(importance) != len(feature_names):
        return pd.Series(0.0, index=feature_names)

    return pd.Series(importance, index=feature_names, name="importance")


def aggregate_fold_importance(
    fold_importances: list[pd.Series],
    fold_labels: list[str | int] | None = None,
) -> pd.DataFrame:
    """Stack per-fold importance into a DataFrame + compute stability metrics.

    Returns a DataFrame indexed by feature with columns:
        mean_importance, std_importance, cv, stability_rank, fold_1 .. fold_N
    """
    if not fold_importances:
        return pd.DataFrame()

    if fold_labels is None:
        fold_labels = [f"fold_{i+1}" for i in range(len(fold_importances))]

    wide = pd.DataFrame(
        {label: imp for label, imp in zip(fold_labels, fold_importances)}
    ).fillna(0.0)

    result = pd.DataFrame(index=wide.index)
    result["mean_importance"] = wide.mean(axis=1)
    result["std_importance"] = wide.std(axis=1)
    result["cv"] = result["std_importance"] / result["mean_importance"].clip(lower=1e-12)
    result["stability_rank"] = result["mean_importance"].rank(ascending=False).astype(int)

    for col in wide.columns:
        result[col] = wide[col]

    return result.sort_values("stability_rank")


def top_n_features(importance_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return the top-N features by mean importance."""
    return importance_df.head(n).copy()
