"""Walk-forward model training and evaluation."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config import SKIP_RANK_FEATURES
from src.models.sample_weights import avg_uniqueness
from src.splits.walkforward import FoldSplit

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def get_models(scale_pos_weight: float = 1.0) -> dict:
    """Build model templates used in each fold."""
    from lightgbm import LGBMClassifier

    models = {
        "LR": LogisticRegression(
            max_iter=1000,
            C=0.1,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
        "RF": RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=20,
            max_features=0.5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LGBM": LGBMClassifier(
            objective="binary",
            n_estimators=800,
            learning_rate=0.05,
            is_unbalance=False,
            verbose=-1,
            random_state=42,
            n_jobs=-1,
        ),
    }
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_raw(X: np.ndarray, medians: np.ndarray = None):
    """Inf → NaN → median impute. Returns (X_clean, medians)."""
    X = np.where(np.isinf(X), np.nan, X)
    if medians is None:
        medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = medians[j]
    return X, medians


_SKIP_RANK = SKIP_RANK_FEATURES


def cross_sectional_rank(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Apply per-date cross-sectional ranks to ticker-specific features."""
    df = df.copy()
    for col in feature_cols:
        if col in _SKIP_RANK:
            continue
        df[col] = df.groupby(level="date")[col].rank(pct=True)
    return df


def _preprocess_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare ranked inputs for tree models and scaled inputs for LR."""
    # Cross-sectional rank for tree models.
    train_ranked = cross_sectional_rank(train_df, feature_cols)
    test_ranked = cross_sectional_rank(test_df, feature_cols)

    X_train_rank = train_ranked[feature_cols].values
    X_test_rank = test_ranked[feature_cols].values

    X_train_rank, med_r = _clean_raw(X_train_rank.copy())
    X_test_rank, _ = _clean_raw(X_test_rank.copy(), med_r)

    # Standard scaling for logistic regression.
    X_train_raw = train_df[feature_cols].values.copy()
    X_test_raw = test_df[feature_cols].values.copy()

    X_train_raw, med = _clean_raw(X_train_raw)
    X_test_raw, _ = _clean_raw(X_test_raw, med)

    # Clip ±5 std
    means = X_train_raw.mean(axis=0)
    stds = X_train_raw.std(axis=0)
    stds[stds == 0] = 1
    X_train_raw = np.clip(X_train_raw, means - 5 * stds, means + 5 * stds)
    X_test_raw = np.clip(X_test_raw, means - 5 * stds, means + 5 * stds)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return X_train_rank, X_test_rank, X_train_scaled, X_test_scaled


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def daily_auc(
    test_df: pd.DataFrame,
    y_prob: np.ndarray,
    target_col: str = "tb_label",
) -> float:
    """Mean cross-sectional AUC by date."""
    if target_col not in test_df.columns:
        return 0.5

    tmp = test_df[[target_col]].copy()
    tmp["prob"] = y_prob

    aucs = []
    for _, grp in tmp.groupby(level="date"):
        grp = grp.dropna(subset=[target_col, "prob"])
        y = grp[target_col].values
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, grp["prob"].values))

    return np.mean(aucs) if aucs else 0.5


def daily_rank_corr(
    test_df: pd.DataFrame,
    y_prob: np.ndarray,
    return_col: str = "tb_return",
) -> float:
    """Mean daily Spearman rank correlation between score and future return."""
    if return_col not in test_df.columns:
        return 0.0

    tmp = test_df[[return_col]].copy()
    tmp["prob"] = y_prob

    corrs = []
    for _, grp in tmp.groupby(level="date"):
        grp = grp.dropna(subset=[return_col, "prob"])
        if len(grp) < 5:
            continue
        r = grp[return_col].values
        p = grp["prob"].values
        from scipy.stats import spearmanr

        corr, _ = spearmanr(r, p)
        if np.isfinite(corr):
            corrs.append(corr)

    return np.mean(corrs) if corrs else 0.0


def top_k_return(
    test_df: pd.DataFrame,
    y_prob: np.ndarray,
    k: int = 5,
    return_col: str = "tb_return",
) -> float:
    """Mean daily forward return of the top-K scored names."""
    if return_col not in test_df.columns:
        return 0.0

    tmp = test_df[[return_col]].copy()
    tmp["prob"] = y_prob

    daily_rets = []
    for _, grp in tmp.groupby(level="date"):
        grp = grp.dropna(subset=[return_col, "prob"])
        if len(grp) < k:
            continue
        top = grp.nlargest(k, "prob")
        daily_rets.append(top[return_col].mean())

    return np.mean(daily_rets) if daily_rets else 0.0


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC with a guard for when the label vector has only one class."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def _daily_auc_from_arrays(
    y_true: np.ndarray,
    y_score: np.ndarray,
    dates: pd.Index | np.ndarray,
) -> float:
    """Daily AUC from array inputs."""
    tmp = pd.DataFrame(
        {
            "date": pd.to_datetime(np.asarray(dates)),
            "y": np.asarray(y_true),
            "score": np.asarray(y_score),
        }
    )

    aucs = []
    for _, grp in tmp.groupby("date", sort=True):
        grp = grp.dropna(subset=["y", "score"])
        if grp["y"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(grp["y"].values, grp["score"].values))

    return float(np.mean(aucs)) if aucs else 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# LIGHTGBM + OPTUNA
# ═══════════════════════════════════════════════════════════════════════════════

def _rank_normalize_by_date(scores: pd.Series) -> pd.Series:
    """Percentile-rank normalize score per date."""
    return scores.groupby(level="date").rank(pct=True)


def _daily_rank_corr_from_arrays(
    y_true: np.ndarray,
    y_score: np.ndarray,
    dates: pd.Index | np.ndarray,
) -> float:
    """Mean daily Spearman correlation from array inputs."""
    from scipy.stats import spearmanr

    tmp = pd.DataFrame(
        {
            "date": pd.to_datetime(np.asarray(dates)),
            "y": np.asarray(y_true),
            "score": np.asarray(y_score),
        }
    )

    corrs = []
    for _, grp in tmp.groupby("date", sort=True):
        grp = grp.dropna(subset=["y", "score"])
        if len(grp) < 5 or grp["y"].nunique() < 2:
            continue
        corr, _ = spearmanr(grp["y"].values, grp["score"].values)
        if np.isfinite(corr):
            corrs.append(float(corr))

    return float(np.mean(corrs)) if corrs else 0.0


def _top_k_return_from_arrays(
    future_returns: np.ndarray,
    y_score: np.ndarray,
    dates: pd.Index | np.ndarray,
    k: int,
) -> float:
    """Mean daily top-K forward return from array inputs."""
    tmp = pd.DataFrame(
        {
            "date": pd.to_datetime(np.asarray(dates)),
            "ret": np.asarray(future_returns),
            "score": np.asarray(y_score),
        }
    )
    vals = []
    for _, grp in tmp.groupby("date", sort=True):
        grp = grp.dropna(subset=["ret", "score"])
        if len(grp) < k:
            continue
        vals.append(float(grp.nlargest(k, "score")["ret"].mean()))
    return float(np.mean(vals)) if vals else 0.0


def _make_ranker_groups(dates_arr: pd.Index | np.ndarray) -> np.ndarray:
    """
    Build LightGBM ranker groups from an ORDERED date array.

    Important: group sizes must follow row order. Using groupby().size() on an
    unsorted array can silently corrupt query grouping.
    """
    dates = pd.Index(pd.to_datetime(np.asarray(dates_arr)))
    if len(dates) == 0:
        return np.array([], dtype=int)
    groups: list[int] = []
    current = 1
    prev = dates[0]
    for d in dates[1:]:
        if d == prev:
            current += 1
        else:
            groups.append(current)
            current = 1
            prev = d
    groups.append(current)
    return np.asarray(groups, dtype=int)


def _return_to_grade_by_date(series: pd.Series, n_grades: int = 5) -> pd.Series:
    """Map forward returns to deterministic percentile grades by date."""
    if n_grades <= 0:
        raise ValueError("n_grades must be > 0")

    def _one_day(x: pd.Series) -> pd.Series:
        out = pd.Series(0, index=x.index, dtype=np.int16)
        valid = x.notna()
        if valid.sum() <= 1:
            return out
        ranks = x.loc[valid].rank(method="first", pct=True)
        grades = np.floor((ranks.to_numpy(dtype=float) - 1e-12) * n_grades)
        grades = np.clip(grades, 0, n_grades - 1).astype(np.int16)
        out.loc[valid] = grades
        return out

    graded = series.groupby(level="date", group_keys=False).apply(_one_day)
    return graded.astype(np.int16)


def _build_lgbm_training_payload(
    train_df: pd.DataFrame,
    X_train: np.ndarray,
    y_binary: np.ndarray,
    val_mask: np.ndarray,
    fit_mask_base: np.ndarray | None = None,
    n_grades: int = 5,
    return_col: str = "tb_return",
    sample_weight: np.ndarray | None = None,
) -> dict[str, object]:
    """Build classifier or ranker inputs for the LightGBM fold."""
    date_vals = train_df.index.get_level_values("date")
    y_binary = np.asarray(y_binary, dtype=float)
    cls_valid = np.isfinite(y_binary)
    if fit_mask_base is None:
        fit_mask_base = ~val_mask
    fit_mask_base = np.asarray(fit_mask_base, dtype=bool)
    fit_cls_mask = fit_mask_base & cls_valid
    val_cls_mask = val_mask & cls_valid
    w_all = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    payload: dict[str, object] = {
        "use_ranker": False,
        "classifier_valid": False,
        "X_fit": X_train[fit_cls_mask],
        "X_val": X_train[val_cls_mask],
        "X_full_train": X_train[cls_valid],
        "y_fit": y_binary[fit_cls_mask].astype(int, copy=False),
        "y_val": y_binary[val_cls_mask].astype(int, copy=False),
        "y_full_train": y_binary[cls_valid].astype(int, copy=False),
        "train_dates_fit": date_vals[fit_cls_mask],
        "train_dates_full": date_vals[cls_valid],
        "val_dates": date_vals[val_cls_mask],
        "val_returns": None,
        "w_fit": None if w_all is None else w_all[fit_cls_mask],
        "w_full_train": None if w_all is None else w_all[cls_valid],
    }

    if (
        fit_cls_mask.sum() > 0
        and val_cls_mask.sum() > 0
        and pd.Index(date_vals[fit_cls_mask]).nunique() >= 2
        and pd.Index(date_vals[val_cls_mask]).nunique() >= 1
        and np.unique(payload["y_fit"]).size >= 2
        and np.unique(payload["y_val"]).size >= 2
    ):
        payload["classifier_valid"] = True

    if return_col not in train_df.columns:
        log.warning("  %s not available -> fallback LGBMClassifier", return_col)
        return payload

    valid_return = train_df[return_col].notna().to_numpy()
    fit_mask = fit_mask_base & valid_return
    val_rank_mask = val_mask & valid_return

    if fit_mask.sum() == 0 or val_rank_mask.sum() == 0:
        log.warning("  %s missing in train/validation -> fallback LGBMClassifier", return_col)
        return payload

    if pd.Index(date_vals[fit_mask]).nunique() < 2 or pd.Index(date_vals[val_rank_mask]).nunique() < 1:
        log.warning("  Not enough valid dates for ranker -> fallback LGBMClassifier")
        return payload

    grades = _return_to_grade_by_date(train_df[return_col], n_grades=n_grades).to_numpy(dtype=np.int16)
    returns = train_df[return_col].to_numpy(dtype=float)

    payload.update(
        {
            "use_ranker": True,
            "X_fit": X_train[fit_mask],
            "X_val": X_train[val_rank_mask],
            "X_full_train": X_train[valid_return],
            "y_fit": grades[fit_mask],
            "y_val": grades[val_rank_mask],
            "y_full_train": grades[valid_return],
            "train_dates_fit": date_vals[fit_mask],
            "train_dates_full": date_vals[valid_return],
            "val_dates": date_vals[val_rank_mask],
            "val_returns": returns[val_rank_mask],
            "w_fit": None if w_all is None else w_all[fit_mask],
            "w_full_train": None if w_all is None else w_all[valid_return],
        }
    )
    return payload


def fit_lgbm_optuna(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_dates: pd.Index | np.ndarray,
    X_full_train: np.ndarray,
    y_full_train: np.ndarray,
    train_dates_fit: pd.Index | np.ndarray | None = None,
    train_dates_full: pd.Index | np.ndarray | None = None,
    scale_pos_weight: float = 1.0,
    n_trials: int = 35,
    random_state: int = 42,
    use_ranker: bool = True,
    val_returns: np.ndarray | None = None,
    top_k_eval: int = 10,
    sample_weight_fit: np.ndarray | None = None,
    sample_weight_full: np.ndarray | None = None,
) -> tuple:
    """
    Optuna-tuned LightGBM per fold.

    Performance/stability fixes:
      1. Ranker queries are built from row-order-aware groups.
      2. Validation objective for ranker aligns with strategy goal
         (daily rank-correlation on future returns), not binary AUC on a coarse
         top-tercile label.
      3. Final refit uses the validated best_iteration instead of forcing
         n_estimators >= 100, which could materially change the chosen model.
      4. Fallback path works even when Optuna is unavailable.
      5. LightGBM uses n_jobs=1 for deterministic, sandbox-safe execution.
    """
    try:
        import optuna
    except ImportError:
        optuna = None
    from lightgbm import LGBMClassifier, LGBMRanker

    _ES_PATIENCE = 80
    _MIN_TREES = 10

    if use_ranker:
        groups_fit = _make_ranker_groups(train_dates_fit) if train_dates_fit is not None else None
        groups_val = _make_ranker_groups(val_dates)
        groups_full = _make_ranker_groups(train_dates_full) if train_dates_full is not None else None
        if groups_fit is None or groups_full is None or groups_fit.sum() != len(X_fit) or groups_full.sum() != len(X_full_train):
            log.warning("  Ranker groups invalid — fallback LGBMClassifier")
            use_ranker = False
    else:
        groups_fit = groups_val = groups_full = None

    if optuna is None:
        use_ranker = bool(use_ranker and groups_full is not None)
        if use_ranker:
            final_model = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                eval_at=[5, 10],
                n_estimators=120,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=30,
                feature_fraction=0.75,
                bagging_fraction=0.80,
                bagging_freq=5,
                lambda_l1=1.0,
                lambda_l2=1.0,
                verbose=-1,
                random_state=random_state,
                n_jobs=1,
            )
            final_model.fit(X_full_train, y_full_train, group=groups_full, sample_weight=sample_weight_full)
            info = {
                "best_params": {},
                "val_metric": np.nan,
                "val_rank_corr": np.nan,
                "val_top_k_ret": np.nan,
                "n_trials": 0,
                "final_n_trees": 120,
                "best_trial_trees": 120,
                "model_type": "LGBMRanker_fallback_no_optuna",
                "metric_name": "rank_corr",
            }
        else:
            final_model = LGBMClassifier(
                objective="binary",
                metric="binary_logloss",
                n_estimators=120,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=30,
                feature_fraction=0.75,
                bagging_fraction=0.80,
                bagging_freq=5,
                lambda_l1=1.0,
                lambda_l2=1.0,
                is_unbalance=False,
                verbose=-1,
                random_state=random_state,
                n_jobs=1,
            )
            final_model.fit(X_full_train, y_full_train, sample_weight=sample_weight_full)
            info = {
                "best_params": {},
                "val_metric": np.nan,
                "val_rank_corr": np.nan,
                "val_top_k_ret": np.nan,
                "n_trials": 0,
                "final_n_trees": 120,
                "best_trial_trees": 120,
                "model_type": "LGBMClassifier_fallback_no_optuna",
                "metric_name": "daily_auc",
            }
        return final_model, info

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        n_est = trial.suggest_int("n_estimators", 100, 600, step=50)
        common_params = {
            "n_estimators": n_est,
            "verbose": -1,
            "random_state": random_state,
            "n_jobs": 1,
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.55, 0.90),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
            "bagging_freq": 5,
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        }

        if use_ranker:
            model = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                eval_at=[5, 10],
                **common_params,
            )
            model.fit(
                X_fit,
                y_fit,
                group=groups_fit,
                eval_set=[(X_val, y_val)],
                eval_group=[groups_val],
                sample_weight=sample_weight_fit,
                callbacks=[
                    _lgbm_early_stopping(_ES_PATIENCE),
                    _lgbm_log_evaluation(-1),
                ],
            )
            val_score = model.predict(X_val)
            val_rank_corr = _daily_rank_corr_from_arrays(
                val_returns if val_returns is not None else y_val,
                val_score,
                val_dates,
            )
            val_top_k = _top_k_return_from_arrays(
                val_returns if val_returns is not None else y_val,
                val_score,
                val_dates,
                k=max(1, top_k_eval),
            )
            metric_name = "rank_corr"
            objective_value = float(val_rank_corr)
        else:
            model = LGBMClassifier(
                objective="binary",
                metric="binary_logloss",
                is_unbalance=False,
                **common_params,
            )
            model.fit(
                X_fit,
                y_fit,
                eval_set=[(X_val, y_val)],
                sample_weight=sample_weight_fit,
                callbacks=[
                    _lgbm_early_stopping(_ES_PATIENCE),
                    _lgbm_log_evaluation(-1),
                ],
            )
            val_score = model.predict_proba(X_val)[:, 1]
            val_rank_corr = np.nan
            val_top_k = np.nan
            metric_name = "daily_auc"
            objective_value = _daily_auc_from_arrays(y_val, val_score, val_dates)

        actual_trees = model.best_iteration_ if model.best_iteration_ else n_est
        actual_trees = int(max(actual_trees, _MIN_TREES))
        trial.set_user_attr("actual_trees", actual_trees)
        trial.set_user_attr("metric_name", metric_name)
        trial.set_user_attr("val_rank_corr", None if pd.isna(val_rank_corr) else float(val_rank_corr))
        trial.set_user_attr("val_top_k_ret", None if pd.isna(val_top_k) else float(val_top_k))
        return float(objective_value)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_metric = float(best_trial.value)
    best_actual_trees = int(best_trial.user_attrs.get("actual_trees", best_params.get("n_estimators", 120)))
    metric_name = best_trial.user_attrs.get("metric_name", "rank_corr" if use_ranker else "daily_auc")
    best_val_rank_corr = best_trial.user_attrs.get("val_rank_corr")
    best_val_top_k = best_trial.user_attrs.get("val_top_k_ret")

    log.info(
        "    LGBM %s Optuna: best %s=%.4f after %d trials",
        "Ranker" if use_ranker else "Classifier",
        metric_name,
        best_metric,
        len(study.trials),
    )
    log.info("    Best params: %s", best_params)

    best_n_trees = int(max(best_actual_trees, _MIN_TREES))
    best_params_clean = {k: v for k, v in best_params.items() if k != "n_estimators"}

    if use_ranker:
        final_model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5, 10],
            n_estimators=best_n_trees,
            verbose=-1,
            random_state=random_state,
            n_jobs=1,
            bagging_freq=5,
            **best_params_clean,
        )
        final_model.fit(X_full_train, y_full_train, group=groups_full, sample_weight=sample_weight_full)
    else:
        final_model = LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            n_estimators=best_n_trees,
            is_unbalance=False,
            verbose=-1,
            random_state=random_state,
            n_jobs=1,
            bagging_freq=5,
            **best_params_clean,
        )
        final_model.fit(X_full_train, y_full_train, sample_weight=sample_weight_full)

    info = {
        "best_params": best_params,
        "val_metric": best_metric,
        "val_rank_corr": best_val_rank_corr,
        "val_top_k_ret": best_val_top_k,
        "n_trials": len(study.trials),
        "final_n_trees": best_n_trees,
        "best_trial_trees": best_actual_trees,
        "model_type": "LGBMRanker" if use_ranker else "LGBMClassifier",
        "metric_name": metric_name,
    }

    del study
    return final_model, info


def _lgbm_early_stopping(stopping_rounds: int):
    """LightGBM early stopping callback."""
    from lightgbm import early_stopping
    return early_stopping(stopping_rounds=stopping_rounds, verbose=False)


def _lgbm_log_evaluation(period: int):
    """LightGBM log evaluation callback."""
    from lightgbm import log_evaluation
    return log_evaluation(period=period)


# ═══════════════════════════════════════════════════════════════════════════════
# FOLD RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    fold: int
    test_year: int
    model_name: str
    global_auc: float
    daily_auc: float
    top_k_ret: float
    rank_corr: float
    train_size: int
    test_size: int


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD LOOP
# ═══════════════════════════════════════════════════════════════════════════════

_USE_RANK = {"RF", "LGBM"}
_USE_SCALE = {"LR"}


def _make_date_block_val_mask(
    date_index: pd.Index, purge_days: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Single train/val split aligned to date boundaries, with intra-fold purge.

    Returns (fit_mask, val_mask). ``fit_mask`` excludes the last ``purge_days``
    unique dates before ``val_start`` to prevent label leakage from overlapping
    triple-barrier windows into the tuner's validation block.
    """
    idx = pd.Index(date_index)
    train_dates_sorted = idx.unique().sort_values()
    n_dates = len(train_dates_sorted)
    if n_dates < 2:
        empty = np.zeros(len(idx), dtype=bool)
        return empty, empty

    if n_dates < 8:
        val_split_idx = max(1, int(n_dates * 0.8))
    else:
        val_split_idx = int(n_dates * 0.85)
    val_split_idx = min(max(val_split_idx, 1), n_dates - 1)

    val_start_date = train_dates_sorted[val_split_idx]
    purge_start_idx = max(0, val_split_idx - int(purge_days))
    purge_start_date = train_dates_sorted[purge_start_idx]

    val_mask = np.asarray(idx >= val_start_date, dtype=bool)
    fit_mask = np.asarray(idx < purge_start_date, dtype=bool)
    return fit_mask, val_mask


def walk_forward_train(
    df: pd.DataFrame,
    splits: list[FoldSplit],
    feature_cols: list[str],
    target: str = "tb_label",
    return_col: str = "tb_return",
    top_k: int = 5,
    n_optuna_trials: int = 35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward training for LR, RF, and LightGBM."""
    all_results: list[FoldResult] = []
    all_preds: list[pd.DataFrame] = []

    for fold in splits:
        log.info(f"{'=' * 60}")
        log.info(f"Fold {fold.fold}: test {fold.test_year}")
        log.info(f"{'=' * 60}")

        train_full_df = df.loc[fold.train_idx].copy().sort_index()
        test_full_df = df.loc[fold.test_idx].copy().sort_index()
        train_cls_df = train_full_df.dropna(subset=[target]).copy()

        if len(train_cls_df) == 0 or len(test_full_df) == 0:
            log.warning(f"  Fold {fold.fold}: skip — empty after filtering")
            continue

        y_train = pd.to_numeric(train_cls_df[target], errors="coerce").values.astype(int)
        y_train_full = pd.to_numeric(train_full_df[target], errors="coerce").to_numpy(dtype=float)

        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        spw = n_neg / max(n_pos, 1)

        log.info(
            f"  Train labeled: {len(train_cls_df):,} (pos={y_train.mean():.1%}) | "
            f"Train scored: {len(train_full_df):,} | Test scored: {len(test_full_df):,}"
        )

        X_tr_rank_full, X_te_rank_full, X_tr_scaled_full, X_te_scaled_full = _preprocess_fold(
            train_full_df,
            test_full_df,
            feature_cols,
        )

        # López de Prado avg_uniqueness weights from triple-barrier t1.
        # Down-weights overlapping labels; improves signal-to-noise and stability
        # (validated 2026-04-13: +35 bps OOS AUC mean, -83 bps OOS std over 3 test years).
        if "t1" in train_full_df.columns:
            w_series = avg_uniqueness(train_full_df, t1_col="t1")
            w_binary = w_series.to_numpy(dtype=float)
            w_mean = np.nanmean(w_binary) if np.isfinite(w_binary).any() else 1.0
            w_binary = np.where(np.isfinite(w_binary), w_binary, w_mean)
            if w_mean > 0:
                w_binary = w_binary / w_mean
        else:
            w_binary = None
        train_cls_mask = train_full_df[target].notna().to_numpy()
        X_tr_rank_cls = X_tr_rank_full[train_cls_mask]
        X_tr_scaled_cls = X_tr_scaled_full[train_cls_mask]
        X_te_rank_cls = X_te_rank_full
        X_te_scaled_cls = X_te_scaled_full

        models = get_models(scale_pos_weight=spw)
        fit_mask_full, val_mask_full = _make_date_block_val_mask(
            train_full_df.index.get_level_values("date"), purge_days=10
        )

        for name, model in models.items():
            if name == "LGBM":
                payload = _build_lgbm_training_payload(
                    train_df=train_full_df,
                    X_train=X_tr_rank_full,
                    y_binary=y_train_full,
                    val_mask=np.asarray(val_mask_full, dtype=bool),
                    fit_mask_base=np.asarray(fit_mask_full, dtype=bool),
                    n_grades=5,
                    return_col=return_col,
                    sample_weight=w_binary,
                )

                if not payload.get("use_ranker") and not payload.get("classifier_valid"):
                    log.warning("  LGBM: skip — no valid ranker or classifier payload")
                    continue

                lgbm_model, lgbm_info = fit_lgbm_optuna(
                    X_fit=payload["X_fit"],
                    y_fit=payload["y_fit"],
                    X_val=payload["X_val"],
                    y_val=payload["y_val"],
                    val_dates=payload["val_dates"],
                    X_full_train=payload["X_full_train"],
                    y_full_train=payload["y_full_train"],
                    train_dates_fit=payload["train_dates_fit"],
                    train_dates_full=payload["train_dates_full"],
                    scale_pos_weight=spw,
                    n_trials=n_optuna_trials,
                    random_state=42,
                    use_ranker=bool(payload["use_ranker"]),
                    val_returns=payload["val_returns"],
                    top_k_eval=top_k,
                    sample_weight_fit=payload.get("w_fit"),
                    sample_weight_full=payload.get("w_full_train"),
                )
                model = lgbm_model
                _use_ranker = bool(payload["use_ranker"])
                X_te = X_te_rank_full
                train_size_used = int(len(payload["X_full_train"]))

                metric_name = lgbm_info.get("metric_name", "val_metric")
                metric_val = lgbm_info.get("val_metric", np.nan)
                extra_rank = lgbm_info.get("val_rank_corr", np.nan)
                extra_topk = lgbm_info.get("val_top_k_ret", np.nan)
                log.info(
                    "    LGBM (%s): %s=%.4f  val_rank_corr=%s  val_top%d_ret=%s  trees=%d  trials=%d",
                    lgbm_info.get("model_type", "?"),
                    metric_name,
                    metric_val,
                    f"{extra_rank:.4f}" if pd.notna(extra_rank) else "nan",
                    top_k,
                    f"{extra_topk:.4f}" if pd.notna(extra_topk) else "nan",
                    lgbm_info["final_n_trees"],
                    lgbm_info["n_trials"],
                )

                # Ranker output: relevance score (predict), not probability
                if _use_ranker:
                    y_prob = model.predict(X_te)
                    y_prob = _rank_normalize_by_date(pd.Series(y_prob, index=test_full_df.index)).values
                else:
                    y_prob = model.predict_proba(X_te)[:, 1]
            else:
                if name in _USE_RANK:
                    X_tr, X_te = X_tr_rank_cls, X_te_rank_cls
                else:
                    X_tr, X_te = X_tr_scaled_cls, X_te_scaled_cls
                model.fit(X_tr, y_train)
                y_prob = model.predict_proba(X_te)[:, 1]
                train_size_used = len(X_tr)

            test_target = pd.to_numeric(test_full_df[target], errors="coerce")
            auc_mask = test_target.notna().to_numpy()
            y_test = test_target[auc_mask].astype(int).to_numpy()
            if len(np.unique(y_test)) < 2:
                g_auc = np.nan
            else:
                g_auc = roc_auc_score(y_test, y_prob[auc_mask])
            d_auc = daily_auc(test_full_df, y_prob, target_col=target)
            tk_ret = top_k_return(test_full_df, y_prob, k=top_k, return_col=return_col)
            rank_cor = daily_rank_corr(test_full_df, y_prob, return_col=return_col)

            result = FoldResult(
                fold=fold.fold,
                test_year=fold.test_year,
                model_name=name,
                global_auc=g_auc,
                daily_auc=d_auc,
                top_k_ret=tk_ret,
                rank_corr=rank_cor,
                train_size=train_size_used,
                test_size=len(X_te),
            )
            all_results.append(result)

            meta_cols = ["adj_close"]
            context_cols = [
                c
                for c in ["p_high_vol", "market_breadth_200d", "vxn_zscore", "yield_spread_zscore"]
                if c in test_full_df.columns
            ]
            pred_df = test_full_df[meta_cols + context_cols].copy()
            pred_df["y_true"] = test_target.values
            pred_df["target_label"] = test_target.values
            if return_col in test_full_df.columns:
                pred_df["target_return"] = test_full_df[return_col].values
            pred_df["y_prob"] = y_prob
            pred_df["model"] = name
            pred_df["fold"] = fold.fold
            all_preds.append(pred_df)

            log.info(
                f"  {name:4s}: global_auc={g_auc:.3f}  "
                f"daily_auc={d_auc:.3f}  rank_corr={rank_cor:+.4f}  "
                f"top{top_k}_ret={tk_ret:.4f}"
            )

    results_df = pd.DataFrame(
        [
            {
                "fold": r.fold,
                "test_year": r.test_year,
                "model": r.model_name,
                "global_auc": r.global_auc,
                "daily_auc": r.daily_auc,
                "top_k_ret": r.top_k_ret,
                "rank_corr": r.rank_corr,
                "train_size": r.train_size,
                "test_size": r.test_size,
            }
            for r in all_results
        ]
    )
    pred_all_df = pd.concat(all_preds)

    log.info(f"\n✓ Done: {len(results_df)} fold×model results")
    return results_df, pred_all_df


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def _adaptive_fold_weights(
    history: pd.DataFrame | None,
    models: list[str],
) -> dict[str, float]:
    """Compute sparse adaptive ensemble weights from prior OOS fold performance."""
    if history is None or len(history) == 0:
        base = {"LR": 0.30, "RF": 0.45, "LGBM": 0.25}
        w = {m: base.get(m, 1.0 / max(len(models), 1)) for m in models}
    else:
        hist = history[history["model"].isin(models)].copy()
        if len(hist) == 0:
            return _adaptive_fold_weights(None, models)

        has_topk = "top_k_ret" in hist.columns
        topk_scale = float(hist["top_k_ret"].abs().median()) if has_topk else 0.0
        if not np.isfinite(topk_scale) or topk_scale < 1e-5:
            topk_scale = 1e-3

        hist["score_auc"] = ((hist["daily_auc"] - 0.5) / 0.02).clip(lower=0)
        hist["score_ic"] = (hist["rank_corr"] / 0.02).clip(lower=0)
        hist["score_topk"] = (hist["top_k_ret"] / topk_scale).clip(lower=0) if has_topk else 0.0
        hist["score"] = (
            0.60 * hist["score_topk"]
            + 0.25 * hist["score_ic"]
            + 0.15 * hist["score_auc"]
        )
        grouped = hist.groupby("model")["score"].mean()
        w = {m: float(grouped.get(m, 0.0)) for m in models}
        if sum(w.values()) <= 0:
            return _adaptive_fold_weights(None, models)

        w_series = pd.Series(w, dtype=float).clip(lower=0.0)
        w_series = w_series[w_series > 0]
        if len(w_series) == 0:
            return _adaptive_fold_weights(None, models)

        ordered = w_series.sort_values(ascending=False)
        if len(ordered) >= 2 and float(ordered.iloc[0]) >= 1.25 * max(float(ordered.iloc[1]), 1e-12):
            leader = ordered.index[0]
            w = {m: 1.0 if m == leader else 0.0 for m in models}
        else:
            keep_mask = ordered >= max(float(ordered.iloc[0]) * 0.85, 1e-12)
            kept = ordered.where(keep_mask, 0.0)
            if float(kept.sum()) <= 0:
                return _adaptive_fold_weights(None, models)
            w = {m: float(kept.get(m, 0.0)) for m in models}
    total = sum(w.values())
    return {m: float(v / total) for m, v in w.items()}



def _build_ensemble_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (date, ticker) with raw model scores in wide format."""
    temp = pred_df.reset_index().copy()
    temp["ensemble_score"] = pd.to_numeric(temp["y_prob"], errors="coerce")
    in_unit_interval = temp["ensemble_score"].between(0.0, 1.0) | temp["ensemble_score"].isna()
    if not bool(in_unit_interval.all()):
        temp["ensemble_score"] = temp.groupby(["model", "date"], sort=False)["ensemble_score"].transform(
            lambda x: x.rank(method="average", pct=True)
        )
    else:
        temp["ensemble_score"] = temp["ensemble_score"].clip(0.0, 1.0)

    wide = temp.pivot_table(
        index=["date", "ticker"],
        columns="model",
        values="ensemble_score",
        aggfunc="first",
    )
    wide.columns = [str(c) for c in wide.columns]

    meta_cols = [
        "y_true",
        "target_label",
        "target_return",
        "adj_close",
        "fold",
        "p_high_vol",
        "market_breadth_200d",
        "vxn_zscore",
        "yield_spread_zscore",
    ]
    meta_aggs = {c: "first" for c in meta_cols if c in temp.columns}
    meta = temp.groupby(["date", "ticker"], sort=False).agg(meta_aggs) if meta_aggs else pd.DataFrame(index=wide.index)
    return wide.join(meta, how="left").sort_index()



def _stacking_features(frame: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Build a compact stacking design matrix from base-model scores."""
    X = frame[model_cols].copy().fillna(0.5)
    X["score_mean"] = X.mean(axis=1)
    X["score_std"] = X.std(axis=1).fillna(0.0)
    X["score_max"] = X.max(axis=1)
    X["score_min"] = X.min(axis=1)
    return X



def _fit_stacking_model(
    history_frame: pd.DataFrame,
    model_cols: list[str],
    label_col: str = "y_true",
) -> LogisticRegression | None:
    """Fit a leakage-safe meta learner on prior OOS folds only."""
    if len(history_frame) == 0 or label_col not in history_frame.columns:
        return None

    hist = history_frame.dropna(subset=[label_col]).copy()
    min_rows = max(100, 20 * max(len(model_cols), 1))
    if len(hist) < min_rows:
        return None

    y = hist[label_col].astype(int)
    if y.nunique() < 2:
        return None

    X = _stacking_features(hist, model_cols)
    sample_weight = None
    if "fold" in hist.columns and hist["fold"].notna().any():
        fold_vals = pd.to_numeric(hist["fold"], errors="coerce").fillna(0.0)
        denom = max(float(fold_vals.max() - fold_vals.min()), 1.0)
        sample_weight = 1.0 + (fold_vals - fold_vals.min()) / denom

    model = LogisticRegression(
        max_iter=1000,
        C=0.5,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X.values, y.values, sample_weight=None if sample_weight is None else sample_weight.values)
    return model



def _adaptive_weighted_scores(
    frame: pd.DataFrame,
    model_cols: list[str],
    fold_weights: dict[str, float],
) -> pd.Series:
    """Blend raw base-model scores with fixed weights without rank amplification."""
    available_models = [m for m in model_cols if m in frame.columns]
    if not available_models:
        return pd.Series(dtype=float)

    weight_sum = sum(fold_weights.get(m, 0.0) for m in available_models)
    if weight_sum <= 0:
        norm_weights = {m: 1.0 / len(available_models) for m in available_models}
    else:
        norm_weights = {m: fold_weights.get(m, 0.0) / weight_sum for m in available_models}

    score = sum(frame[m].fillna(0.5) * norm_weights.get(m, 0.0) for m in available_models)
    return score.clip(lower=0.0, upper=1.0)



def _regime_bucket_from_context(values: pd.Series) -> pd.Series:
    """Bucket regime context into low / mid / high stress states."""
    context = pd.to_numeric(values, errors="coerce")
    bucket = pd.Series("mid", index=context.index, dtype="object")
    bucket.loc[context <= 0.33] = "low"
    bucket.loc[context >= 0.67] = "high"
    bucket.loc[context.isna()] = "unknown"
    return bucket



def _contextual_fold_weights(
    history_frame: pd.DataFrame | None,
    model_cols: list[str],
    *,
    context_col: str = "p_high_vol",
    bucket: str | None = None,
    top_k: int = 10,
) -> dict[str, float] | None:
    """Estimate model weights from prior OOS rows in a similar regime bucket."""
    if history_frame is None or len(history_frame) == 0 or "target_return" not in history_frame.columns:
        return None

    hist = history_frame.copy()
    if bucket is not None and context_col in hist.columns and bucket != "all":
        bucket_labels = _regime_bucket_from_context(hist[context_col])
        hist = hist.loc[bucket_labels == bucket]

    min_dates = max(20, 2 * int(top_k))
    if hist.index.get_level_values("date").nunique() < min_dates:
        return None

    metrics: dict[str, tuple[float, float]] = {}
    for model_col in model_cols:
        if model_col not in hist.columns:
            continue
        view = hist[[model_col, "target_return"]].dropna()
        if view.index.get_level_values("date").nunique() < min_dates:
            continue

        daily_topk: list[float] = []
        daily_corr: list[float] = []
        for _, grp in view.groupby(level="date", sort=True):
            if len(grp) < top_k:
                continue
            daily_topk.append(float(grp.nlargest(top_k, model_col)["target_return"].mean()))
            if len(grp) >= 5:
                corr = grp[model_col].corr(grp["target_return"], method="spearman")
                if pd.notna(corr) and np.isfinite(corr):
                    daily_corr.append(float(corr))

        if len(daily_topk) < max(10, top_k):
            continue
        metrics[model_col] = (
            float(np.mean(daily_topk)),
            float(np.mean(daily_corr)) if daily_corr else 0.0,
        )

    if not metrics:
        return None

    topk_scale = float(np.median([abs(v[0]) for v in metrics.values()]))
    if not np.isfinite(topk_scale) or topk_scale < 1e-5:
        topk_scale = 1e-3

    raw = {
        model_col: max(0.0, 0.80 * (topk_val / topk_scale) + 0.20 * (rank_corr / 0.02))
        for model_col, (topk_val, rank_corr) in metrics.items()
    }
    raw_series = pd.Series(raw, dtype=float).clip(lower=0.0)
    if float(raw_series.sum()) <= 0:
        return None

    ordered = raw_series.sort_values(ascending=False)
    if len(ordered) >= 2 and float(ordered.iloc[0]) >= 1.20 * max(float(ordered.iloc[1]), 1e-12):
        leader = ordered.index[0]
        return {m: 1.0 if m == leader else 0.0 for m in model_cols}

    keep_mask = ordered >= max(float(ordered.iloc[0]) * 0.85, 1e-12)
    kept = ordered.where(keep_mask, 0.0)
    if float(kept.sum()) <= 0:
        return None

    return {m: float(kept.get(m, 0.0) / kept.sum()) for m in model_cols}



def _adaptive_regime_scores(
    fold_frame: pd.DataFrame,
    history_frame: pd.DataFrame | None,
    model_cols: list[str],
    default_weights: dict[str, float],
    *,
    context_col: str = "p_high_vol",
    top_k: int = 10,
) -> tuple[pd.Series, dict[str, dict[str, float]]]:
    """Blend scores with weights conditioned on prior OOS performance by regime bucket."""
    if context_col not in fold_frame.columns or history_frame is None or len(history_frame) == 0:
        return _adaptive_weighted_scores(fold_frame, model_cols, default_weights), {}

    bucket_labels = _regime_bucket_from_context(fold_frame[context_col])
    pieces: list[pd.Series] = []
    used_weights: dict[str, dict[str, float]] = {}
    for bucket in pd.Index(bucket_labels).dropna().unique():
        mask = bucket_labels == bucket
        bucket_frame = fold_frame.loc[mask]
        if len(bucket_frame) == 0:
            continue
        bucket_weights = _contextual_fold_weights(
            history_frame,
            model_cols,
            context_col=context_col,
            bucket=str(bucket),
            top_k=top_k,
        )
        if bucket_weights is None:
            bucket_weights = default_weights
        used_weights[str(bucket)] = bucket_weights
        pieces.append(_adaptive_weighted_scores(bucket_frame, model_cols, bucket_weights))

    if not pieces:
        return _adaptive_weighted_scores(fold_frame, model_cols, default_weights), {}

    score = pd.concat(pieces).reindex(fold_frame.index)
    return score, used_weights



def build_ensemble(
    pred_df: pd.DataFrame,
    results_df: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
    method: str = "adaptive",
    *,
    top_k: int = 10,
    context_col: str = "p_high_vol",
) -> pd.DataFrame:
    """Build adaptive, regime-aware, or leakage-safe stacked ensembles."""
    if len(pred_df) == 0:
        return pd.DataFrame()

    frame = _build_ensemble_frame(pred_df)
    models = [m for m in pred_df["model"].drop_duplicates().tolist() if m in frame.columns]
    fold_ids = sorted(frame["fold"].dropna().unique().tolist()) if "fold" in frame.columns else [None]

    label_col = "target_label" if "target_label" in frame.columns else "y_true"

    pieces = []
    for fold_id in fold_ids:
        fold_frame = frame if fold_id is None else frame[frame["fold"] == fold_id].copy()
        if len(fold_frame) == 0:
            continue

        method_used = "adaptive"
        if weights is not None:
            ens_score = _adaptive_weighted_scores(fold_frame, models, weights)
            method_used = "manual_weights"
        else:
            hist_results = None
            hist_frame = None
            if fold_id is not None and "fold" in frame.columns:
                hist_frame = frame[frame["fold"] < fold_id].copy()
            if results_df is not None and fold_id is not None and "fold" in results_df.columns:
                hist_results = results_df[results_df["fold"] < fold_id]

            use_stacking = method == "stacked" and fold_id is not None and "fold" in frame.columns
            stack_model = None
            if use_stacking:
                stack_model = _fit_stacking_model(hist_frame, models, label_col=label_col)

            if stack_model is not None:
                X_cur = _stacking_features(fold_frame, models)
                ens_score = pd.Series(
                    stack_model.predict_proba(X_cur)[:, 1],
                    index=fold_frame.index,
                    dtype=float,
                )
                ens_score = _rank_normalize_by_date(ens_score)
                method_used = "stacked"
            else:
                if use_stacking:
                    method_used = "adaptive_fallback"
                fold_weights = _adaptive_fold_weights(hist_results, models)
                if method == "adaptive_regime":
                    ens_score, bucket_weights = _adaptive_regime_scores(
                        fold_frame,
                        history_frame=hist_frame,
                        model_cols=models,
                        default_weights=fold_weights,
                        context_col=context_col,
                        top_k=top_k,
                    )
                    method_used = "adaptive_regime" if bucket_weights else "adaptive"
                else:
                    ens_score = _adaptive_weighted_scores(fold_frame, models, fold_weights)

        cols = [
            c
            for c in [
                "y_true",
                "target_label",
                "target_return",
                "adj_close",
                "p_high_vol",
                "market_breadth_200d",
                "vxn_zscore",
                "yield_spread_zscore",
            ]
            if c in fold_frame.columns
        ]
        ens_df = fold_frame[cols].copy()
        ens_df["y_prob"] = ens_score
        ens_df["model"] = "ENS"
        ens_df["ensemble_method"] = method_used
        if fold_id is not None:
            ens_df["fold"] = fold_id
        pieces.append(ens_df)
        log.info("Ensemble fold %s: %s", fold_id, method_used)

    ensemble_df = pd.concat(pieces).sort_index() if pieces else pd.DataFrame()
    log.info("Ensemble: %s predictions", f"{len(ensemble_df):,}" if len(ensemble_df) else 0)
    return ensemble_df
