"""
Walk-forward training where each fold tunes LR/RF/LGBM via Optuna with purged
3-fold inner CV, then refits with best params and predicts OOS.

Outputs in `results/`:
- walkforward_v2.csv      (same schema as outputs/metrics/walkforward_full.csv)
- best_params_per_fold.json
- tuning_history.csv      (every Optuna trial, for inspection)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from experiments.v2_hyperparam_tuning.purged_kfold import (
    cv_score_classifier,
    daily_auc_score,
)
from src.config import load_config, get_feature_cols, prepare_feature_sets
from src.models.sample_weights import avg_uniqueness
from src.models.train import (
    _preprocess_fold,
    daily_auc,
    daily_rank_corr,
    top_k_return,
)
from src.splits.walkforward import make_expanding_splits
from src.utils.io import load

log = logging.getLogger("v2_tune")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tuners
# ──────────────────────────────────────────────────────────────────────────────

def _make_study(seed: int):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )


def tune_lr(X, y, dates, w, n_trials: int, inner_cv: int, purge_days: int, seed: int = 42):
    study = _make_study(seed)

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])

        def factory():
            return LogisticRegression(
                C=C,
                class_weight=class_weight,
                max_iter=1000,
                solver="lbfgs",
                random_state=seed,
            )

        return cv_score_classifier(factory, X, y, dates, sample_weight=w,
                                   n_splits=inner_cv, purge_days=purge_days)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value), study.trials_dataframe()


def tune_rf(X, y, dates, w, n_trials: int, inner_cv: int, purge_days: int, seed: int = 42):
    study = _make_study(seed)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
            max_features=trial.suggest_float("max_features", 0.3, 0.8),
        )

        def factory():
            return RandomForestClassifier(
                **params,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )

        return cv_score_classifier(factory, X, y, dates, sample_weight=w,
                                   n_splits=inner_cv, purge_days=purge_days)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value), study.trials_dataframe()


def tune_lgbm(X, y, dates, w, n_trials: int, inner_cv: int, purge_days: int, seed: int = 42):
    from lightgbm import LGBMClassifier
    study = _make_study(seed)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 60),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.55, 0.90),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 0.95),
            lambda_l1=trial.suggest_float("lambda_l1", 0.0, 5.0),
            lambda_l2=trial.suggest_float("lambda_l2", 0.0, 5.0),
            bagging_freq=5,
        )

        def factory():
            return LGBMClassifier(
                objective="binary",
                metric="binary_logloss",
                is_unbalance=False,
                verbose=-1,
                random_state=seed,
                n_jobs=-1,
                **params,
            )

        return cv_score_classifier(factory, X, y, dates, sample_weight=w,
                                   n_splits=inner_cv, purge_days=purge_days)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value), study.trials_dataframe()


# ──────────────────────────────────────────────────────────────────────────────
# Refit + predict
# ──────────────────────────────────────────────────────────────────────────────

def _refit_lr(params, X_tr, y_tr, w_tr, X_te, seed=42):
    m = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed, **params)
    m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m.predict_proba(X_te)[:, 1]


def _refit_rf(params, X_tr, y_tr, w_tr, X_te, seed=42):
    m = RandomForestClassifier(
        class_weight="balanced", random_state=seed, n_jobs=-1, **params
    )
    m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m.predict_proba(X_te)[:, 1]


def _refit_lgbm(params, X_tr, y_tr, w_tr, X_te, seed=42):
    from lightgbm import LGBMClassifier
    m = LGBMClassifier(
        objective="binary", metric="binary_logloss", is_unbalance=False,
        verbose=-1, random_state=seed, n_jobs=-1, **params,
    )
    m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m.predict_proba(X_te)[:, 1]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _eval(test_df, y_prob, target, return_col, top_k):
    test_target = pd.to_numeric(test_df[target], errors="coerce")
    auc_mask = test_target.notna().to_numpy()
    y_test = test_target[auc_mask].astype(int).to_numpy()
    g_auc = roc_auc_score(y_test, y_prob[auc_mask]) if len(np.unique(y_test)) >= 2 else np.nan
    d_auc = daily_auc(test_df, y_prob, target_col=target)
    tk_ret = top_k_return(test_df, y_prob, k=top_k, return_col=return_col)
    rc = daily_rank_corr(test_df, y_prob, return_col=return_col)
    return g_auc, d_auc, tk_ret, rc


def _build_sample_weights(train_full_df) -> np.ndarray | None:
    if "t1" not in train_full_df.columns:
        return None
    w = avg_uniqueness(train_full_df, t1_col="t1").to_numpy(dtype=float)
    mean = np.nanmean(w) if np.isfinite(w).any() else 1.0
    w = np.where(np.isfinite(w), w, mean)
    if mean > 0:
        w = w / mean
    return w


def run(
    models: list[str],
    n_trials_lr: int,
    n_trials_rf: int,
    n_trials_lgbm: int,
    inner_cv: int,
    purge_days: int,
    smoke: bool,
    config_path: str | None,
):
    cfg = load_config(config_path) if config_path else load_config()

    labeled_path = cfg.dir_processed / "dataset_labeled.parquet"
    if not labeled_path.exists():
        raise FileNotFoundError(f"Missing {labeled_path}. Run scripts/run_labeling.py first.")

    log.info("Loading %s", labeled_path)
    df = load(labeled_path)

    feature_cols = get_feature_cols(list(df.columns))
    feature_sets = prepare_feature_sets(df, feature_cols)
    feature_cols_full = feature_sets["full_cols"]
    log.info("Features: %d", len(feature_cols_full))

    splits = make_expanding_splits(
        df,
        first_test_year=cfg.walkforward.first_test_year,
        horizon=cfg.labeling.horizon,
        max_train_years=cfg.walkforward.max_train_years,
    )
    if smoke:
        splits = splits[:1]
        n_trials_lr = min(n_trials_lr, 3)
        n_trials_rf = min(n_trials_rf, 3)
        n_trials_lgbm = min(n_trials_lgbm, 3)
        log.warning("SMOKE MODE: 1 fold × 3 trials per model")

    target = "alpha_ext_label" if "alpha_ext_label" in df.columns else "alpha_label"
    return_col = "alpha_ret"
    top_k = cfg.strategy.top_k

    log.info("Target=%s | return=%s | top_k=%d | inner_cv=%d × purge=%d",
             target, return_col, top_k, inner_cv, purge_days)

    all_results = []
    best_params_log: dict = {}
    history_frames: list[pd.DataFrame] = []

    for fold in splits:
        log.info("=" * 70)
        log.info("Fold %d (test %d) | train=%d test=%d", fold.fold, fold.test_year,
                 len(fold.train_idx), len(fold.test_idx))

        train_full_df = df.loc[fold.train_idx].copy().sort_index()
        test_full_df = df.loc[fold.test_idx].copy().sort_index()

        X_tr_rank, X_te_rank, X_tr_scaled, X_te_scaled = _preprocess_fold(
            train_full_df, test_full_df, feature_cols_full
        )

        # Restrict to labeled rows for classifier training.
        train_cls_mask = train_full_df[target].notna().to_numpy()
        y_train = pd.to_numeric(train_full_df[target], errors="coerce").to_numpy()
        y_train_cls = y_train[train_cls_mask].astype(int)
        dates_tr = train_full_df.index.get_level_values("date")[train_cls_mask]

        sample_weights_full = _build_sample_weights(train_full_df)
        w_tr_cls = sample_weights_full[train_cls_mask] if sample_weights_full is not None else None

        X_tr_rank_cls = X_tr_rank[train_cls_mask]
        X_tr_scaled_cls = X_tr_scaled[train_cls_mask]

        fold_key = f"fold_{fold.fold}_test_{fold.test_year}"
        best_params_log[fold_key] = {}

        train_size_used = int(len(X_tr_rank_cls))
        test_size = int(len(X_te_rank))

        # ───── LR ─────
        if "lr" in models:
            t0 = time.time()
            log.info("  [LR] tuning %d trials × %d inner CV", n_trials_lr, inner_cv)
            best_p, best_v, hist = tune_lr(X_tr_scaled_cls, y_train_cls, dates_tr, w_tr_cls,
                                           n_trials_lr, inner_cv, purge_days)
            log.info("  [LR] best inner-CV daily-AUC=%.4f params=%s time=%.1fs",
                     best_v, best_p, time.time() - t0)
            y_prob = _refit_lr(best_p, X_tr_scaled_cls, y_train_cls, w_tr_cls, X_te_scaled)
            g, d, tk, rc = _eval(test_full_df, y_prob, target, return_col, top_k)
            log.info("  [LR] OOS  global_auc=%.3f  daily_auc=%.3f  rank_corr=%+.4f  top%d_ret=%.4f",
                     g, d, rc, top_k, tk)
            all_results.append({
                "fold": fold.fold, "test_year": fold.test_year, "model": "LR",
                "global_auc": g, "daily_auc": d, "top_k_ret": tk, "rank_corr": rc,
                "train_size": train_size_used, "test_size": test_size,
                "inner_cv_auc": best_v,
            })
            best_params_log[fold_key]["LR"] = {"params": best_p, "inner_cv_auc": best_v}
            hist["fold"] = fold.fold; hist["model"] = "LR"
            history_frames.append(hist)

        # ───── RF ─────
        if "rf" in models:
            t0 = time.time()
            log.info("  [RF] tuning %d trials × %d inner CV", n_trials_rf, inner_cv)
            best_p, best_v, hist = tune_rf(X_tr_rank_cls, y_train_cls, dates_tr, w_tr_cls,
                                           n_trials_rf, inner_cv, purge_days)
            log.info("  [RF] best inner-CV daily-AUC=%.4f params=%s time=%.1fs",
                     best_v, best_p, time.time() - t0)
            y_prob = _refit_rf(best_p, X_tr_rank_cls, y_train_cls, w_tr_cls, X_te_rank)
            g, d, tk, rc = _eval(test_full_df, y_prob, target, return_col, top_k)
            log.info("  [RF] OOS  global_auc=%.3f  daily_auc=%.3f  rank_corr=%+.4f  top%d_ret=%.4f",
                     g, d, rc, top_k, tk)
            all_results.append({
                "fold": fold.fold, "test_year": fold.test_year, "model": "RF",
                "global_auc": g, "daily_auc": d, "top_k_ret": tk, "rank_corr": rc,
                "train_size": train_size_used, "test_size": test_size,
                "inner_cv_auc": best_v,
            })
            best_params_log[fold_key]["RF"] = {"params": best_p, "inner_cv_auc": best_v}
            hist["fold"] = fold.fold; hist["model"] = "RF"
            history_frames.append(hist)

        # ───── LGBM ─────
        if "lgbm" in models:
            t0 = time.time()
            log.info("  [LGBM] tuning %d trials × %d inner CV", n_trials_lgbm, inner_cv)
            best_p, best_v, hist = tune_lgbm(X_tr_rank_cls, y_train_cls, dates_tr, w_tr_cls,
                                             n_trials_lgbm, inner_cv, purge_days)
            log.info("  [LGBM] best inner-CV daily-AUC=%.4f params=%s time=%.1fs",
                     best_v, best_p, time.time() - t0)
            y_prob = _refit_lgbm(best_p, X_tr_rank_cls, y_train_cls, w_tr_cls, X_te_rank)
            g, d, tk, rc = _eval(test_full_df, y_prob, target, return_col, top_k)
            log.info("  [LGBM] OOS  global_auc=%.3f  daily_auc=%.3f  rank_corr=%+.4f  top%d_ret=%.4f",
                     g, d, rc, top_k, tk)
            all_results.append({
                "fold": fold.fold, "test_year": fold.test_year, "model": "LGBM",
                "global_auc": g, "daily_auc": d, "top_k_ret": tk, "rank_corr": rc,
                "train_size": train_size_used, "test_size": test_size,
                "inner_cv_auc": best_v,
            })
            best_params_log[fold_key]["LGBM"] = {"params": best_p, "inner_cv_auc": best_v}
            hist["fold"] = fold.fold; hist["model"] = "LGBM"
            history_frames.append(hist)

    # Save outputs
    results_df = pd.DataFrame(all_results)
    out_csv = RESULTS_DIR / ("walkforward_v2_smoke.csv" if smoke else "walkforward_v2.csv")
    results_df.to_csv(out_csv, index=False)
    log.info("Saved %s (%d rows)", out_csv, len(results_df))

    out_params = RESULTS_DIR / ("best_params_per_fold_smoke.json" if smoke else "best_params_per_fold.json")
    with open(out_params, "w", encoding="utf-8") as f:
        json.dump(best_params_log, f, indent=2, default=str)
    log.info("Saved %s", out_params)

    if history_frames:
        hist_all = pd.concat(history_frames, ignore_index=True)
        out_hist = RESULTS_DIR / ("tuning_history_smoke.csv" if smoke else "tuning_history.csv")
        hist_all.to_csv(out_hist, index=False)
        log.info("Saved %s (%d trials)", out_hist, len(hist_all))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="all",
                   help="Comma list: lr,rf,lgbm,all (default: all)")
    p.add_argument("--n-trials-lr", type=int, default=25)
    p.add_argument("--n-trials-rf", type=int, default=20)
    p.add_argument("--n-trials-lgbm", type=int, default=25)
    p.add_argument("--inner-cv", type=int, default=3)
    p.add_argument("--purge-days", type=int, default=10)
    p.add_argument("--smoke", action="store_true",
                   help="1 fold × 3 trials per model (sanity check)")
    p.add_argument("--config", default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    raw = [m.strip().lower() for m in args.models.split(",")]
    if "all" in raw:
        models = ["lr", "rf", "lgbm"]
    else:
        models = [m for m in raw if m in {"lr", "rf", "lgbm"}]
    if not models:
        raise SystemExit("No valid model selected. Choose from lr,rf,lgbm,all")

    run(
        models=models,
        n_trials_lr=args.n_trials_lr,
        n_trials_rf=args.n_trials_rf,
        n_trials_lgbm=args.n_trials_lgbm,
        inner_cv=args.inner_cv,
        purge_days=args.purge_days,
        smoke=args.smoke,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
