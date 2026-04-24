# v2 Hyperparameter Tuning Experiment

Isolated tuning of LR + RF + LGBM with **Purged K-Fold Inner CV** to reduce
single-val-block bias. v1 (`src/`) is untouched — promote only if criteria pass.

## Files

| File | Purpose |
|---|---|
| `purged_kfold.py` | Time-ordered K-fold splitter with purge buffer between fit/val |
| `tune_walkforward.py` | Walk-forward training where each fold tunes LR/RF/LGBM via Optuna |
| `compare_v1_v2.py` | Side-by-side metric comparison vs `outputs/metrics/walkforward_full.csv` |
| `results/` | Outputs: `walkforward_v2.csv`, `best_params_per_fold.json`, `tuning_history.csv` |

## Bias controls

| Bias source | Mitigation |
|---|---|
| Test leakage | Per-fold tuning uses only that fold's `train_idx` (already purged from test) |
| Single-val-block overfitting | Inner CV = 3 purged folds × 10d purge between fit/val |
| Multiple comparisons | Search spaces are narrow (LR=2, RF=4, LGBM=7 params); n_trials kept low |
| Refit-on-full-train drift | Standard trade-off, accepted; same as v1 LGBM |

## Promote criterion (decided before run)

A model is **promoted** to v1 only if:
- Mean OOS daily-AUC on the **6 full-year folds (2020–2025)** improves by **≥ +50bps**
- AND improves on **≥ 4 of 6** folds individually

Fold 7 (2026 partial) is excluded from promotion judgment (only 1,904 rows; high variance).

## Run

```bash
# Tune all three models (default ~1.5–2.5h on CPU, RAM-heavy)
python -m experiments.v2_hyperparam_tuning.tune_walkforward

# Or restrict to one model
python -m experiments.v2_hyperparam_tuning.tune_walkforward --models lr
python -m experiments.v2_hyperparam_tuning.tune_walkforward --models rf
python -m experiments.v2_hyperparam_tuning.tune_walkforward --models lgbm

# Optional: smoke test (1 fold, 3 trials)
python -m experiments.v2_hyperparam_tuning.tune_walkforward --smoke

# Compare results
python -m experiments.v2_hyperparam_tuning.compare_v1_v2
```

## Notes

- LGBM here is tuned as **classifier** (binary `alpha_ext_label`), not ranker, to keep
  the search consistent with LR/RF. v1 LGBM ranker stays separate; if v2 classifier
  beats v1 ranker on daily-AUC, promotion still uses ranker design plus the v2 params
  that transfer (num_leaves, lambdas, etc.).
- GPU not used: sklearn doesn't support GPU; LGBM-on-Windows-GPU is finicky and the
  dataset (~150k rows × 58 features) is small enough that CPU + `n_jobs=-1` is faster.
- All results land in `results/` — delete the folder to discard the experiment cleanly.
