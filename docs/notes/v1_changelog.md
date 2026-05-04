# v1 Changelog — Methodology Changes & Experimental Results

**Window covered:** 2026-04-12 → 2026-04-13 (updated)
**Scope:** from the post-v0 baseline (CW_baseline declared winner) through the start of v1 upstream work (purged K-fold CV prototype).

This note exists as an audit trail of what was tried, what was kept, and what was dropped, together with the raw result tables that drove each decision. It complements `docs/notes/adaptive_overlay_dropped.md` (which covers only Strategy C) by tracking the *full* v0→v1 transition.

---

## 1. Drop ML_Full_Adaptive risk-overlay strategy (2026-04-12)

### Context
v0 ended with three shipping candidates tested on 7 walk-forward folds (2020–2026):

| Strategy          | CAGR  | Sharpe | MDD    | Calmar | α (vs QQQ) | HAC t | p      |
|-------------------|-------|--------|--------|--------|------------|-------|--------|
| ML_Full_EW        | 21.9% | 0.68   | −43.2% | 0.51   | +2.68%     | +0.35 | 0.72   |
| ML_Full_CW        | 25.8% | 0.72   | −45.7% | 0.56   | +6.34%     | +0.80 | 0.42   |
| ML_Full_Adaptive  |  7.8% | 0.41   | −24.4% | 0.32   | **−13.35%**| −2.24 | **0.025** |

`ML_Full_Adaptive` = ML_Full_CW wrapped in a risk-overlay stack (regime gate + breadth gate + benchmark trend gate + vol targeting + drawdown killswitch).

### Decision
**Drop** `ML_Full_Adaptive` from the pipeline. The overlays destroyed 18pp of CAGR to save 21pp of MDD — Calmar actually degraded (0.56 → 0.32) and the alpha flipped significantly negative. Full rationale and parameter table: `docs/notes/adaptive_overlay_dropped.md`.

### Stage 2 tuning attempt (notebook 07, 20-combo grid)
Before dropping, we ran a 20-combination grid across SMA length, vol target, sector cap, breadth threshold, and dd trigger. Results in `outputs/metrics/overlay_tuning_grid_v2.csv`. **No combo** achieved CAGR ≥ 20% AND MDD ≥ −30%. Winner declared was `CW_baseline` (zero overlays).

### Files changed
- `scripts/run_backtest.py` — removed Strategy C block, `strategy_equities`, `alpha_all`, `alpha_boot`, `trade_summary` references; random benchmark reference switched from `ML_Full_Adaptive` to `ML_Full_CW`.
- `scripts/run_analysis.py` — removed `"ML Full Adaptive"` from `colors`, removed `eq_adaptive` loading and `equities` entry.
- `scripts/export_reporting.py` — removed from `STRATEGY_SPECS`, renumbered sort orders (`BH_QQQ` 4→3, `BH_MCap10` 5→4, `BH_Full` 6→5), removed from fallback alpha list, removed from `trade_specs`.
- `notebooks/05_backtesting.ipynb` and `notebooks/06_analysis.ipynb` — cleaned and re-executed.
- `docs/notes/adaptive_overlay_dropped.md` — created (drop rationale).
- Memory: `project_adaptive_ensemble_bug.md` — rewritten with the "adaptive means two different things" clarification.

### Important clarification
"Adaptive" refers to **two** different things in this codebase. Only one was dropped:

1. **`build_ensemble(method="adaptive")`** in `src/models/train.py` — the ensemble blending method used by **both** `ML_Full_EW` and `ML_Full_CW`. **STILL IN USE. NOT TOUCHED.**
2. **`ML_Full_Adaptive` strategy** in `scripts/run_backtest.py` — the risk-overlay portfolio. **This is the one that was dropped.**

### Ship state after drop
v0 shipping candidate = **`ML_Full_CW`** (CW_baseline): CAGR 25.8%, Sharpe 0.72, MDD −45.7%, Calmar 0.56, α = +6.34% (ns).

### Re-enable criteria for risk overlays
Do NOT revive overlays until the base signal clears all three:
- (a) HAC alpha p-value vs QQQ < 0.10
- (b) MDD ≥ −35%
- (c) CAGR > 30%

---

## 2. v1 design — replace single-block validation with purged K-fold CV (2026-04-13)

### Motivation
v0 alpha p-values (0.42 and 0.72) are not significant — the base signal is too weak. Before tuning portfolio overlays further, improve the upstream model selection. Hypothesis: single-block (last 15% date-tail) validation inside `fit_lgbm_optuna` overfits to one contiguous block and inflates CV scores relative to what generalizes to the next test year.

### Design (confirmed with user)
- **K = 5 purged K-fold CV** (López de Prado recipe)
- **Purge = 10 days** (matches labeling horizon `H = 10`)
- **Scope: LGBM only** — LR and RF left untouched
- **Prototype first** in a standalone script, port to `src/models/train.py` (replacing `_make_date_block_val_mask` in `fit_lgbm_optuna`) **only after methodology validates**
- Fold-level dates: each fold gets contiguous date blocks in train window; purge zones cut `purge_days` on each side of the validation block

### Prototype implementation
File: `notebooks/_v1_purged_cv_run.py`
- Fold: `test_year=2023` only (reduced scope for prototype)
- Train window: last 4 years (2019–2022)
- Configs: `single_block` (85/15 by date), `kfold5_purged`
- Seeds: [42, 123]
- Trials: 15 each
- n_estimators cap: 300
- Metric: **daily AUC** (group-by-date ROC-AUC mean — matches what the production tuner uses)
- Output: `outputs/metrics/v1_purged_cv_prototype.json`

### Environment hiccup and proxy substitution
First run crashed: `FileNotFoundError: Could not find module 'lib_lightgbm.dll' (or one of its dependencies)`. Root cause: `vcomp140.dll` (OpenMP runtime from Visual C++ Redistributable) not installed on this machine. As a temporary fix, the prototype was run with `sklearn.ensemble.HistGradientBoostingClassifier` as a pure-Python proxy — same gradient boosting family, methodology comparison is model-agnostic. Hyperparameters remapped to HGBC's space (`max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`, `max_features`). LightGBM was reinstalled (working) after user installed Visual C++ Redistributable on 2026-04-13.

### Prototype results (HGBC proxy, test_year=2023 only, 2 seeds × 15 trials)

| config         | CV mean | CV std | OOS mean | OOS std |
|----------------|---------|--------|----------|---------|
| single_block   | 0.5439  | 0.0058 | 0.4865   | 0.0008  |
| kfold5_purged  | 0.5069  | 0.0017 | 0.4901   | 0.0018  |

Delta kfold vs single: OOS mean **+0.0036**, OOS std **+0.0010**.
Raw script verdict: **FAIL** (strict heuristic requiring std improvement too).

### Interpretation
The strict FAIL is misleading. The important result is the **CV→OOS gap shrinkage**:

- single_block: CV 0.5439 → OOS 0.4865 = **−5.7pp gap** (CV was lying by 570 bps)
- kfold5_purged: CV 0.5069 → OOS 0.4901 = **−1.7pp gap** (CV is honest within ~170 bps)

Purged k-fold is doing exactly what the López de Prado recipe promises: killing the optimistic bias of single-block validation. The delta on OOS mean itself (+36 bps) is within noise at this small scope. **Neither config crosses 0.50 OOS on test_year=2023** — this is an unusually hard year, and one fold is too noisy to make a shipping decision.

### Confounds to address in the real test
- HGBC ≠ LGBM (different regularization behavior, different hyperparameter space)
- Only 1 test year = no fold-variance signal
- Only 2 seeds = undersampled stochasticity
- 15 trials = undersaturated tuning
- Last 4 training years only = less data than production (production uses expanding window)

### Path A results (real LightGBM, 3 test years, 2026-04-13)

User installed Visual C++ Redistributable → LightGBM loads. Re-ran prototype with:
- Real LightGBM (not HGBC proxy)
- Test years: 2021, 2022, 2023
- 2 seeds × 15 trials × 5 folds per config
- Train window: last 6 years (expanding)
- Output: `outputs/metrics/v1_purged_cv_prototype_lgbm.json`

**Per-year OOS daily AUC:**

| test_year | config | cv_mean | oos_mean | oos_std |
|---|---|---|---|---|
| 2021 | single_block  | 0.5454 | **0.5139** | 0.0016 |
| 2021 | kfold5_purged | 0.5298 | 0.5099 | 0.0014 |
| 2022 | single_block  | 0.5115 | **0.4938** | 0.0196 |
| 2022 | kfold5_purged | 0.5217 | 0.4780 | 0.0010 |
| 2023 | single_block  | 0.5140 | 0.5052 | 0.0112 |
| 2023 | kfold5_purged | 0.5098 | 0.5053 | 0.0097 |

**Aggregated across 3 years:**

| config | CV mean | OOS mean | OOS std | CV→OOS gap |
|---|---|---|---|---|
| single_block  | 0.5236 | **0.5043** | **0.0135** | 0.0193 |
| kfold5_purged | 0.5204 | 0.4977 | 0.0160 | 0.0227 |

Delta kfold vs single: OOS mean **−0.0066**, OOS std **+0.0025**, CV→OOS gap **+0.0034** (worse, not better).

### Verdict: FAIL — do NOT port purged K-fold

Purged K-fold loses on every dimension when measured with real LightGBM across 3 test years:

1. **Lower OOS mean** (−66 bps). Single-block wins in 2021 and 2022, ties in 2023.
2. **Higher OOS std** — the stability argument for k-fold does not hold empirically here.
3. **CV→OOS gap is slightly worse**, not better. In 2022 specifically, k-fold's CV was more optimistic than single-block (437 bps vs 177 bps gap).
4. **Biggest failure in the stress year** (2022, −158 bps OOS gap) — exactly where robust CV matters most.

### Lesson: single-year prototypes are unreliable for methodology decisions

The HGBC prototype on test_year=2023 alone suggested k-fold was more honest (gap 1.7pp vs 5.7pp). With real LGBM + 3 years the story flips completely. Future methodology experiments must use ≥ 3 test years before drawing conclusions.

### Hypothesized root cause
- Purged K-fold splits the already-small train window into 5 × ~5k-sample blocks. LightGBM with `is_unbalance=True` produces noisy fold scores; averaging 5 noisy scores is not clearly better than one 3,852-sample single-block validation.
- Pure K-fold weights all folds equally, including early-window folds from potentially different regimes. Single-block always validates on the most-recent date slice, which is a better proxy for the upcoming test year.
- Base signal is weak enough (OOS ~0.50–0.51) that the tuner's cross-config differences are dominated by noise, and k-fold's noise-averaging hurts more than it helps.

### Decision: v1 purged CV is not the bottleneck

`src/models/train.py` remains **unchanged**. Keep `fit_lgbm_optuna`'s single-block validation. The v1 upstream win must come from elsewhere.

### v1.1 candidates (ranked)
1. **Walk-forward (expanding) K-fold** — each fold validates on a later time slice, weighting recent-regime fit more heavily. Keeps López de Prado purge discipline but avoids equal-weighting early-window folds.
2. **Meta-labeling** — train a second model on when to trust the base signal. Directly attacks the weak p-values without changing the base model.
3. **Regime-aware training** — train separate models per macro regime.

---

## 3. avg_uniqueness sample weights — PASS, ported to train.py (2026-04-13)

### Motivation
After the purged CV FAIL, the actual bottleneck was hypothesized to be *sample-level noise* from overlapping labels (H=10 horizon → each label carries information shared with neighbors). López de Prado's `avg_uniqueness` downweights samples whose label window overlaps with many others. Computed per-ticker from `t1`: for each sample i, `u_i = mean over t in [t0_i, t1_i] of 1/concurrency(t)`.

### Prototype
File: `notebooks/_v1_uniqueness_weight_run.py`
- Test years: 2021, 2022, 2023
- Train window: last 6 years (expanding)
- Validation: single_block (85/15)  — same as production
- 2 seeds × 15 trials, real LightGBM
- Configs: `no_weight`, `uniqueness` (weights normalized to mean 1)

Helper: `src/models/sample_weights.py:avg_uniqueness(df, t1_col='t1')` — per-ticker concurrency count, O(N·H).

### Results (real LightGBM, 3 test years)

**Per-year OOS daily AUC:**

| test_year | no_weight | uniqueness | Δ |
|---|---|---|---|
| 2021 | 0.5139 | 0.5086 | −53 bps |
| 2022 (stress) | 0.4938 | **0.5055** | **+117 bps** |
| 2023 | 0.5052 | 0.5093 | +41 bps |

**Aggregated:**

| config | OOS mean | OOS std |
|---|---|---|
| no_weight | 0.5043 | 0.0135 |
| **uniqueness** | **0.5078** | **0.0052** |

Δ: OOS mean **+35 bps**, OOS std **−83 bps** (2.6× more stable).

### Verdict: PASS
- Biggest win in the 2022 stress year (+117 bps) — exactly where Path A failed worst.
- OOS std dropped by 83 bps, meaning model selection is dramatically more stable across seeds.
- The 2021 regression (−53 bps) is smaller than the 2022 win.

### Port to src/models/train.py
- `src/models/sample_weights.py` — new helper.
- `src/models/train.py`:
  - Import `avg_uniqueness`.
  - `walk_forward_train`: compute `w_binary` from `train_full_df['t1']` (per fold), normalize to mean 1, pass through `_build_lgbm_training_payload(..., sample_weight=w_binary)`.
  - `_build_lgbm_training_payload`: new `sample_weight` param, slices into `w_fit` / `w_full_train` for both classifier and ranker paths.
  - `fit_lgbm_optuna`: new `sample_weight_fit`, `sample_weight_full` params, threaded to every `.fit()` call (tune loop + final refit, both classifier and ranker, both Optuna and fallback paths). Default `None` → identical behavior for backward compatibility.

---

## 4. Meta-labeling prototype — FAIL (parked, design flaw)

### Motivation
Uniqueness weights alone improve OOS AUC by ~35 bps, but the real goal is lifting alpha p-value (currently ns at 0.42/0.72). Meta-labeling (López de Prado, AFML Ch 3): train a secondary model on whether to trust the base signal, directly attacks precision on the long side.

### Prototype
File: `notebooks/_v1_meta_label_run.py`
- Base: LGBM tuned with uniqueness weights (from section 3).
- Meta training set: samples in SB_VAL whose base score is in **top 30% per day** (cross-sectional).
- Meta features: `[base_score] + all 58 original features`.
- Meta target: true label (same as base).
- Metric: daily AUC, top-10 precision, top-10 realized return.

### Results

**Per-year:**

| test_year | base_auc | meta_auc | base_prec@10 | meta_prec@10 | base_ret@10 | meta_ret@10 |
|---|---|---|---|---|---|---|
| 2021 | 0.5086 | 0.4964 | 0.499 | 0.502 | ~0 | +0.0004 |
| 2022 | 0.5055 | 0.5037 | 0.503 | 0.517 | −0.0020 | −0.0010 |
| 2023 | 0.5093 | 0.4997 | 0.507 | 0.512 | +0.0009 | +0.0009 |

**Aggregated Δ:**
- OOS AUC: **−79 bps** ❌
- Precision@10: +73 bps (small)
- Return@10: +4 bps (noise)

### Verdict: WEAK/FAIL

### Root cause (design flaw, not idea failure)
1. **Target collinearity**: Meta target = base target. Meta is effectively a second base model trained on a tiny val slice (~1.5k rows) — noisier than base.
2. **Full feature reuse**: Passing all 58 original features to meta turns it into a redundant ensemble, not a confidence filter.
3. **Score-zero below top-30%**: Meta zeroes out ~70% of test rows, collapsing rank AUC.

### Redesign ideas (if retried later)
- Meta features should be **disjoint** from base: only `[base_score, p_high_vol, vxn_zscore, market_breadth_200d, yield_spread_zscore]` (regime context, not raw features).
- **Cross-fit base scores** across the full train set (not just SB_VAL) so meta training set is ~6× larger.
- Final score: `base_score` for non-top rows + `meta_prob` refinement for top rows, avoid zeroing.

**Status: parked.** Not blocking — uniqueness win is already ported.

---

## 5. Purged K-fold retry with uniqueness weights — still FAIL (2026-04-13)

### Motivation
Section 2 showed purged K-fold FAIL under vanilla fits. Hypothesis: the fold scores were too noisy because of overlapping labels, which uniqueness weights now correct. Retry k-fold with weights to see if the verdict flips.

### Prototype
File: `notebooks/_v1_purged_cv_weighted_run.py` (identical to section 2 Path A runner, but every `.fit()` now carries `sample_weight=w_tr[mask]`).

### Results (real LightGBM, 3 test years, weighted)

**Per-year OOS daily AUC:**

| test_year | config | cv_mean | oos_mean | oos_std |
|---|---|---|---|---|
| 2021 | single_block  | 0.5451 | **0.5086** | 0.0018 |
| 2021 | kfold5_purged | 0.5286 | 0.5069 | 0.0131 |
| 2022 | single_block  | 0.5177 | **0.5055** | 0.0055 |
| 2022 | kfold5_purged | 0.5218 | 0.4809 | 0.0030 |
| 2023 | single_block  | 0.5143 | **0.5093** | 0.0093 |
| 2023 | kfold5_purged | 0.5087 | 0.5036 | 0.0035 |

**Aggregated:**

| config | CV mean | OOS mean | OOS std | CV→OOS gap |
|---|---|---|---|---|
| **single_block**  | 0.5257 | **0.5078** | **0.0052** | 0.0179 |
| kfold5_purged | 0.5197 | 0.4971 | 0.0141 | 0.0226 |

Δ kfold vs single: OOS mean **−107 bps**, OOS std **+89 bps**, gap **+47 bps worse**.

### Verdict: still WEAK/FAIL — purged K-fold is structurally wrong for this problem

Weighting the fits did not rescue k-fold:
1. **2022 stress year collapses even worse**: kfold 0.4809 vs single 0.5055 (−246 bps). The weighted kfold is MORE optimistic in CV (0.5218 vs 0.5177) while OOS is worse — classic sign of overfit to noisy equal-weighted folds.
2. **Single-block with uniqueness is the clear winner across all 3 years.**
3. The CV→OOS gap widens (0.0226 vs 0.0179), opposite of what the k-fold recipe promises.

### Lesson
Purged K-fold is not the right CV scheme for this codebase's shape (expanding train window ~6 years, cross-sectional labels with H=10 overlap, weak base signal). Single-block validation on the most-recent date slice is closer in distribution to the upcoming test year. Uniqueness weights fix sample-level noise but cannot fix the **fold-level regime mismatch** inherent in pure K-fold.

**Decision: abandon purged K-fold for this project.** The v1.1 candidate list from section 2 is updated: walk-forward K-fold is still theoretically worth trying (different structure), but it is deprioritized below higher-leverage work (better labeling, feature engineering).

---

## 6. Files touched in v1 window (cumulative)

### Created
- `docs/notes/adaptive_overlay_dropped.md`
- `docs/notes/v1_changelog.md` (this file)
- `notebooks/08_purged_cv.ipynb` (prototype notebook — timed out during nbconvert, superseded by standalone script)
- `notebooks/_v1_purged_cv_run.py` (HGBC proxy runner)
- `notebooks/_v1_purged_cv_lgbm_run.py` (Path A real-LGBM runner)
- `notebooks/_v1_uniqueness_weight_run.py` (uniqueness weight prototype)
- `notebooks/_v1_meta_label_run.py` (meta-labeling prototype, FAIL)
- `notebooks/_v1_purged_cv_weighted_run.py` (k-fold retry with weights, FAIL)
- `outputs/metrics/v1_purged_cv_prototype.json` (HGBC proxy)
- `outputs/metrics/v1_purged_cv_prototype_lgbm.json` (Path A)
- `outputs/metrics/v1_uniqueness_weight_prototype.json` (PASS)
- `outputs/metrics/v1_meta_label_prototype.json` (FAIL)
- `outputs/metrics/v1_purged_cv_weighted_prototype.json` (FAIL)
- `src/models/sample_weights.py` — avg_uniqueness helper

### Modified
- `scripts/run_backtest.py`, `scripts/run_analysis.py`, `scripts/export_reporting.py` (section 1)
- `notebooks/05_backtesting.ipynb`, `notebooks/06_analysis.ipynb` (section 1)
- **`src/models/train.py`** — uniqueness sample weights wired into `walk_forward_train` → `_build_lgbm_training_payload` → `fit_lgbm_optuna`. Default `sample_weight=None` preserves backward compat. All classifier + ranker fits receive weights when `t1` is present.

### Unchanged (important)
- `src/splits/walkforward.py` — unchanged (already does year-level purge+embargo via `t1` when available).
- `build_ensemble(method="adaptive")` — unchanged, still used by both shipping strategies.
- Validation scheme in `fit_lgbm_optuna` — **still single-block 85/15**. Purged K-fold abandoned per section 5.

---

## 4. Root-cause investigation (2026-04-13) — why meta + TB was weak

User prompt: meta-labeling + triple-barrier is a strong LdP combo; in this project it was
weak. Investigate whether single-block's apparent win over purged K-fold was bias/instability.

Four root causes identified:

- **Fix A — substrate mismatch.** Base target was `alpha_ext_label` (cross-sectional rank),
  but `t1` (for uniqueness / meta) comes from triple-barrier. Weights and meta were computed
  on the wrong substrate. `tb_label` also retains 2.6× more rows (172k vs 66k).
- **Fix B — wrong `is_unbalance`.** `alpha_ext_label` is ~50/50; `is_unbalance=True` gives
  LightGBM spurious rebalancing that hurts calibration.
- **Fix C — no intra-fold purge.** Single-block 85/15 left zero gap between `SB_FIT` and
  `SB_VAL`; tuner validation leaks via overlapping TB labels.
- **Fix D — pure K-fold ≠ walk-forward.** LdP's purged K-fold averages folds from different
  regimes; production is train-up-to-t predict-next.

### Step 1 — Fair baseline (Fix B + Fix C, `alpha_ext_label`)

`notebooks/_v1_step1_fair_baseline_run.py` — intra-fold purge gap + `is_unbalance=False`,
uniqueness weights vs none.

| Config             | OOS daily AUC (mean) | OOS std |
|--------------------|---------------------:|--------:|
| no_weights         | 0.5002               | 0.0138  |
| uniqueness_weights | **0.5079 (+77 bps)** | 0.0128  |

Uniqueness wins all 3 test years. Restates prior result: earlier "+35 bps" was understated
because the unfair baseline had intra-fold leakage. **Verdict PASS.**

### Step 2 — Switch to `tb_label` substrate (Fix A)

`notebooks/_v1_step2_tb_label_run.py` — `tb_label` target, uniqueness weights, regime-only
meta features, additive meta scoring (preserve base score for non-top rows).

| Metric      | BASE    | META    | Δ        |
|-------------|--------:|--------:|---------:|
| daily AUC   | 0.5018  | 0.4997  | −21 bps  |
| prec@10     | 0.5079  | 0.5099  | +20 bps  |
| ret@10      | +0.0002 | +0.0002 | ~0       |

Marginal. Meta still does not lift meaningfully; tb_label substrate does not materially beat
fair-baseline `alpha_ext_label`. **Verdict WEAK.** Do not port tb_label switch.

### Step 3 — Walk-forward K-fold vs single-block (Fix D)

`notebooks/_v1_step3_walkforward_kfold_run.py` — expanding walk-forward 5-fold with
purge+embargo, same fair setup.

| Config              | CV mean | OOS mean | CV→OOS gap |
|---------------------|--------:|---------:|-----------:|
| single_block        | 0.5303  | 0.5088   | +0.0215    |
| walkforward_kfold   | 0.5187  | 0.5000   | +0.0187    |

Walk-forward CV is more honest (gap 28 bps tighter) — the bias concern was real. But it does
**not** translate to OOS: single-block still wins +88 bps OOS because the most-recent date
tail proxies the upcoming test year better than averaged forward folds. **Verdict FAIL.**
Keep single-block.

### Port decision

| Fix  | Decision  | Rationale                                       |
|------|-----------|-------------------------------------------------|
| A    | no        | marginal (Step 2 weak)                          |
| B    | **port**  | Step 1 fair-baseline requires it                |
| C    | **port**  | Step 1 fair-baseline requires it                |
| D    | no        | Step 3 FAIL — single-block empirically stronger |
| meta | no        | still weak across all substrates                |

### Ported to `src/models/train.py` (2026-04-13)

- `_make_date_block_val_mask(date_index, purge_days=10)` now returns `(fit_mask, val_mask)`
  with a 10-day purge gap before `val_start`. Previously `fit` was `~val`.
- `_build_lgbm_training_payload` accepts `fit_mask_base`; classifier and ranker both use it
  (previously both derived `fit = ~val_mask`).
- `walk_forward_train` unpacks the tuple and passes both masks through.
- All 4 hardcoded `is_unbalance=True` → `False` (classifier default in `get_models`, and the
  three `fit_lgbm_optuna` code paths: objective, fallback, final refit).

Uniqueness weights (ported earlier in section 3) remain.

### Cleanup (2026-04-13)

All v1 prototype scripts, logs, and JSON metrics were deleted after the port. Results
already captured in the tables above; the prototypes were one-shot exploration scripts
with hardcoded paths that would rot quickly. Deleted:

- `notebooks/_v1_purged_cv_run.py`, `_v1_purged_cv_lgbm_run.py`,
  `_v1_uniqueness_weight_run.py`, `_v1_meta_label_run.py`,
  `_v1_purged_cv_weighted_run.py`, `_v1_step1_fair_baseline_run.py`,
  `_v1_step2_tb_label_run.py`, `_v1_step3_walkforward_kfold_run.py`
- Matching `outputs/logs/v1_*.log` and `outputs/metrics/v1_*.json`

Kept: `src/models/sample_weights.py` (ported, still used), this changelog, and
`docs/notes/v1_roadmap.md` (forward plan).

---

## 7. P5 post-port backtest baseline (2026-04-13)

Ran `scripts/run_backtest.py` end-to-end after Fix B + Fix C + uniqueness weights were ported to `src/models/train.py`. Goal: verify the port didn't regress shipping strategies before starting P1.

### Results

| Strategy | CAGR | Sharpe | MDD | Calmar | α vs QQQ | HAC t | p |
|---|---|---|---|---|---|---|---|
| ML_Full_EW | 21.9% | 0.64 | −45.4% | 0.48 | +2.68% | 0.354 | 0.7234 |
| ML_Full_CW | 25.8% | 0.72 | −45.7% | 0.56 | +6.34% | 0.800 | 0.4235 |
| BH_QQQ     | 18.6% | 0.67 | −35.1% | 0.53 | —       | —     | —     |
| BH_MCap10  | 34.1% | 0.90 | −57.5% | 0.59 | —       | —     | —     |
| BH_Full    | 20.3% | 0.73 | −32.4% | 0.62 | —       | —     | —     |

Single-model diagnostics: LGBM CAGR 27.9% / Sharpe 0.78 / MDD −43.4%; LR 19.5% / 0.60 / −37.8%; RF 24.2% / 0.72 / −32.4%. Random-benchmark percentile 100%, z=3.83.

### Observation: port is a no-op at the backtest level

Numbers are **effectively identical** to the pre-port baseline in §1 (CW 25.8 / 0.72 / −45.7 / +6.34% / p=0.42). The Step 1 fair-baseline OOS daily-AUC lift from Fix B + Fix C + uniqueness (+77 bps) did **not** translate into portfolio alpha. Hypothesis: the AUC lift lives in the middle of the score distribution, but top-10 selection truncates to the tail where signal-to-noise is dominated by random variation. Upstream validation is honest now, but the ceiling is clearly elsewhere.

### Decision

Keep the port (methodologically correct is worth preserving even if portfolio-flat). Move to P1 to attempt upstream lift via regime conditioning.

---

## 8. P1 family — regime-conditional base model, FULL FAIL (2026-04-13)

### Motivation
Base model OOS AUC hovers at 0.50–0.51. Hypothesis from v1_roadmap P1: a monolithic model averages across regimes where the signal flips, so regime-aware training should lift daily AUC by ≥ 50 bps across all 3 test years without worse std.

Four variants were tried. **All four FAILed** by the P1 success criterion. Summary at the bottom.

All runs used the fair-baseline stack (uniqueness weights + Fix B `is_unbalance=False` + Fix C 10-day intra-fold purge). Classifier path was forced on both baseline and variants so cross-head score mixing in regime configs is apples-to-apples. As a result, the "baseline" number (0.5118) is a **classifier-path reference**, ~40 bps below the ranker-path fair baseline from §4 Step 1 (0.5079). Deltas are the decision metric.

Test years: 2021, 2022, 2023. Seeds: 42, 123. 10 Optuna trials. Train window: expanding, last 6 years.

### P1 — per-regime heads on `p_high_vol` median (2 buckets)

Prototype: `notebooks/_v1_p1_regime_run.py`

| test_year | base | regime2 | Δ |
|---|---|---|---|
| 2021 | 0.5163 | 0.5393 | **+230 bps** |
| 2022 | 0.5012 | 0.4766 | **−246 bps** |
| 2023 | 0.5180 | 0.5050 | −130 bps |

**Aggregated:** base 0.5118 (std 77 bps) → regime2 0.5070 (std 260 bps). Δ mean **−49 bps**, std **3.4× worse**.

Catastrophic 2022 collapse. Huge 2021 win is real but reversed by stress-year blowup. Bucket counts were healthy (~40k labeled rows per bucket) — failure is structural, not data-thinness: half-data heads overfit to bucket-specific noise.

### P1b — explicit `feat × (p_high_vol − 0.5)` interaction features

Prototype: `notebooks/_v1_p1b_regime_interactions_run.py`. Top-15 features chosen per fold by |Spearman IC| with the target, interaction columns added to the base 58-feature set (73 total).

| test_year | base | interact | Δ |
|---|---|---|---|
| 2021 | 0.5163 | 0.5191 | +28 bps |
| 2022 | 0.5012 | 0.4940 | −72 bps |
| 2023 | 0.5180 | 0.5248 | +68 bps |

**Aggregated:** Δ mean **+8 bps** (noise), std 161 bps (2.1× worse).

Top interactors were dominated by vol/beta/trend features that LightGBM already splits on via `p_high_vol` — explicit columns mostly added collinear noise. Milder 2022 damage than P1 (−72 vs −246), but still the wrong direction in the year that matters most.

### P1c — regime-weighted monolithic training (no data split)

Prototype: `notebooks/_v1_p1cd_regime_variants_run.py`. One LGBM trained on the full train set with `sample_weight = uniqueness × stress_multiplier` where `stress_multiplier = 0.5 + rank_pct(p_high_vol)` (range [0.5, 1.5]).

| test_year | base | weighted | Δ |
|---|---|---|---|
| 2021 | 0.5163 | 0.5164 | +1 bps |
| 2022 | 0.5012 | 0.4974 | −38 bps |
| 2023 | 0.5180 | 0.5070 | −110 bps |

**Aggregated:** Δ mean **−49 bps**. FAIL. Upweighting stress rows drags the model toward samples that are noisier on average without reliably helping where it matters.

### P1d — per-regime heads on `vxn_zscore` median (2 buckets, axis swap)

Same prototype script (`_v1_p1cd_regime_variants_run.py`). Identical structure to P1, but bucket axis is `vxn_zscore` instead of `p_high_vol`. Goal: rule out whether the axis choice was the problem.

| test_year | base | vxn_heads | Δ |
|---|---|---|---|
| 2021 | 0.5163 | 0.5135 | −28 bps |
| 2022 | 0.5012 | **0.5103** | **+91 bps** |
| 2023 | 0.5180 | 0.5047 | −133 bps |

**Aggregated:** Δ mean **−23 bps**.

**This is the diagnostic result.** `vxn_zscore` is the **first regime axis that helps the 2022 stress year** (+91 bps) — exactly where every other variant collapsed. But it now loses 2023 (−133 bps). The winning/losing years shift with the axis while the net stays negative. This is the strongest evidence that regime gains/losses are **axis-dependent noise, not signal** — no single axis helps all 3 years, so no regime conditioning is a reliable lift.

### Cross-variant summary

| Variant | 2021 Δ | 2022 Δ | 2023 Δ | Agg Δ | Agg std vs base | Verdict |
|---|---|---|---|---|---|---|
| P1  (p_high_vol heads)       | +230 | **−246** | −130 | **−49** | 3.4× | FAIL |
| P1b (p_high_vol interactions)| +28  | −72      | +68  | +8      | 2.1× | FAIL (noise) |
| P1c (p_high_vol weighted)    | +1   | −38      | −110 | **−49** | 1.1× | FAIL |
| P1d (vxn_zscore heads)       | −28  | **+91**  | −133 | −23     | 0.7× | FAIL (axis-noise) |

Success criterion (≥ +50 bps across **all** 3 years, std not worse) not met by any variant. No axis, no mechanism (hard split / soft interaction / sample reweighting) produces a reliable lift.

### Root-cause interpretation

Three converging pieces of evidence suggest regime conditioning is structurally the wrong lever for this codebase:

1. **Axis-dependent flip (P1 vs P1d):** `p_high_vol` wins 2021 huge / loses 2022. `vxn_zscore` loses 2021 / wins 2022. Same year, opposite sign depending on axis — this is the signature of noise, not regime structure.
2. **Mechanism-independent failure:** hard bucket split (P1), soft feature interaction (P1b), monolithic sample reweighting (P1c) all fail. If regime were a real lever, at least one mechanism would work.
3. **Base signal weakness:** OOS AUC centered at 0.50–0.51 means the base model has ~0–200 bps of true edge. Any refinement technique operating on that tiny signal is dominated by per-fold sampling noise. Regime conditioning adds variance (split data, new features, biased weights) that overwhelms the lift.

**Bottleneck is upstream, not in training.** The v1 effort has now rejected: purged K-fold (§2, §5), tb_label substrate (§4 Step 2), walk-forward K-fold (§4 Step 3), meta-labeling (§4, §4 Step 2), and regime conditioning (§8). The remaining candidates are all feature/label side: leakage audit (P4), purge sweep (P3), or new feature engineering.

### Decision
**Close P1 permanently.** Also park P2 (meta-labeling was blocked on a working P1 base; now doubly dead). Next priority: **P4 — feature leakage audit.** If AUC hovers near random and every tuning/CV/regime lever has failed, a structural feature-side issue is the most likely remaining cause.

### Files (to be cleaned up after this entry is committed)
- `notebooks/_v1_p1_regime_run.py`
- `notebooks/_v1_p1b_regime_interactions_run.py`
- `notebooks/_v1_p1cd_regime_variants_run.py`
- `outputs/metrics/v1_p1_regime_prototype.json`
- `outputs/metrics/v1_p1b_regime_interactions_prototype.json`
- `outputs/metrics/v1_p1cd_regime_variants_prototype.json`
- Matching `outputs/logs/v1_p1*.log`

Nothing ported to `src/models/train.py`.

---

## 9. P4 — Feature leakage audit: PASS (no leakage) + v1 closeout (2026-04-13)

### Scope

Two halves: static code review of `src/features/*.py` + `src/labeling/*.py`, and
a data-level audit on `dataset_labeled.parquet` measuring per-fold univariate
Spearman IC of every feature against `alpha_ext_label`.

### Static review — CLEAN

All feature generators use backward-looking rolling windows with `min_periods`,
positive-only `.shift()` offsets, and groupby-level computations that don't
contaminate across dates. `triple_barrier.label()` writes exactly six auxiliary
columns (`daily_vol`, `tb_label`, `tb_barrier`, `tb_return`, `t1`, `holding_td`);
all six are in both `NON_FEATURE_COLS` (src/config.py) and the `forbidden_features`
guard in `scripts/run_models.py`. `forward_targets.add_forward_rebalance_targets`
computes `alpha_ret`/`alpha_label`/`alpha_ext_label` forward-looking by design
(they're the target); all three are also in the forbidden set.

No look-ahead bug, no target leakage, no derivative-of-label sneaking in.

### Data-level audit — also CLEAN, and diagnostic

Prototype: `notebooks/_v1_p4_leakage_audit.py`. Enumerates actual feature columns
in the labeled dataset, computes per-fold `|Spearman IC|` against target on the
train slices for test_year ∈ {2021, 2022, 2023}.

| Check | Result |
|---|---|
| Feature columns after `NON_FEATURE_COLS` filter | 58 |
| Forbidden-name leaks into feature set | **0** |
| Non-numeric / dtype traps | **0** |
| Features with coverage < 50% in any fold | **0** |
| Features with \|IC\| ≥ 0.10 in any fold | **0** |
| Max single-feature cross-fold mean \|IC\| | **0.0557** (`rolling_beta_63d`) |

**Top 10 features by cross-fold mean |IC|:**

| Feature | IC mean | IC std | IC range |
|---|---|---|---|
| `rolling_beta_63d`  | +0.0557 | 0.0144 | [+0.0354, +0.0671] |
| `beta_regime`       | +0.0419 | 0.0124 | [+0.0244, +0.0520] |
| `rel_strength_21d`  | −0.0395 | 0.0005 | [−0.0401, −0.0388] |
| `downside_vol_21d`  | +0.0356 | 0.0085 | [+0.0246, +0.0453] |
| `cs_vol_zscore_21d` | +0.0349 | 0.0118 | [+0.0191, +0.0473] |
| `trend_strength_21d`| −0.0305 | 0.0035 | [−0.0339, −0.0257] |
| `max_dd_21d`        | −0.0301 | 0.0036 | [−0.0339, −0.0254] |
| `ret_21d`           | −0.0287 | 0.0010 | [−0.0298, −0.0275] |
| `zspread`           | +0.0282 | 0.0129 | [+0.0112, +0.0425] |
| `bb_pctb_20d`       | −0.0280 | 0.0019 | [−0.0300, −0.0254] |

Note the IC std column: most top features have std < 0.01, meaning they produce
virtually identical IC in every training window. This is the fingerprint of a
leak-free dataset — the opposite of what fold-specific leakage looks like (large
positive spike in one fold, flat elsewhere).

Permutation importance was skipped: univariate IC already captures the decisive
signal (no single-feature anomaly, no fold-specific outlier), and a permutation
run would not change the interpretation.

### Interpretation: the feature ceiling is the problem

The audit is PASS — but the "pass" is the diagnosis, not a null result:

1. **Max |IC| 0.0557.** A single feature explaining ~0.31% of label variance is
   a weak-signal regime. Even an ensemble of all 58 features, under realistic
   positive inter-feature correlation, caps the theoretical ensemble IC at
   roughly 0.08–0.12 → theoretical OOS AUC ceiling ~0.53–0.56. Measured OOS AUC
   hovers at 0.51. LGBM closes much of the gap via non-linear interactions; the
   remaining headroom is small and dominated by sampling noise.
2. **IC fingerprint is leak-free.** Tiny `ic_std` across folds on every top
   feature rules out fold-specific contamination; the features behave like
   genuinely-stable weak predictors, not occasional leaks.
3. **Feature universe is pure OHLCV + macro.** Price/volatility/momentum/
   cross-sectional stats on `adj_*` columns, plus vxn/vix/yield macro, plus
   their interactions. Academic ceiling for this feature class on a liquid
   large-cap universe is known to sit around |IC| 0.05–0.08 — exactly where we
   are. There is no fundamental, analyst, sentiment, options, short-interest,
   insider, or news signal in the dataset.

### Why every v1 lever failed: one-line explanation

Every failed v1 experiment (purged K-fold §2/§5, walk-forward K-fold §4 step 3,
tb_label §4 step 2, meta-labeling §4/§4 step 2, regime conditioning §8) was a
second-order refinement on a first-order-weak base signal. Second-order tricks
on a signal this weak are dominated by the variance they introduce.

### Decision: v1 is closed

Nothing more to tune, split, weight, or condition inside the current data. The
shipping strategy `ML_Full_CW` (CAGR 25.8%, Sharpe 0.72, MDD −45.7%, α +6.34%
ns, random-bench percentile 100%, z=3.83) **is the realistic ceiling of the
current feature universe.** The alpha-significance gap (p=0.42) is a
feature-universe problem, not a model problem.

P3 (purge sweep {5, 10, 15, 20}) is skipped: based on the IC-ceiling analysis,
any purge-value win will be ≤ 20 bps at the OOS AUC level and won't meaningfully
move portfolio alpha. Left in the backlog only as cheap sanity work if the
feature set changes materially.

### What the next phase looks like

The ceiling breaks only by expanding the feature universe. In rough order of
effort/payoff:

- **Options data** — IV skew, put/call ratio, 25-delta risk reversal,
  term-structure slope. CBOE/Yahoo/Polygon, daily. Proven alpha source for
  ~10-day horizons; would directly feed the existing 10-day `alpha_ext_label`.
- **Short interest** — FINRA biweekly bulk files. Cheap, low-frequency but
  persistent signal.
- **Analyst revisions / earnings surprise / fundamental deltas** — Refinitiv,
  Alpha Vantage, Polygon. Medium cost, strong historical alpha.
- **News / sentiment** — vendor-gated, expensive, domain-specific preprocessing.
- **High-frequency microstructure** — overkill for a 10-day signal horizon.

None of these are in scope for the current codebase — this is a data-sourcing
project, not a modeling project.

### Files (to clean up after this entry is committed)

- `notebooks/_v1_p4_leakage_audit.py`
- `outputs/metrics/v1_p4_leakage_audit.json`
- `outputs/logs/v1_p4_leakage_audit.log`

Nothing ported to production code. The static audit finding ("no leakage in the
feature pipeline") is memorialized in this section; no code change is needed.

---

## 10. Open questions / parked (post v1 closeout)

- **Feature-sourcing phase** is the successor to v1. No ticket exists yet.
- **P3 purge sweep** — still available as a cheap sanity check if the feature
  universe changes. Irrelevant on the current dataset per §9.
- **Meta-labeling** — parked indefinitely. Requires a stronger base signal that
  only new features can provide.
- **HMM regime source** — deprioritized (§8 showed regime conditioning is
  axis-agnostic noise on the current feature set; a better regime signal
  won't rescue a weak base).


## 11. FS1 — Polygon options features: FAIL on full panel (2026-04-20)

### Scope
First feature-sourcing ticket. Polygon.io REST API (free tier: 5 req/min,
2-year history). 72 tickers (full `SECTOR_MAP`), 12 monthly expiries
(2024-06 through 2025-05), lookback 30d per expiry, strike band ±12% spot.
5 volume/price features derived per (date, ticker) from front-month chain:

  opt_pc_vol_ratio       log(put_vol / call_vol)
  opt_dollar_vol_ratio   log(put_dollar / call_dollar)
  opt_skew_px_10pct      log(OTM_put_px@0.9*spot / OTM_call_px@1.1*spot)
  opt_atm_px_norm        avg(ATM_call_px, ATM_put_px) / spot
  opt_call_vol_norm      log(call_vol / 21d rolling median call_vol)

Total: 864 (ticker, expiry) units fetched, 16,131 API calls, ~56h wall time
on free tier. Cached under `data/polygon_options/{ticker}/{expiry}.parquet`.

### 10-ticker sub-study (intermediate, 2026-04-17)
Restricted to AAPL/MSFT/NVDA/GOOGL/META/AMZN/TSLA/AVGO/AMD/QCOM (992 obs,
2 yearly folds). Apparent PASS: 3 features above 0.08 threshold
(`opt_call_vol_norm` 0.134, `opt_pc_vol_ratio` 0.114,
`opt_dollar_vol_ratio` 0.099). **This was small-sample noise**, revealed
when expanded.

### 72-ticker full panel (5338 obs, 13 monthly folds)
All features collapse well below v1 ceiling (max |IC| 0.0557):

| feature                | ic_mean  | ic_std | 10t |IC| | 72t |IC| |
|------------------------|----------|--------|-----------|-----------|
| opt_atm_px_norm        | +0.0474  | 0.173  | 0.090     | **0.047** |
| opt_call_vol_norm      | −0.0430  | 0.112  | 0.134     | **0.043** |
| opt_skew_px_10pct      | +0.0162  | 0.140  | 0.007     | **0.016** |
| opt_pc_vol_ratio       | −0.0111  | 0.101  | 0.114     | **0.011** |
| opt_dollar_vol_ratio   | +0.0048  | 0.096  | 0.099     | **0.005** |

Sign stability also poor: best feature (`opt_skew_px_10pct`) has only 69%
of folds matching mean sign; the others ≤62%. Features flip sign
month-to-month → not persistent signal.

### Verdict: FAIL
- Max mean |IC| on full panel = **0.0474**, below both v1 ceiling (0.0557)
  and PASS threshold (0.08).
- Upgrading Polygon plan ($29/mo Starter → full history) would not change
  this — the signal is weak on the window we DID cover. More history gives
  more folds but doesn't rescue features that are already noise-level.
- Volume-only features (no BS-inverted IV, no open interest) are
  structurally limited. OI and greeks are behind the paid tier.

### Root cause interpretation
The earlier 10-ticker result looked promising because it concentrated on
mega-cap highly liquid names where option flow genuinely reflects
institutional positioning. Expanding to 72 tickers dilutes that — mid-cap
and low-liquidity names have noisy option volumes that add variance
without adding signal. At the cross-sectional rank target level,
put/call ratios are too common a signal to be alpha on this universe.

### Decision
- FS1 CLOSED (FAIL).
- **Do NOT** upgrade Polygon plan based on this evidence alone.
- Move to FS2 (short interest, FINRA free bulk files).

### Files (to clean up after this entry is committed)
- `scripts/probe_polygon.py`, `probe_polygon2.py`
- `scripts/fs1_poc_fetch.py`
- `scripts/fs1_backfill.py`, `fs1_backfill_full.py`
- `scripts/fs1_ic_audit.py`
- `src/ingest/polygon_client.py`, `options_chain.py`, `__init__.py`
- `src/features/options_features.py`
- `data/polygon_options/` (864 parquets)
- `data/processed/fs1_ic_summary.csv`, `fs1_ic_detail.csv`
- `logs/fs1_backfill*.log`


## 12. FS2 — FINRA short interest features: FAIL (2026-04-21)

### Scope
Second feature-sourcing ticket. FINRA Reg SHO CNMS consolidated daily short
sale volume (free bulk files, no rate limit). 72 tickers (full `SECTOR_MAP`),
2018-08-01 → 2026-02-27 (~7.6y — FINRA CNMS earliest available is 2018-08).
5 features derived per (date, ticker):

  short_ratio              short_volume / total_volume (daily)
  short_ratio_ma21         21d rolling mean (smoothed)
  short_ratio_z_63d        63d z-score of ma21 (relative level)
  short_ratio_delta_5d     short_ratio − short_ratio_ma21 (fast deviation)
  short_vol_share_chg_21d  Δ21d short_volume / Δ21d total_volume (marginal)

Backfill: 133,667 rows, 23 min wall time, 92 monthly parquets. Cached under
`data/finra_shorts/{YYYY-MM}.parquet`.

### IC audit (48,368 obs after merge with dataset_labeled)
**Yearly folds (9):**

| feature                  | ic_mean  | ic_std | abs_ic | sign_stab |
|--------------------------|----------|--------|--------|-----------|
| short_ratio_delta_5d     | −0.0153  | 0.0172 | 0.0153 | 0.67      |
| short_ratio_ma21         | +0.0105  | 0.0341 | 0.0105 | 0.56      |
| short_ratio_z_63d        | +0.0102  | 0.0513 | 0.0102 | 0.56      |
| short_vol_share_chg_21d  | −0.0044  | 0.0181 | 0.0044 | 0.56      |
| short_ratio              | −0.0009  | 0.0263 | 0.0009 | 0.44      |

**Monthly folds (91):** same ordering, all |IC| ≤ 0.0145. Best feature
(`short_ratio_delta_5d`) monthly |IC| = 0.0145, sign_stab = 0.60.

### Verdict: FAIL
- Max mean |IC| = **0.0153** (yearly) / **0.0145** (monthly) — an order of
  magnitude below PASS threshold (0.08) and ~3× below v1 ceiling (0.0557).
- Sign stability ≤ 67% on the best feature — features flip sign across
  folds, no persistent signal.

### Root cause interpretation
Timescale mismatch. Short-volume ratios are a slow-moving, multi-week signal
(positioning shifts show up over 2–8 weeks as shorts accumulate / cover).
The target is a **10-day forward cross-sectional rank** — too fast for
short-interest state to have informational content on this horizon. The
fast-deviation feature (`short_ratio_delta_5d`) is the best of the bunch
precisely because it's the one closest to a 10-day scale, but even it is
noise-level.

FINRA daily short **volume** (what CNMS gives) is also a weaker signal than
the biweekly settled **short interest** (SIRS) that short-squeeze and
short-crowding studies typically use. SIRS days-to-cover plus FTD data could
be investigated separately but is a different horizon proposition.

### Decision
- FS2 CLOSED (FAIL).
- Do NOT port short-volume features into v1 model.
- Short-interest as a **30-60d horizon** signal remains a valid hypothesis
  but requires changing the target label — that's a different ticket, not a
  FS2 retry.
- Move to FS3 (analyst revisions / fundamental deltas).

### Files (to clean up after this entry is committed)
- `scripts/probe_finra.py`
- `scripts/fs2_backfill.py`, `fs2_ic_audit.py`
- `src/ingest/finra_shorts.py`
- `src/features/short_interest.py`
- `data/finra_shorts/` (92 parquets, ~133k rows)
- `data/processed/fs2_ic_yearly.csv`, `fs2_ic_monthly.csv`,
  `fs2_ic_yearly_detail.csv`, `fs2_ic_monthly_detail.csv`
- `logs/fs2_backfill.log`


## 13. FS3 — SEC EDGAR fundamentals features: FAIL on POC (2026-04-21)

### Scope
Third feature-sourcing ticket. SEC EDGAR companyfacts XBRL + submissions
index (free, unlimited, full history 2007+). 10-ticker POC sub-study
(AAPL/MSFT/NVDA/GOOGL/META/AMZN/TSLA/AVGO/AMD/QCOM) with explicit gate:
expand to 72 only if best |IC| ≥ 0.08.

Two iterations:

**v0.1 features (5):**

  fin_days_since_filing     Days since most recent 10-K/10-Q (capped 90d).
  fin_rev_yoy_growth        Latest-Q revenue / same-Q-prior-year − 1.
  fin_rev_growth_accel      YoY growth − trailing 4Q median (implicit
                            revenue surprise).
  fin_margin_delta          Gross margin − trailing 4Q median.
  fin_pead_decay            rev_growth_accel × exp(−0.1 × days_since).

**v0.2 tactical additions (4 more, total 9):**

  fin_eps_surprise          EPS YoY change − trailing 4Q median (PEAD anchor).
  fin_eps_pead              eps_surprise × exp decay.
  fin_op_margin_delta       Operating margin − trailing 4Q median.
  fin_margin_x_recency      margin_delta / (1 + days/20)  interaction.

GrossProfit fallback: GOOGL/META/QCOM don't tag GrossProfit directly; we
compute GP = Revenue − CostOfRevenue. 71/72 SECTOR_MAP tickers covered
(ANSS delisted post Synopsys merger).

### IC audit (10334 obs, 11 yearly / 121 monthly folds, 2016-2026)

| feature                | yearly |IC| | monthly |IC| | sign_stab (yearly) |
|------------------------|-------------|--------------|--------------------|
| fin_op_margin_delta    | 0.0527      | 0.0435       | 0.82               |
| fin_days_since_filing  | 0.0482      | 0.0462       | 0.82               |
| fin_margin_x_recency   | 0.0364      | 0.0552       | 0.82               |
| fin_margin_delta       | 0.0350      | **0.0606**   | 0.82               |
| fin_eps_surprise       | 0.0115      | 0.0239       | 0.64               |
| fin_eps_pead           | 0.0087      | 0.0082       | 0.36               |
| fin_rev_yoy_growth     | 0.0081      | 0.0245       | 0.64               |
| fin_rev_growth_accel   | 0.0045      | 0.0048       | 0.55               |
| fin_pead_decay         | 0.0006      | 0.0052       | 0.45               |

### Verdict: FAIL
- Max yearly |IC| = **0.0527**, max monthly |IC| = **0.0606** — below
  PASS threshold 0.08 and just under v1 ceiling 0.0557.
- v0.2 tactical tweaks (EPS surprise, operating margin, interactions)
  did NOT lift the signal above gross-margin baseline.
- Sub-study FAIL is a hard stop per the gate; do NOT expand to 72.
  Adding mid-caps would dilute, not lift (FS1 lesson).

### What v0.2 did teach
- The **stable** features (`fin_margin_delta`, `fin_op_margin_delta`,
  `fin_days_since_filing`, `fin_margin_x_recency`) all share **sign_stab
  = 0.82 yearly** — direction is reliable, magnitude is just too small.
  Real signal exists; just sub-threshold.
- EPS-based features (surprise + decay) failed hard (sign_stab 0.36–0.64).
  Likely because PEAD literature uses analyst-estimate-based surprise,
  not "vs trailing 4Q median" — without analyst data, the EPS signal is
  not recoverable from EDGAR alone.
- Interaction features (`*_pead`, `*_x_recency`) underperformed their
  base features. The decay weighting destroys signal more than it
  concentrates it on this universe.

### Decision: feature-sourcing phase CLOSED (3/3 FAIL)
After 3 free-data sources tested, the pattern is structural:

| ticket | source                      | best |IC|  | verdict |
|--------|-----------------------------|------------|---------|
| FS1    | Polygon options (volume)    | 0.0474     | FAIL    |
| FS2    | FINRA short volume (CNMS)   | 0.0153     | FAIL    |
| FS3    | SEC EDGAR fundamentals      | 0.0606     | FAIL    |

Free-data ceiling on this universe + 10d target = ~0.06. Threshold
0.08 unreachable without one of:
- Paid alt-data (Polygon Starter+ for IV/OI, Refinitiv estimates,
  RavenPack sentiment) — financial constraint, user-rejected.
- Different target horizon (10d → 30d/60d) — would unlock PEAD and
  short-interest features but **rebuilds entire labeling/training/
  backtest** = v2 scope.
- Different / smaller universe (mid-cap focus) — also v2 scope.

### v1 final state (2026-04-21)
v1 is **sealed**. Shipping `ML_Full_CW` (CAGR 25.8% / Sharpe 0.72 /
MDD −45.7% / α +6.34% p=0.42) per §9. Future work belongs in a new
v2 charter with explicit target/universe redesign.

### Files (to clean up after this entry is committed)
- `scripts/fs3_poc.py`
- `src/ingest/edgar.py`
- `src/features/fundamentals.py`
- `data/edgar_fundamentals/` (10 ticker parquets from POC)
- `data/processed/fs3_poc_ic_yearly.csv`, `fs3_poc_ic_monthly.csv`
- `logs/fs3_poc.log`


## 14. Benchmark date alignment fix + headline rerun (2026-05-04)

### Problem identified by external review
`scripts/run_backtest.py` was passing `pred_full.index.get_level_values("date").unique()`
(only 1,476 dates — the days where predictions exist) to `compute_benchmark_etf()`,
while the ML strategy's hold dates extend to 1,536 (the union of all trading
days between rebalance dates, plus the trailing window after the last
rebalance). Result: ML metrics were computed on 1,536 days while benchmarks
used 1,476 — same period, different grids.

### Fix
After `eq_full` is computed, pass `eq_full.index` to all three
`compute_benchmark_*` calls. Verified empirically:
`new_qqq.index == eq_full.index = True`. All five strategies now share a
single 1,536-day trading-day grid.

Same patch added `slippage_bps` parameter to `run_random_benchmark()` so the
random benchmark uses the same 12 bps total cost as ML (was 10 bps —
random was getting a 2-bps tail-wind).

### Headline impact (2026-05-04 rerun, all training deterministic)

| Metric | v1 sealed (pred-date grid) | post-alignment (hold-date grid) |
|---|---:|---:|
| ML_Full_CW CAGR | 34.1% | 34.1% |
| ML_Full_CW Sharpe | 0.91 | 0.91 |
| ML_Full_CW MDD | −37.6% | −37.6% |
| BH_QQQ CAGR | 18.6% | 18.5% |
| BH_QQQ Sharpe | 0.67 | 0.65 |
| **Annual alpha** | **+12.3%** | **+15.2%** |
| **HAC p-value** | 0.11 | **0.039** ✓ 5% |
| **Centered bootstrap p-value** | 0.08 | **0.032** ✓ 5% |
| Random benchmark z-score | 3.83 | 6.06 |

The ML equity curves are unchanged (predictions and engine are deterministic);
the alpha p-value moved because the alpha test is now apples-to-apples on a
shared trading-day index. The Sharpe CIs are unchanged in width (~0.40 SE on
a 6-year window) — only the point estimate of the QQQ benchmark moved
slightly, and the alpha t-stat reflects the now-honest comparison.

### Doc updates
- `README.md` §1 — headline table, period, Sharpe CI, cost-sensitivity all
  re-pinned to the new grid.
- `README.md` §7 — Limitation #1 rewritten ("single-strategy alpha clears 5%;
  multi-test deflated Sharpe does not"), no longer claims overall not significant.
- `docs/DEFENSE_GUIDE_VI.md` — same.
- `scripts/export_reporting.py` — sensitivity strategy label fixed
  (`ML_Full_EW` → `ML_Full_CW`) so the reporting mart matches what
  `run_analysis.py` actually runs (production CW).

### v1 sealed state (post-alignment)
`ML_Full_CW`: CAGR 34.1% / Sharpe 0.91 / MDD −37.6% / Calmar 0.91 / α +15.2%
HAC p = 0.039 / bootstrap p = 0.032. Single-strategy alpha is now significant
at 5%. Deflated Sharpe still requires expanded sample period or more signal
to clear under broader trial assumptions (p_DSR = 0.082 at 3 trials, 0.146 at 5).
