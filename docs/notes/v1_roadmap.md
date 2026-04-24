# v1 Roadmap — Ongoing & Planned Improvements

**Purpose:** survive chat resets. Anyone (human or new Claude session) should be
able to read this file plus [v1_changelog.md](v1_changelog.md) and pick up the
work in the same direction without re-deriving context.

Last updated: 2026-04-21 (v1 SEALED; FS1+FS2+FS3 all FAIL; feature-sourcing phase CLOSED).

---

## v1 STATUS: CLOSED (2026-04-13)

Every improvement lever on the current feature universe has been tried. The P4
audit (v1_changelog §9) confirmed no leakage and revealed the ceiling: max
single-feature cross-fold |IC| is 0.0557 (`rolling_beta_63d`), and 58-feature
ensemble theoretical ceiling is OOS AUC ~0.53–0.56. Measured OOS AUC at 0.51
is within striking distance of that ceiling, and the alpha p-value gap
(0.42/0.72) is a feature-universe problem, not a model problem.

**Shipping state locked in:** `ML_Full_EW` + `ML_Full_CW` with Fix B
(is_unbalance=False) + Fix C (intra-fold 10-day purge) + avg_uniqueness sample
weights, all in `src/models/train.py`. Backtest CAGR 25.8% / Sharpe 0.72 /
MDD −45.7% / α +6.34% ns (ML_Full_CW).

**Do NOT retry any v1 P-item** without a materially different feature source.
All P1–P5 are closed or done; see rejected list below for details.

**Next phase** is not v1.x — it's a data-sourcing project. See the
"Feature-sourcing phase" section below.

---

## Current project state (snapshot)

- **Shipping strategies:** `ML_Full_EW`, `ML_Full_CW`. `ML_Full_Adaptive` risk
  overlay was dropped 2026-04-12 (α=−13.35%, p=0.025). The ensemble method
  `adaptive` (different thing, confusingly named) is still used by EW/CW.
- **Base model:** LightGBM via `fit_lgbm_optuna` in [src/models/train.py](../../src/models/train.py).
  Classifier path + ranker path, Optuna tuning, single-block 85/15 validation.
- **Target:** `alpha_ext_label` (cross-sectional rank). `tb_label` available but
  not used (see v1_changelog §4 Step 2 — marginal).
- **Validation:** single-block 85/15 date-tail with **10-day intra-fold purge**
  (Fix C, ported 2026-04-13). Walk-forward K-fold explored and rejected (Fix D
  FAIL — single-block OOS +88 bps despite looser CV gap).
- **Sample weights:** López de Prado `avg_uniqueness` from triple-barrier `t1`,
  active whenever `t1` column is present. Validated +77 bps OOS mean, −10 bps
  std. Module: [src/models/sample_weights.py](../../src/models/sample_weights.py).
- **`is_unbalance`:** hardcoded `False` everywhere in train.py (Fix B). Correct
  for ~50/50 `alpha_ext_label`. If ever switching to `tb_label` (54% pos),
  reconsider — but prefer `scale_pos_weight` over `is_unbalance` regardless.
- **Meta-labeling:** not in production. Tried twice, both weak (see §4 Step 2).

## What's already locked in

See [v1_changelog.md](v1_changelog.md) §3–§4. Short version: uniqueness weights
+ Fix B (is_unbalance=False) + Fix C (intra-fold purge) are in `src/models/train.py`.
Everything else from the v1 exploration was rejected or parked.

---

## Planned / in-flight work

Ordered by expected value. Each item lists **why**, **how**, **success criterion**
so a future session can judge whether it's still worth doing.

### P1 — Regime-conditional base model — **CLOSED, FULL FAIL (2026-04-13)**

Four variants tested, all failed the success criterion. See v1_changelog §8 for
tables, per-variant deltas, and root-cause interpretation.

- **P1**  per-regime heads on `p_high_vol` — Δ agg −49 bps (catastrophic 2022)
- **P1b** `feat × p_high_vol` interactions — Δ agg +8 bps (noise), 2× std
- **P1c** monolithic LGBM, regime-weighted samples — Δ agg −49 bps
- **P1d** per-regime heads on `vxn_zscore` — Δ agg −23 bps (but +91 bps in 2022!)

**Why this is structurally closed, not just "not today":** P1d's 2022 win flipped
2023's loss vs P1's pattern. Same year, opposite sign depending on regime axis →
axis-dependent noise, not signal. Combined with mechanism-independent failure
(split / interact / reweight all lose), this rules out regime conditioning as a
reliable lever on top of the current feature set.

Do NOT retry without: (a) a materially stronger base signal (upstream work), or
(b) a regime axis not considered (HMM hidden state is the only remaining one,
but the axis-flip evidence makes it a long shot).

### P2 — Meta-labeling v3 — **CLOSED (blocked permanently on P1)**

P2's precondition was a working regime base to refine. P1 is dead, so P2 is
doubly dead. Two prior meta attempts (see v1_changelog §4 Step 2) also failed
on their own merits. Revisit only if a structural upstream change (new features,
new labeling) lifts base AUC materially above ~0.51.

### P4 — Feature leakage audit — **DONE, PASS (2026-04-13)**

Static code review of `src/features/*.py` + `src/labeling/*.py`: clean, no
look-ahead, all target-adjacent columns in the `forbidden_features` guard.
Data-level audit: no forbidden-name leaks, 0 non-numeric traps, 0 coverage
flags, 0 features with |IC| ≥ 0.10 in any fold. Max cross-fold mean |IC| =
0.0557 (`rolling_beta_63d`), and per-feature `ic_std` is uniformly tiny —
the fingerprint of a leak-free but genuinely weak feature set.

This PASS is the diagnosis, not a null result: the feature universe caps OOS
AUC at roughly 0.53–0.56, which explains why every P1/P2/CV experiment bounced
off the 0.51 ceiling. See v1_changelog §9 for full tables and interpretation.

### P3 — Purge parameter sweep — **SKIPPED (deprioritized by P4)**

Any purge-value win would be ≤ 20 bps at the OOS AUC level and wouldn't move
the portfolio-level alpha p-value — not worth the cycles on the current feature
universe. Left in the backlog only as cheap sanity work if the feature set
changes materially. `purge_days = 10` remains locked in `_make_date_block_val_mask`.

### P5 — Backtest re-run with ported train.py — **DONE (2026-04-13)**

Ran end-to-end. Results match pre-port baseline exactly (ML_Full_CW 25.8% CAGR,
Sharpe 0.72, MDD −45.7%, α +6.34% p=0.42). The Fix B/C/uniqueness port is
metric-level correct but did not translate to portfolio alpha — the +77 bps OOS
daily-AUC lift is not visible through top-10 selection. See v1_changelog §7.

---

## Feature-sourcing phase — CLOSED 2026-04-21 (3/3 FAIL)

After v1 closed, three free-data feature sources were tested. All failed the
PASS threshold (cross-fold mean |IC| ≥ 0.08). Pattern is structural:

| ticket | source                      | best |IC|  | verdict |
|--------|-----------------------------|------------|---------|
| FS1    | Polygon options (volume)    | 0.0474     | FAIL    |
| FS2    | FINRA short volume (CNMS)   | 0.0153     | FAIL    |
| FS3    | SEC EDGAR fundamentals      | 0.0606     | FAIL    |

Free-data ceiling on this universe + 10d target = ~0.06. Threshold 0.08
unreachable without one of: paid alt-data (rejected), different target
horizon (= v2), different universe (= v2). See v1_changelog §11–§13.

**v1 is SEALED 2026-04-21.** Shipping `ML_Full_CW` final state per §9 and §13.
Future improvement work belongs to a v2 charter — see "v2 candidate scopes"
below.

### FS1 — Options-derived features — **CLOSED, FAIL (2026-04-20)**
Attempted with Polygon.io free tier (5 req/min, 2-year window, no IV/OI).
72 tickers × 12 monthly expiries (~864 units, 16k API calls, ~56h).
Volume-only features (put/call vol ratio, dollar vol ratio, ATM price
norm, call vol z, strike skew). Max cross-fold mean |IC| = 0.047 on
5,338 obs / 13 monthly folds — below both v1 ceiling (0.0557) and PASS
threshold (0.08). See v1_changelog §11.

**Do NOT retry** with paid Polygon tier unless a fundamentally different
option feature (BS-inverted IV, 25-delta greeks, term structure) is
proposed — volume-only signal is too weak on this universe.

### FS2 — Short interest — **CLOSED, FAIL (2026-04-21)**
Attempted with FINRA Reg SHO CNMS daily short-volume bulk files (free,
2018-08-01+ coverage). 72 tickers × ~7.6y, 133k rows, 5 features
(short_ratio and derivatives). Max cross-fold mean |IC| = 0.0153 on
48,368 obs / 9 yearly folds / 91 monthly folds — order of magnitude
below PASS threshold (0.08) and ~3× below v1 ceiling (0.0557). See
v1_changelog §12.

**Root cause:** timescale mismatch — short-volume state is a multi-week
signal, target is 10-day forward rank. **Do NOT retry** on the same
target. Short-interest as a 30-60d signal remains a valid hypothesis but
requires changing the target label (separate ticket, not FS2 retry).
Biweekly SIRS short-interest + FTD (not what CNMS gives) is also a
different proposition worth separate investigation if horizon changes.

### FS3 — Fundamentals (SEC EDGAR) — **CLOSED, FAIL (2026-04-21)**
Attempted with SEC EDGAR companyfacts XBRL + submissions index (free,
unlimited, full history). 10-ticker POC sub-study, 9 features across two
iterations (v0.1 revenue/margin/PEAD-decay; v0.2 EPS surprise + operating
margin + interactions). Max yearly |IC| = 0.0527 (`fin_op_margin_delta`),
monthly |IC| = 0.0606 (`fin_margin_delta`) on 10,334 obs / 11 yearly /
121 monthly folds. Sub-study FAIL → no full-panel expansion. See
v1_changelog §13.

**Root cause:** EPS surprise without analyst estimates is structurally
weak (PEAD literature uses analyst-estimate-based surprise; "vs trailing
4Q median" is too noisy a proxy). Margin deltas have stable direction
(sign_stab 0.82) but magnitudes are sub-threshold. Interactions
(decay-weighted, recency-scaled) underperformed base features.

## v2 candidate scopes (NEW project — not v1.x)

Feature-sourcing phase exhausted free-data avenues at the v1 ceiling.
Two structural changes could unlock new signal classes; each is its
own project:

### v2-A: Different target horizon (10d → 30d/60d)
- Unlocks PEAD/short-interest features that have multi-week effective
  horizons (FS2 + FS3 features that failed at 10d may pass at 30-60d).
- Cost: rebuild labeling pipeline, retrain all models, redo backtest,
  redo regime analysis. ~2-3 weeks of work.
- Risk: longer-horizon labels also have lower data density per ticker.

### v2-B: Smaller / different universe
- Mid-cap focus (drop the mega-caps where alpha is heavily arbitraged).
- Cost: changes research foundation; everything from features through
  backtest needs re-validation on new universe.
- Risk: smaller universe → fewer cross-sectional folds → noisier IC.

### v2-C: Paid alt-data
- Polygon Starter+ ($29/mo) for IV/OI; Refinitiv estimates (~$1k+/mo);
  RavenPack sentiment (vendor-gated).
- Already rejected by user (financial constraint). If reconsidered,
  start with Polygon Starter+ (lowest commitment).

**Rule:** pick exactly ONE of v2-A/B/C — combining is too many
moving parts for a single project iteration.

---

## Rejected / do not retry without new info

- **Pure K-fold CV** (contiguous blocks, LdP style): unstable across regimes,
  OOS underperformed single-block.
- **Walk-forward expanding K-fold with embargo:** honest CV but OOS −88 bps
  vs single-block. Don't port unless the feature set or target structure
  changes meaningfully.
- **`tb_label` as base target:** marginal (Step 2). The +2.6× data was not
  enough to beat `alpha_ext_label`.
- **Meta-labeling with full feature set:** turns meta into redundant base.
- **Regime-conditional base model (all four P1 variants):** per-regime heads,
  feature interactions, regime-weighted training, alt regime axis — all
  FAILed (v1_changelog §8). Evidence is structural (axis-dependent noise),
  not tactical. Do not retry without a materially stronger base signal or a
  fundamentally new regime signal source.
- **Meta-labeling v3:** was P2, blocked on P1 which is now dead.

---

## How to resume in a new chat session

Paste this block into the new conversation's first message:

```
This project is at c:\Users\Main 1.9\Desktop\8.
Continue v1 work per docs/notes/v1_roadmap.md and docs/notes/v1_changelog.md.

State:
- Fix B (is_unbalance=False) and Fix C (intra-fold purge) are already
  ported into src/models/train.py along with avg_uniqueness weights.
- Shipping strategies ML_Full_EW, ML_Full_CW. ML_Full_Adaptive dropped.
- Base target alpha_ext_label. tb_label exists but not used.
- Validation: single-block 85/15 with 10-day purge gap. Walk-forward
  k-fold was tried and rejected.
- Meta-labeling tried twice, both weak — parked behind P1.

v1 is CLOSED (2026-04-13). Every P-item is done or closed:
  P1/P2: CLOSED FAIL (regime conditioning is axis-agnostic noise)
  P3:    SKIPPED (won't move the ceiling)
  P4:    DONE, PASS (audit CLEAN — feature universe is the ceiling)
  P5:    DONE (backtest unchanged vs pre-port)

Shipping: ML_Full_CW (CAGR 25.8% / Sharpe 0.72 / α +6.34% p=0.42)
is the realistic ceiling of the current OHLCV/macro feature set.

Next phase is NOT v1.x. It's a feature-sourcing project. Backlog
in docs/notes/v1_roadmap.md: FS1 (options data, highest payoff),
FS2 (short interest, cheapest), FS3 (analyst / fundamentals),
FS4 (sentiment, defer), FS5 (microstructure, out of scope).

Before acting on anything v1-labeled, read v1_changelog.md §9
(P4 audit, v1 closeout, ceiling explanation). Do NOT retry any
v1 P-item without a materially new feature source.
```

### Context that isn't in code / git

- **User works autonomously**: except for deletions, don't ask permission to
  edit. Just act and report. (User feedback 2026-04-13.)
- **User wants results first, cleanup second**: run experiments, report
  PASS/FAIL, only then port + delete prototypes. Don't interleave.
- **Language**: user writes in Vietnamese; respond in Vietnamese when they do.
- **"adaptive" is overloaded**: `ML_Full_Adaptive` (strategy, dropped) ≠
  `build_ensemble(method="adaptive")` (ensemble combiner, still used). Don't
  conflate.

---

## File map (what lives where)

- [src/models/train.py](../../src/models/train.py) — walk-forward loop,
  `fit_lgbm_optuna`, `_make_date_block_val_mask`, `_build_lgbm_training_payload`.
- [src/models/sample_weights.py](../../src/models/sample_weights.py) —
  `avg_uniqueness(df, t1_col)`.
- [src/splits/walkforward.py](../../src/splits/walkforward.py) — year-level
  fold generator with its own purge+embargo (separate from intra-fold purge).
- [configs/base.yaml](../../configs/base.yaml) — `walkforward.first_test_year`,
  `labeling.horizon`.
- [docs/notes/v1_changelog.md](v1_changelog.md) — audit trail, §1–§5.
- [docs/notes/adaptive_overlay_dropped.md](adaptive_overlay_dropped.md) —
  Strategy C drop rationale (2026-04-12).
- [scripts/run_backtest.py](../../scripts/run_backtest.py) — end-to-end
  backtest entry point.
