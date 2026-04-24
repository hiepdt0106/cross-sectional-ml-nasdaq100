# ML_Full_Adaptive (Risk Overlay) — Dropped from Pipeline

**Date:** 2026-04-12
**Decision:** Remove Strategy C (`ML_Full_Adaptive`) from `run_backtest.py` and all downstream artifacts.
**Scope:** Drops only the risk-overlay portfolio. Does NOT touch `build_ensemble(method="adaptive")` (EW/CW still depend on it).

---

## What ML_Full_Adaptive was

Strategy C in `scripts/run_backtest.py` — same Top-10 signal as EW/CW, but with a stack of risk overlays intended to reduce exposure in unfavorable regimes:

| Overlay | Config value |
|---|---|
| Regime targeting (`p_high_vol`) | sensitivity 0.45, exposure range 0.55–1.00 |
| Breadth gate (% stocks > 200d SMA) | threshold 0.48, low-exposure 0.65 |
| Benchmark trend gate (QQQ vs 200d SMA) | low-exposure 0.70 |
| Vol targeting | target 14% annualized |
| Drawdown killswitch | arm at −8%, exit at −4%, exposure 0.70 |

## v0 results (post-ML_Base refactor, 7 folds 2020-2026)

| Strategy | CAGR | Sharpe | MDD | Calmar | α vs QQQ | t-stat | p-value |
|---|---|---|---|---|---|---|---|
| ML_Full_EW | 21.9% | 0.64 | −45.4% | 0.48 | +2.68% | 0.35 | 0.72 |
| ML_Full_CW | **25.8%** | **0.72** | −45.7% | **0.56** | +6.34% | 0.80 | 0.42 |
| **ML_Full_Adaptive** | **7.8%** | **0.32** | −24.4% | **0.32** | **−13.35%** | **−2.24** | **0.025** |
| BH_QQQ | 18.6% | 0.67 | −35.1% | 0.53 | — | — | — |

ML_Full_Adaptive is the **only strategy with a statistically significant alpha — and it's significantly negative** at 5%.

## Root cause

Risk overlays cut exposure during the 2020–2025 bull run and did not recover the missed upside. Breakdown:

1. **MDD reduction was real but insufficient**: −45.7% → −24.4% (−21 pp improvement).
2. **CAGR destruction was catastrophic**: 25.8% → 7.8% (−18 pp). Cost/benefit ratio ~0.9:1 on absolute percentages, but on Calmar: 0.56 → 0.32 (**overlay actively harms Calmar**).
3. **Stacking overlays compounded exposure drag**: regime + breadth + trend + vol + dd all firing simultaneously in mid-2022 drawdown → overlay exited near local bottom, re-entered after recovery started → classic whipsaw.
4. **QQQ as alpha benchmark**: overlays reduced exposure during QQQ bull phases (2020-2021, 2023-2024) → overlay lags benchmark by construction even when base signal has positive alpha.

## What was tried (Stage 2 overlay tuning, `notebooks/07_overlay_tuning.ipynb`)

20 combinations tested across 4 lever axes, measured on design (2020-2023) and holdout (2024-2026):

| Lever | Values tried |
|---|---|
| Base weighting | EW, CW |
| Benchmark SMA gate | off, @0.30, @0.50, @0.70 exposure |
| Vol targeting | off, 20% annual |
| Sector cap | off, 0.25, 0.30 |
| Breadth gate | off, on |
| Drawdown stop | off, on |

### Key findings from grid (see `outputs/metrics/overlay_tuning_grid_v2.csv`)

Ranked by full-period Calmar:

| Combo | CAGR | MDD | Sharpe | Calmar |
|---|---|---|---|---|
| **CW_baseline** (no overlay) | **25.8%** | −45.7% | 0.72 | **0.56** |
| CW+sector0.30 | 25.1% | −45.7% | 0.71 | 0.55 |
| EW+vol0.20 | 19.6% | −37.8% | 0.64 | 0.52 |
| CW+SMA@0.50+vol0.20 | 17.8% | −36.4% | 0.66 | 0.49 |
| CW+SMA@0.70 | 21.7% | −44.5% | 0.67 | 0.49 |
| EW_baseline | 21.9% | −45.4% | 0.64 | 0.48 |
| CW+SMA@0.30+vol0.20 | 15.2% | −35.7% | 0.58 | 0.42 |
| CW+SMA@0.30+sector0.30+vol0.20 | 13.8% | −35.7% | 0.52 | 0.39 |
| *(14 more combos)* | *all worse* | | | |

**Pareto wall**: No combo achieved CAGR ≥ 20% AND MDD ≥ −30% simultaneously. Every MDD improvement cost more CAGR than it saved in drawdown terms.

**Winner declared by `overlay_tuning_winner_v2.json`**: `CW_baseline` (no overlay). The tuning notebook itself chose "unconstrained max Calmar_d" and landed on the no-overlay baseline.

## Why drop instead of tune further

- Overlay tuning already sampled a 20-point grid across 4 axes. The Pareto frontier is flat: downside protection and upside participation are substitutes, not complements, in this signal+overlay configuration.
- The base signal (Top-10 ranking on ML_Full ensemble) does not generalize strongly enough that clipping its exposure preserves alpha. Any clip proportionally reduces alpha.
- Upstream improvements (purged CV, meta-labeling, regime-aware training — v1/v2/v3 roadmap) will produce different base signal properties. Overlay tuning should be revisited ONLY after the base signal improves, not in the current regime.
- Shipping candidate (CW_baseline: CAGR 25.8%, Sharpe 0.72, Calmar 0.56) already Pareto-dominates every overlay combo. Keeping Strategy C as a comparison point adds noise to reports and invites re-litigation.

## Action taken

1. Removed Strategy C construction, equity/trade saving, metrics, alpha stats from `scripts/run_backtest.py`.
2. Removed `ML_Full_Adaptive` from `scripts/run_analysis.py` and `scripts/export_reporting.py`.
3. Removed from notebooks 05 (backtesting), 06 (analysis), 07 (overlay tuning).
4. Deleted stale `outputs/equity_adaptive.parquet`, `outputs/metrics/trade_log_adaptive.csv`.
5. Kept `build_ensemble(method="adaptive")` untouched — EW/CW still use it as the ensemble blender.
6. Kept `configs/base.yaml` overlay parameters in place (unused for now; reference config in case overlay revisit happens post-v3).

## Re-enable criteria

Reconsider risk overlays only if ALL of these hold on v1+ base signal:
- Base signal has statistically significant alpha (p < 0.10, HAC t-test vs QQQ).
- Base signal MDD ≥ −35% (overlay headroom to improve meaningfully).
- Base signal CAGR > 30% (overlay CAGR drag becomes tolerable in relative terms).

Until then, ship CW without overlay and invest research budget in upstream (alpha generation, not alpha gating).
