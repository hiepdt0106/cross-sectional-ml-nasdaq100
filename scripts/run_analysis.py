"""
scripts/run_analysis.py
────────────────────────
Consolidated analysis:

1. Sensitivity analysis (top-K, cost, rebalance freq)
2. Feature importance (RF + LGBM, 7 folds)
3. Conditional analysis (stress vs normal)
4. Benchmark comparisons
5. Generate figures
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts._common import add_common_args, setup_logging
from src.config import load_config
from src.backtest import (
    BacktestEngineConfig,
    run_backtest,
    compute_metrics,
)
from src.utils.io import load

log = logging.getLogger(__name__)


def sensitivity_top_k(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    ks: list[int] = None,
    base_cfg: BacktestEngineConfig = None,
) -> pd.DataFrame:
    """Sensitivity analysis: vary top-K. All other settings inherit from base_cfg
    so the comparison is against the same production strategy."""
    from dataclasses import replace
    if ks is None:
        ks = [3, 5, 8, 10, 15, 20]
    if base_cfg is None:
        base_cfg = BacktestEngineConfig()

    results = []
    for k in ks:
        cfg_k = replace(base_cfg, top_k=k)
        eq, _ = run_backtest(df, pred_df, cfg_k)
        m = compute_metrics(eq)
        m["top_k"] = k
        results.append(m)
        log.info(f"  K={k}: CAGR={m.get('CAGR', 0):.1%} Sharpe={m.get('Sharpe', 0):.2f}")

    return pd.DataFrame(results)


def sensitivity_cost(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    costs: list[float] = None,
    base_cfg: BacktestEngineConfig = None,
) -> pd.DataFrame:
    """Sensitivity analysis: vary transaction cost. All other settings inherit
    from base_cfg so the comparison is against the same production strategy."""
    from dataclasses import replace
    if costs is None:
        costs = [0, 5, 10, 15, 20, 30]
    if base_cfg is None:
        base_cfg = BacktestEngineConfig()

    results = []
    for c in costs:
        cfg_c = replace(base_cfg, cost_bps=c)
        eq, _ = run_backtest(df, pred_df, cfg_c)
        m = compute_metrics(eq)
        m["cost_bps"] = c
        results.append(m)
        log.info(f"  Cost={c}bps: CAGR={m.get('CAGR', 0):.1%} Sharpe={m.get('Sharpe', 0):.2f}")

    return pd.DataFrame(results)


def sensitivity_rebalance(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    freqs: list[int] = None,
    base_cfg: BacktestEngineConfig = None,
) -> pd.DataFrame:
    """Sensitivity analysis: vary rebalance frequency. All other settings inherit
    from base_cfg so the comparison is against the same production strategy."""
    from dataclasses import replace
    if freqs is None:
        freqs = [5, 10, 15, 21]
    if base_cfg is None:
        base_cfg = BacktestEngineConfig()

    results = []
    for f in freqs:
        cfg_f = replace(base_cfg, rebalance_days=f)
        eq, _ = run_backtest(df, pred_df, cfg_f)
        m = compute_metrics(eq)
        m["rebalance_days"] = f
        results.append(m)
        log.info(f"  Rebal={f}d: CAGR={m.get('CAGR', 0):.1%} Sharpe={m.get('Sharpe', 0):.2f}")

    return pd.DataFrame(results)


def plot_equity_curves(
    equities: dict[str, pd.DataFrame],
    fig_path: Path,
    title: str = "Equity Curves — ML Trading Strategy",
):
    """Plot equity curves for multiple strategies/benchmarks."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {
        "ML Full EW": "#2196F3",
        "ML Full CW": "#4CAF50",
        "B&H QQQ": "#FF9800",
        "B&H MCap Top-10": "#F44336",
        "B&H Full": "#9E9E9E",
    }

    for name, eq in equities.items():
        if len(eq) == 0:
            continue
        color = colors.get(name, None)
        ax.plot(eq.index, eq["equity"], label=name, linewidth=1.5, color=color)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {fig_path}")


def plot_drawdown(
    equities: dict[str, pd.DataFrame],
    fig_path: Path,
):
    """Plot drawdown curves."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for name, eq in equities.items():
        if len(eq) == 0:
            continue
        running_max = eq["equity"].cummax()
        dd = eq["equity"] / running_max - 1
        ax.fill_between(dd.index, dd, 0, alpha=0.3, label=name)

    ax.set_title("Drawdown — ML Trading Strategy", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {fig_path}")


def plot_annual_returns(
    equities: dict[str, pd.DataFrame],
    fig_path: Path,
):
    """Plot annual returns bar chart."""
    annual = {}
    for name, eq in equities.items():
        if len(eq) == 0:
            continue
        rets = eq["daily_ret"].copy()
        rets.index = pd.to_datetime(rets.index)
        yearly = (1 + rets).groupby(rets.index.year).prod() - 1
        annual[name] = yearly

    if not annual:
        return

    ann_df = pd.DataFrame(annual)
    ax = ann_df.plot(kind="bar", figsize=(14, 6), width=0.75)
    ax.set_title("Annual Returns — ML Trading Strategy", fontsize=14)
    ax.set_ylabel("Return")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {fig_path}")


def run_analysis(config_path: str | Path | None = None):
    """Run the reporting and analysis pipeline."""
    cfg = load_config(config_path) if config_path else load_config()

    # ── Load data ──
    log.info("Loading backtest results ...")
    df = load(cfg.dir_processed / "dataset_featured.parquet")
    pred_full = load(cfg.dir_processed / "predictions_ens_full.parquet")

    eq_full = load(cfg.dir_outputs / "equity_full.parquet")

    # Load optional benchmark outputs if present.
    try:
        eq_bh_qqq = load(cfg.dir_outputs / "equity_benchmark_qqq.parquet")
    except FileNotFoundError:
        eq_bh_qqq = pd.DataFrame()
    try:
        eq_bh_mcap = load(cfg.dir_outputs / "equity_benchmark_mcap10.parquet")
    except FileNotFoundError:
        eq_bh_mcap = pd.DataFrame()
    try:
        eq_bh_full = load(cfg.dir_outputs / "equity_benchmark_full.parquet")
    except FileNotFoundError:
        eq_bh_full = pd.DataFrame()
    try:
        eq_cw = load(cfg.dir_outputs / "equity_cw.parquet")
    except FileNotFoundError:
        eq_cw = pd.DataFrame()

    fig_dir = cfg.dir_figures
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = cfg.dir_outputs / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Base config for sensitivity must match production CW so the comparison
    # is meaningful (otherwise sensitivity_cost.csv reports a stripped-down
    # EW baseline whose Sharpe is structurally lower than the headline).
    from src.config import SECTOR_MAP
    base_cfg = BacktestEngineConfig(
        top_k=cfg.strategy.top_k,
        rebalance_days=cfg.strategy.rebalance_days,
        cost_bps=cfg.backtest.cost_bps,
        slippage_bps=cfg.backtest.slippage_bps,
        initial_capital=cfg.backtest.initial_capital,
        confidence_weighted=True,  # match production CW headline
        max_weight_cap=cfg.strategy.max_weight_cap,
        hold_buffer=cfg.strategy.hold_buffer,
        hold_score_tolerance=cfg.strategy.hold_score_tolerance,
        signal_anchor_weight=cfg.strategy.signal_anchor_weight,
        signal_anchor_features=tuple(cfg.strategy.signal_anchor_features),
        sector_mode=cfg.backtest.sector_mode,
        sector_max_weight=cfg.backtest.sector_max_weight,
        sector_map=SECTOR_MAP,
    )

    # ══════════════════════════════════════════════════════════════
    # 1. SENSITIVITY ANALYSIS
    # ══════════════════════════════════════════════════════════════
    log.info("\n" + "═" * 60)
    log.info("▶ Sensitivity Analysis")
    log.info("═" * 60)

    log.info("\n  Top-K sensitivity:")
    sens_k = sensitivity_top_k(df, pred_full, base_cfg=base_cfg)
    sens_k.to_csv(metrics_dir / "sensitivity_topk.csv", index=False)

    log.info("\n  Cost sensitivity:")
    sens_cost = sensitivity_cost(df, pred_full, base_cfg=base_cfg)
    sens_cost.to_csv(metrics_dir / "sensitivity_cost.csv", index=False)

    log.info("\n  Rebalance frequency sensitivity:")
    sens_reb = sensitivity_rebalance(df, pred_full, base_cfg=base_cfg)
    sens_reb.to_csv(metrics_dir / "sensitivity_rebalance.csv", index=False)

    # ══════════════════════════════════════════════════════════════
    # 1b. DEFLATED SHARPE RATIO (Bailey-LdP 2014)
    # ══════════════════════════════════════════════════════════════
    log.info("\n" + "═" * 60)
    log.info("▶ Deflated Sharpe Ratio (multi-test correction)")
    log.info("═" * 60)
    from scipy import stats as sps
    from src.backtest.engine import deflated_sharpe_ratio
    rf_annual = 0.04  # matches compute_metrics default
    dsr_rows = []
    for strat_name, eq in [("ML_Full_CW", eq_cw), ("ML_Full_EW", eq_full)]:
        if eq is None or len(eq) == 0:
            continue
        ret = eq["daily_ret"].dropna()
        excess = ret - (rf_annual / 252)
        N = len(excess)
        sr_ann = excess.mean() / excess.std(ddof=1) * np.sqrt(252)
        skew = float(sps.skew(excess))
        kurt_ex = float(sps.kurtosis(excess, fisher=True))
        for n_trials in [3, 5, 10, 20]:
            d = deflated_sharpe_ratio(sr_ann, n_days=N, n_trials=n_trials,
                                       skew=skew, kurtosis_excess=kurt_ex)
            d["strategy"] = strat_name
            dsr_rows.append(d)
            log.info(f"  {strat_name} n_trials={n_trials:>2d}: obs_SR={d['observed_sharpe']:.3f} "
                     f"E[max]={d['expected_max_sharpe']:.3f} p_DSR={d['p_value']:.4f}")
    pd.DataFrame(dsr_rows).to_csv(metrics_dir / "deflated_sharpe.csv", index=False)
    log.info(f"  Saved: {metrics_dir / 'deflated_sharpe.csv'}")

    # ══════════════════════════════════════════════════════════════
    # 2. FIGURES
    # ══════════════════════════════════════════════════════════════
    log.info("\n" + "═" * 60)
    log.info("▶ Generating Figures")
    log.info("═" * 60)

    equities = {"ML Full EW": eq_full}
    if len(eq_cw) > 0:
        equities["ML Full CW"] = eq_cw
    if len(eq_bh_qqq) > 0:
        equities["B&H QQQ"] = eq_bh_qqq
    if len(eq_bh_mcap) > 0:
        equities["B&H MCap Top-10"] = eq_bh_mcap
    if len(eq_bh_full) > 0:
        equities["B&H Full"] = eq_bh_full

    plot_equity_curves(equities, fig_dir / "equity_curves.png")
    plot_drawdown(equities, fig_dir / "drawdown.png")
    plot_annual_returns(equities, fig_dir / "annual_returns.png")

    log.info("\n✓ Analysis complete!")


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_analysis(args.config)


if __name__ == "__main__":
    main()
