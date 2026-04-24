"""Run the research backtests and export comparison tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import logging

import pandas as pd

from scripts._common import add_common_args, setup_logging
from src.backtest import (
    BacktestEngineConfig,
    block_bootstrap_alpha,
    compute_alpha_stats,
    compute_benchmark,
    compute_benchmark_etf,
    compute_benchmark_mcap_top10,
    compute_metrics,
    run_backtest,
    run_random_benchmark,
    summarize_trade_log,
)
from src.config import SECTOR_MAP, load_config
from src.utils.io import load, save

log = logging.getLogger(__name__)



def _annual_return_series(eq_df: pd.DataFrame) -> pd.Series:
    """Aggregate daily returns into calendar-year returns."""
    if len(eq_df) == 0:
        return pd.Series(dtype=float)
    rets = eq_df["daily_ret"].copy()
    rets.index = pd.to_datetime(rets.index)
    yearly = (1 + rets).groupby(rets.index.to_period("Y")).prod() - 1
    yearly.index = yearly.index.to_timestamp("Y")
    yearly.index.name = "year_end"
    return yearly



def _engine_from_cfg(cfg, **overrides) -> BacktestEngineConfig:
    """Construct an engine config from project config plus overrides."""
    params = dict(
        top_k=cfg.strategy.top_k,
        rebalance_days=cfg.strategy.rebalance_days,
        cost_bps=cfg.backtest.cost_bps,
        slippage_bps=cfg.backtest.slippage_bps,
        initial_capital=cfg.backtest.initial_capital,
        confidence_weighted=False,
        max_weight_cap=cfg.strategy.max_weight_cap,
        hold_buffer=cfg.strategy.hold_buffer,
        hold_score_tolerance=cfg.strategy.hold_score_tolerance,
        signal_anchor_weight=cfg.strategy.signal_anchor_weight,
        signal_anchor_features=tuple(cfg.strategy.signal_anchor_features),
        regime_targeting=False,
        regime_col=cfg.backtest.regime_col,
        regime_sensitivity=cfg.backtest.regime_sensitivity,
        regime_min_exposure=cfg.backtest.regime_min_exposure,
        regime_max_exposure=cfg.backtest.regime_max_exposure,
        vol_targeting=False,
        target_vol=cfg.backtest.target_vol,
        vol_lookback_days=cfg.backtest.vol_lookback_days,
        vol_min_scale=cfg.backtest.vol_min_scale,
        vol_max_scale=cfg.backtest.vol_max_scale,
        dd_threshold=0.0,
        dd_exit=cfg.backtest.dd_exit,
        dd_exposure=cfg.backtest.dd_exposure,
        sector_max_weight=cfg.backtest.sector_max_weight,
        sector_map=SECTOR_MAP,
    )
    params.update(overrides)
    return BacktestEngineConfig(**params)



def run_all_backtests(config_path: str | Path | None = None):
    """Run baseline and risk-managed backtests for the current ensemble outputs."""
    cfg = load_config(config_path) if config_path else load_config()

    pred_full_path = cfg.dir_processed / "predictions_ens_full.parquet"
    pred_all_full_path = cfg.dir_processed / "predictions_all_full.parquet"
    dataset_path = cfg.dir_processed / "dataset_featured.parquet"

    log.info("Loading data ...")
    df = load(dataset_path)
    pred_full = load(pred_full_path)
    ml_dates = pred_full.index.get_level_values("date").unique()

    log.info("\n" + "═" * 60)
    log.info("▶ Strategy A: ML Top-10 Equal-Weight (Full, pure signal)")
    log.info("═" * 60)
    cfg_ew = _engine_from_cfg(cfg, confidence_weighted=False, signal_anchor_weight=0.0)
    eq_full, trades_full = run_backtest(df, pred_full, cfg_ew)
    save(eq_full, cfg.dir_outputs / "equity_full.parquet")

    log.info("\n" + "═" * 60)
    log.info("▶ Strategy B: ML Top-10 Confidence-Weighted (Full, pure signal)")
    log.info("═" * 60)
    cfg_cw = _engine_from_cfg(
        cfg,
        confidence_weighted=True,
        max_weight_cap=cfg.strategy.max_weight_cap,
        signal_anchor_weight=0.0,
    )
    eq_cw, trades_cw = run_backtest(df, pred_full, cfg_cw)
    save(eq_cw, cfg.dir_outputs / "equity_cw.parquet")

    log.info("\n" + "═" * 60)
    log.info("▶ Benchmark 1: Buy & Hold QQQ ETF (primary, no look-ahead)")
    log.info("═" * 60)
    eq_bh_qqq = compute_benchmark_etf(
        df,
        ml_dates,
        initial_capital=cfg.backtest.initial_capital,
        bench_col="bench_close",
    )
    save(eq_bh_qqq, cfg.dir_outputs / "equity_benchmark_qqq.parquet")
    save(eq_bh_qqq, cfg.dir_outputs / "equity_benchmark.parquet")

    log.info("\n▶ Benchmark 2: Buy & Hold Top-10 Market Cap (look-ahead, reference only)")
    eq_bh_mcap = compute_benchmark_mcap_top10(
        df,
        ml_dates,
        mcap_tickers=cfg.benchmark_mcap.tickers,
        initial_capital=cfg.backtest.initial_capital,
    )
    save(eq_bh_mcap, cfg.dir_outputs / "equity_benchmark_mcap10.parquet")

    log.info("\n▶ Benchmark 3: Buy & Hold Full Universe")
    eq_bh_full = compute_benchmark(
        df,
        ml_dates,
        initial_capital=cfg.backtest.initial_capital,
    )
    save(eq_bh_full, cfg.dir_outputs / "equity_benchmark_full.parquet")

    log.info("\n" + "═" * 60)
    log.info("▶ Benchmark 3: Random Top-10")
    log.info("═" * 60)
    random_results = run_random_benchmark(
        df,
        pred_full,
        n_iterations=cfg.random_benchmark.n_iterations,
        top_k=cfg.strategy.top_k,
        rebalance_days=cfg.strategy.rebalance_days,
        cost_bps=cfg.backtest.cost_bps,
        initial_capital=cfg.backtest.initial_capital,
        seed=cfg.random_benchmark.seed,
    )
    random_results.to_csv(cfg.dir_outputs / "metrics" / "random_benchmark_stats.csv", index=False)

    log.info("\n" + "═" * 60)
    log.info("▶ Metrics Comparison")
    log.info("═" * 60)
    all_metrics = {}
    strategy_equities = {
        "ML_Full_EW": eq_full,
        "ML_Full_CW": eq_cw,
        "BH_QQQ": eq_bh_qqq,
        "BH_MCap10": eq_bh_mcap,
        "BH_Full": eq_bh_full,
    }
    for name, eq_df in strategy_equities.items():
        metrics = compute_metrics(eq_df)
        all_metrics[name] = metrics
        log.info(
            "  %-16s CAGR=%6.1f%%  Sharpe=%5.2f  MDD=%6.1f%%  Calmar=%5.2f",
            name,
            metrics.get("CAGR", 0.0) * 100,
            metrics.get("Sharpe", 0.0),
            metrics.get("Max_Drawdown", 0.0) * 100,
            metrics.get("Calmar", 0.0),
        )

    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(cfg.dir_outputs / "metrics" / "backtest_metrics.csv")

    annual_returns = pd.concat(
        {name: _annual_return_series(eq_df) for name, eq_df in strategy_equities.items()},
        axis=1,
    )
    annual_returns.to_csv(cfg.dir_outputs / "metrics" / "annual_returns.csv")

    if len(random_results) > 0:
        ml_cagr = all_metrics["ML_Full_CW"].get("CAGR", 0.0)
        rand_cagrs = random_results["CAGR"]
        percentile = float((rand_cagrs < ml_cagr).mean() * 100)
        z_score = (ml_cagr - rand_cagrs.mean()) / max(rand_cagrs.std(), 1e-10)
        log.info("\n  Random benchmark: CW strategy at percentile %.1f%%  z-score=%.2f", percentile, z_score)

    log.info("\n" + "═" * 60)
    log.info("▶ Alpha Testing (vs B&H QQQ ETF — fair, no look-ahead)")
    log.info("═" * 60)
    # Primary alpha test: vs QQQ (no look-ahead bias)
    alpha_bench = eq_bh_qqq if len(eq_bh_qqq) > 0 else eq_bh_mcap
    alpha_all = {
        "ML_Full_EW": compute_alpha_stats(eq_full, alpha_bench, hac_lags=5),
        "ML_Full_CW": compute_alpha_stats(eq_cw, alpha_bench, hac_lags=5),
    }
    alpha_boot = {
        "ML_Full_EW": block_bootstrap_alpha(eq_full, alpha_bench, block_size=10, n_bootstrap=5000),
        "ML_Full_CW": block_bootstrap_alpha(eq_cw, alpha_bench, block_size=10, n_bootstrap=5000),
    }

    for name, stats in alpha_all.items():
        log.info(
            "  %-16s HAC t=%6.3f  p=%7.4f  annual_alpha=%6.2f%%",
            name,
            stats.get("t_stat_alpha", 0.0),
            stats.get("p_value_alpha", 1.0),
            stats.get("Avg_Annual_Alpha", 0.0) * 100,
        )

    alpha_rows = []
    for name in alpha_all:
        row = {"strategy": name}
        row.update(alpha_all[name])
        row.update({f"bootstrap_{k}": v for k, v in alpha_boot[name].items()})
        alpha_rows.append(row)
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(cfg.dir_outputs / "metrics" / "alpha_stats.csv", index=False)

    with open(cfg.dir_outputs / "metrics" / "alpha_stats.json", "w", encoding="utf-8") as f:
        json.dump({"hac": alpha_all, "bootstrap": alpha_boot}, f, indent=2, default=str)

    trades_full.to_csv(cfg.dir_outputs / "metrics" / "trade_log_full.csv", index=False)
    trades_cw.to_csv(cfg.dir_outputs / "metrics" / "trade_log_cw.csv", index=False)

    trade_summary = {
        "ML_Full_EW": summarize_trade_log(trades_full),
        "ML_Full_CW": summarize_trade_log(trades_cw),
    }
    pd.DataFrame(trade_summary).T.to_csv(cfg.dir_outputs / "metrics" / "trade_summary.csv")

    if pred_all_full_path.exists():
        log.info("\n" + "═" * 60)
        log.info("▶ Single-model diagnostics (Full feature set)")
        log.info("═" * 60)
        pred_all_full = load(pred_all_full_path)
        diag_rows = []
        for model_name, pred_model in pred_all_full.groupby("model"):
            pred_model = pred_model.copy()
            eq_model, trades_model = run_backtest(df, pred_model, cfg_ew)
            metrics = compute_metrics(eq_model)
            diag_rows.append(
                {
                    "model": model_name,
                    **metrics,
                    **summarize_trade_log(trades_model),
                }
            )
            save(eq_model, cfg.dir_outputs / f"equity_model_{str(model_name).lower()}.parquet")
            log.info(
                "  %-8s CAGR=%6.1f%%  Sharpe=%5.2f  MDD=%6.1f%%",
                model_name,
                metrics.get("CAGR", 0.0) * 100,
                metrics.get("Sharpe", 0.0),
                metrics.get("Max_Drawdown", 0.0) * 100,
            )
        pd.DataFrame(diag_rows).to_csv(cfg.dir_outputs / "metrics" / "single_model_diagnostics.csv", index=False)

    log.info("\n✓ All backtests complete!")


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_all_backtests(args.config)


if __name__ == "__main__":
    main()
