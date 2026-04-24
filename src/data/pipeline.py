from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import TreasuryConfig, load_config
from src.data.build_dataset import build_dataset
from src.data.clean import align_panel, filter_valid, quality_check
from src.data.fetch_benchmark import fetch_benchmark
from src.data.fetch_macro import fetch_macro
from src.data.schemas import validate_dataset, validate_macro, validate_stock_panel
from src.data.tiingo_client import fetch_ticker
from src.utils.io import save

log = logging.getLogger(__name__)


def _fetch_treasury_safe(
    start: str,
    end: str,
    treasury_cfg: TreasuryConfig,
    cache_dir: Path | None = None,
) -> pd.DataFrame | None:
    """Fetch Treasury yields and return None if every source fails."""
    try:
        from src.data.fetch_treasury import fetch_treasury

        return fetch_treasury(
            start=start,
            end=end,
            symbols=treasury_cfg.symbols,
            cache_dir=cache_dir,
            ffill_limit=treasury_cfg.ffill_limit,
        )
    except Exception as exc:
        log.warning(
            "Could not fetch treasury data: %s\nPipeline will continue without yield-curve features.",
            exc,
        )
        return None


def run_data_pipeline(
    config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fetch, alignment, QC, and dataset assembly."""
    cfg = load_config(config_path) if config_path else load_config()

    stocks: dict[str, pd.DataFrame] = {}
    failed_tickers: list[str] = []
    for ticker in cfg.data.tickers:
        try:
            stocks[ticker] = fetch_ticker(
                ticker=ticker,
                start=cfg.data.start_date,
                end=cfg.data.end_date,
                cache_dir=cfg.dir_cache / "tiingo",
            )
        except Exception as exc:
            log.warning("[%s] fetch failed: %s", ticker, exc)
            failed_tickers.append(ticker)

    if failed_tickers:
        log.warning("Failed tickers (%s): %s", len(failed_tickers), failed_tickers)

    if not stocks:
        raise RuntimeError("No tickers were fetched successfully")

    stock_panel = pd.concat(stocks, names=["ticker"]).swaplevel(0, 1).sort_index()
    stock_panel.index = stock_panel.index.set_names(["date", "ticker"])

    macro = fetch_macro(
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        cache_dir=cfg.dir_cache / "macro",
    )

    bench = fetch_benchmark(
        ticker=cfg.data.benchmark_ticker,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        cache_dir=cfg.dir_cache / "benchmark",
    )

    treasury = _fetch_treasury_safe(
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        treasury_cfg=cfg.treasury,
        cache_dir=cfg.dir_cache / "treasury",
    )

    validate_macro(macro)

    stocks_aligned, macro_aligned, bench_aligned = align_panel(
        stock_panel,
        macro,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        bench=bench,
    )

    validate_stock_panel(stocks_aligned)

    qc_report, valid_tickers = quality_check(
        stocks_aligned,
        min_days=cfg.data.min_trading_days,
        max_nan_ratio=cfg.data.max_nan_ratio,
        max_consec_nan=cfg.data.max_consec_nan,
    )
    stocks_filtered = filter_valid(stocks_aligned, valid_tickers)

    log.info(
        "Universe: %s tickers passed QC / %s candidates",
        len(valid_tickers),
        len(cfg.data.tickers),
    )
    if failed_tickers:
        log.info("  Failed fetch: %s", failed_tickers)

    dataset = build_dataset(
        stocks_filtered,
        macro_aligned,
        bench=bench_aligned,
        treasury=treasury,
    )

    validate_dataset(dataset)

    save(stocks_filtered, cfg.dir_interim / "stock_panel_qc.parquet")
    save(dataset, cfg.dir_processed / "dataset.parquet")
    save(qc_report, cfg.dir_outputs / "metrics" / "qc_report.parquet")

    log.info("Pipeline complete")
    return dataset, qc_report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_data_pipeline()
