# Project audit and risk notes

## Current health checks
- `scripts/diagnose_processed_data.py` reports whether aligned alpha targets are present.
- The same script now emits explicit alerts when `p_high_vol` is constant or `yield_spread_zscore` is constant.
- `scripts/run_models.py` already fails fast when aligned alpha targets are missing.

## Backtest risk controls available
- Regime targeting via `backtest.regime_*`
- Volatility targeting via `backtest.target_vol` and related fields
- Drawdown exposure scaling via `backtest.dd_*`
- Breadth gate via:
  - `backtest.breadth_gate`
  - `backtest.breadth_col`
  - `backtest.breadth_threshold`
  - `backtest.breadth_low_exposure`
- Benchmark trend gate via:
  - `backtest.benchmark_trend_gate`
  - `backtest.benchmark_col`
  - `backtest.benchmark_sma_window`
  - `backtest.benchmark_low_exposure`

## Practical MDD reduction order
1. Make sure processed data health checks are clean before training or backtesting.
2. Use broader portfolios and stronger hold buffers before applying hard overlays.
3. Add breadth and benchmark-trend gates only as exposure caps, not as full risk-off switches.
4. Tighten target volatility only after the signal and turnover are stable.
