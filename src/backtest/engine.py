"""Backtest engine, benchmarks, and performance analytics."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class BacktestEngineConfig:
    """Configuration for the research backtest engine."""
    top_k: int = 10
    rebalance_days: int = 10
    cost_bps: float = 10.0
    slippage_bps: float = 0.0
    initial_capital: float = 10_000
    confidence_weighted: bool = False
    max_weight_cap: float = 0.25
    hold_buffer: int = 3
    hold_score_tolerance: float = 0.02
    signal_anchor_weight: float = 0.0
    signal_anchor_features: tuple[str, ...] = (
        "mom_63d",
        "rel_strength_63d",
        "trend_strength_21d",
        "price_sma200",
    )
    regime_targeting: bool = False
    regime_col: str = "p_high_vol"
    regime_sensitivity: float = 0.35
    regime_min_exposure: float = 0.65
    regime_max_exposure: float = 1.00
    breadth_gate: bool = False
    breadth_col: str = "market_breadth_200d"
    breadth_threshold: float = 0.35
    breadth_low_exposure: float = 0.60
    benchmark_trend_gate: bool = False
    benchmark_col: str = "bench_close"
    benchmark_sma_window: int = 200
    benchmark_low_exposure: float = 0.70
    vol_targeting: bool = False
    target_vol: float = 0.16
    vol_lookback_days: int = 63
    vol_min_scale: float = 0.75
    vol_max_scale: float = 1.15
    dd_threshold: float = 0.0
    dd_exit: float = -0.10
    dd_exposure: float = 0.80
    sector_max_weight: float = 0.40
    sector_map: dict[str, str] | None = None



def _compute_weights(
    signal_pool: pd.Series,
    top_k: int,
    confidence_weighted: bool = False,
    max_weight_cap: float = 0.25,
    selected_names: list[str] | None = None,
    sector_map: dict[str, str] | None = None,
    sector_max_weight: float = 0.40,
) -> dict[str, float]:
    """Compute equal-weight or confidence-weighted portfolio weights.

    When *sector_map* is provided, each GICS sector is capped at
    *sector_max_weight* of total portfolio weight.  Excess is
    redistributed proportionally to uncapped sectors.
    """
    top_k_actual = min(int(top_k), len(signal_pool))
    if top_k_actual <= 0:
        return {}

    if selected_names is None:
        selected = signal_pool.nlargest(top_k_actual)
    else:
        ordered = [t for t in selected_names if t in signal_pool.index]
        if not ordered:
            return {}
        selected = signal_pool.loc[ordered].sort_values(ascending=False)

    if not confidence_weighted:
        w = 1.0 / top_k_actual
        weights = {tkr: w for tkr in selected.index}
    else:
        confidence = (selected - 0.5).clip(lower=1e-6)
        ws = confidence / confidence.sum()

        locked = pd.Series(False, index=ws.index)
        for _ in range(len(ws)):
            over = (ws > max_weight_cap + 1e-12) & ~locked
            if not over.any():
                break
            locked = locked | over
            ws[locked] = max_weight_cap
            remaining = 1.0 - locked.sum() * max_weight_cap
            unlocked = ~locked
            if unlocked.any() and remaining > 0:
                ws[unlocked] = remaining * (confidence[unlocked] / confidence[unlocked].sum())
            else:
                break
        weights = ws.to_dict()

    # ── Sector concentration cap ───────────────────────────────────
    if sector_map and sector_max_weight < 1.0:
        weights = _apply_sector_cap(weights, sector_map, sector_max_weight)

    return weights


def _apply_sector_cap(
    weights: dict[str, float],
    sector_map: dict[str, str],
    sector_max_weight: float,
) -> dict[str, float]:
    """Iteratively rescale weights so no single sector exceeds *sector_max_weight*."""
    w = pd.Series(weights)
    sectors = pd.Series({tkr: sector_map.get(tkr, "Other") for tkr in w.index})

    for _ in range(10):  # converges in 2-3 iterations
        sector_totals = w.groupby(sectors).sum()
        over = sector_totals[sector_totals > sector_max_weight + 1e-12]
        if over.empty:
            break
        for sec in over.index:
            mask = sectors == sec
            scale = sector_max_weight / float(sector_totals[sec])
            w[mask] *= scale
        # renormalize to 1.0
        w /= w.sum()

    return w.to_dict()



def _select_with_buffer(
    signal_pool: pd.Series,
    top_k: int,
    prev_holdings: set[str] | None = None,
    hold_buffer: int = 0,
    hold_score_tolerance: float = 0.0,
) -> list[str]:
    """Select names with rank and score hysteresis to reduce churn."""
    top_k_actual = min(int(top_k), len(signal_pool))
    if top_k_actual <= 0:
        return []

    ranked = signal_pool.sort_values(ascending=False)
    if not prev_holdings or (hold_buffer <= 0 and hold_score_tolerance <= 0):
        return ranked.index[:top_k_actual].tolist()

    rank_pos = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
    cutoff_score = float(ranked.iloc[top_k_actual - 1])
    keep: list[str] = []
    for tkr in ranked.index:
        if tkr not in prev_holdings:
            continue
        within_rank = hold_buffer > 0 and int(rank_pos[tkr]) <= top_k_actual + int(hold_buffer)
        within_score = hold_score_tolerance > 0 and float(ranked.loc[tkr]) >= cutoff_score - float(hold_score_tolerance)
        if within_rank or within_score:
            keep.append(tkr)

    selected = list(keep)
    for tkr in ranked.index:
        if tkr in selected:
            continue
        selected.append(tkr)
        if len(selected) >= top_k_actual:
            break
    return selected[:top_k_actual]



def _estimate_trade_cost(
    prev_holdings: set[str],
    new_holdings: set[str],
    total_cost_rate: float,
) -> tuple[set[str], set[str], float, float]:
    """Approximation of turnover and proportional trading cost for research backtests."""
    if not prev_holdings:
        bought = set(new_holdings)
        turnover_est = 1.0 if new_holdings else 0.0
        return set(), bought, turnover_est, turnover_est * total_cost_rate

    sold = prev_holdings - new_holdings
    bought = new_holdings - prev_holdings
    n_prev = max(len(prev_holdings), 1)
    n_new = max(len(new_holdings), 1)

    turnover_est = (len(sold) / n_prev) + (len(bought) / n_new)
    cost = turnover_est * total_cost_rate
    return sold, bought, turnover_est, cost



def _portfolio_return_for_day(
    df: pd.DataFrame,
    weights: dict[str, float],
    hold_dates: pd.Index,
    j: int,
    d: pd.Timestamp,
) -> float:
    """Compute one-day portfolio return for a fixed basket of weights."""
    total_ret = 0.0
    total_weight = 0.0

    for tkr, w in weights.items():
        try:
            if j == 0:
                p0 = (
                    df.loc[(d, tkr), "adj_open"]
                    if "adj_open" in df.columns
                    else df.loc[(d, tkr), "adj_close"]
                )
                p1 = df.loc[(d, tkr), "adj_close"]
            else:
                prev_d = hold_dates[j - 1]
                p0 = df.loc[(prev_d, tkr), "adj_close"]
                p1 = df.loc[(d, tkr), "adj_close"]

            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                total_ret += w * float(p1 / p0 - 1.0)
                total_weight += w
        except KeyError:
            continue

    if 0 < total_weight < 0.99:
        total_ret = total_ret / total_weight

    return total_ret



def _get_date_slice(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame | None:
    """Return the ticker slice for one date, or None when missing."""
    try:
        return df.xs(date, level="date")
    except KeyError:
        return None



def _blend_signal_with_anchor(
    df: pd.DataFrame,
    signal_date: pd.Timestamp,
    signal_pool: pd.Series,
    cfg: BacktestEngineConfig,
) -> tuple[pd.Series, bool]:
    """Blend model scores with a simple factor anchor when available."""
    if cfg.signal_anchor_weight <= 0:
        return signal_pool, False

    date_slice = _get_date_slice(df, signal_date)
    if date_slice is None:
        return signal_pool, False

    anchor_cols = [c for c in cfg.signal_anchor_features if c in date_slice.columns]
    if not anchor_cols:
        return signal_pool, False

    anchor_ranks = []
    for col in anchor_cols:
        values = pd.to_numeric(date_slice[col], errors="coerce")
        values = values.reindex(signal_pool.index)
        if values.notna().sum() < 3:
            continue
        anchor_ranks.append(values.rank(method="average", pct=True))

    if not anchor_ranks:
        return signal_pool, False

    ml_rank = signal_pool.rank(method="average", pct=True)
    anchor_rank = pd.concat(anchor_ranks, axis=1).mean(axis=1, skipna=True)
    aligned = ml_rank.to_frame("ml_rank").join(anchor_rank.rename("anchor_rank"), how="left")
    if aligned["anchor_rank"].notna().sum() < 3:
        return signal_pool, False

    blended = aligned["ml_rank"].copy()
    w = float(np.clip(cfg.signal_anchor_weight, 0.0, 1.0))
    valid = aligned["anchor_rank"].notna()
    blended.loc[valid] = (1.0 - w) * aligned.loc[valid, "ml_rank"] + w * aligned.loc[valid, "anchor_rank"]
    blended = blended.rank(method="average", pct=True)
    return blended, True



def _compute_regime_scale(
    df: pd.DataFrame,
    signal_date: pd.Timestamp,
    selected: list[str],
    cfg: BacktestEngineConfig,
) -> tuple[float, float | None]:
    """Map regime stress to an exposure scale."""
    if not cfg.regime_targeting:
        return 1.0, None

    date_slice = _get_date_slice(df, signal_date)
    if date_slice is None or cfg.regime_col not in date_slice.columns:
        return 1.0, None

    if selected:
        available = [t for t in selected if t in date_slice.index]
        values = date_slice.loc[available, cfg.regime_col] if available else date_slice[cfg.regime_col]
    else:
        values = date_slice[cfg.regime_col]

    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return 1.0, None

    regime_score = float(np.clip(values.median(), 0.0, 1.0))
    raw_scale = cfg.regime_max_exposure - cfg.regime_sensitivity * regime_score
    scale = float(np.clip(raw_scale, cfg.regime_min_exposure, cfg.regime_max_exposure))
    return scale, regime_score



def _compute_defense_scale(
    df: pd.DataFrame,
    signal_date: pd.Timestamp,
    cfg: BacktestEngineConfig,
) -> tuple[float, bool, bool]:
    """Cap exposure using market breadth and benchmark trend filters."""
    scale = 1.0
    breadth_triggered = False
    trend_triggered = False

    date_slice = _get_date_slice(df, signal_date)
    if date_slice is None:
        return scale, breadth_triggered, trend_triggered

    if cfg.breadth_gate and cfg.breadth_col in date_slice.columns:
        breadth_values = pd.to_numeric(date_slice[cfg.breadth_col], errors="coerce").dropna()
        if len(breadth_values) > 0:
            breadth = float(breadth_values.median())
            if breadth < cfg.breadth_threshold:
                scale = min(scale, float(cfg.breadth_low_exposure))
                breadth_triggered = True

    if cfg.benchmark_trend_gate and cfg.benchmark_col in df.columns:
        bench_daily = df.groupby(level="date")[cfg.benchmark_col].first().sort_index()
        hist = pd.to_numeric(bench_daily.loc[bench_daily.index <= signal_date], errors="coerce").dropna()
        if len(hist) >= cfg.benchmark_sma_window:
            sma = hist.rolling(cfg.benchmark_sma_window, min_periods=cfg.benchmark_sma_window).mean().iloc[-1]
            price = hist.iloc[-1]
            if pd.notna(sma) and price < sma:
                scale = min(scale, float(cfg.benchmark_low_exposure))
                trend_triggered = True

    return scale, breadth_triggered, trend_triggered



def _estimate_portfolio_vol(
    df: pd.DataFrame,
    signal_date: pd.Timestamp,
    weights: dict[str, float],
    lookback_days: int = 63,
) -> float:
    """Estimate ex-ante annualized volatility from trailing close-to-close returns."""
    if not weights or lookback_days <= 1 or "adj_close" not in df.columns:
        return np.nan

    all_dates = df.index.get_level_values("date").unique().sort_values()
    hist_dates = all_dates[all_dates <= signal_date]
    if len(hist_dates) < 3:
        return np.nan
    hist_dates = hist_dates[-(lookback_days + 1):]
    min_obs = max(10, lookback_days // 3)
    if len(hist_dates) < min_obs:
        return np.nan

    tickers = list(weights.keys())
    mask = (
        df.index.get_level_values("date").isin(hist_dates)
        & df.index.get_level_values("ticker").isin(tickers)
    )
    close_px = df.loc[mask, "adj_close"].unstack("ticker").reindex(hist_dates).ffill()
    if close_px.shape[1] == 0:
        return np.nan

    rets = close_px.pct_change().dropna(how="all")
    if len(rets) < min_obs:
        return np.nan

    weight_s = pd.Series(weights, dtype=float).reindex(rets.columns).fillna(0.0)
    weight_s = weight_s[weight_s > 0]
    if len(weight_s) == 0 or float(weight_s.sum()) <= 0:
        return np.nan
    weight_s = weight_s / weight_s.sum()

    available_weight = (~rets.isna()).mul(weight_s, axis=1).sum(axis=1)
    valid_rows = available_weight >= 0.80
    if valid_rows.sum() < min_obs:
        return np.nan

    port_ret = rets.fillna(0.0).mul(weight_s, axis=1).sum(axis=1)
    port_ret = port_ret.loc[valid_rows] / available_weight.loc[valid_rows]
    if len(port_ret) < min_obs:
        return np.nan

    vol = float(port_ret.std(ddof=1) * np.sqrt(252))
    return vol if np.isfinite(vol) and vol > 0 else np.nan



def _compute_vol_target_scale(
    df: pd.DataFrame,
    signal_date: pd.Timestamp,
    weights: dict[str, float],
    cfg: BacktestEngineConfig,
) -> tuple[float, float | None]:
    """Convert trailing basket volatility into an exposure multiplier."""
    if not cfg.vol_targeting:
        return 1.0, None

    est_vol = _estimate_portfolio_vol(
        df,
        signal_date=signal_date,
        weights=weights,
        lookback_days=cfg.vol_lookback_days,
    )
    if not np.isfinite(est_vol) or est_vol <= 1e-8:
        return 1.0, None

    raw_scale = cfg.target_vol / est_vol
    scale = float(np.clip(raw_scale, cfg.vol_min_scale, cfg.vol_max_scale))
    return scale, float(est_vol)



def run_backtest(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cfg: BacktestEngineConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg is None:
        cfg = BacktestEngineConfig()

    if "model" in pred_df.columns and pred_df["model"].nunique() != 1:
        raise ValueError(
            f"pred_df contains {pred_df['model'].nunique()} models. "
            f"Use build_ensemble() or filter pred_df first."
        )

    total_cost_rate = (cfg.cost_bps + cfg.slippage_bps) / 10_000

    all_dates = df.index.get_level_values("date").unique().sort_values()
    pred_dates = pred_df.index.get_level_values("date").unique().sort_values()
    pred_date_list = pred_dates.tolist()
    rebalance_dates = pred_date_list[::cfg.rebalance_days]

    log.info(
        "Backtest: %s rebalances, every %s days, %s→%s | %s",
        len(rebalance_dates),
        cfg.rebalance_days,
        pred_dates[0].date() if len(pred_dates) else None,
        pred_dates[-1].date() if len(pred_dates) else None,
        "confidence-weighted" if cfg.confidence_weighted else "equal-weight",
    )

    equity = float(cfg.initial_capital)
    prev_holdings: set[str] = set()
    eq_records: list[dict] = []
    trade_records: list[dict] = []

    peak_equity = equity
    dd_scale_state = 1.0
    use_dd_control = cfg.dd_threshold < 0

    for i, reb_date in enumerate(rebalance_dates):
        available = pred_dates[pred_dates <= reb_date]
        if len(available) == 0:
            continue
        signal_date = available[-1]

        mask = pred_df.index.get_level_values("date") == signal_date
        signal_slice = pred_df.loc[mask]
        signal_pool = signal_slice.set_index(signal_slice.index.get_level_values("ticker"))["y_prob"]
        if len(signal_pool) == 0:
            continue

        signal_pool = signal_pool[~signal_pool.index.duplicated(keep="first")]
        signal_pool, anchor_used = _blend_signal_with_anchor(df, signal_date, signal_pool, cfg)

        selected_names = _select_with_buffer(
            signal_pool,
            cfg.top_k,
            prev_holdings=prev_holdings,
            hold_buffer=cfg.hold_buffer,
            hold_score_tolerance=cfg.hold_score_tolerance,
        )

        weights = _compute_weights(
            signal_pool,
            cfg.top_k,
            confidence_weighted=cfg.confidence_weighted,
            max_weight_cap=cfg.max_weight_cap,
            selected_names=selected_names,
            sector_map=cfg.sector_map,
            sector_max_weight=cfg.sector_max_weight,
        )
        if not weights:
            continue

        selected = list(weights.keys())
        new_holdings = set(selected)

        if i + 1 < len(rebalance_dates):
            next_reb = rebalance_dates[i + 1]
            hold_dates = all_dates[(all_dates > reb_date) & (all_dates <= next_reb)]
        else:
            hold_dates = all_dates[all_dates > reb_date]

        entry_date = hold_dates[0] if len(hold_dates) > 0 else pd.NaT
        if len(hold_dates) > 0 and "adj_open" in df.columns:
            valid_tickers = [
                tkr for tkr in selected
                if ((entry_date, tkr) in df.index)
                and pd.notna(df.loc[(entry_date, tkr), "adj_open"])
                and float(df.loc[(entry_date, tkr), "adj_open"]) > 0
            ]
            if len(valid_tickers) < len(selected):
                filtered_pool = signal_pool[signal_pool.index.isin(valid_tickers)]
                selected_names = _select_with_buffer(
                    filtered_pool,
                    cfg.top_k,
                    prev_holdings=prev_holdings,
                    hold_buffer=cfg.hold_buffer,
                    hold_score_tolerance=cfg.hold_score_tolerance,
                )
                weights = _compute_weights(
                    filtered_pool,
                    cfg.top_k,
                    confidence_weighted=cfg.confidence_weighted,
                    max_weight_cap=cfg.max_weight_cap,
                    selected_names=selected_names,
                    sector_map=cfg.sector_map,
                    sector_max_weight=cfg.sector_max_weight,
                )
                selected = list(weights.keys())
                new_holdings = set(selected)

        if not weights:
            trade_records.append(
                {
                    "rebalance_date": reb_date,
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "holdings": [],
                    "n_holdings": 0,
                    "n_sold": 0,
                    "n_bought": 0,
                    "turnover_est": 0.0,
                    "cost": 0.0,
                    "exposure": 0.0,
                    "regime_scale": 1.0,
                    "vol_scale": 1.0,
                    "dd_scale": dd_scale_state,
                    "estimated_vol": np.nan,
                    "regime_score": np.nan,
                    "anchor_used": anchor_used,
                    "skipped_no_hold_dates": True,
                    "skipped_empty_weights": True,
                }
            )
            continue

        if len(hold_dates) == 0:
            trade_records.append(
                {
                    "rebalance_date": reb_date,
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "holdings": selected,
                    "n_holdings": len(selected),
                    "n_sold": 0,
                    "n_bought": 0,
                    "turnover_est": 0.0,
                    "cost": 0.0,
                    "exposure": 0.0,
                    "regime_scale": 1.0,
                    "vol_scale": 1.0,
                    "dd_scale": dd_scale_state,
                    "estimated_vol": np.nan,
                    "regime_score": np.nan,
                    "anchor_used": anchor_used,
                    "skipped_no_hold_dates": True,
                }
            )
            continue

        sold, bought, turnover_est, cost = _estimate_trade_cost(
            prev_holdings,
            new_holdings,
            total_cost_rate,
        )

        regime_scale, regime_score = _compute_regime_scale(df, signal_date, selected, cfg)
        defense_scale, breadth_gate_triggered, benchmark_trend_triggered = _compute_defense_scale(df, signal_date, cfg)
        vol_scale, estimated_vol = _compute_vol_target_scale(df, signal_date, weights, cfg)

        dd_scale = 1.0
        if use_dd_control:
            peak_equity = max(peak_equity, equity)
            current_dd = equity / peak_equity - 1.0

            prev_dd_scale = dd_scale_state
            if dd_scale_state >= 0.999 and current_dd <= cfg.dd_threshold:
                dd_scale_state = cfg.dd_exposure
            elif dd_scale_state < 0.999 and current_dd >= cfg.dd_exit:
                dd_scale_state = 1.0
            dd_scale = dd_scale_state

            if dd_scale != prev_dd_scale:
                log.debug(
                    "  DD overlay: dd=%.1f%% exposure %.0f%% -> %.0f%% at %s",
                    current_dd * 100,
                    prev_dd_scale * 100,
                    dd_scale * 100,
                    reb_date.date(),
                )

        exposure = float(regime_scale * defense_scale * vol_scale * dd_scale)

        for j, d in enumerate(hold_dates):
            gross_ret = _portfolio_return_for_day(df, weights, hold_dates, j, d)

            if j == 0:
                net_ret = (1.0 + exposure * gross_ret) * (1.0 - exposure * cost) - 1.0
                cost_ret = net_ret - exposure * gross_ret
            else:
                net_ret = exposure * gross_ret
                cost_ret = 0.0

            equity *= (1.0 + net_ret)
            eq_records.append(
                {
                    "date": d,
                    "equity": equity,
                    "daily_ret": net_ret,
                    "gross_ret": gross_ret,
                    "cost_ret": cost_ret,
                    "is_entry_day": bool(j == 0),
                    "rebalance_date": reb_date,
                    "signal_date": signal_date,
                    "exposure": exposure,
                    "regime_scale": regime_scale,
                    "vol_scale": vol_scale,
                    "dd_scale": dd_scale,
                    "estimated_vol": estimated_vol,
                    "regime_score": regime_score,
                    "anchor_used": anchor_used,
                }
            )

        trade_records.append(
            {
                "rebalance_date": reb_date,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "holdings": selected,
                "n_holdings": len(selected),
                "n_sold": len(sold),
                "n_bought": len(bought),
                "turnover_est": turnover_est,
                "cost": cost,
                "exposure": exposure,
                "regime_scale": regime_scale,
                "vol_scale": vol_scale,
                "dd_scale": dd_scale,
                "estimated_vol": estimated_vol,
                "regime_score": regime_score,
                "anchor_used": anchor_used,
                "skipped_no_hold_dates": False,
            }
        )
        prev_holdings = new_holdings

    if eq_records:
        equity_df = pd.DataFrame(eq_records).set_index("date")
    else:
        equity_df = pd.DataFrame(
            columns=[
                "equity",
                "daily_ret",
                "gross_ret",
                "cost_ret",
                "is_entry_day",
                "rebalance_date",
                "signal_date",
                "exposure",
                "regime_scale",
                "defense_scale",
                "vol_scale",
                "dd_scale",
                "estimated_vol",
                "regime_score",
                "breadth_gate",
                "benchmark_trend_gate",
                "anchor_used",
            ]
        )

    trades_df = pd.DataFrame(trade_records)
    equity_df.attrs["initial_capital"] = float(cfg.initial_capital)

    if len(equity_df) > 0:
        log.info(
            "  Final: $%s (return: %.1f%%)",
            f"{equity_df['equity'].iloc[-1]:,.0f}",
            (equity_df["equity"].iloc[-1] / cfg.initial_capital - 1) * 100,
        )

    return equity_df, trades_df


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_benchmark(
    df: pd.DataFrame,
    dates: pd.Index,
    initial_capital: float = 10_000,
) -> pd.DataFrame:
    """Equal-weight buy-and-hold benchmark."""
    all_dates = df.index.get_level_values("date").unique().sort_values()
    target_dates = all_dates[all_dates.isin(dates)]

    if len(target_dates) == 0:
        return pd.DataFrame(columns=["equity", "daily_ret"])

    close_px = df["adj_close"].unstack("ticker").reindex(target_dates).sort_index()
    if "adj_open" in df.columns:
        open_px = df["adj_open"].unstack("ticker").reindex(target_dates).sort_index()
    else:
        open_px = close_px.copy()

    start_open = open_px.iloc[0]
    valid_tickers = start_open[start_open.notna() & (start_open > 0)].index

    if len(valid_tickers) == 0:
        return pd.DataFrame(columns=["equity", "daily_ret"], index=target_dates)

    close_px = close_px[valid_tickers].ffill().dropna(axis=1, how="any")
    open_px = open_px[valid_tickers][close_px.columns]

    if close_px.shape[1] == 0:
        return pd.DataFrame(columns=["equity", "daily_ret"], index=target_dates)

    capital_per_asset = initial_capital / close_px.shape[1]
    shares = capital_per_asset / open_px.iloc[0]
    equity_series = close_px.mul(shares, axis=1).sum(axis=1)

    benchmark = pd.DataFrame(index=equity_series.index)
    benchmark["equity"] = equity_series
    benchmark["daily_ret"] = benchmark["equity"].pct_change()
    benchmark.loc[benchmark.index[0], "daily_ret"] = (
        benchmark["equity"].iloc[0] / initial_capital - 1.0
    )
    benchmark["gross_ret"] = benchmark["daily_ret"]
    benchmark["cost_ret"] = 0.0
    benchmark["is_entry_day"] = False
    benchmark.attrs["initial_capital"] = float(initial_capital)
    return benchmark


def compute_benchmark_mcap_top10(
    df: pd.DataFrame,
    dates: pd.Index,
    mcap_tickers: list[str],
    initial_capital: float = 10_000,
) -> pd.DataFrame:
    """
    Buy-and-hold benchmark for the fixed top-10 market-cap basket.

    NOTE: This benchmark uses a static ticker list chosen with hindsight.
    It is retained for backward compatibility but should NOT be the primary
    comparison.  Use compute_benchmark_etf (QQQ) for a fair comparison.
    """
    available = df.index.get_level_values("ticker").unique()
    valid_mcap = [t for t in mcap_tickers if t in available]

    if not valid_mcap:
        log.warning("No MCap ticker found in data!")
        return pd.DataFrame(columns=["equity", "daily_ret"])

    log.info(f"B&H MCap Top-10: {valid_mcap}")

    mask = df.index.get_level_values("ticker").isin(valid_mcap)
    df_mcap = df[mask]

    return compute_benchmark(df_mcap, dates, initial_capital)


def compute_benchmark_etf(
    df: pd.DataFrame,
    dates: pd.Index,
    initial_capital: float = 10_000,
    bench_col: str = "bench_close",
) -> pd.DataFrame:
    """Buy-and-hold benchmark using the actual ETF price (e.g. QQQ).

    This is the fairest passive comparison: no look-ahead ticker selection,
    no rebalancing, minimal survivorship bias.
    """
    if bench_col not in df.columns:
        log.warning("bench_col '%s' not in df — cannot build ETF benchmark", bench_col)
        return pd.DataFrame(columns=["equity", "daily_ret"])

    all_dates = df.index.get_level_values("date").unique().sort_values()
    target_dates = all_dates[all_dates.isin(dates)]
    if len(target_dates) == 0:
        return pd.DataFrame(columns=["equity", "daily_ret"])

    bench_daily = df.groupby(level="date")[bench_col].first().sort_index()
    bench_daily = bench_daily.reindex(target_dates).ffill().dropna()
    if len(bench_daily) < 2:
        return pd.DataFrame(columns=["equity", "daily_ret"])

    equity = initial_capital * bench_daily / bench_daily.iloc[0]
    benchmark = pd.DataFrame(index=equity.index)
    benchmark["equity"] = equity
    benchmark["daily_ret"] = benchmark["equity"].pct_change()
    benchmark.loc[benchmark.index[0], "daily_ret"] = (
        benchmark["equity"].iloc[0] / initial_capital - 1.0
    )
    benchmark["gross_ret"] = benchmark["daily_ret"]
    benchmark["cost_ret"] = 0.0
    benchmark["is_entry_day"] = False
    benchmark.attrs["initial_capital"] = float(initial_capital)
    return benchmark


def run_random_benchmark(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    n_iterations: int = 200,
    top_k: int = 10,
    rebalance_days: int = 10,
    cost_bps: float = 10.0,
    initial_capital: float = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Random top-10 benchmark averaged over multiple runs.

    Each rebalance period: random 10 tickers, equal-weighted.
    Returns DataFrame with columns=[iteration, CAGR, Sharpe, MDD, ...].
    """
    rng = np.random.RandomState(seed)

    all_dates = df.index.get_level_values("date").unique().sort_values()
    pred_dates = pred_df.index.get_level_values("date").unique().sort_values()
    pred_date_list = pred_dates.tolist()
    rebalance_dates = pred_date_list[::rebalance_days]

    total_cost_rate = cost_bps / 10_000
    results = []

    for it in range(n_iterations):
        equity = float(initial_capital)
        prev_holdings: set[str] = set()
        eq_records: list[dict] = []

        for i, reb_date in enumerate(rebalance_dates):
            available = pred_dates[pred_dates <= reb_date]
            if len(available) == 0:
                continue
            signal_date = available[-1]

            # Get available tickers at this date
            mask = pred_df.index.get_level_values("date") == signal_date
            available_tickers = pred_df.loc[mask].index.get_level_values("ticker").tolist()

            if len(available_tickers) < top_k:
                selected = available_tickers
            else:
                selected = rng.choice(available_tickers, size=top_k, replace=False).tolist()

            new_holdings = set(selected)
            weights = {t: 1.0 / len(selected) for t in selected}

            if i + 1 < len(rebalance_dates):
                next_reb = rebalance_dates[i + 1]
                hold_dates = all_dates[(all_dates > reb_date) & (all_dates <= next_reb)]
            else:
                hold_dates = all_dates[all_dates > reb_date]

            if len(hold_dates) == 0:
                continue

            # Filter valid entry
            entry_date = hold_dates[0]
            if "adj_open" in df.columns:
                selected = [
                    t for t in selected
                    if ((entry_date, t) in df.index)
                    and pd.notna(df.loc[(entry_date, t), "adj_open"])
                    and float(df.loc[(entry_date, t), "adj_open"]) > 0
                ]
                if not selected:
                    continue
                new_holdings = set(selected)
                weights = {t: 1.0 / len(selected) for t in selected}

            _, _, turnover_est, cost = _estimate_trade_cost(
                prev_holdings, new_holdings, total_cost_rate,
            )

            for j, d in enumerate(hold_dates):
                gross_ret = _portfolio_return_for_day(df, weights, hold_dates, j, d)
                if j == 0:
                    net_ret = (1.0 + gross_ret) * (1.0 - cost) - 1.0
                else:
                    net_ret = gross_ret
                equity *= (1.0 + net_ret)
                eq_records.append({"date": d, "equity": equity, "daily_ret": net_ret})

            prev_holdings = new_holdings

        if eq_records:
            eq_df = pd.DataFrame(eq_records).set_index("date")
            eq_df.attrs["initial_capital"] = initial_capital
            metrics = compute_metrics(eq_df)
            metrics["iteration"] = it
            results.append(metrics)

        if (it + 1) % 50 == 0:
            log.info(f"  Random benchmark: {it + 1}/{n_iterations} done")

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(equity_df: pd.DataFrame, risk_free_rate: float = 0.04) -> dict:
    """CAGR, Sharpe, Sortino, Max Drawdown, Calmar, VaR, Win Rate, MDD Duration."""
    if len(equity_df) < 2:
        return {}

    eq = equity_df["equity"]
    rets = equity_df["daily_ret"].fillna(0.0)

    initial_equity = equity_df.attrs.get("initial_capital")
    if initial_equity is None:
        first_ret = float(rets.iloc[0])
        denom = max(1.0 + first_ret, 1e-12)
        initial_equity = float(eq.iloc[0] / denom)

    n_days = (eq.index[-1] - eq.index[0]).days
    n_years = max(n_days / 365.25, 0.01)
    total_return = float(eq.iloc[-1] / initial_equity)
    cagr = total_return ** (1 / n_years) - 1

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = rets - daily_rf
    sharpe = excess.mean() / max(excess.std(), 1e-10) * np.sqrt(252)

    downside = rets[rets < 0]
    sortino = (rets.mean() - daily_rf) / max(downside.std(), 1e-10) * np.sqrt(252)

    running_max = eq.cummax()
    drawdown = eq / running_max - 1
    max_dd = float(drawdown.min())

    calmar = cagr / max(abs(max_dd), 1e-10)
    var_95 = float(np.percentile(rets, 5))
    win_rate = float((rets > 0).mean())

    # Maximum drawdown recovery duration.
    mdd_duration = _compute_mdd_duration(eq)

    # Tail ratio.
    tail_ratio = float(np.percentile(rets, 95) / max(abs(np.percentile(rets, 5)), 1e-10))

    return {
        "Total_Return": total_return - 1,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_Drawdown": max_dd,
        "Calmar": calmar,
        "VaR_95": var_95,
        "Win_Rate": win_rate,
        "MDD_Duration": mdd_duration,
        "Tail_Ratio": tail_ratio,
        "Avg_Daily_Ret": float(rets.mean()),
        "Std_Daily_Ret": float(rets.std()),
        "N_Days": len(eq),
    }


def _compute_mdd_duration(equity: pd.Series) -> int:
    """Longest number of trading days to recover from a drawdown."""
    running_max = equity.cummax()
    in_drawdown = equity < running_max

    max_duration = 0
    current_duration = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


def summarize_trade_log(trades_df: pd.DataFrame) -> dict:
    """Useful implementation stats for the research notebook."""
    if len(trades_df) == 0:
        return {}

    valid = trades_df.copy()
    if "skipped_no_hold_dates" in valid.columns:
        valid = valid[~valid["skipped_no_hold_dates"].fillna(False)]
    if len(valid) == 0:
        return {}

    return {
        "N_Rebalances": int(len(valid)),
        "Avg_Holdings": float(valid["n_holdings"].mean()),
        "Avg_N_Sold": float(valid["n_sold"].mean()),
        "Avg_N_Bought": float(valid["n_bought"].mean()),
        "Avg_Turnover_Est": float(valid["turnover_est"].mean()),
        "Avg_Cost_Per_Rebalance": float(valid["cost"].mean()),
        "Pct_Full_Refresh": float((valid["n_sold"] == valid["n_holdings"]).mean()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ALPHA TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_alpha_stats(
    ml_equity: pd.DataFrame,
    bh_equity: pd.DataFrame,
    hac_lags: int | None = 5,
) -> dict:
    """
    HAC t-test for mean daily alpha.
    """
    common = ml_equity.index.intersection(bh_equity.index)
    if len(common) == 0:
        return {}

    ml_ret = ml_equity.loc[common, "daily_ret"]
    bh_ret = bh_equity.loc[common, "daily_ret"]

    daily_alpha = (ml_ret - bh_ret).dropna()
    n = len(daily_alpha)
    if n == 0:
        return {}

    tracking_error = float(daily_alpha.std(ddof=1))
    ir = float(daily_alpha.mean() / max(tracking_error, 1e-10) * np.sqrt(252))
    avg_annual_alpha = float(daily_alpha.mean() * 252)

    if hac_lags is None:
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(daily_alpha, 0.0, nan_policy="omit")
        method = "naive_ttest_mean_alpha"
    else:
        import statsmodels.api as sm
        y = daily_alpha.to_numpy(dtype=float)
        X = np.ones((len(y), 1), dtype=float)
        model = sm.OLS(y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": int(hac_lags)},
        )
        t_stat = float(model.tvalues[0])
        p_value = float(model.pvalues[0])
        method = f"HAC_mean_alpha_maxlags={int(hac_lags)}"

    return {
        "Information_Ratio": ir,
        "t_stat_alpha": float(t_stat),
        "p_value_alpha": float(p_value),
        "Avg_Annual_Alpha": avg_annual_alpha,
        "Tracking_Error_Annual": float(tracking_error * np.sqrt(252)),
        "N_Days": int(n),
        "Alpha_Test_Method": method,
    }


def block_bootstrap_alpha(
    ml_equity: pd.DataFrame,
    bh_equity: pd.DataFrame,
    block_size: int = 10,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Block bootstrap test for mean alpha.

    Block size = 10 (match rebalance period) → respect autocorrelation.
    Returns: p_value, confidence_interval_95, mean_bootstrap_alpha.
    """
    common = ml_equity.index.intersection(bh_equity.index)
    if len(common) < block_size * 2:
        return {"bootstrap_error": "insufficient data"}

    ml_ret = ml_equity.loc[common, "daily_ret"].values
    bh_ret = bh_equity.loc[common, "daily_ret"].values
    daily_alpha = ml_ret - bh_ret

    n = len(daily_alpha)
    observed_mean = np.mean(daily_alpha)

    rng = np.random.RandomState(seed)
    n_blocks = n // block_size

    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        # Sample blocks with replacement
        block_starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([
            daily_alpha[s:s + block_size] for s in block_starts
        ])
        boot_means[b] = np.mean(sample)

    # Two-sided p-value: proportion of bootstrap means <= 0
    p_value = float(np.mean(boot_means <= 0))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return {
        "bootstrap_mean_alpha_daily": float(np.mean(boot_means)),
        "bootstrap_std": float(np.std(boot_means)),
        "bootstrap_p_value": p_value,
        "bootstrap_ci_95_lower": ci_lower,
        "bootstrap_ci_95_upper": ci_upper,
        "bootstrap_ci_95_lower_annual": ci_lower * 252,
        "bootstrap_ci_95_upper_annual": ci_upper * 252,
        "observed_mean_alpha_daily": float(observed_mean),
        "observed_mean_alpha_annual": float(observed_mean * 252),
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
    }


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_days: int,
    n_trials: int,
    skew: float = 0.0,
    kurtosis_excess: float = 0.0,
) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Adjusts the observed Sharpe ratio for multiple testing by comparing
    it against the expected maximum Sharpe under the null hypothesis
    that all *n_trials* strategies have zero true Sharpe.

    Parameters
    ----------
    observed_sharpe : Annualised Sharpe ratio of the chosen strategy.
    n_days : Number of daily observations.
    n_trials : Number of strategy variants tried (configs, feature sets, etc.).
    skew : Skewness of daily returns (default 0 = normal).
    kurtosis_excess : Excess kurtosis of daily returns (default 0 = normal).

    Returns
    -------
    dict with deflated_sharpe, expected_max_sharpe, p_value, is_significant.
    """
    from scipy import stats as sp_stats

    if n_days < 2 or n_trials < 1:
        return {
            "deflated_sharpe": np.nan,
            "expected_max_sharpe": np.nan,
            "p_value": np.nan,
            "is_significant": False,
        }

    # Expected maximum Sharpe under the null (Bailey & López de Prado 2014, eq. 6).
    # The expression below is in z-score units; convert to ANNUALISED Sharpe units
    # by scaling with sqrt(252 / n_days), since SR_daily ~ N(0, 1/T) under the null
    # and SR_annual = SR_daily * sqrt(252).
    euler_mascheroni = 0.5772156649
    e_max_z = (
        (1.0 - euler_mascheroni) * sp_stats.norm.ppf(1.0 - 1.0 / n_trials)
        + euler_mascheroni * sp_stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    )
    sr0 = e_max_z * np.sqrt(252.0 / n_days)  # expected max SR, annualised

    # Variance of the Sharpe estimator (Lo, 2002), annualised.
    # Lo's formula gives Var(SR_daily); multiply by 252 to get annualised variance.
    # The skew/kurtosis correction uses the DAILY Sharpe.
    sr = observed_sharpe
    sr_daily = sr / np.sqrt(252.0)
    var_sr_ann = 252.0 * (
        1.0
        - skew * sr_daily
        + (kurtosis_excess / 4.0) * sr_daily ** 2
    ) / (n_days - 1)
    se_sr = max(np.sqrt(var_sr_ann), 1e-12)

    # Test statistic: how far observed is above the expected maximum
    psr_stat = (sr - sr0) / se_sr
    p_value = 1.0 - float(sp_stats.norm.cdf(psr_stat))

    return {
        "deflated_sharpe": float(psr_stat),
        "expected_max_sharpe": float(sr0),
        "p_value": float(p_value),
        "is_significant": bool(p_value < 0.05),
        "observed_sharpe": float(sr),
        "n_trials": int(n_trials),
        "n_days": int(n_days),
    }