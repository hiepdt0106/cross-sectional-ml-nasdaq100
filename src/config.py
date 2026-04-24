"""Project configuration, validation, and feature registry."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MCAP_TOP10 = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "INTC", "NFLX", "PEP", "COST", "AVGO",
]


@dataclass(frozen=True)
class DataConfig:
    start_date: str
    end_date: str
    tickers: List[str]
    benchmark_ticker: str = "QQQ"
    min_trading_days: int = 200
    max_nan_ratio: float = 0.02
    max_consec_nan: int = 5


@dataclass(frozen=True)
class TreasuryConfig:
    symbols: dict = field(default_factory=lambda: {"us10y": "DGS10", "us2y": "DGS2"})
    ffill_limit: int = 5


@dataclass(frozen=True)
class LabelingConfig:
    horizon: int = 10
    pt_sl_mult: float = 1.5
    vol_window: int = 20


@dataclass(frozen=True)
class StrategyConfig:
    top_k: int = 10
    rebalance_days: int = 10
    confidence_weighted: bool = False
    max_weight_cap: float = 0.25
    hold_buffer: int = 3
    hold_score_tolerance: float = 0.02
    signal_anchor_weight: float = 0.0
    signal_anchor_features: List[str] = field(
        default_factory=lambda: [
            "mom_63d",
            "rel_strength_63d",
            "trend_strength_21d",
            "price_sma200",
        ]
    )


@dataclass(frozen=True)
class RegimeConfig:
    lookback: int = 504
    refit_frequency: int = 63


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps: float = 10.0
    slippage_bps: float = 0.0
    initial_capital: float = 10_000
    regime_targeting: bool = True
    regime_col: str = "p_high_vol"
    regime_sensitivity: float = 0.45
    regime_min_exposure: float = 0.55
    regime_max_exposure: float = 1.00
    breadth_gate: bool = False
    breadth_col: str = "market_breadth_200d"
    breadth_threshold: float = 0.35
    breadth_low_exposure: float = 0.60
    benchmark_trend_gate: bool = False
    benchmark_col: str = "bench_close"
    benchmark_sma_window: int = 200
    benchmark_low_exposure: float = 0.70
    vol_targeting: bool = True
    target_vol: float = 0.14
    vol_lookback_days: int = 63
    vol_min_scale: float = 0.75
    vol_max_scale: float = 1.15
    dd_threshold: float = -0.08
    dd_exit: float = -0.04
    dd_exposure: float = 0.70
    sector_max_weight: float = 0.40


@dataclass(frozen=True)
class RandomBenchmarkConfig:
    n_iterations: int = 200
    seed: int = 42


@dataclass
class WalkForwardConfig:
    first_test_year: int = 2020
    max_train_years: int | None = None


@dataclass
class BenchmarkMcapConfig:
    reference_date: str = "2020-01-01"
    tickers: List[str] = field(default_factory=lambda: list(DEFAULT_MCAP_TOP10))


@dataclass
class ProjectConfig:
    data: DataConfig
    labeling: LabelingConfig
    strategy: StrategyConfig
    walkforward: WalkForwardConfig
    regime: RegimeConfig
    backtest: BacktestConfig
    treasury: TreasuryConfig = field(default_factory=TreasuryConfig)
    random_benchmark: RandomBenchmarkConfig = field(default_factory=RandomBenchmarkConfig)
    benchmark_mcap: BenchmarkMcapConfig = field(default_factory=BenchmarkMcapConfig)

    dir_raw: Path = field(default_factory=lambda: ROOT / "data" / "raw")
    dir_interim: Path = field(default_factory=lambda: ROOT / "data" / "interim")
    dir_processed: Path = field(default_factory=lambda: ROOT / "data" / "processed")
    dir_cache: Path = field(default_factory=lambda: ROOT / "data" / "raw" / "cache")
    dir_outputs: Path = field(default_factory=lambda: ROOT / "outputs")
    dir_figures: Path = field(default_factory=lambda: ROOT / "outputs" / "figures")


# ── GICS Sector Mapping (NASDAQ-100 subset) ────────────────────────────────
SECTOR_MAP: dict[str, str] = {
    # Information Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "AMD": "Technology", "QCOM": "Technology",
    "AMAT": "Technology", "LRCX": "Technology", "MU": "Technology",
    "INTC": "Technology", "TXN": "Technology", "KLAC": "Technology",
    "MCHP": "Technology", "ADI": "Technology", "ADBE": "Technology",
    "CRM": "Technology", "ORCL": "Technology", "PANW": "Technology",
    "SNPS": "Technology", "CDNS": "Technology", "INTU": "Technology",
    "WDAY": "Technology", "ANSS": "Technology", "FTNT": "Technology",
    # Communication Services / Internet
    "AMZN": "Consumer Discretionary", "GOOGL": "Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services",
    "CMCSA": "Communication Services", "TMUS": "Communication Services",
    "CHTR": "Communication Services", "WBD": "Communication Services",
    "EA": "Communication Services",
    # Consumer Discretionary
    "EBAY": "Consumer Discretionary", "PYPL": "Financials",
    "BKNG": "Consumer Discretionary", "MELI": "Consumer Discretionary",
    "JD": "Consumer Discretionary", "ABNB": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary", "RIVN": "Consumer Discretionary",
    "LULU": "Consumer Discretionary", "ROST": "Consumer Discretionary",
    "DLTR": "Consumer Discretionary",
    # Health Care
    "AMGN": "Health Care", "GILD": "Health Care", "REGN": "Health Care",
    "VRTX": "Health Care", "ISRG": "Health Care", "MRNA": "Health Care",
    "DXCM": "Health Care", "IDXX": "Health Care", "BIIB": "Health Care",
    "ILMN": "Health Care", "AZN": "Health Care",
    # Consumer Staples
    "COST": "Consumer Staples", "PEP": "Consumer Staples",
    "SBUX": "Consumer Staples", "MDLZ": "Consumer Staples",
    "MNST": "Consumer Staples", "KDP": "Consumer Staples",
    "KHC": "Consumer Staples",
    # Industrials
    "HON": "Industrials", "PCAR": "Industrials", "ODFL": "Industrials",
    "CTAS": "Industrials", "VRSK": "Industrials", "CSX": "Industrials",
    "FAST": "Industrials", "PAYX": "Industrials", "ADP": "Industrials",
    # Utilities
    "XEL": "Utilities",
}


NON_FEATURE_COLS = frozenset({
    "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume",
    "vix", "vxn",
    "treasury_10y", "treasury_2y",
    "tb_label", "tb_barrier", "tb_return", "daily_vol", "t1", "holding_td",
    "alpha_ret", "alpha_label", "alpha_ext_label",
    "bench_close",
})

REMOVED_FEATURES = frozenset({
    "ret_1d_lag4",
    "ret_1d_lag5",
    "sma_cross_20_50",
    "vol_change_1d",
    "vol_change_5d",
    "vol_change_21d",
    "atr_14d",
    "downside_beta_63d",
    "vol_confirmation",
    "cs_ret_x_regime",
    "cs_mom_x_regime",
    "p_high_x_resid_ret",
    "vxn_accel",
})

MACRO_FEATURE_NAMES = frozenset({
    "vix_ret_1d", "vxn_ret_1d", "vxn_ret_5d",
    "vxn_zscore", "vxn_ma5_ma21",
    "vix_vxn_spread",
    "market_breadth_200d",
})

YIELD_FEATURE_NAMES = frozenset({
    "yield_spread_10y2y",
    "yield_spread_change_5d",
})

SKIP_RANK_FEATURES = MACRO_FEATURE_NAMES | YIELD_FEATURE_NAMES | frozenset({
    "market_dispersion_21d",
    "p_high_vol",
    "rolling_beta_63d",
    "resid_ret_21d",
    "idio_vol_21d",
    "rel_strength_21d",
    "cs_ret_zscore_1d",
    "cs_vol_zscore_21d",
    "cs_mom_zscore_63d",
    "overnight_gap",
    "close_position",
    "mom_quality",
    "stress_reversal",
    "beta_regime",
    "spread_momentum",
    "p_high_x_mom_63d",
    "p_high_x_vol_21d",
    "zspread",
    "zspread_ma5",
    "zspread_change_5d",
    "dispersion_adjusted_vol",
    "cs_mom_x_yield",
})

MACRO_FEATURE_PREFIXES = (
    "vix_", "vxn_", "zspread", "vix_vxn_", "p_high",
    "yield_spread", "market_breadth",
)

MACRO_DEPENDENT_FEATURES = frozenset({
    "vix_ret_1d",
    "vxn_ret_1d",
    "vxn_ret_5d",
    "vxn_zscore",
    "vxn_ma5_ma21",
    "vix_vxn_spread",
    "yield_spread_10y2y",
    "yield_spread_change_5d",
    "yield_spread_zscore",
    "market_breadth_200d",
    "market_dispersion_21d",
    "p_high_vol",
    "zspread",
    "zspread_ma5",
    "zspread_change_5d",
    "p_high_x_mom_63d",
    "p_high_x_vol_21d",
    "stress_reversal",
    "beta_regime",
    "spread_momentum",
    "dispersion_adjusted_vol",
    "cs_mom_x_yield",
})


def get_feature_cols(df_columns: list[str]) -> list[str]:
    """Return model feature columns after removing metadata and pruned inputs."""
    return [
        c for c in df_columns
        if c not in NON_FEATURE_COLS and c not in REMOVED_FEATURES
    ]



def split_feature_cols(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Split features into base and macro-dependent buckets by meaning."""
    macro_cols = [
        c for c in feature_cols
        if c.startswith(MACRO_FEATURE_PREFIXES) or c in MACRO_DEPENDENT_FEATURES
    ]
    base_cols = [c for c in feature_cols if c not in macro_cols]
    return base_cols, macro_cols


def infer_date_level_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    same_share_threshold: float = 0.95,
    value_tol: float = 1e-12,
    max_sample_dates: int = 252,
) -> list[str]:
    """Infer features that are effectively identical across tickers on most dates."""
    if len(feature_cols) == 0:
        return []

    work = df[feature_cols].copy()
    unique_dates = pd.Index(work.index.get_level_values("date")).unique().sort_values()
    if len(unique_dates) > max_sample_dates:
        sample_idx = np.linspace(0, len(unique_dates) - 1, max_sample_dates, dtype=int)
        sample_dates = unique_dates[sample_idx]
        work = work.loc[work.index.get_level_values("date").isin(sample_dates)]

    grouped_min = work.groupby(level="date").min()
    grouped_max = work.groupby(level="date").max()
    same_mask = (grouped_max - grouped_min).abs() <= value_tol
    same_share = same_mask.mean(axis=0)
    return sorted(same_share[same_share >= same_share_threshold].index.tolist())


def prune_redundant_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    corr_threshold: float = 0.9999,
    max_rows: int = 50000,
    min_pair_obs: int = 50,
) -> tuple[list[str], list[str]]:
    """Drop near-constant or near-duplicate features while preserving order."""
    if len(feature_cols) <= 1:
        return feature_cols, []

    work = df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if len(work) == 0:
        return feature_cols, []
    if len(work) > max_rows:
        work = work.sample(n=max_rows, random_state=42)

    nunique = work.nunique(dropna=True)
    keep: list[str] = []
    dropped: list[str] = []
    for col in feature_cols:
        if int(nunique.get(col, 0)) <= 1:
            dropped.append(col)
            continue
        is_dup = False
        for kept in keep:
            pair = work[[kept, col]].dropna()
            if len(pair) < min_pair_obs:
                continue
            corr = pair[kept].corr(pair[col])
            if pd.notna(corr) and abs(float(corr)) >= corr_threshold:
                dropped.append(col)
                is_dup = True
                break
        if not is_dup:
            keep.append(col)
    return keep, dropped


def prepare_feature_sets(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, list[str]]:
    """Prepare selector feature sets and prune dead / duplicate features."""
    base_cols_raw, macro_cols_raw = split_feature_cols(feature_cols)
    date_level = set(infer_date_level_features(df, feature_cols))

    base_cols = [c for c in base_cols_raw if c not in date_level]
    macro_cols = [c for c in macro_cols_raw if c not in date_level]
    raw_date_cols = sorted([c for c in feature_cols if c in date_level])

    ordered_full = base_cols + macro_cols
    full_cols, dropped_redundant = prune_redundant_features(df, ordered_full)
    base_cols = [c for c in base_cols if c in full_cols]
    macro_cols = [c for c in macro_cols if c in full_cols]

    return {
        "base_cols": base_cols,
        "macro_cols": macro_cols,
        "full_cols": full_cols,
        "raw_date_cols": raw_date_cols,
        "dropped_redundant": dropped_redundant,
    }


def _validate(raw: dict, config_path: Path | None = None) -> None:
    import pandas as pd

    data = raw.get("data", {})
    start = data.get("start_date")
    end = data.get("end_date")
    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
    except Exception as exc:
        raise ValueError("Invalid start_date/end_date") from exc
    if end_ts <= start_ts:
        raise ValueError("start_date/end_date order invalid")

    if not data.get("tickers"):
        raise ValueError("data.tickers must not be empty")
    if int(data.get("min_trading_days", 1)) <= 0:
        raise ValueError("min_trading_days must be > 0")
    nan_ratio = float(data.get("max_nan_ratio", 0.02))
    if not (0.0 <= nan_ratio <= 1.0):
        raise ValueError("max_nan_ratio must be in [0,1]")
    if int(data.get("max_consec_nan", 1)) < 0:
        raise ValueError("max_consec_nan must be >= 0")

    labeling = raw.get("labeling", {})
    if int(labeling.get("horizon", 1)) <= 0:
        raise ValueError("labeling.horizon must be > 0")
    if float(labeling.get("pt_sl_mult", 0.0)) <= 0:
        raise ValueError("labeling.pt_sl_mult must be > 0")
    if int(labeling.get("vol_window", 1)) <= 0:
        raise ValueError("labeling.vol_window must be > 0")

    strategy = raw.get("strategy", {})
    if int(strategy.get("top_k", 1)) <= 0:
        raise ValueError("strategy.top_k must be > 0")
    if int(strategy.get("rebalance_days", 1)) <= 0:
        raise ValueError("strategy.rebalance_days must be > 0")
    max_cap = float(strategy.get("max_weight_cap", 0.25))
    if not (0.0 < max_cap <= 1.0):
        raise ValueError("strategy.max_weight_cap must be in (0,1]")
    hold_buffer = int(strategy.get("hold_buffer", 0))
    if hold_buffer < 0:
        raise ValueError("strategy.hold_buffer must be >= 0")
    hold_score_tolerance = float(strategy.get("hold_score_tolerance", 0.0))
    if hold_score_tolerance < 0:
        raise ValueError("strategy.hold_score_tolerance must be >= 0")
    anchor_weight = float(strategy.get("signal_anchor_weight", 0.0))
    if not (0.0 <= anchor_weight <= 1.0):
        raise ValueError("strategy.signal_anchor_weight must be in [0,1]")
    anchor_features = strategy.get("signal_anchor_features", [])
    if anchor_features is not None and not isinstance(anchor_features, list):
        raise ValueError("strategy.signal_anchor_features must be a list")

    walk = raw.get("walkforward", {})
    if int(walk.get("first_test_year", 1900)) < 1900:
        raise ValueError("walkforward.first_test_year invalid")
    max_train_years = walk.get("max_train_years")
    if max_train_years is not None and int(max_train_years) <= 0:
        raise ValueError("walkforward.max_train_years must be > 0")

    regime = raw.get("regime", {})
    if int(regime.get("lookback", 1)) <= 0:
        raise ValueError("regime.lookback must be > 0")
    if int(regime.get("refit_frequency", 1)) <= 0:
        raise ValueError("regime.refit_frequency must be > 0")

    backtest = raw.get("backtest", {})
    if float(backtest.get("cost_bps", 0.0)) < 0:
        raise ValueError("backtest.cost_bps must be >= 0")
    if float(backtest.get("slippage_bps", 0.0)) < 0:
        raise ValueError("backtest.slippage_bps must be >= 0")
    if float(backtest.get("initial_capital", 0.0)) <= 0:
        raise ValueError("backtest.initial_capital must be > 0")

    regime_col = str(backtest.get("regime_col", "p_high_vol"))
    if not regime_col:
        raise ValueError("backtest.regime_col must not be empty")

    regime_sensitivity = float(backtest.get("regime_sensitivity", 0.35))
    if regime_sensitivity < 0:
        raise ValueError("backtest.regime_sensitivity must be >= 0")

    regime_min = float(backtest.get("regime_min_exposure", 0.65))
    regime_max = float(backtest.get("regime_max_exposure", 1.0))
    if not (0.0 < regime_min <= regime_max):
        raise ValueError("backtest regime exposure bounds invalid")

    breadth_threshold = float(backtest.get("breadth_threshold", 0.35))
    if not (0.0 <= breadth_threshold <= 1.0):
        raise ValueError("backtest.breadth_threshold must be in [0,1]")
    breadth_low_exposure = float(backtest.get("breadth_low_exposure", 0.60))
    if not (0.0 < breadth_low_exposure <= 1.0):
        raise ValueError("backtest.breadth_low_exposure must be in (0,1]")

    benchmark_sma_window = int(backtest.get("benchmark_sma_window", 200))
    if benchmark_sma_window <= 1:
        raise ValueError("backtest.benchmark_sma_window must be > 1")
    benchmark_low_exposure = float(backtest.get("benchmark_low_exposure", 0.70))
    if not (0.0 < benchmark_low_exposure <= 1.0):
        raise ValueError("backtest.benchmark_low_exposure must be in (0,1]")

    target_vol = float(backtest.get("target_vol", 0.16))
    if target_vol <= 0:
        raise ValueError("backtest.target_vol must be > 0")
    vol_lookback = int(backtest.get("vol_lookback_days", 63))
    if vol_lookback <= 1:
        raise ValueError("backtest.vol_lookback_days must be > 1")
    vol_min = float(backtest.get("vol_min_scale", 0.75))
    vol_max = float(backtest.get("vol_max_scale", 1.15))
    if not (0.0 < vol_min <= vol_max):
        raise ValueError("backtest volatility scale bounds invalid")

    dd_threshold = float(backtest.get("dd_threshold", 0.0))
    dd_exit = float(backtest.get("dd_exit", -0.05))
    dd_exposure = float(backtest.get("dd_exposure", 0.80))
    if dd_threshold < 0 and not (0.0 < dd_exposure <= 1.0):
        raise ValueError("backtest.dd_exposure must be in (0,1]")
    if dd_threshold < 0 and dd_exit < dd_threshold:
        raise ValueError("backtest.dd_exit must be >= backtest.dd_threshold")



def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    if config_path is None:
        config_path = ROOT / "configs" / "base.yaml"
    config_path = Path(config_path)

    if not config_path.exists() and config_path.name == "base_v2.yaml":
        config_path = ROOT / "configs" / "base.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _validate(raw)

    data_cfg = DataConfig(**raw["data"])
    labeling_cfg = LabelingConfig(**raw.get("labeling", {}))
    strategy_cfg = StrategyConfig(**raw.get("strategy", {}))
    walkforward_cfg = WalkForwardConfig(**raw.get("walkforward", {}))
    regime_cfg = RegimeConfig(**raw.get("regime", {}))
    backtest_cfg = BacktestConfig(**raw.get("backtest", {}))

    treasury_raw = raw.get("treasury", {})
    treasury_cfg = TreasuryConfig(**treasury_raw) if treasury_raw else TreasuryConfig()

    random_raw = raw.get("random_benchmark", {})
    random_cfg = RandomBenchmarkConfig(**random_raw) if random_raw else RandomBenchmarkConfig()

    mcap_raw = raw.get("benchmark_mcap", raw.get("benchmark_mcap_top10", {}))
    if mcap_raw:
        mcap_cfg = BenchmarkMcapConfig(
            reference_date=mcap_raw.get("reference_date", "2020-01-01"),
            tickers=mcap_raw.get("tickers", list(DEFAULT_MCAP_TOP10)),
        )
    else:
        mcap_cfg = BenchmarkMcapConfig()

    cfg = ProjectConfig(
        data=data_cfg,
        labeling=labeling_cfg,
        strategy=strategy_cfg,
        walkforward=walkforward_cfg,
        regime=regime_cfg,
        backtest=backtest_cfg,
        treasury=treasury_cfg,
        random_benchmark=random_cfg,
        benchmark_mcap=mcap_cfg,
    )

    for d in [
        cfg.dir_raw,
        cfg.dir_interim,
        cfg.dir_processed,
        cfg.dir_cache,
        cfg.dir_outputs,
        cfg.dir_figures,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    return cfg
