from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

_TREASURY_XML_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
)
_TREASURY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ml-trading/1.0)",
}


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _normalize_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _guess_treasury_column(name: str, symbol: str) -> str | None:
    token = f"{name} {symbol}".lower()
    if "10" in token:
        return "BC_10YEAR"
    if " 2" in token or "2y" in token or "2yr" in token or token.endswith("2"):
        return "BC_2YEAR"
    return None


def _resolve_column(columns: list[str], target: str) -> str | None:
    wanted = _normalize_token(target)
    for col in columns:
        if _normalize_token(col) == wanted:
            return col
    return None


def _parse_treasury_xml(text: str, symbols: dict[str, str]) -> pd.DataFrame:
    """Parse the Treasury XML feed into a date-indexed DataFrame."""
    root = ET.fromstring(text)

    rows: list[dict[str, str | None]] = []
    for elem in root.iter():
        if _strip_ns(elem.tag).lower() != "entry":
            continue

        props = None
        for child in elem.iter():
            if _strip_ns(child.tag).lower() == "properties":
                props = child
                break
        if props is None:
            continue

        row = {_strip_ns(child.tag): child.text for child in list(props)}
        if row:
            rows.append(row)

    if not rows:
        raise RuntimeError("Treasury XML feed returned empty")

    raw = pd.DataFrame(rows)
    raw.columns = [str(c).strip() for c in raw.columns]

    date_col = next(
        (c for c in raw.columns if _normalize_token(c) in {"NEWDATE", "DATE"}),
        None,
    )
    if date_col is None:
        date_col = next((c for c in raw.columns if "DATE" in c.upper()), None)
    if date_col is None:
        raise RuntimeError("Date column not found in Treasury XML feed")

    df = pd.DataFrame(index=pd.to_datetime(raw[date_col], errors="coerce").dt.normalize())
    df.index.name = "date"

    for output_name, source_symbol in symbols.items():
        treasury_col = _guess_treasury_column(output_name, source_symbol)
        if treasury_col is None:
            continue
        actual_col = _resolve_column(list(raw.columns), treasury_col)
        if actual_col is None:
            continue
        df[output_name] = pd.to_numeric(raw[actual_col], errors="coerce").to_numpy()

    df = df[~df.index.isna()].sort_index()
    if df.empty or df.shape[1] == 0:
        raise RuntimeError("Treasury XML feed is missing required maturity columns")
    return df


def _fetch_treasury_xml(symbols: dict[str, str], start: str, end: str) -> pd.DataFrame:
    """Fetch 2Y/10Y yields from the official U.S. Treasury XML feed."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    frames: list[pd.DataFrame] = []
    for year in range(start_ts.year, end_ts.year + 1):
        params = {
            "data": "daily_treasury_yield_curve",
            "field_tdr_date_value": str(year),
        }
        log.info("  Fetching Treasury XML feed for %s ...", year)
        resp = requests.get(
            _TREASURY_XML_URL,
            params=params,
            headers=_TREASURY_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        frames.append(_parse_treasury_xml(resp.text, symbols))

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[(df.index >= start_ts.normalize()) & (df.index <= end_ts.normalize())]
    return df


def _fetch_fredapi(symbols: dict[str, str], start: str, end: str) -> pd.DataFrame:
    """Fetch data from FRED using fredapi."""
    from fredapi import Fred

    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        raise EnvironmentError("FRED_API_KEY is not set in the environment")

    fred = Fred(api_key=key)
    series = {}
    for name, fred_symbol in symbols.items():
        log.info("  Fetching %s (%s) from FRED ...", fred_symbol, name)
        s = fred.get_series(fred_symbol, observation_start=start, observation_end=end)
        s.index = pd.to_datetime(s.index).normalize()
        s.index.name = "date"
        s.name = name
        series[name] = s

    return pd.concat(series.values(), axis=1).sort_index()


def _fetch_datareader(symbols: dict[str, str], start: str, end: str) -> pd.DataFrame:
    """Fetch data from FRED using pandas-datareader."""
    from pandas_datareader import data as web

    fred_codes = list(symbols.values())
    name_map = {v: k for k, v in symbols.items()}

    log.info("  Fetching %s from FRED via pandas-datareader ...", fred_codes)
    df = web.DataReader(fred_codes, "fred", start, end)
    df = df.rename(columns=name_map)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    return df.sort_index()


def _fetch_yfinance_proxy(start: str, end: str) -> pd.DataFrame:
    """Fetch an approximate fallback from Yahoo Finance."""
    import yfinance as yf

    log.warning("  Fallback: fetching ^TNX from Yahoo Finance (10Y only)")
    raw = yf.download("^TNX", start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError("Could not fetch ^TNX from Yahoo Finance")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    s = raw["Close"].squeeze().copy()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index = s.index.normalize()
    s.index.name = "date"

    df = pd.DataFrame(index=s.index)
    df["us10y"] = s
    df["us2y"] = np.nan
    log.warning("  2Y Treasury unavailable in Yahoo proxy -> leaving us2y as NaN so yield-spread features are disabled instead of fabricated")
    return df.sort_index()


def fetch_treasury(
    start: str,
    end: str,
    symbols: dict[str, str] | None = None,
    cache_dir: Path | None = None,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """Fetch Treasury yields and return a date-indexed DataFrame."""
    if symbols is None:
        symbols = {"us10y": "DGS10", "us2y": "DGS2"}

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fp = cache_dir / f"treasury_{'_'.join(symbols.keys())}_{start}_{end}.parquet"
        if fp.exists():
            log.info("fetch_treasury: cache hit -> %s", fp.name)
            return pd.read_parquet(fp)

    df = None
    errors: list[str] = []

    for label, loader in (
        ("treasury_xml", lambda: _fetch_treasury_xml(symbols, start, end)),
        ("datareader", lambda: _fetch_datareader(symbols, start, end)),
        ("fredapi", lambda: _fetch_fredapi(symbols, start, end)),
        ("yfinance", lambda: _fetch_yfinance_proxy(start, end)),
    ):
        try:
            df = loader()
            log.info("fetch_treasury: OK via %s", label)
            break
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            log.debug("  %s failed: %s", label, exc)

    if df is None:
        raise RuntimeError(
            "Could not fetch treasury data from any source. "
            f"Errors: {errors}"
        )

    n_nan_before = int(df.isna().sum().sum())
    df = df.ffill(limit=ffill_limit)
    n_nan_after = int(df.isna().sum().sum())
    log.info("  ffill(limit=%s): NaN %s -> %s", ffill_limit, n_nan_before, n_nan_after)

    df = df.dropna(how="all")

    if cache_dir is not None:
        df.to_parquet(fp)
        log.info("fetch_treasury: cached -> %s", fp.name)

    stats = []
    for col in df.columns:
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        stats.append(f"{col} [{valid.min():.2f}-{valid.max():.2f}%]")
    log.info("fetch_treasury OK: %s days | %s", len(df), " | ".join(stats))
    return df
