"""
Free BTC OHLCV data fetcher -- strategy-agnostic, built ahead of picking the
actual Part B strategy so tomorrow's session starts with data already in
hand rather than spent on API plumbing.

Supports both Binance markets via the `market` param:
  - "spot"    -> /api/v3/klines on api.binance.com (the original default).
  - "futures" -> /fapi/v1/klines on fapi.binance.com (USDT-M perpetuals).
Both are PUBLIC endpoints -- no API key needed, this is market data, not
account data, so it works from anywhere without the testnet/auth machinery
part_a/binance_broker.py needs.

The Part B strategy (BTC/BNB stat-arb, see strategy_stat_arb_btc.py) is run
here on "futures" to match Part A's venue (Binance USDT-M futures) --
keeping data and cost assumptions on the same instrument family the
"connect A and B" section will compare against. The test's own instructions
say to "note basis differences vs. CME if you think they matter" -- spot
capability is kept here too so that comparison is one function argument
away, not a rewrite.

Caches every pull to CSV under part_b/data/ so repeated backtest runs (and
parameter sweeps) don't re-hit the API or risk hitting rate limits mid-sweep.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

MARKET_BASE_URLS = {
    "spot": "https://api.binance.com",
    "futures": "https://fapi.binance.com",
}
MARKET_KLINES_ENDPOINTS = {
    "spot": "/api/v3/klines",
    "futures": "/fapi/v1/klines",
}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _market_urls(market: str) -> tuple[str, str]:
    if market not in MARKET_BASE_URLS:
        raise ValueError(f"market must be one of {list(MARKET_BASE_URLS)}, got {market!r}")
    return MARKET_BASE_URLS[market], MARKET_KLINES_ENDPOINTS[market]

# Binance caps each klines response at this many rows -- larger ranges are
# paginated internally by walking startTime forward.
MAX_LIMIT = 1000

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def _interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    n = int(interval[:-1])
    unit_ms = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    if unit not in unit_ms:
        raise ValueError(f"Unsupported interval unit in {interval!r}")
    return n * unit_ms[unit]


def _cache_path(symbol: str, interval: str, start: str, end: str, market: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    safe = lambda s: s.replace("-", "").replace(":", "").replace(" ", "T")
    prefix = "" if market == "spot" else f"{market}_"
    return os.path.join(DATA_DIR, f"{prefix}{symbol}_{interval}_{safe(start)}_{safe(end)}.csv")


def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1h",
                  start: str = "2020-01-01", end: str | None = None,
                  use_cache: bool = True, market: str = "spot") -> pd.DataFrame:
    """Fetch historical OHLCV between `start` and `end` (date strings, UTC),
    paginating past Binance's 1000-row-per-request cap automatically.

    market: "spot" (default, api.binance.com) or "futures" (USDT-M perps,
    fapi.binance.com) -- see module docstring for why "futures" is what the
    Part B strategy actually runs on.

    Returns a DataFrame indexed by UTC timestamp with columns:
    open, high, low, close, volume (all float), plus n_trades (int).
    """
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base_url, klines_endpoint = _market_urls(market)
    cache_file = _cache_path(symbol, interval, start, end, market)
    if use_cache and os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    step_ms = _interval_to_ms(interval)

    rows = []
    cursor = start_ms
    session = requests.Session()
    while cursor < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": MAX_LIMIT,
        }
        resp = session.get(base_url + klines_endpoint, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_open_time = batch[-1][0]
        cursor = last_open_time + step_ms
        if len(batch) < MAX_LIMIT:
            break
        time.sleep(0.2)  # polite pacing, well under Binance's public rate limit

    if not rows:
        raise RuntimeError(
            f"No klines returned for {symbol} {interval} {start}->{end} (market={market}). "
            f"Check the symbol/interval or your network access to {base_url}."
        )

    df = _klines_to_df(rows)

    if use_cache:
        df.to_csv(cache_file)
    return df


def _klines_to_df(rows: list) -> pd.DataFrame:
    """Pure transform: raw Binance klines rows -> clean OHLCV DataFrame.
    Factored out from fetch_klines so it's unit-testable with fabricated
    rows, no network needed (see tests/test_data_fetch.py)."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["n_trades"] = df["n_trades"].astype(int)
    df = df[["open", "high", "low", "close", "volume", "n_trades"]]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


FUNDING_ENDPOINT = "/fapi/v1/fundingRate"  # USDT-M perpetuals only -- no spot equivalent


def _funding_cache_path(symbol: str, start: str, end: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    safe = lambda s: s.replace("-", "").replace(":", "").replace(" ", "T")
    return os.path.join(DATA_DIR, f"funding_{symbol}_{safe(start)}_{safe(end)}.csv")


def _funding_rows_to_df(rows: list) -> pd.DataFrame:
    """Pure transform: raw /fapi/v1/fundingRate rows -> a Series-shaped
    DataFrame indexed by the funding SETTLEMENT timestamp (fundingTime),
    one row per ~8h funding event. Factored out for offline testability,
    same pattern as _klines_to_df."""
    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df.set_index("funding_time")[["funding_rate"]]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def fetch_funding_rates(symbol: str = "BTCUSDT", start: str = "2020-01-01",
                         end: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Historical funding-rate settlements for a USDT-M perpetual (public
    endpoint, no API key needed -- same pagination/caching pattern as
    fetch_klines). Convention: funding_rate > 0 means longs pay shorts at
    that settlement -- see costs.py's funding cost function for how this is
    actually applied to a trade's P&L."""
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache_file = _funding_cache_path(symbol, start, end)
    if use_cache and os.path.exists(cache_file):
        # Explicit to_datetime rather than relying on read_csv's
        # parse_dates=True heuristic -- that heuristic behaves differently
        # for a single-data-column CSV (this one) than for the
        # multi-column klines cache, and silently leaves the index as
        # plain strings here on pandas 3.x. Confirmed directly: klines
        # cache round-trips to datetime64 fine with parse_dates=True;
        # this file's index came back as dtype=str.
        df = pd.read_csv(cache_file, index_col=0)
        # format="ISO8601": funding timestamps occasionally carry a few ms
        # of sub-second precision (Binance's own settlement clock, not a
        # parsing artifact here) while most don't -- a fixed-format parse
        # chokes on that mix, ISO8601 handles variable precision.
        df.index = pd.to_datetime(df.index, utc=True, format="ISO8601")
        return df

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    rows = []
    cursor = start_ms
    session = requests.Session()
    while cursor < end_ms:
        params = {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": MAX_LIMIT}
        resp = session.get(MARKET_BASE_URLS["futures"] + FUNDING_ENDPOINT, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_funding_time = batch[-1]["fundingTime"]
        cursor = last_funding_time + 1
        if len(batch) < MAX_LIMIT:
            break
        time.sleep(0.2)

    if not rows:
        raise RuntimeError(f"No funding rate history returned for {symbol} {start}->{end}.")

    df = _funding_rows_to_df(rows)
    if use_cache:
        df.to_csv(cache_file)
    return df


if __name__ == "__main__":
    # Smoke test -- pulls a small recent window so it's fast to sanity-check.
    for market in ("spot", "futures"):
        df = fetch_klines(symbol="BTCUSDT", interval="1h", start="2026-08-01", end="2026-09-01", market=market)
        print(market, df.shape)
        print(df.head())
        print(df.tail())
        assert df.index.is_monotonic_increasing
        assert not df.isnull().any().any()
    print("\nPASS: fetch_klines smoke test")

    fr = fetch_funding_rates(symbol="BTCUSDT", start="2026-08-01", end="2026-09-01")
    print("\nfunding", fr.shape)
    print(fr.head())
    assert fr.index.is_monotonic_increasing
    assert not fr.isnull().any().any()
    print("\nPASS: fetch_funding_rates smoke test")
