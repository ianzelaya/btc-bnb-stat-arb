"""
Generic, strategy-agnostic indicator primitives -- pure functions, no
strategy-specific composition logic (Wilder RSI/ATR/ADX, Donchian channels,
rolling percentile rank), reused by the BTC/BNB stat-arb strategy's regime
filter and stop-loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_wilder(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI (RMA smoothing, adjust=False)."""
    close = pd.Series(close).astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    rsi.name = f"RSI_{window}"
    return rsi


def donchian_channel(high: pd.Series, low: pd.Series, lower_length: int, upper_length: int
                      ) -> tuple[pd.Series, pd.Series]:
    """Rolling-min(low) / rolling-max(high) Donchian bands. Returns
    (lower, upper), NOT shifted -- callers shift as needed for lookahead
    safety."""
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)
    dcu = high.rolling(window=upper_length, min_periods=upper_length).max()
    dcl = low.rolling(window=lower_length, min_periods=lower_length).min()
    return dcl, dcu


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's ATR (RMA of true range)."""
    high, low, close = pd.Series(high).astype(float), pd.Series(low).astype(float), pd.Series(close).astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's ADX -- an unsigned, smoothed measure of trend strength
    (0-100), independent of direction. +DM/-DM (one-bar directional
    movement, whichever direction dominates that bar) are RMA-smoothed and
    normalized by RMA-smoothed true range into +DI/-DI; DX is their
    normalized absolute spread; ADX is DX's own RMA smoothing. High ADX =
    a strong, persistent one-sided move under way (regardless of which
    direction); low ADX = no clear trend, i.e. safe conditions for a
    mean-reversion entry. Same RMA-smoothing convention as atr_wilder/
    rsi_wilder above."""
    high, low, close = pd.Series(high).astype(float), pd.Series(low).astype(float), pd.Series(close).astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def rolling_percentile_rank(series: pd.Series, lookback: int) -> pd.Series:
    """Percentile rank (0-1) of each value within its own trailing
    `lookback` window -- 1.0 means "the highest this series has been in the
    last `lookback` bars," 0.0 the lowest. Vectorized (pandas' rolling
    rank), not a Python-level rolling apply -- fast even over
    million-row series."""
    return pd.Series(series).rolling(lookback).rank(pct=True)


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta of y on x via the covariance/variance shortcut
    (cov(y,x)/var(x)) -- the standard rolling hedge-ratio estimator for a
    pairs/spread trade."""
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var
