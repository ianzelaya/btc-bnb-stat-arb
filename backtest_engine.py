"""
Strategy-agnostic vectorized-pandas backtest engine, built ahead of picking
the actual Part B strategy so the plumbing exists before the signal does.

Design choice: this loops over ENTRY EVENTS, not over every bar -- the test
explicitly says "vectorised pandas is fine, no need for an event-driven
engine," which this respects in spirit: there is no full bar-by-bar state
machine simulating an order book or limit orders. For BTC OHLCV at hourly/
daily resolution, entries are sparse (tens to low thousands over years of
history), so looping over entries while using vectorized slicing/boolean
masks WITHIN each trade's forward window is both fast and far simpler to
verify correctly than a "clever" fully-vectorized-across-trades version.

Deliberately excluded (cut for scope, matching "keep it simple"):
- Pyramiding / adds. Part A's add-to-winner logic is about LIVE trade
  management, not backtest measurement -- this engine measures one clip in,
  one clip out, per trade. Extending it to size adds would duplicate Part
  A's job in a place the test doesn't ask for it.
- A full order book / slippage-per-fill model. Costs are applied uniformly
  after the fact in measurement.py's cost-sensitivity function instead,
  which is what the spec actually asks for ("a cost assumption... and how
  results change if you double it") -- modeling slippage per-bar here would
  be more machinery than the honest-measurement requirement calls for.

Ambiguity this engine resolves conservatively, on purpose: if both a stop
and a take-profit would be touched within the SAME bar (only possible with
OHLC data, since we can't see intra-bar order), the STOP is assumed to have
been hit first. This is the standard conservative convention in backtesting
-- it never overstates results -- and is called out here rather than left
implicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    entry_fill: str = "next_open"       # "next_open" or "this_close"
    take_profit_r: Optional[float] = None   # fixed R target; None = no take-profit exit
    max_holding_bars: Optional[int] = None  # None = no time-based exit
    allow_overlapping: bool = False     # if False, skip new entries while already in a trade


def run_backtest(
    df: pd.DataFrame,
    entry_signal: pd.Series,
    stop_r_price: Union[pd.Series, float],
    direction: Union[pd.Series, int] = 1,
    cfg: Optional[BacktestConfig] = None,
    dynamic_exit_long: Optional[pd.Series] = None,
    dynamic_exit_short: Optional[pd.Series] = None,
    exit_signal_long: Optional[pd.Series] = None,
    exit_signal_short: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    df: OHLCV DataFrame, columns open/high/low/close, sorted ascending by a
        datetime index (exactly what data_fetch.fetch_klines returns).
    entry_signal: bool Series aligned to df.index. True = a signal fired at
        this bar (the actual fill happens per cfg.entry_fill).
    stop_r_price: distance from entry to initial stop, in PRICE units (not
        R -- this defines what R means for the trade, same convention as
        Part A's compute_r_price). Scalar (fixed stop distance) or a Series
        aligned to df.index (e.g. an ATR-based stop that varies per entry --
        read at the entry bar). Must be > 0.
    direction: +1 for long, -1 for short. Scalar (single-direction strategy)
        or a Series aligned to df.index, read at the entry bar (for a
        strategy that goes both ways).
    cfg: BacktestConfig; defaults to next-bar-open fills, no take-profit, no
        time exit, no overlapping positions.
    dynamic_exit_long / dynamic_exit_short: optional Series aligned to
        df.index, giving a PER-BAR exit trigger LEVEL (not frozen at entry --
        re-read at every bar the trade is open), used for strategies whose
        exit is a moving level rather than a fixed R target -- e.g. a
        Donchian channel that keeps rolling forward for the life of the
        trade. Which series applies is chosen by the TRADE's own direction
        (not the current bar's regime, which can differ or flip mid-trade):
        dynamic_exit_long for a long trade (exits when high >= level),
        dynamic_exit_short for a short trade (exits when low <= level).
        NaN at a given bar means "no dynamic exit level yet" (e.g. still
        inside the channel's warmup window) -- skipped, not treated as 0.
        Independent of cfg.take_profit_r; pass at most one exit mechanism
        per strategy in practice, but both are checked if both are given.
    exit_signal_long / exit_signal_short: optional bool Series aligned to
        df.index -- an INDICATOR-CONDITION exit (not a price level), e.g.
        "a mean-reversion spread's percentile rank has crossed back through
        its midpoint." Which series applies is chosen by the trade's own
        direction, same convention as dynamic_exit_long/short. Only checked
        from the bar AFTER entry onward (never on the entry bar itself,
        matching the convention of evaluating entry and exit as mutually
        exclusive within a single bar). Exits at that bar's CLOSE (the
        condition is evaluated from close-based data, so close is the
        earliest realistic fill -- same convention as the time-exit below).

    Returns a DataFrame, one row per trade, columns:
      entry_time, entry_price, exit_time, exit_price, direction,
      r_price, pnl_price, pnl_r, mae_r, mfe_r, bars_held, exit_reason,
      is_closed (False only for a trade still open at the end of the data)
    """
    cfg = cfg or BacktestConfig()
    if cfg.entry_fill not in ("next_open", "this_close"):
        raise ValueError("entry_fill must be 'next_open' or 'this_close'")

    n = len(df)
    opens, highs, lows, closes = (df["open"].to_numpy(), df["high"].to_numpy(),
                                   df["low"].to_numpy(), df["close"].to_numpy())
    idx = df.index

    stop_series = (pd.Series(stop_r_price, index=df.index) if np.isscalar(stop_r_price)
                   else stop_r_price)
    dir_series = (pd.Series(direction, index=df.index) if np.isscalar(direction)
                  else direction)
    dyn_exit_long_arr = dynamic_exit_long.to_numpy() if dynamic_exit_long is not None else None
    dyn_exit_short_arr = dynamic_exit_short.to_numpy() if dynamic_exit_short is not None else None
    exit_sig_long_arr = exit_signal_long.to_numpy(dtype=bool) if exit_signal_long is not None else None
    exit_sig_short_arr = exit_signal_short.to_numpy(dtype=bool) if exit_signal_short is not None else None

    signal_positions = np.flatnonzero(entry_signal.to_numpy(dtype=bool))

    trades = []
    in_trade_until = -1  # positional index of the last bar of the current open trade

    for sig_pos in signal_positions:
        entry_pos = sig_pos + 1 if cfg.entry_fill == "next_open" else sig_pos
        if entry_pos >= n:
            continue  # signal on the last bar with next_open fill -- no bar to fill on
        if not cfg.allow_overlapping and entry_pos <= in_trade_until:
            continue  # still in a prior trade -- skip this signal

        d = int(np.sign(dir_series.iloc[sig_pos])) or 1
        r_price = float(stop_series.iloc[sig_pos])
        if not np.isfinite(r_price) or r_price <= 0:
            continue  # can't size a trade with no valid stop distance

        entry_price = opens[entry_pos] if cfg.entry_fill == "next_open" else closes[entry_pos]
        stop_price = entry_price - d * r_price
        tp_price = (entry_price + d * cfg.take_profit_r * r_price
                    if cfg.take_profit_r is not None else None)

        max_pos = n - 1
        if cfg.max_holding_bars is not None:
            max_pos = min(max_pos, entry_pos + cfg.max_holding_bars)

        exit_pos, exit_price, exit_reason = None, None, None
        running_mae_price, running_mfe_price = 0.0, 0.0  # adverse/favorable, in price units, >= 0

        dyn_arr = dyn_exit_long_arr if d == 1 else dyn_exit_short_arr
        exit_sig_arr = exit_sig_long_arr if d == 1 else exit_sig_short_arr

        for pos in range(entry_pos, max_pos + 1):
            hi, lo, cl = highs[pos], lows[pos], closes[pos]
            adverse = (entry_price - lo) if d == 1 else (hi - entry_price)
            favorable = (hi - entry_price) if d == 1 else (entry_price - lo)
            running_mae_price = max(running_mae_price, adverse)
            running_mfe_price = max(running_mfe_price, favorable)

            stop_touched = (lo <= stop_price) if d == 1 else (hi >= stop_price)
            dyn_level = dyn_arr[pos] if dyn_arr is not None else np.nan
            dyn_touched = (np.isfinite(dyn_level) and
                           ((hi >= dyn_level) if d == 1 else (lo <= dyn_level)))
            tp_touched = (tp_price is not None and
                          ((hi >= tp_price) if d == 1 else (lo <= tp_price)))
            sig_exit = (exit_sig_arr is not None and pos > entry_pos and exit_sig_arr[pos])

            if stop_touched:  # conservative: stop wins ties within the same bar
                exit_pos, exit_price, exit_reason = pos, stop_price, "stop"
                break
            if dyn_touched:
                exit_pos, exit_price, exit_reason = pos, dyn_level, "channel_exit"
                break
            if sig_exit:
                exit_pos, exit_price, exit_reason = pos, cl, "signal_exit"
                break
            if tp_touched:
                exit_pos, exit_price, exit_reason = pos, tp_price, "take_profit"
                break
            if cfg.max_holding_bars is not None and pos == max_pos:
                exit_pos, exit_price, exit_reason = pos, cl, "time_exit"
                break

        is_closed = exit_pos is not None
        if not is_closed:
            exit_pos, exit_price, exit_reason = max_pos, closes[max_pos], "end_of_data"

        pnl_price = (exit_price - entry_price) * d
        trades.append({
            "entry_time": idx[entry_pos], "entry_price": entry_price,
            "exit_time": idx[exit_pos], "exit_price": exit_price,
            "direction": d, "r_price": r_price,
            "pnl_price": pnl_price, "pnl_r": pnl_price / r_price,
            "mae_r": running_mae_price / r_price, "mfe_r": running_mfe_price / r_price,
            "bars_held": exit_pos - entry_pos, "exit_reason": exit_reason,
            "is_closed": is_closed,
        })
        in_trade_until = exit_pos

    return pd.DataFrame(trades)


if __name__ == "__main__":
    # Smoke test with a fabricated price path -- no network, no real strategy.
    idx = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open":  [100, 101, 102, 103, 97, 96, 95, 108, 109, 110],
        "high":  [101, 102, 103, 104, 98, 97, 96, 109, 110, 111],
        "low":   [99, 100, 101, 96, 95, 94, 94, 107, 108, 109],
        "close": [101, 102, 103, 97, 96, 95, 95, 108, 109, 110],
    }, index=idx)
    entry_signal = pd.Series([True] + [False] * 9, index=idx)  # one signal, at bar 0
    trades = run_backtest(df, entry_signal, stop_r_price=3.0, direction=1)
    print(trades.to_string())
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_price"] == 101.0  # next_open fill, bar 1's open
    assert row["exit_reason"] == "stop"  # low=96 at bar idx 3 breaches stop=101-3=98
    assert abs(row["pnl_r"] - (-1.0)) < 1e-9  # exited exactly at the stop -> -1.0R
    print("\nPASS: backtest_engine smoke test")
