"""
BTC/BNB stat-arb signal, adapted to trade the BTC leg only.

What this is: a classic pairs mean-reversion construction -- a rolling OLS
hedge ratio (beta) of BNB on BTC defines a spread (BNB_close -
beta*BTC_close); the spread's own rolling empirical percentile rank is the
signal. When the spread is unusually LOW relative to its recent history
(bottom `low_q`), the pairs construction goes long BNB / short BTC, betting
the spread reverts up; unusually HIGH (top `high_q`) goes short BNB / long
BTC. A BTC-only regime filter (ADX, see indicators.adx_wilder) restricts
entries to periods where BTC isn't in a strong one-sided trend, since the
hedge ratio (and the spread's own mean-reversion thesis) is least reliable
exactly when BTC itself is making a violent directional move. A genuinely
different regime construction this project has also worked with is kept
out of this public module on purpose -- see graveyard.md and README.

Why BTC only, here: this adaptation targets execution on CME Micro Bitcoin
futures (MBT) via IBKR, which doesn't offer BNB futures at all -- so only
the BTC leg is actually tradeable through that venue. BNB is used purely as
a signal input; see ../both_legs_adx/strategy_pairs.py for the full two-leg
variant (Binance-only, since that's the only venue that can actually
execute both legs), which required its own dedicated re-optimization
rather than reusing these parameters -- see that module and graveyard.md
for why a parameter set tuned for one leg configuration doesn't transfer
to the other.

No stop-loss in the original pairs construction -- both an ATR-based and a
fixed-percentage stop were implemented and tested here as additions;
ATR won a head-to-head comparison (see ../graveyard.md).

The rolling empirical percentile is computed via pandas' native vectorized
`.rolling(lookback).rank(pct=True)` (indicators.rolling_percentile_rank) --
fast even over million-row series.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators import adx_wilder, atr_wilder, rolling_beta, rolling_percentile_rank


@dataclass(frozen=True)
class StatArbBtcParams:
    # --- Hedge ratio / spread ---
    beta_window: int = 100          # bars, rolling OLS beta of BNB on BTC
    pct_lookback: int = 2500        # bars, spread's own percentile-rank window

    # --- Entry/exit thresholds on the spread's percentile rank ---
    low_q: float = 0.08   # pct < this -> pairs "BUY" (long BNB/short BTC) -> we go SHORT btc
    high_q: float = 0.70  # pct > this -> pairs "SELL" (short BNB/long BTC) -> we go LONG btc
    mid_q: float = 0.45   # exit when pct reverts back through this

    max_hold_bars: int = 300  # hard time exit (~3.1 days @ 15m), regardless of pct

    # --- BTC-only regime filter (ADX -- see indicators.adx_wilder). adx_window
    # is fixed at the standard default, same convention as this project's
    # other Wilder-smoothed indicators, not searched by the optimizer. ---
    adx_window: int = 14
    adx_threshold: float = 25.0

    # --- Stop-loss (both modes tested, see graveyard.md) ---
    stop_mode: Literal["atr", "fixed_pct"] = "atr"
    atr_window: int = 14
    atr_stop_multiple: float = 15.0
    fixed_stop_pct: float = 0.003


# Result of the ADX re-optimization for the BTC leg (see optimize_btc_only.py
# and graveyard.md) -- 2,645 backtests (245 stage 1 + 2,400 stage 2), selected
# by min(IS Sharpe, OOS Sharpe): min_sharpe=0.813 (IS 0.813 / OOS 1.094),
# 1,270 trades (1,072 IS / 198 OOS). A naive port of the pairs construction's
# own parameters onto BTC-only execution fails outright (confirmed: a
# 635-run one-at-a-time sweep, zero in-sample-positive combinations -- see
# graveyard.md) because the edge, at those settings, lives disproportionately
# in the BNB leg's own reversion, not in BTC's counter-move -- meaning the
# parameter region that works well for BTC-only execution is a genuinely
# different region of the search space, not a small perturbation.
DEFAULT_PARAMS = StatArbBtcParams()


@dataclass
class Signals:
    entry_signal: pd.Series
    direction: pd.Series           # +1 (long BTC) / -1 (short BTC) / 0
    stop_r_price: pd.Series
    exit_signal_long: pd.Series    # bool: exit a long-BTC trade (pairs "SELL") when pct reverts down
    exit_signal_short: pd.Series   # bool: exit a short-BTC trade (pairs "BUY") when pct reverts up


def build_signals(df_btc: pd.DataFrame, df_bnb: pd.DataFrame,
                   params: StatArbBtcParams = DEFAULT_PARAMS) -> Signals:
    """df_btc / df_bnb: OHLCV DataFrames (open/high/low/close), sorted
    ascending, UTC index -- exactly what data_fetch.fetch_klines returns.
    Only df_btc's OHLC is used for the stop/regime/trade simulation itself
    (BNB contributes its close price only, as a signal input). Aligned on
    the intersection of both indices.

    Returns a Signals bundle ready to hand to backtest_engine.run_backtest(
    df_btc, sig.entry_signal, sig.stop_r_price, sig.direction,
    exit_signal_long=sig.exit_signal_long, exit_signal_short=sig.exit_signal_short,
    cfg=BacktestConfig(max_holding_bars=params.max_hold_bars)).
    """
    common_idx = df_btc.index.intersection(df_bnb.index)
    btc = df_btc.loc[common_idx]
    bnb_close = df_bnb.loc[common_idx, "close"]

    beta = rolling_beta(bnb_close, btc["close"], params.beta_window).ffill()
    spread = bnb_close - beta * btc["close"]
    pct = rolling_percentile_rank(spread, params.pct_lookback)

    adx = adx_wilder(btc["high"], btc["low"], btc["close"], params.adx_window)
    regime_ok = adx <= params.adx_threshold

    pairs_buy = regime_ok & (pct < params.low_q)     # long BNB / short BTC
    pairs_sell = regime_ok & (pct > params.high_q)   # short BNB / long BTC

    entry_signal = (pairs_buy | pairs_sell).fillna(False)
    direction = pd.Series(
        np.where(pairs_buy, -1, np.where(pairs_sell, 1, 0)),
        index=common_idx,
    )

    exit_signal_long = (pct <= params.mid_q).fillna(False)   # exits a long-BTC (pairs "SELL") trade
    exit_signal_short = (pct >= params.mid_q).fillna(False)  # exits a short-BTC (pairs "BUY") trade

    high, low, close = btc["high"], btc["low"], btc["close"]
    if params.stop_mode == "atr":
        atr = atr_wilder(high, low, close, params.atr_window)
        stop_r_price = (atr * params.atr_stop_multiple).rename("stop_r_price")
    elif params.stop_mode == "fixed_pct":
        stop_r_price = (close * params.fixed_stop_pct).rename("stop_r_price")
    else:
        raise ValueError(f"stop_mode must be 'atr' or 'fixed_pct', got {params.stop_mode!r}")

    return Signals(
        entry_signal=entry_signal,
        direction=direction,
        stop_r_price=stop_r_price,
        exit_signal_long=exit_signal_long,
        exit_signal_short=exit_signal_short,
    )


if __name__ == "__main__":
    # Smoke test with two fabricated, mildly-correlated random-walk price
    # paths -- no network, just proves the vectorized signal math runs
    # end-to-end and produces sane shapes/types before real data.
    rng = np.random.default_rng(11)
    n = 6000
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    btc_walk = 60_000 + np.cumsum(rng.normal(0, 30, n))
    bnb_walk = 400 + 0.003 * (btc_walk - 60_000) + np.cumsum(rng.normal(0, 0.5, n))

    def _ohlc(walk, noise_scale):
        high = walk + np.abs(rng.normal(noise_scale, noise_scale / 2, n))
        low = walk - np.abs(rng.normal(noise_scale, noise_scale / 2, n))
        close = walk + rng.normal(0, noise_scale / 4, n)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    df_btc = _ohlc(btc_walk, 20)
    df_bnb = _ohlc(bnb_walk, 0.3)

    sig = build_signals(df_btc, df_bnb, StatArbBtcParams(beta_window=50, pct_lookback=200))
    assert sig.entry_signal.dtype == bool
    assert sig.entry_signal.index.equals(df_btc.index.intersection(df_bnb.index))
    assert set(sig.direction.unique()) <= {-1, 0, 1}
    n_signals = int(sig.entry_signal.sum())
    print(f"n bars={n}, n entry signals={n_signals}")
    print("\nPASS: strategy_btc_only smoke test")
