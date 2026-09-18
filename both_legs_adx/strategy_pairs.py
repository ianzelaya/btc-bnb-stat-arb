"""
Full two-leg BTC/BNB pairs variant -- both legs actually traded (long one /
short the other), unlike the BTC-only adaptation in strategy_btc_only.py.
Only tradeable on Binance: this venue choice isn't a simplification, it's a
hard constraint, since no major futures venue offers BNB derivatives outside
Binance itself.

Signal construction is identical in form to the BTC-only variant (same
rolling-beta hedge ratio, same percentile-rank spread signal, same BTC-only
regime filter -- see that module's docstring for the full mechanism and
thesis) -- what's different here is EXECUTION (both legs traded, not just
BTC) and therefore MEASUREMENT (two_leg_backtest.py's combiner, not a single
run_backtest() call) and PARAMETERS (this variant's own dedicated
re-optimization -- see optimize_pairs.py and graveyard.md -- landing on a
genuinely different region of parameter space than the BTC-only variant,
same way that adaptation itself needed its own search rather than reusing
values tuned for a different leg configuration).

Regime filter: ADX (indicators.adx_wilder) on BTC's own price -- entries
are only allowed when ADX sits below `adx_threshold`, i.e. BTC isn't in a
strong one-sided trend, since the hedge ratio (and the spread's own
mean-reversion thesis) is least reliable exactly when BTC itself is making
a violent directional move. A genuinely different construction from a
separate regime filter this project has also worked with -- see
graveyard.md for the honest comparison -- kept out of this public module on
purpose (see README).

Sizing/risk are NOT signal concerns and live entirely in
two_leg_backtest.py: `beta` (this module's hedge-ratio series, exposed on
PairsSignals so the backtest can read beta_at_entry) sizes both legs
dollar-for-dollar against a fixed capital allocation, and the stop is a
max-loss fraction of that same capital -- there is no ATR-based stop
anywhere in this variant's parameters. See two_leg_backtest.py's docstring
for why: an ATR-based BTC-price-level stop doesn't make sense for a hedged
spread trade, since it can force-close a position that's net flat/
profitable (BTC alone moved against the stop while BNB moved favorably),
and it doesn't control dollar risk-at-stop consistently across BTC's price
history the way risk-based sizing did in the old (discarded) single-leg-ATR
design.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators import adx_wilder, rolling_beta, rolling_percentile_rank


@dataclass(frozen=True)
class PairsParams:
    # --- Hedge ratio / spread (same construction as the BTC-only variant) ---
    beta_window: int = 500
    pct_lookback: int = 2500

    # --- Entry/exit thresholds on the spread's percentile rank ---
    low_q: float = 0.30
    high_q: float = 0.92
    mid_q: float = 0.55

    max_hold_bars: int = 300

    # --- BTC-only regime filter (ADX -- see module docstring). adx_window
    # is fixed at the standard default (not searched by optimize_pairs.py),
    # same convention as this project's other Wilder-smoothed indicators. ---
    adx_window: int = 14
    adx_threshold: float = 30.0

    # --- Risk: max combined loss (both legs, after beta-hedge sizing) as a
    # fraction of the pair's total allocated capital -- see
    # two_leg_backtest.py's CAP constant and stop-check loop. Not an ATR
    # multiple; this variant has no ATR-based sizing or stop at all. ---
    max_loss_pct_of_cap: float = 0.01


# Result of the ADX re-optimization (see optimize_pairs.py and graveyard.md)
# -- 2,645 backtests (245 stage 1 + 2,400 stage 2), selected by min(IS
# Sharpe, OOS Sharpe): min_sharpe=1.153 (IS 1.153 / OOS 1.380), 682 trades
# (594 IS / 88 OOS) -- close to the redacted real-filter result's own
# min_sharpe of 1.228, a useful confirmation that the core mechanism (hedge
# + percentile-rank spread) carries most of the edge, not the specific
# choice of regime filter.
DEFAULT_PARAMS = PairsParams()


@dataclass
class PairsSignals:
    entry_signal: pd.Series
    btc_direction: pd.Series        # +1 (long BTC) / -1 (short BTC) / 0
    beta: pd.Series                 # hedge ratio -- read at entry bar for position sizing
    exit_signal_long: pd.Series     # exits a long-BTC / short-BNB trade
    exit_signal_short: pd.Series    # exits a short-BTC / long-BNB trade


def build_signals(df_btc: pd.DataFrame, df_bnb: pd.DataFrame,
                   params: PairsParams = DEFAULT_PARAMS) -> PairsSignals:
    common_idx = df_btc.index.intersection(df_bnb.index)
    btc = df_btc.loc[common_idx]
    bnb = df_bnb.loc[common_idx]

    beta = rolling_beta(bnb["close"], btc["close"], params.beta_window).ffill()
    spread = bnb["close"] - beta * btc["close"]
    pct = rolling_percentile_rank(spread, params.pct_lookback)

    adx = adx_wilder(btc["high"], btc["low"], btc["close"], params.adx_window)
    regime_ok = adx <= params.adx_threshold

    pairs_buy = regime_ok & (pct < params.low_q)     # long BNB / short BTC
    pairs_sell = regime_ok & (pct > params.high_q)   # short BNB / long BTC

    entry_signal = (pairs_buy | pairs_sell).fillna(False)
    btc_direction = pd.Series(np.where(pairs_buy, -1, np.where(pairs_sell, 1, 0)), index=common_idx)

    exit_signal_long = (pct <= params.mid_q).fillna(False)
    exit_signal_short = (pct >= params.mid_q).fillna(False)

    return PairsSignals(
        entry_signal=entry_signal,
        btc_direction=btc_direction,
        beta=beta,
        exit_signal_long=exit_signal_long,
        exit_signal_short=exit_signal_short,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n = 6000
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    btc_walk = 60_000 + np.cumsum(rng.normal(0, 30, n))
    bnb_walk = 400 + 0.003 * (btc_walk - 60_000) + np.cumsum(rng.normal(0, 0.5, n))

    def _ohlc(walk, noise):
        high = walk + np.abs(rng.normal(noise, noise / 2, n))
        low = walk - np.abs(rng.normal(noise, noise / 2, n))
        close = walk + rng.normal(0, noise / 4, n)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    df_btc, df_bnb = _ohlc(btc_walk, 20), _ohlc(bnb_walk, 0.3)
    sig = build_signals(df_btc, df_bnb, PairsParams(beta_window=50, pct_lookback=200))
    assert sig.entry_signal.dtype == bool
    assert set(sig.btc_direction.unique()) <= {-1, 0, 1}
    assert len(sig.beta) == n
    print(f"n bars={n}, n entry signals={int(sig.entry_signal.sum())}")
    print("\nPASS: strategy_pairs smoke test")
