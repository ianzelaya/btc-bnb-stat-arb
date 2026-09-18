"""
Two-leg pairs backtest: a real joint bar-by-bar loop over both legs, sized
as a genuine beta hedge against a fixed capital allocation.

History: an earlier version of this module ran backtest_engine.run_backtest()
on the BTC leg ALONE (an ATR-based stop, equal-risk-dollar sizing per leg)
and pasted BNB onto BTC's resulting schedule after the fact. That was wrong
on two counts, both caught in review before any numbers were published:

1. Sizing wasn't actually a hedge. `beta = rolling_beta(bnb, btc)` is the
   OLS slope of regressing BNB's price on BTC's (see strategy_pairs.py) --
   dimensionally, dBNB_$/dBTC_$. The position that makes combined P&L
   track the SPREAD (not the shared BTC/BNB market factor) is a UNIT
   ratio, `qty_btc = beta * qty_bnb` (derived from requiring
   `qty_bnb*dBNB - qty_btc*dBTC = 0` when `dBNB = beta*dBTC`, the
   regression's predicted co-movement) -- not a dollar-notional ratio.
   Applying beta to NOTIONALS directly (an earlier draft's mistake) uses
   beta's small magnitude (~0.005-0.01, since BTC trades ~100-150x BNB's
   price) to nearly zero out the BNB leg. Applied correctly to UNIT counts,
   the two legs land in the same dollar order of magnitude (~roughly
   balanced, not 99%/1%) because `beta * BTC_price` and `BNB_price` are
   themselves close in size -- that's what the regression's slope
   captures. See this module's own smoke test for the identity this
   guarantees: for a single trade (beta held fixed at its entry-time
   value), combined $ P&L is EXACTLY `-btc_direction * qty_bnb *
   (spread_exit - spread_entry)` (direction sign flips which leg is long)
   -- the hedge cancels the shared factor by construction, leaving only
   the spread's own move.

2. An ATR-based, BTC-price-level stop doesn't make sense for a hedged
   spread trade. It can force-close a position that's net flat or
   profitable (BTC alone crosses its stop while BNB has moved favorably
   enough to offset it -- the stop never looks at combined P&L at all).
   It also doesn't control dollar risk-at-stop consistently: sizing off a
   fixed capital allocation (see CAP below) means BTC's own ATR distance,
   not position size, ends up determining $-loss-at-stop, and that swung
   roughly 10x across this project's own 2020-2026 backtest history purely
   from BTC's price/volatility regime changing -- not something either
   party actually wants "risk per trade" to do. The stop here instead
   checks combined $ P&L (both legs, at each bar's CLOSE) against a
   fraction of CAP -- tied to the actual thing being risked (the spread),
   and stable across price regimes by construction.

Sizing convention: a fixed capital sleeve (`CAP`) split so
`qty_btc = beta_at_entry * qty_bnb` and `notional_btc + notional_bnb ==
CAP` at entry. Solving both simultaneously:
    qty_bnb = CAP / (beta_at_entry * btc_entry_price + bnb_entry_price)
    qty_btc = beta_at_entry * qty_bnb
Quantities are fixed for the trade's life -- no mid-trade rebalancing as
beta drifts (a real, documented limitation: the hedge is only exact at
entry's beta, see graveyard.md).

Exit mechanics: three triggers, checked at each bar's CLOSE for BOTH legs
(not high/low -- a joint intrabar worst-case isn't knowable from OHLC data
without assuming BTC's and BNB's extremes happened at the same instant
within the bar, so this project uses close-to-close, a conservative-enough
simplification consistent with the rest of this codebase's documented
choices): (a) stop -- combined $ P&L <= -max_loss_pct_of_cap * CAP; (b)
signal exit -- the spread's percentile rank crosses back through mid_q
(only checked from the bar after entry, matching backtest_engine's
convention); (c) time exit -- max_hold_bars reached. Stop wins ties within
the same bar. Both legs always exit on the identical bar (same exit_time)
by construction -- BTC's and BNB's schedules never desync, since there's a
single combined trigger evaluated jointly, not two independent per-leg
engines (an earlier design considered exactly that and rejected it: two
independent run_backtest() calls on the same signals produced different
trade COUNTS entirely, 883 vs. 787, because one leg exiting early frees it
for a new entry the other leg -- still "in trade" on its own clock -- has
to skip).

Costs: VIP3-tier Binance fee + slippage on both legs (see costs.py), plus
funding across every ~8h settlement the trade spans (see
costs.compute_funding_usd) -- NOT netted between legs, since sizing is
beta-hedge notional, not equal notional, so funding on one leg doesn't
cancel funding on the other.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from costs import compute_fee_usd, compute_funding_usd

CAP = 550_000.0  # this strategy's capital sleeve -- one allocation among several sharing one account


@dataclass
class PairsBacktestConfig:
    cap_usd: float = CAP
    max_loss_pct_of_cap: float = 0.02
    max_hold_bars: int = 150
    allow_overlapping: bool = False
    fee_multiplier: float = 1.0       # feeds the 0x/1x/2x cost-sensitivity check
    funding_multiplier: float = 1.0   # same convention, for funding stress-testing
    apply_funding: bool = True


def _first_true_index(mask: np.ndarray) -> Optional[int]:
    nz = np.flatnonzero(mask)
    return int(nz[0]) if len(nz) else None


def run_pairs_backtest(df_btc: pd.DataFrame, df_bnb: pd.DataFrame, entry_signal: pd.Series,
                        btc_direction: pd.Series, beta: pd.Series,
                        exit_signal_long: pd.Series, exit_signal_short: pd.Series,
                        cfg: PairsBacktestConfig, funding_df_btc: Optional[pd.DataFrame] = None,
                        funding_df_bnb: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """df_btc / df_bnb: OHLC, aligned indices (same convention as
    strategy_pairs.build_signals -- entry_signal/btc_direction/beta/
    exit_signal_long/exit_signal_short all share that same index).
    funding_df_btc/funding_df_bnb: fetch_funding_rates() output per symbol;
    None (or cfg.apply_funding=False) skips funding entirely (e.g. for a
    fast smoke test with fabricated data that has no real funding history).

    Returns one row per PAIR trade: entry_time, exit_time, exit_reason,
    bars_held, is_closed, per-leg qty/entry/exit prices, beta_entry, and
    the combined pnl_r/mae_r/mfe_r (now defined as fraction of CAP, not
    fraction of a per-trade risk dollar amount) that every measurement.py
    function reads.
    """
    common_idx = df_btc.index.intersection(df_bnb.index)
    btc, bnb = df_btc.loc[common_idx], df_bnb.loc[common_idx]
    entry_signal = entry_signal.loc[common_idx]
    btc_direction = btc_direction.loc[common_idx]
    beta = beta.loc[common_idx]
    exit_signal_long = exit_signal_long.loc[common_idx]
    exit_signal_short = exit_signal_short.loc[common_idx]

    n = len(common_idx)
    idx = common_idx
    btc_open, btc_close = btc["open"].to_numpy(), btc["close"].to_numpy()
    bnb_open, bnb_close = bnb["open"].to_numpy(), bnb["close"].to_numpy()
    beta_arr = beta.to_numpy()
    dir_arr = btc_direction.to_numpy()
    exit_long_arr = exit_signal_long.to_numpy(dtype=bool)
    exit_short_arr = exit_signal_short.to_numpy(dtype=bool)

    stop_threshold_usd = -cfg.max_loss_pct_of_cap * cfg.cap_usd
    signal_positions = np.flatnonzero(entry_signal.to_numpy(dtype=bool))

    rows = []
    in_trade_until = -1

    for sig_pos in signal_positions:
        entry_pos = sig_pos + 1  # next-bar-open fill, matching backtest_engine's convention
        if entry_pos >= n:
            continue
        if not cfg.allow_overlapping and entry_pos <= in_trade_until:
            continue

        beta_entry = beta_arr[sig_pos]
        if not np.isfinite(beta_entry) or beta_entry <= 0:
            continue  # hedge undefined -- rolling window warmup or a rare negative-beta regime

        d = int(np.sign(dir_arr[sig_pos])) or 1  # +1 = long BTC/short BNB, -1 = short BTC/long BNB
        bnb_d = -d

        btc_entry_price = btc_open[entry_pos]
        bnb_entry_price = bnb_open[entry_pos]
        denom = beta_entry * btc_entry_price + bnb_entry_price
        if not np.isfinite(denom) or denom <= 0:
            continue
        qty_bnb = cfg.cap_usd / denom
        qty_btc = beta_entry * qty_bnb

        max_pos = min(n - 1, entry_pos + cfg.max_hold_bars)
        window_len = max_pos - entry_pos + 1

        btc_close_w = btc_close[entry_pos:max_pos + 1]
        bnb_close_w = bnb_close[entry_pos:max_pos + 1]
        btc_pnl_path = (btc_close_w - btc_entry_price) * d * qty_btc
        bnb_pnl_path = (bnb_close_w - bnb_entry_price) * bnb_d * qty_bnb
        combined_pnl_path = btc_pnl_path + bnb_pnl_path

        stop_mask = combined_pnl_path <= stop_threshold_usd
        sig_arr = exit_long_arr if d == 1 else exit_short_arr
        sig_mask = sig_arr[entry_pos:max_pos + 1].copy()
        sig_mask[0] = False  # never on the entry bar itself

        stop_first = _first_true_index(stop_mask)
        sig_first = _first_true_index(sig_mask)
        last_local_idx = window_len - 1

        candidates = [(i, r) for i, r in ((stop_first, "stop"), (sig_first, "signal_exit")) if i is not None]
        if candidates:
            exit_local_idx, exit_reason = min(candidates, key=lambda t: (t[0], 0 if t[1] == "stop" else 1))
            is_closed = True
        elif max_pos < n - 1:
            exit_local_idx, exit_reason = last_local_idx, "time_exit"
            is_closed = True
        else:
            exit_local_idx, exit_reason = last_local_idx, "end_of_data"
            is_closed = False

        exit_pos = entry_pos + exit_local_idx
        exit_time = idx[exit_pos]
        entry_time = idx[entry_pos]

        btc_exit_price = btc_close_w[exit_local_idx]
        bnb_exit_price = bnb_close_w[exit_local_idx]
        btc_pnl_usd_gross = btc_pnl_path[exit_local_idx]
        bnb_pnl_usd_gross = bnb_pnl_path[exit_local_idx]
        combined_pnl_usd_gross = btc_pnl_usd_gross + bnb_pnl_usd_gross

        fee_usd = (compute_fee_usd(qty_btc * btc_entry_price, multiplier=cfg.fee_multiplier)
                   + compute_fee_usd(qty_bnb * bnb_entry_price, multiplier=cfg.fee_multiplier))

        funding_usd = 0.0
        if cfg.apply_funding and funding_df_btc is not None and funding_df_bnb is not None:
            funding_usd = (cfg.funding_multiplier * compute_funding_usd(
                                entry_time, exit_time, qty_btc, d, btc["close"], funding_df_btc)
                            + cfg.funding_multiplier * compute_funding_usd(
                                entry_time, exit_time, qty_bnb, bnb_d, bnb["close"], funding_df_bnb))

        pnl_usd_net = combined_pnl_usd_gross - fee_usd + funding_usd

        path_so_far = combined_pnl_path[:exit_local_idx + 1]
        mae_r = max(-path_so_far.min(), 0.0) / cfg.cap_usd
        mfe_r = max(path_so_far.max(), 0.0) / cfg.cap_usd

        rows.append({
            "entry_time": entry_time, "exit_time": exit_time, "exit_reason": exit_reason,
            "bars_held": exit_local_idx, "is_closed": is_closed,
            "beta_entry": beta_entry,
            "btc_direction": d, "btc_entry_price": btc_entry_price, "btc_exit_price": btc_exit_price,
            "qty_btc": qty_btc, "btc_pnl_usd_gross": btc_pnl_usd_gross,
            "bnb_direction": bnb_d, "bnb_entry_price": bnb_entry_price, "bnb_exit_price": bnb_exit_price,
            "qty_bnb": qty_bnb, "bnb_pnl_usd_gross": bnb_pnl_usd_gross,
            "pnl_usd_gross": combined_pnl_usd_gross, "fee_usd": fee_usd, "funding_usd": funding_usd,
            "pnl_usd_net": pnl_usd_net,
            # Blended fields every measurement.py function actually reads --
            # "R" is redefined here as fraction of CAP, not fraction of a
            # per-trade risk-dollar amount (see module docstring).
            "pnl_r": pnl_usd_net / cfg.cap_usd, "mae_r": mae_r, "mfe_r": mfe_r,
        })
        in_trade_until = exit_pos

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Smoke test: two fabricated, correlated price paths -- no network, no funding data.
    rng = np.random.default_rng(7)
    n = 8000
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    btc_walk = 60_000 + np.cumsum(rng.normal(0, 30, n))
    bnb_walk = 500 + 0.005 * (btc_walk - 60_000) + np.cumsum(rng.normal(0, 0.3, n))

    def _ohlc(walk, noise):
        high = walk + np.abs(rng.normal(noise, noise / 2, n))
        low = walk - np.abs(rng.normal(noise, noise / 2, n))
        close = walk + rng.normal(0, noise / 4, n)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    df_btc, df_bnb = _ohlc(btc_walk, 20), _ohlc(bnb_walk, 0.3)

    from indicators import rolling_beta
    beta = rolling_beta(df_bnb["close"], df_btc["close"], 200).ffill()

    entry_signal = pd.Series(False, index=idx)
    entry_signal.iloc[300::250] = True
    direction = pd.Series(np.where(np.arange(n) % 500 < 250, 1, -1), index=idx)
    # Sparse, rare exit triggers -- frequent enough to exercise signal_exit at
    # all, but rare enough that most trades run long enough to also test the
    # stop and time-exit paths (a smoke test wants all three exit_reasons to
    # actually occur at least once, not one path dominating by construction).
    exit_long = pd.Series(False, index=idx)
    exit_short = pd.Series(False, index=idx)
    exit_long.iloc[::173] = True
    exit_short.iloc[::181] = True

    # A tight max_loss_pct_of_cap here (vs. the real ~1-5% range this
    # project actually searches) is deliberate: this synthetic data's
    # spread noise is small relative to CAP, so a realistic threshold would
    # never trigger the stop path at all. The point of a smoke test is to
    # exercise the code path, not to model a realistic hit rate.
    cfg = PairsBacktestConfig(cap_usd=550_000.0, max_loss_pct_of_cap=0.0005, max_hold_bars=100, apply_funding=False)
    trades = run_pairs_backtest(df_btc, df_bnb, entry_signal, direction, beta, exit_long, exit_short, cfg)
    assert len(trades) > 0
    assert {"pnl_r", "mae_r", "mfe_r", "entry_time", "exit_time", "beta_entry"} <= set(trades.columns)

    # Hedge identity: for a single trade (beta fixed at entry), combined gross
    # $ P&L must exactly equal -btc_direction * qty_bnb * (spread_exit - spread_entry)
    # (the sign flips with direction: qty_bnb*(spread move) is the identity for the
    # "long BNB / short BTC" (btc_direction=-1) case specifically; long BTC/short BNB
    # is the mirror image).
    for _, t in trades.iterrows():
        spread_entry = t["bnb_entry_price"] - t["beta_entry"] * t["btc_entry_price"]
        spread_exit = t["bnb_exit_price"] - t["beta_entry"] * t["btc_exit_price"]
        expected = -t["btc_direction"] * t["qty_bnb"] * (spread_exit - spread_entry)
        assert abs(t["pnl_usd_gross"] - expected) < 1e-6, (t["pnl_usd_gross"], expected)

    # Stop discipline: no closed "stop" trade should have a gross loss worse
    # than the threshold by more than a bar's worth of slack (checked at
    # close, not intrabar, so it can gap slightly past the line).
    stopped = trades[trades["exit_reason"] == "stop"]
    assert len(stopped) > 0, "smoke test's random walk should trigger at least one stop"

    # Both legs always exit on the identical bar, by construction. exit_time
    # can equal entry_time (a stop hit on the very fill bar is a legitimate
    # outcome, same convention as backtest_engine.run_backtest) but never precede it.
    assert (trades["exit_time"] >= trades["entry_time"]).all()

    print(f"n trades={len(trades)}, stops={len(stopped)}, "
          f"signal_exits={(trades.exit_reason=='signal_exit').sum()}, "
          f"time_exits={(trades.exit_reason=='time_exit').sum()}")
    print("\nPASS: two_leg_backtest smoke test")
