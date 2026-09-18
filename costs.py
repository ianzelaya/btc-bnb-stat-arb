"""
Strategy-agnostic execution-cost model -- extracted so any Part B strategy's
driver script can import the same cost basis instead of redefining it. The
cost model itself doesn't depend on which strategy is being costed, only on
which venue/instrument is being traded.

---------------------------------------------------------------------------
Cost assumption -- IBKR, not Binance. IBKR is the intended live execution
venue via CME Micro Bitcoin futures (MBT, 0.1 BTC/contract), matching Part
A. Binance is still used as the free, long-history price/signal data
source for BOTH legs of a signal (e.g. a BTC/BNB pairs-derived signal) --
market data and execution venue are independent choices, and only the
actually-TRADED instrument (BTC) needs a cost model; a signal-only
reference asset (e.g. BNB) never touches costs at all.

IBKR CME Micro Bitcoin futures commission: could NOT be confirmed to the
exact cent -- interactivebrokers.com returned HTTP 403 to automated
fetches. Public search results bracket CME crypto futures between CME
Micro Ether's $0.10/contract and full-size Bitcoin (BRR)'s up to
$5/contract. IBKR_FEE_PER_CONTRACT_PER_SIDE_USD = $2.00 is used as an
explicitly-labeled ESTIMATE (all-in, including assumed CME exchange/
regulatory pass-through) pending confirmation once Ian's account clears --
update this the moment a real number is available, do not treat as verified.

Unlike Binance's %-of-notional fee, IBKR's is FLAT PER CONTRACT -- it does
NOT scale with price (a $2/contract fee on a fixed 0.1 BTC contract costs
the same in dollar terms whether BTC is at $5k or $100k). Converted to a
per-BTC-of-exposure cost: IBKR_ROUNDTRIP_USD_PER_BTC = 2 sides x $2.00 /
0.1 BTC = $40/BTC round-trip. Since position size in BTC = risk_per_trade_usd
/ r_price, the $ cost of a trade is (risk_per_trade_usd/r_price) x
IBKR_ROUNDTRIP_USD_PER_BTC, and dividing by risk_per_trade_usd to express in
R gives cost_r = IBKR_ROUNDTRIP_USD_PER_BTC / r_price -- notably it does NOT
depend on entry_price at all (unlike the Binance formula below), which is
the real structural difference between a flat-per-contract and a
%-of-notional fee schedule.

Binance kept as an alternative cost_basis for comparison: 0.05% taker/side
+ 0.02% slippage/side = 0.14% round-trip, cost_r = BINANCE_BASE_COST_PCT *
entry_price / r_price (scales with price, unlike IBKR's).

---------------------------------------------------------------------------
VIP3 tier (pairs variant) -- the two-leg BTC/BNB variant assumes a VIP3
Binance USDT-M futures account (>=$50M 30D volume, >=100 BNB balance), not
the retail tier above: this fleet trades >100 trades/day across 8+ stat-arb
strategies sharing one account, which clears VIP3 volume easily. No BNB
10%-fee-discount applied (conservative: 0.0320% taker, not 0.0288%) -- see
BINANCE_VIP3_TAKER_FEE_PCT. Slippage stays at the same 0.02%/side as the
retail assumption: 15m-bar signals on $1-5M AUM clips are non-HFT and small
relative to BTC/BNB perp daily volume, so the retail slippage estimate
(confirmed against 1-min-candle intra-minute range) already covers it --
only the fee schedule changes with tier, not the market-impact/slippage
component.

Funding -- perpetuals charge/pay funding roughly every 8h (see
data_fetch.fetch_funding_rates), and the two-leg variant holds positions up
to max_hold_bars long enough to span several funding events (150-300 bars
x 15m = 37.5-75h, i.e. up to ~9 events). This is NOT netted between legs:
sizing is beta-hedge-notional (see two_leg_backtest.py), not equal-notional,
so BTC's and BNB's funding payments don't cancel. Convention: funding_rate
> 0 means longs pay shorts at that settlement (Binance's own convention),
so a leg's funding P&L contribution is -direction * notional * funding_rate
summed over every funding event strictly within (entry_time, exit_time].
---------------------------------------------------------------------------
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- IBKR cost basis (primary, BTC-only variant) ---
IBKR_FEE_PER_CONTRACT_PER_SIDE_USD = 2.00   # ESTIMATE -- see module docstring
MBT_BTC_PER_CONTRACT = 0.1
IBKR_ROUNDTRIP_USD_PER_BTC = 2 * IBKR_FEE_PER_CONTRACT_PER_SIDE_USD / MBT_BTC_PER_CONTRACT  # $40/BTC

# --- Binance cost basis, retail tier (alternative, for comparison) ---
BINANCE_TAKER_FEE_PCT = 0.0005
BINANCE_SLIPPAGE_PCT = 0.0002
BINANCE_BASE_COST_PCT = 2 * (BINANCE_TAKER_FEE_PCT + BINANCE_SLIPPAGE_PCT)  # 0.14% round-trip

# --- Binance cost basis, VIP3 tier (pairs variant's actual assumption) ---
BINANCE_VIP3_TAKER_FEE_PCT = 0.00032   # no BNB 10% discount -- see module docstring
BINANCE_VIP3_BASE_COST_PCT = 2 * (BINANCE_VIP3_TAKER_FEE_PCT + BINANCE_SLIPPAGE_PCT)  # 0.104% round-trip

COST_BASIS = "ibkr"  # "ibkr" (default -- matches Part A's venue) or "binance"

# Matches Part A's own convention ($100k account, 0.5% risk/trade -> $500/R)
# so equity curves stay directly comparable in dollar terms across parts.
# BTC-only variant only -- the pairs variant uses its own CAP allocation,
# see two_leg_backtest.py's CAP constant.
ACCOUNT_EQUITY_USD = 100_000.0
RISK_PER_TRADE_USD = 500.0


def compute_cost_r(trades: pd.DataFrame, cost_basis: str = COST_BASIS, multiplier: float = 1.0) -> pd.Series:
    """Per-trade round-trip cost expressed in R. multiplier feeds the spec's
    required 0x/1x/2x cost-sensitivity check (see cost_sensitivity() in
    measurement.py). trades must have 'r_price' (and 'entry_price' for the
    binance bases) columns -- exactly what backtest_engine.run_backtest()
    produces."""
    if cost_basis == "ibkr":
        return pd.Series(multiplier * IBKR_ROUNDTRIP_USD_PER_BTC / trades["r_price"], index=trades.index)
    elif cost_basis == "binance":
        return multiplier * BINANCE_BASE_COST_PCT * trades["entry_price"] / trades["r_price"]
    elif cost_basis == "binance_vip3":
        return multiplier * BINANCE_VIP3_BASE_COST_PCT * trades["entry_price"] / trades["r_price"]
    raise ValueError(f"cost_basis must be 'ibkr', 'binance', or 'binance_vip3', got {cost_basis!r}")


def compute_fee_usd(notional_usd: float, multiplier: float = 1.0) -> float:
    """Round-trip fee + slippage in dollars for one leg of the pairs
    variant, VIP3 tier. multiplier feeds the same 0x/1x/2x cost-sensitivity
    convention as compute_cost_r."""
    return multiplier * BINANCE_VIP3_BASE_COST_PCT * notional_usd


def compute_funding_usd(entry_time, exit_time, qty: float, direction: int,
                         price_series: pd.Series, funding_df: pd.DataFrame) -> float:
    """Funding P&L (can be positive or negative) for one leg held from
    entry_time to exit_time. direction: +1 long, -1 short. price_series:
    that leg's own close price series (indexed like funding_df/df_btc/
    df_bnb), used to mark notional at each funding event rather than
    freezing it at entry -- funding is paid on current notional, not the
    entry-time notional. funding_df: fetch_funding_rates() output for that
    symbol (funding_rate > 0 = longs pay shorts, Binance's convention).

    Only counts events strictly after entry_time and up to/including
    exit_time -- a position must be open AT a funding settlement to owe or
    receive it. Vectorized (no iterrows) -- called per-leg, per-trade,
    inside a grid search that runs millions of trades total, so the
    per-call cost matters."""
    mask = (funding_df.index > entry_time) & (funding_df.index <= exit_time)
    if not mask.any():
        return 0.0
    event_times = funding_df.index[mask]
    rates = funding_df.loc[mask, "funding_rate"].to_numpy()
    prices_at_events = price_series.reindex(event_times, method="ffill").to_numpy()
    notional = qty * prices_at_events
    return float(np.nansum(-direction * notional * rates))
