"""Offline tests for two_leg_backtest.py -- fabricated prices, no network."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "both_legs_adx"))

import numpy as np
import pandas as pd

from two_leg_backtest import PairsBacktestConfig, run_pairs_backtest


def _flat_ohlc(closes: list[float], idx: pd.DatetimeIndex) -> pd.DataFrame:
    closes = np.array(closes, dtype=float)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    return pd.DataFrame({"open": opens, "high": closes, "low": closes, "close": closes}, index=idx)


def _series(idx, **kwargs):
    n = len(idx)
    entry_signal = kwargs.get("entry_signal", pd.Series(False, index=idx))
    btc_direction = kwargs.get("btc_direction", pd.Series(0, index=idx))
    beta = kwargs.get("beta", pd.Series(1.0, index=idx))
    exit_long = kwargs.get("exit_long", pd.Series(False, index=idx))
    exit_short = kwargs.get("exit_short", pd.Series(False, index=idx))
    return entry_signal, btc_direction, beta, exit_long, exit_short


def test_hedge_identity_holds_both_directions():
    """Combined gross $ P&L must exactly equal -direction*qty_bnb*(spread move)
    -- the whole point of beta-hedge sizing, for both trade directions."""
    n = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    btc = _flat_ohlc([60_000 + i * 50 for i in range(n)], idx)
    bnb = _flat_ohlc([500 + i * 0.3 for i in range(n)], idx)
    beta = pd.Series(0.006, index=idx)

    entry_signal = pd.Series(False, index=idx)
    entry_signal.iloc[[0, 10]] = True
    btc_direction = pd.Series(0, index=idx)
    btc_direction.iloc[0] = 1    # long BTC / short BNB
    btc_direction.iloc[10] = -1  # short BTC / long BNB
    exit_long = pd.Series(False, index=idx)
    exit_short = pd.Series(False, index=idx)

    cfg = PairsBacktestConfig(cap_usd=550_000.0, max_loss_pct_of_cap=0.5, max_hold_bars=5, apply_funding=False)
    trades = run_pairs_backtest(btc, bnb, entry_signal, btc_direction, beta, exit_long, exit_short, cfg)
    assert len(trades) == 2
    for _, t in trades.iterrows():
        spread_entry = t["bnb_entry_price"] - t["beta_entry"] * t["btc_entry_price"]
        spread_exit = t["bnb_exit_price"] - t["beta_entry"] * t["btc_exit_price"]
        expected = -t["btc_direction"] * t["qty_bnb"] * (spread_exit - spread_entry)
        assert abs(t["pnl_usd_gross"] - expected) < 1e-6
    print("PASS: test_hedge_identity_holds_both_directions")


def test_capital_constraint_satisfied_at_entry():
    """notional_btc + notional_bnb must equal CAP at entry, and qty_btc must
    equal beta_entry * qty_bnb (the unit-ratio hedge, not a notional ratio)."""
    n = 10
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    btc = _flat_ohlc([70_000] * n, idx)
    bnb = _flat_ohlc([600] * n, idx)
    beta = pd.Series(0.0055, index=idx)
    entry_signal = pd.Series(False, index=idx)
    entry_signal.iloc[0] = True
    btc_direction = pd.Series(0, index=idx)
    btc_direction.iloc[0] = 1
    exit_long = pd.Series(False, index=idx)
    exit_short = pd.Series(False, index=idx)

    cfg = PairsBacktestConfig(cap_usd=550_000.0, max_loss_pct_of_cap=0.5, max_hold_bars=5, apply_funding=False)
    trades = run_pairs_backtest(btc, bnb, entry_signal, btc_direction, beta, exit_long, exit_short, cfg)
    t = trades.iloc[0]
    notional_btc = t["qty_btc"] * t["btc_entry_price"]
    notional_bnb = t["qty_bnb"] * t["bnb_entry_price"]
    assert abs((notional_btc + notional_bnb) - 550_000.0) < 1e-6
    assert abs(t["qty_btc"] - t["beta_entry"] * t["qty_bnb"]) < 1e-9
    print("PASS: test_capital_constraint_satisfied_at_entry")


def test_stop_triggers_on_combined_pnl_not_either_leg_alone():
    """The scenario that motivated the redesign: BTC alone makes a huge move
    (big enough to have blown through any old BTC-only ATR stop), but BNB
    co-moves exactly per the beta relationship, so the SPREAD -- and
    therefore combined P&L -- barely moves. The pair must NOT be closed,
    even though BTC's own leg moved ~7%."""
    n = 10
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    beta_val = 0.0055
    btc_entry, bnb_entry = 70_000.0, 600.0
    d_btc = -5_000.0                    # BTC drops ~7% -- would trip most ATR stops
    d_bnb = beta_val * d_btc            # BNB co-moves exactly as the regression predicts -> spread flat
    btc_closes = [btc_entry] + [btc_entry + d_btc] * (n - 1)
    bnb_closes = [bnb_entry] + [bnb_entry + d_bnb] * (n - 1)
    btc = _flat_ohlc(btc_closes, idx)
    bnb = _flat_ohlc(bnb_closes, idx)
    beta = pd.Series(beta_val, index=idx)
    entry_signal = pd.Series(False, index=idx)
    entry_signal.iloc[0] = True
    btc_direction = pd.Series(0, index=idx)
    btc_direction.iloc[0] = 1  # long BTC / short BNB -- BTC leg alone loses big here
    exit_long = pd.Series(False, index=idx)
    exit_short = pd.Series(False, index=idx)

    cfg = PairsBacktestConfig(cap_usd=550_000.0, max_loss_pct_of_cap=0.02, max_hold_bars=5, apply_funding=False)
    trades = run_pairs_backtest(btc, bnb, entry_signal, btc_direction, beta, exit_long, exit_short, cfg)
    t = trades.iloc[0]
    # BTC leg alone lost a huge amount, but the hedge kept the spread (and
    # thus combined P&L) essentially flat -- confirm the pair was never
    # actually at risk of the stop despite BTC's own dramatic move.
    assert t["exit_reason"] != "stop"
    assert t["btc_pnl_usd_gross"] < -10_000  # BTC leg alone lost a lot in isolation
    assert abs(t["pnl_usd_gross"]) < 1.0      # but combined P&L is ~0 -- the hedge worked
    print("PASS: test_stop_triggers_on_combined_pnl_not_either_leg_alone")


def test_allow_overlapping_false_skips_signal_during_open_trade():
    n = 15
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    btc = _flat_ohlc([70_000 + i for i in range(n)], idx)
    bnb = _flat_ohlc([600 + i * 0.01 for i in range(n)], idx)
    beta = pd.Series(0.0055, index=idx)
    entry_signal = pd.Series(False, index=idx)
    entry_signal.iloc[[0, 1, 2]] = True  # signals while already in the first trade
    btc_direction = pd.Series(1, index=idx)
    exit_long = pd.Series(False, index=idx)
    exit_short = pd.Series(False, index=idx)

    cfg = PairsBacktestConfig(cap_usd=550_000.0, max_loss_pct_of_cap=0.5, max_hold_bars=10,
                               allow_overlapping=False, apply_funding=False)
    trades = run_pairs_backtest(btc, bnb, entry_signal, btc_direction, beta, exit_long, exit_short, cfg)
    assert len(trades) == 1, "overlapping signals during an open trade must be skipped"
    print("PASS: test_allow_overlapping_false_skips_signal_during_open_trade")


def test_funding_reduces_net_pnl_for_long_when_rate_positive():
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    btc = _flat_ohlc([70_000] * n, idx)
    bnb = _flat_ohlc([600] * n, idx)
    beta = pd.Series(0.0055, index=idx)
    entry_signal = pd.Series(False, index=idx)
    entry_signal.iloc[0] = True
    btc_direction = pd.Series(0, index=idx)
    btc_direction.iloc[0] = 1  # long BTC
    exit_long = pd.Series(False, index=idx)
    exit_short = pd.Series(False, index=idx)

    funding_idx = idx[[10, 20]]  # two funding events inside the trade's window
    funding_btc = pd.DataFrame({"funding_rate": [0.0002, 0.0002]}, index=funding_idx)
    funding_bnb = pd.DataFrame({"funding_rate": [0.0, 0.0]}, index=funding_idx)  # isolate the BTC leg's funding

    cfg = PairsBacktestConfig(cap_usd=550_000.0, max_loss_pct_of_cap=0.5, max_hold_bars=30, apply_funding=True)
    trades = run_pairs_backtest(btc, bnb, entry_signal, btc_direction, beta, exit_long, exit_short, cfg,
                                 funding_df_btc=funding_btc, funding_df_bnb=funding_bnb)
    t = trades.iloc[0]
    assert t["funding_usd"] < 0, "long BTC leg must pay (negative funding P&L) when funding_rate > 0"
    assert abs(t["pnl_usd_net"] - (t["pnl_usd_gross"] - t["fee_usd"] + t["funding_usd"])) < 1e-6
    print("PASS: test_funding_reduces_net_pnl_for_long_when_rate_positive")


def test_negative_or_zero_beta_skips_trade():
    n = 5
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    btc = _flat_ohlc([70_000] * n, idx)
    bnb = _flat_ohlc([600] * n, idx)
    beta = pd.Series(-0.01, index=idx)  # invalid hedge -- must be skipped, not crash
    entry_signal = pd.Series(False, index=idx)
    entry_signal.iloc[0] = True
    btc_direction = pd.Series(1, index=idx)
    exit_long = pd.Series(False, index=idx)
    exit_short = pd.Series(False, index=idx)

    cfg = PairsBacktestConfig(cap_usd=550_000.0, max_loss_pct_of_cap=0.5, max_hold_bars=3, apply_funding=False)
    trades = run_pairs_backtest(btc, bnb, entry_signal, btc_direction, beta, exit_long, exit_short, cfg)
    assert len(trades) == 0
    print("PASS: test_negative_or_zero_beta_skips_trade")


if __name__ == "__main__":
    test_hedge_identity_holds_both_directions()
    test_capital_constraint_satisfied_at_entry()
    test_stop_triggers_on_combined_pnl_not_either_leg_alone()
    test_allow_overlapping_false_skips_signal_during_open_trade()
    test_funding_reduces_net_pnl_for_long_when_rate_positive()
    test_negative_or_zero_beta_skips_trade()
    print("\nAll two_leg_backtest tests passed.")
