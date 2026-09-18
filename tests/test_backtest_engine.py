"""
Offline tests for backtest_engine.py -- fabricated OHLCV, no network, no
real strategy. Verifies the engine's mechanics (fills, stop/take-profit
detection, overlap skipping, short-side mirroring) are correct BEFORE any
real strategy gets built on top of them tomorrow.

Run: python -m pytest part_b/tests/ -v   (from part_b/, with part_b/ on PYTHONPATH)
  or: python tests/test_backtest_engine.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from backtest_engine import BacktestConfig, run_backtest


def _df(rows):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])


def test_long_trade_hits_stop():
    rows = [
        [100, 101, 99, 100],   # 0: signal bar
        [101, 102, 100, 102],  # 1: entry fill (next_open=101), no touch
        [102, 103, 101, 103],  # 2
        [103, 104, 96, 97],    # 3: low=96 breaches stop = 101-3=98
        [97, 98, 96, 97],      # 4 (should never be reached)
    ]
    df = _df(rows)
    entry_signal = pd.Series([True, False, False, False, False], index=df.index)
    trades = run_backtest(df, entry_signal, stop_r_price=3.0, direction=1)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["entry_price"] == 101.0
    assert t["exit_reason"] == "stop"
    assert abs(t["pnl_r"] - (-1.0)) < 1e-9
    print("PASS: test_long_trade_hits_stop")


def test_long_trade_hits_take_profit():
    rows = [
        [100, 101, 99, 100],
        [101, 102, 100, 102],   # entry @ 101
        [102, 108, 101, 107],   # high=108 -> +2R target = 101+2*3=107 touched
        [107, 109, 106, 108],
    ]
    df = _df(rows)
    entry_signal = pd.Series([True, False, False, False], index=df.index)
    cfg = BacktestConfig(take_profit_r=2.0)
    trades = run_backtest(df, entry_signal, stop_r_price=3.0, direction=1, cfg=cfg)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "take_profit"
    assert abs(t["pnl_r"] - 2.0) < 1e-9
    print("PASS: test_long_trade_hits_take_profit")


def test_short_trade_mirrors_long():
    rows = [
        [100, 101, 99, 100],
        [99, 100, 98, 99],     # entry @ 99 (short)
        [99, 103, 98, 102],    # high=103 breaches short stop = 99+3=102
        [102, 104, 101, 103],
    ]
    df = _df(rows)
    entry_signal = pd.Series([True, False, False, False], index=df.index)
    trades = run_backtest(df, entry_signal, stop_r_price=3.0, direction=-1)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["entry_price"] == 99.0
    assert t["exit_reason"] == "stop"
    assert abs(t["pnl_r"] - (-1.0)) < 1e-9
    print("PASS: test_short_trade_mirrors_long")


def test_overlapping_signals_are_skipped_by_default():
    rows = [
        [100, 101, 99, 100],
        [101, 102, 100, 101],  # entry @ 101 from bar-0 signal
        [101, 102, 100, 101],  # a second signal here should be IGNORED (still in trade)
        [101, 102, 100, 101],
        [101, 102, 100, 101],
        [101, 150, 100, 149],  # eventually a big favorable move, way past any reasonable stop
    ]
    df = _df(rows)
    entry_signal = pd.Series([True, False, True, False, False, False], index=df.index)
    trades = run_backtest(df, entry_signal, stop_r_price=3.0, direction=1,
                           cfg=BacktestConfig(max_holding_bars=10))
    assert len(trades) == 1, "the second signal while already in a trade must be skipped"
    print("PASS: test_overlapping_signals_are_skipped_by_default")


def test_signal_exit_fires_on_condition_not_price():
    rows = [
        [100, 101, 99, 100],
        [101, 102, 100, 101],   # entry @ 101, bar index 1
        [101, 102, 100, 101],   # bar 2: exit_signal_long True here -- exits at close=101
        [101, 300, 100, 299],   # should never be reached
    ]
    df = _df(rows)
    entry_signal = pd.Series([True, False, False, False], index=df.index)
    exit_sig = pd.Series([False, False, True, False], index=df.index)
    trades = run_backtest(df, entry_signal, stop_r_price=50.0, direction=1,
                           exit_signal_long=exit_sig)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "signal_exit"
    assert t["exit_price"] == 101  # exits at bar 2's close, not price-driven
    print("PASS: test_signal_exit_fires_on_condition_not_price")


def test_signal_exit_never_fires_on_the_entry_bar_itself():
    rows = [
        [100, 101, 99, 100],
        [101, 102, 100, 101],   # entry @ 101, bar index 1 -- exit_signal is True HERE too
        [101, 103, 100, 102],   # but must not exit until this bar at the earliest
    ]
    df = _df(rows)
    entry_signal = pd.Series([True, False, False], index=df.index)
    exit_sig = pd.Series([False, True, True], index=df.index)
    trades = run_backtest(df, entry_signal, stop_r_price=50.0, direction=1,
                           exit_signal_long=exit_sig)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "signal_exit"
    assert t["bars_held"] == 1, "must not exit on the same bar as entry, even if the signal is already true there"
    print("PASS: test_signal_exit_never_fires_on_the_entry_bar_itself")


def test_signal_exit_stop_still_wins_same_bar_tie():
    rows = [
        [100, 101, 99, 100],
        [101, 102, 100, 101],   # entry @ 101, stop = 101-3=98
        [101, 102, 95, 96],     # low=95 breaches stop; exit_signal also true here -- stop wins
    ]
    df = _df(rows)
    entry_signal = pd.Series([True, False, False], index=df.index)
    exit_sig = pd.Series([False, False, True], index=df.index)
    trades = run_backtest(df, entry_signal, stop_r_price=3.0, direction=1,
                           exit_signal_long=exit_sig)
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "stop"
    print("PASS: test_signal_exit_stop_still_wins_same_bar_tie")


def test_dynamic_exit_long_uses_per_bar_moving_level():
    rows = [
        [100, 101, 99, 100],
        [101, 102, 100, 101],   # entry @ 101
        [101, 105, 100, 104],   # dyn level 106 here -- not touched (high=105)
        [104, 107, 103, 106],   # dyn level drops to 106.5 -- touched (high=107 >= 106.5)
        [106, 110, 105, 109],   # should never be reached
    ]
    df = _df(rows)
    entry_signal = pd.Series([True, False, False, False, False], index=df.index)
    dyn_exit = pd.Series([float("nan"), float("nan"), 106.0, 106.5, 105.0], index=df.index)
    trades = run_backtest(df, entry_signal, stop_r_price=10.0, direction=1,
                           dynamic_exit_long=dyn_exit)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "channel_exit"
    assert t["exit_price"] == 106.5
    print("PASS: test_dynamic_exit_long_uses_per_bar_moving_level")


def test_dynamic_exit_stop_still_wins_same_bar_tie():
    rows = [
        [100, 101, 99, 100],
        [101, 102, 100, 101],   # entry @ 101, stop = 101-3=98
        [101, 106, 95, 96],     # both stop (low=95<=98) and dyn (high=106>=105) touched -- stop wins
    ]
    df = _df(rows)
    entry_signal = pd.Series([True, False, False], index=df.index)
    dyn_exit = pd.Series([float("nan"), float("nan"), 105.0], index=df.index)
    trades = run_backtest(df, entry_signal, stop_r_price=3.0, direction=1,
                           dynamic_exit_long=dyn_exit)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "stop"
    print("PASS: test_dynamic_exit_stop_still_wins_same_bar_tie")


def test_end_of_data_marks_trade_as_still_open():
    rows = [
        [100, 101, 99, 100],
        [101, 102, 100, 101],  # entry, never hits stop or takes profit before data ends
        [101, 102, 100, 101],
    ]
    df = _df(rows)
    entry_signal = pd.Series([True, False, False], index=df.index)
    trades = run_backtest(df, entry_signal, stop_r_price=50.0, direction=1)  # huge stop, never touched
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "end_of_data"
    assert t["is_closed"] == False
    print("PASS: test_end_of_data_marks_trade_as_still_open")


if __name__ == "__main__":
    test_long_trade_hits_stop()
    test_long_trade_hits_take_profit()
    test_short_trade_mirrors_long()
    test_overlapping_signals_are_skipped_by_default()
    test_signal_exit_fires_on_condition_not_price()
    test_signal_exit_never_fires_on_the_entry_bar_itself()
    test_signal_exit_stop_still_wins_same_bar_tie()
    test_dynamic_exit_long_uses_per_bar_moving_level()
    test_dynamic_exit_stop_still_wins_same_bar_tie()
    test_end_of_data_marks_trade_as_still_open()
    print("\nAll backtest_engine tests passed.")
