"""
Offline tests for strategy_btc_only.py -- fabricated OHLCV, no network.
Focused on the parts unique to this strategy's composition (the indicator
math itself is covered in test_indicators.py): the BUY/SELL -> SHORT/LONG
BTC direction flip, the regime gate, and the exit-signal direction mapping.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "single_leg_adx"))

import numpy as np
import pandas as pd

from indicators import adx_wilder, rolling_beta, rolling_percentile_rank
from strategy_btc_only import StatArbBtcParams, build_signals


def _idx(n, freq="15min"):
    return pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")


def _fabricate(n=6000, seed=11):
    rng = np.random.default_rng(seed)
    idx = _idx(n)
    btc_walk = 60_000 + np.cumsum(rng.normal(0, 30, n))
    bnb_walk = 400 + 0.003 * (btc_walk - 60_000) + np.cumsum(rng.normal(0, 0.5, n))

    def _ohlc(walk, noise_scale):
        high = walk + np.abs(rng.normal(noise_scale, noise_scale / 2, n))
        low = walk - np.abs(rng.normal(noise_scale, noise_scale / 2, n))
        close = walk + rng.normal(0, noise_scale / 4, n)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    return _ohlc(btc_walk, 20), _ohlc(bnb_walk, 0.3)


def test_build_signals_output_shape_and_types():
    df_btc, df_bnb = _fabricate()
    params = StatArbBtcParams(beta_window=50, pct_lookback=200)
    sig = build_signals(df_btc, df_bnb, params)
    expected_idx = df_btc.index.intersection(df_bnb.index)
    assert sig.entry_signal.index.equals(expected_idx)
    assert sig.entry_signal.dtype == bool
    assert sig.exit_signal_long.dtype == bool
    assert sig.exit_signal_short.dtype == bool
    assert set(sig.direction.unique()) <= {-1, 0, 1}
    assert (sig.stop_r_price.dropna() > 0).all()
    print("PASS: test_build_signals_output_shape_and_types")


def test_entry_signal_and_direction_are_consistent():
    """Wherever direction is nonzero, entry_signal must be True, and vice
    versa -- the two are derived from the same underlying masks and must
    never disagree."""
    df_btc, df_bnb = _fabricate()
    params = StatArbBtcParams(beta_window=50, pct_lookback=200)
    sig = build_signals(df_btc, df_bnb, params)
    assert (sig.entry_signal == (sig.direction != 0)).all()
    print("PASS: test_entry_signal_and_direction_are_consistent")


def test_direction_flip_convention():
    """Pairs 'BUY' (spread cheap, pct < low_q -> long BNB/short BTC in the
    underlying pairs construction) must map to direction=-1 (SHORT btc)
    here, per the explicit direction-flip convention that isolates the BTC
    leg. Pairs 'SELL' (pct > high_q) must map to direction=+1 (LONG btc).
    Verified against the actual computed pct/regime series, not just
    entry_signal, so this test would fail if the sign got flipped by
    accident."""
    df_btc, df_bnb = _fabricate()
    params = StatArbBtcParams(beta_window=50, pct_lookback=200)
    sig = build_signals(df_btc, df_bnb, params)

    common_idx = df_btc.index.intersection(df_bnb.index)
    btc, bnb_close = df_btc.loc[common_idx], df_bnb.loc[common_idx, "close"]
    beta = rolling_beta(bnb_close, btc["close"], params.beta_window).ffill()
    spread = bnb_close - beta * btc["close"]
    pct = rolling_percentile_rank(spread, params.pct_lookback)
    adx = adx_wilder(btc["high"], btc["low"], btc["close"], params.adx_window)
    regime_ok = adx <= params.adx_threshold

    shorts = sig.direction == -1
    longs = sig.direction == 1
    assert shorts.any() and longs.any(), "fixture should produce both short and long entries"
    assert (pct[shorts] < params.low_q).all()
    assert regime_ok[shorts].all()
    assert (pct[longs] > params.high_q).all()
    assert regime_ok[longs].all()
    print("PASS: test_direction_flip_convention")


def test_regime_filter_blocks_entries_when_threshold_impossible():
    """No entries at all when the ADX threshold can never be satisfied
    (ADX is always >= 0, so a negative threshold excludes every bar)."""
    df_btc, df_bnb = _fabricate()
    params = StatArbBtcParams(beta_window=50, pct_lookback=200, adx_threshold=-1.0)
    sig = build_signals(df_btc, df_bnb, params)
    assert not sig.entry_signal.any()
    print("PASS: test_regime_filter_blocks_entries_when_threshold_impossible")


def test_exit_signal_direction_mapping():
    """exit_signal_long (closes a LONG btc / pairs-SELL trade) must equal
    pct <= mid_q; exit_signal_short (closes a SHORT btc / pairs-BUY trade)
    must equal pct >= mid_q -- exact equality, not just 'roughly'."""
    df_btc, df_bnb = _fabricate()
    params = StatArbBtcParams(beta_window=50, pct_lookback=200)
    sig = build_signals(df_btc, df_bnb, params)

    common_idx = df_btc.index.intersection(df_bnb.index)
    btc, bnb_close = df_btc.loc[common_idx], df_bnb.loc[common_idx, "close"]
    beta = rolling_beta(bnb_close, btc["close"], params.beta_window).ffill()
    spread = bnb_close - beta * btc["close"]
    pct = rolling_percentile_rank(spread, params.pct_lookback)

    assert (sig.exit_signal_long == (pct <= params.mid_q).fillna(False)).all()
    assert (sig.exit_signal_short == (pct >= params.mid_q).fillna(False)).all()
    print("PASS: test_exit_signal_direction_mapping")


def test_stop_mode_fixed_pct_scales_with_price():
    df_btc, df_bnb = _fabricate()
    params = StatArbBtcParams(beta_window=50, pct_lookback=200,
                               stop_mode="fixed_pct", fixed_stop_pct=0.01)
    sig = build_signals(df_btc, df_bnb, params)
    common_idx = df_btc.index.intersection(df_bnb.index)
    close = df_btc.loc[common_idx, "close"]
    assert np.allclose(sig.stop_r_price.to_numpy(), (close * 0.01).to_numpy())
    print("PASS: test_stop_mode_fixed_pct_scales_with_price")


if __name__ == "__main__":
    test_build_signals_output_shape_and_types()
    test_entry_signal_and_direction_are_consistent()
    test_direction_flip_convention()
    test_regime_filter_blocks_entries_when_threshold_impossible()
    test_exit_signal_direction_mapping()
    test_stop_mode_fixed_pct_scales_with_price()
    print("\nAll strategy_btc_only tests passed.")
