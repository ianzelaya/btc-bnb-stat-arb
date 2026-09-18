"""
Offline tests for indicators.py -- fabricated data, no network.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from indicators import (
    adx_wilder, atr_wilder, donchian_channel, rolling_beta,
    rolling_percentile_rank, rsi_wilder,
)


def _idx(n, freq="15min"):
    return pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")


def test_donchian_channel_matches_hand_calc():
    high = pd.Series([10, 12, 11, 15, 9, 8], index=_idx(6))
    low = pd.Series([8, 9, 10, 11, 7, 6], index=_idx(6))
    lower, upper = donchian_channel(high, low, lower_length=3, upper_length=3)
    assert upper.iloc[2] == 12
    assert upper.iloc[3] == 15
    assert lower.iloc[2] == 8
    assert lower.iloc[3] == 9
    assert pd.isna(upper.iloc[1]) and pd.isna(lower.iloc[1])
    print("PASS: test_donchian_channel_matches_hand_calc")


def test_atr_wilder_matches_hand_recursion():
    high = pd.Series([10.0, 11.0, 12.0, 11.5, 13.0], index=_idx(5))
    low = pd.Series([9.0, 10.0, 10.5, 10.0, 11.0], index=_idx(5))
    close = pd.Series([9.5, 10.5, 11.5, 10.5, 12.5], index=_idx(5))
    window = 3
    atr = atr_wilder(high, low, close, window=window)

    tr = [1.0]
    tr.append(max(11 - 10, abs(11 - 9.5), abs(10 - 9.5)))
    tr.append(max(12 - 10.5, abs(12 - 10.5), abs(10.5 - 10.5)))
    tr.append(max(11.5 - 10, abs(11.5 - 11.5), abs(10 - 11.5)))
    tr.append(max(13 - 11, abs(13 - 10.5), abs(11 - 10.5)))
    alpha = 1 / window
    y = tr[0]
    expected = [y]
    for t in tr[1:]:
        y = alpha * t + (1 - alpha) * y
        expected.append(y)

    assert pd.isna(atr.iloc[0]) and pd.isna(atr.iloc[1])
    assert abs(atr.iloc[2] - expected[2]) < 1e-9
    assert abs(atr.iloc[3] - expected[3]) < 1e-9
    assert abs(atr.iloc[4] - expected[4]) < 1e-9
    print("PASS: test_atr_wilder_matches_hand_recursion")


def test_rsi_wilder_bounds_and_direction():
    n = 40
    rising = pd.Series(np.linspace(100, 200, n), index=_idx(n))
    falling = pd.Series(np.linspace(200, 100, n), index=_idx(n))
    rsi_up = rsi_wilder(rising, window=14)
    rsi_down = rsi_wilder(falling, window=14)
    assert (rsi_up.dropna() >= 0).all() and (rsi_up.dropna() <= 100).all()
    assert rsi_up.iloc[-1] > 90
    assert rsi_down.iloc[-1] < 10
    print("PASS: test_rsi_wilder_bounds_and_direction")


def test_adx_wilder_bounds():
    n = 200
    rng = np.random.default_rng(9)
    walk = 100 + np.cumsum(rng.normal(0, 1, n))
    high = pd.Series(walk + np.abs(rng.normal(0.5, 0.3, n)), index=_idx(n))
    low = pd.Series(walk - np.abs(rng.normal(0.5, 0.3, n)), index=_idx(n))
    close = pd.Series(walk, index=_idx(n))
    adx = adx_wilder(high, low, close, window=14)
    valid = adx.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()
    print("PASS: test_adx_wilder_bounds")


def test_adx_wilder_high_for_persistent_trend_low_for_chop():
    n = 300
    trending = pd.Series(np.linspace(100, 300, n), index=_idx(n))
    adx_trend = adx_wilder(trending + 1, trending - 1, trending, window=14)

    rng = np.random.default_rng(11)
    choppy = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)) - np.cumsum(rng.normal(0, 1, n)), index=_idx(n))
    # A pure zig-zag (alternating up/down every bar) never sustains one-sided
    # directional movement -- the case ADX should read as "no trend."
    zigzag = pd.Series(100 + np.tile([1.0, -1.0], n // 2).cumsum(), index=_idx(n))
    adx_chop = adx_wilder(zigzag + 0.5, zigzag - 0.5, zigzag, window=14)

    assert adx_trend.iloc[-1] > adx_chop.iloc[-1]
    assert adx_trend.iloc[-1] > 50   # a clean straight-line trend should read as strongly trending
    assert adx_chop.iloc[-1] < 20    # a pure zig-zag should read as close to no trend
    print("PASS: test_adx_wilder_high_for_persistent_trend_low_for_chop")


def test_rolling_percentile_rank_bounds_and_extremes():
    n = 300
    rng = np.random.default_rng(4)
    s = pd.Series(rng.normal(0, 1, n), index=_idx(n))
    rank = rolling_percentile_rank(s, lookback=100)
    valid = rank.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()
    # force a known max at the end of a window
    s2 = s.copy()
    s2.iloc[250] = s2.iloc[150:250].max() + 100  # unambiguous new high
    rank2 = rolling_percentile_rank(s2, lookback=100)
    assert rank2.iloc[250] == 1.0
    print("PASS: test_rolling_percentile_rank_bounds_and_extremes")


def test_rolling_beta_matches_known_linear_relationship():
    n = 200
    x = pd.Series(np.linspace(1, 100, n), index=_idx(n))
    y = 2.5 * x + 10  # exact linear relationship, beta should be ~2.5
    beta = rolling_beta(y, x, window=30)
    assert abs(beta.iloc[-1] - 2.5) < 1e-6
    print("PASS: test_rolling_beta_matches_known_linear_relationship")


if __name__ == "__main__":
    test_donchian_channel_matches_hand_calc()
    test_atr_wilder_matches_hand_recursion()
    test_rsi_wilder_bounds_and_direction()
    test_adx_wilder_bounds()
    test_adx_wilder_high_for_persistent_trend_low_for_chop()
    test_rolling_percentile_rank_bounds_and_extremes()
    test_rolling_beta_matches_known_linear_relationship()
    print("\nAll indicators tests passed.")
