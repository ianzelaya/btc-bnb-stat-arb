"""Offline tests for costs.py -- fabricated trades, no network."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from costs import (BINANCE_BASE_COST_PCT, BINANCE_VIP3_BASE_COST_PCT, IBKR_ROUNDTRIP_USD_PER_BTC,
                    compute_cost_r, compute_fee_usd, compute_funding_usd)


def _trades():
    return pd.DataFrame({
        "entry_price": [50000.0, 80000.0],
        "r_price": [200.0, 400.0],
    })


def test_ibkr_cost_independent_of_price():
    trades = _trades()
    cost_r = compute_cost_r(trades, cost_basis="ibkr")
    # IBKR cost_r = ROUNDTRIP_USD_PER_BTC / r_price -- doesn't touch entry_price
    assert abs(cost_r.iloc[0] - IBKR_ROUNDTRIP_USD_PER_BTC / 200.0) < 1e-9
    assert abs(cost_r.iloc[1] - IBKR_ROUNDTRIP_USD_PER_BTC / 400.0) < 1e-9
    print("PASS: test_ibkr_cost_independent_of_price")


def test_binance_cost_scales_with_price():
    trades = _trades()
    cost_r = compute_cost_r(trades, cost_basis="binance")
    assert abs(cost_r.iloc[0] - BINANCE_BASE_COST_PCT * 50000.0 / 200.0) < 1e-9
    assert abs(cost_r.iloc[1] - BINANCE_BASE_COST_PCT * 80000.0 / 400.0) < 1e-9
    print("PASS: test_binance_cost_scales_with_price")


def test_multiplier_scales_linearly():
    trades = _trades()
    base = compute_cost_r(trades, cost_basis="ibkr", multiplier=1.0)
    doubled = compute_cost_r(trades, cost_basis="ibkr", multiplier=2.0)
    assert (abs(doubled - 2 * base) < 1e-9).all()
    print("PASS: test_multiplier_scales_linearly")


def test_unknown_basis_raises():
    trades = _trades()
    try:
        compute_cost_r(trades, cost_basis="coinbase")
        assert False, "should have raised on an unknown cost basis"
    except ValueError:
        pass
    print("PASS: test_unknown_basis_raises")


def test_vip3_cost_cheaper_than_retail():
    trades = _trades()
    retail = compute_cost_r(trades, cost_basis="binance")
    vip3 = compute_cost_r(trades, cost_basis="binance_vip3")
    assert (vip3 < retail).all(), "VIP3 tier must be cheaper than the retail tier"
    assert abs(vip3.iloc[0] - BINANCE_VIP3_BASE_COST_PCT * 50000.0 / 200.0) < 1e-9
    print("PASS: test_vip3_cost_cheaper_than_retail")


def test_compute_fee_usd_scales_with_notional():
    fee = compute_fee_usd(notional_usd=100_000.0)
    assert abs(fee - BINANCE_VIP3_BASE_COST_PCT * 100_000.0) < 1e-9
    assert abs(compute_fee_usd(100_000.0, multiplier=2.0) - 2 * fee) < 1e-9
    print("PASS: test_compute_fee_usd_scales_with_notional")


def test_compute_funding_usd_long_pays_positive_rate():
    idx = pd.date_range("2026-01-01", periods=5, freq="8h", tz="UTC")
    funding_df = pd.DataFrame({"funding_rate": [0.0001, 0.0002, -0.0001, 0.0, 0.00005]}, index=idx)
    price_series = pd.Series([100.0] * 5, index=idx)
    # Position open from just before idx[0] to just after idx[2] -> covers events at idx[0..2]
    entry_time = idx[0] - pd.Timedelta(minutes=1)
    exit_time = idx[2] + pd.Timedelta(minutes=1)
    funding_usd = compute_funding_usd(entry_time, exit_time, qty=10.0, direction=1,
                                       price_series=price_series, funding_df=funding_df)
    # long pays when rate > 0: -1 * 10*100*0.0001 + -1*10*100*0.0002 + -1*10*100*(-0.0001)
    expected = -(10 * 100 * 0.0001) - (10 * 100 * 0.0002) - (10 * 100 * -0.0001)
    assert abs(funding_usd - expected) < 1e-9
    print("PASS: test_compute_funding_usd_long_pays_positive_rate")


def test_compute_funding_usd_short_is_sign_flipped():
    idx = pd.date_range("2026-01-01", periods=2, freq="8h", tz="UTC")
    funding_df = pd.DataFrame({"funding_rate": [0.0001, 0.0002]}, index=idx)
    price_series = pd.Series([100.0, 100.0], index=idx)
    entry_time = idx[0] - pd.Timedelta(minutes=1)
    exit_time = idx[1] + pd.Timedelta(minutes=1)
    long_usd = compute_funding_usd(entry_time, exit_time, qty=10.0, direction=1,
                                    price_series=price_series, funding_df=funding_df)
    short_usd = compute_funding_usd(entry_time, exit_time, qty=10.0, direction=-1,
                                     price_series=price_series, funding_df=funding_df)
    assert abs(long_usd + short_usd) < 1e-9, "long and short funding P&L must be exact opposites"
    print("PASS: test_compute_funding_usd_short_is_sign_flipped")


def test_compute_funding_usd_excludes_events_outside_window():
    idx = pd.date_range("2026-01-01", periods=3, freq="8h", tz="UTC")
    funding_df = pd.DataFrame({"funding_rate": [0.0001, 0.0001, 0.0001]}, index=idx)
    price_series = pd.Series([100.0, 100.0, 100.0], index=idx)
    # Entry AT idx[0] (excluded, must be strictly after) to exit AT idx[1] (included).
    funding_usd = compute_funding_usd(idx[0], idx[1], qty=1.0, direction=1,
                                       price_series=price_series, funding_df=funding_df)
    expected = -(1 * 100 * 0.0001)  # only idx[1]'s event counts
    assert abs(funding_usd - expected) < 1e-9
    print("PASS: test_compute_funding_usd_excludes_events_outside_window")


def test_compute_funding_usd_no_events_returns_zero():
    idx = pd.date_range("2026-01-01", periods=2, freq="8h", tz="UTC")
    funding_df = pd.DataFrame({"funding_rate": [0.0001, 0.0001]}, index=idx)
    price_series = pd.Series([100.0, 100.0], index=idx)
    funding_usd = compute_funding_usd(idx[0] - pd.Timedelta(minutes=10), idx[0] - pd.Timedelta(minutes=5),
                                       qty=1.0, direction=1, price_series=price_series, funding_df=funding_df)
    assert funding_usd == 0.0
    print("PASS: test_compute_funding_usd_no_events_returns_zero")


if __name__ == "__main__":
    test_ibkr_cost_independent_of_price()
    test_binance_cost_scales_with_price()
    test_multiplier_scales_linearly()
    test_unknown_basis_raises()
    test_vip3_cost_cheaper_than_retail()
    test_compute_fee_usd_scales_with_notional()
    test_compute_funding_usd_long_pays_positive_rate()
    test_compute_funding_usd_short_is_sign_flipped()
    test_compute_funding_usd_excludes_events_outside_window()
    test_compute_funding_usd_no_events_returns_zero()
    print("\nAll costs tests passed.")
