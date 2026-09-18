"""
Offline tests for data_fetch.py's pure/parsing logic. The actual HTTP call
(fetch_klines) needs network access to api.binance.com, which this sandbox's
egress policy blocks (same category of limitation as testnet.binancefuture.com
in Part A -- see NOTES.md) -- so these tests exercise everything that doesn't
need the network: interval parsing and the raw-rows -> DataFrame transform,
using fabricated kline rows shaped exactly like Binance's real response.
Run this for real (fetch_klines itself) from your own machine once you're
ready to pull actual history.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetch import _funding_rows_to_df, _interval_to_ms, _klines_to_df, _market_urls


def test_interval_to_ms():
    assert _interval_to_ms("1m") == 60_000
    assert _interval_to_ms("15m") == 15 * 60_000
    assert _interval_to_ms("1h") == 3_600_000
    assert _interval_to_ms("4h") == 4 * 3_600_000
    assert _interval_to_ms("1d") == 86_400_000
    print("PASS: test_interval_to_ms")


def test_interval_to_ms_rejects_unknown_unit():
    try:
        _interval_to_ms("1x")
        assert False, "should have raised on an unknown interval unit"
    except ValueError:
        pass
    print("PASS: test_interval_to_ms_rejects_unknown_unit")


def _fake_kline_row(open_time_ms, o, h, l, c, v):
    # Shaped exactly like Binance's real /api/v3/klines response rows.
    return [open_time_ms, str(o), str(h), str(l), str(c), str(v),
            open_time_ms + 3_600_000 - 1, "0", 10, "0", "0", "0"]


def test_klines_to_df_parses_and_types_correctly():
    rows = [
        _fake_kline_row(1_700_000_000_000, 100.0, 101.5, 99.0, 100.5, 12.3),
        _fake_kline_row(1_700_003_600_000, 100.5, 102.0, 100.0, 101.0, 8.1),
    ]
    df = _klines_to_df(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "n_trades"]
    assert df["open"].dtype == float
    assert df["n_trades"].dtype == int
    assert df.index.is_monotonic_increasing
    assert df.iloc[0]["close"] == 100.5
    print("PASS: test_klines_to_df_parses_and_types_correctly")


def test_klines_to_df_dedupes_overlapping_pagination():
    # Pagination can re-request the boundary candle -- must not double-count it.
    row = _fake_kline_row(1_700_000_000_000, 100.0, 101.0, 99.0, 100.5, 5.0)
    df = _klines_to_df([row, row])
    assert len(df) == 1, "duplicate open_time rows (pagination overlap) must be deduped"
    print("PASS: test_klines_to_df_dedupes_overlapping_pagination")


def test_market_urls_selects_correct_host():
    spot_base, spot_ep = _market_urls("spot")
    fut_base, fut_ep = _market_urls("futures")
    assert spot_base == "https://api.binance.com" and spot_ep == "/api/v3/klines"
    assert fut_base == "https://fapi.binance.com" and fut_ep == "/fapi/v1/klines"
    print("PASS: test_market_urls_selects_correct_host")


def test_market_urls_rejects_unknown_market():
    try:
        _market_urls("dexes")
        assert False, "should have raised on an unknown market"
    except ValueError:
        pass
    print("PASS: test_market_urls_rejects_unknown_market")


def _fake_funding_row(funding_time_ms, rate):
    # Shaped like a real /fapi/v1/fundingRate response row.
    return {"symbol": "BTCUSDT", "fundingTime": funding_time_ms, "fundingRate": str(rate), "markPrice": "0"}


def test_funding_rows_to_df_parses_and_types_correctly():
    rows = [
        _fake_funding_row(1_700_000_000_000, 0.0001),
        _fake_funding_row(1_700_028_800_000, -0.00005),
    ]
    df = _funding_rows_to_df(rows)
    assert list(df.columns) == ["funding_rate"]
    assert df["funding_rate"].dtype == float
    assert df.index.is_monotonic_increasing
    assert abs(df.iloc[0]["funding_rate"] - 0.0001) < 1e-12
    print("PASS: test_funding_rows_to_df_parses_and_types_correctly")


def test_funding_rows_to_df_dedupes_overlapping_pagination():
    row = _fake_funding_row(1_700_000_000_000, 0.0001)
    df = _funding_rows_to_df([row, row])
    assert len(df) == 1, "duplicate fundingTime rows (pagination overlap) must be deduped"
    print("PASS: test_funding_rows_to_df_dedupes_overlapping_pagination")


if __name__ == "__main__":
    test_interval_to_ms()
    test_interval_to_ms_rejects_unknown_unit()
    test_klines_to_df_parses_and_types_correctly()
    test_klines_to_df_dedupes_overlapping_pagination()
    test_market_urls_selects_correct_host()
    test_market_urls_rejects_unknown_market()
    test_funding_rows_to_df_parses_and_types_correctly()
    test_funding_rows_to_df_dedupes_overlapping_pagination()
    print("\nAll data_fetch tests passed.")
