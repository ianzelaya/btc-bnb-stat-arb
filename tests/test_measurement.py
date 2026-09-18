"""
Offline tests for measurement.py -- fabricated trade logs, no network, no
real strategy. Verifies the honest-measurement primitives are correct
before real Part B results get computed through them.

Run: python -m pytest part_b/tests/ -v   (from part_b/, with part_b/ on PYTHONPATH)
  or: python tests/test_measurement.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from measurement import (apply_cost, calmar_ratio, cost_sensitivity, equity_curve_from_trades,
                          mae_mfe_distribution, max_drawdown, sharpe_from_equity_curve,
                          train_test_split_by_date, trade_stats, walk_forward_correlation)


def _fixed_trades():
    """4 known trades: +2R, +1R, -1R, -1R -> hit_rate=0.5, easy to hand-check."""
    times = pd.date_range("2026-01-01", periods=4, freq="10D", tz="UTC")
    return pd.DataFrame({
        "entry_time": times, "exit_time": times + pd.Timedelta(days=1),
        "pnl_r": [2.0, 1.0, -1.0, -1.0],
        "mae_r": [0.2, 0.3, 0.9, 0.95],
        "mfe_r": [2.1, 1.2, 0.3, 0.1],
        "is_closed": True,
    })


def test_trade_stats_matches_hand_calc():
    stats = trade_stats(_fixed_trades())
    assert stats.n_trades == 4
    assert abs(stats.hit_rate - 0.5) < 1e-9
    assert abs(stats.avg_win_r - 1.5) < 1e-9    # (2+1)/2
    assert abs(stats.avg_loss_r - (-1.0)) < 1e-9
    assert abs(stats.payoff_ratio - 1.5) < 1e-9
    expected_expectancy = 0.5 * 1.5 + 0.5 * (-1.0)  # = 0.25
    assert abs(stats.expectancy_r - expected_expectancy) < 1e-9
    print("PASS: test_trade_stats_matches_hand_calc")


def test_trade_stats_empty_trades_does_not_crash():
    empty = pd.DataFrame(columns=["pnl_r", "is_closed"])
    stats = trade_stats(empty)
    assert stats.n_trades == 0
    print("PASS: test_trade_stats_empty_trades_does_not_crash")


def test_cost_sensitivity_monotonically_worsens_expectancy():
    trades = _fixed_trades()
    table = cost_sensitivity(trades, base_cost_r=0.1)
    zero = table.loc[table.scenario == "zero_cost", "expectancy_r"].iloc[0]
    base = table.loc[table.scenario == "base_cost", "expectancy_r"].iloc[0]
    double = table.loc[table.scenario == "2x_cost", "expectancy_r"].iloc[0]
    assert zero > base > double, "expectancy must strictly worsen as cost increases"
    assert abs((zero - base) - 0.1) < 1e-9  # cost is a flat per-trade R subtraction
    print("PASS: test_cost_sensitivity_monotonically_worsens_expectancy")


def test_apply_cost_does_not_mutate_input():
    trades = _fixed_trades()
    original = trades["pnl_r"].copy()
    _ = apply_cost(trades, cost_r=0.5)
    assert (trades["pnl_r"] == original).all(), "apply_cost must not mutate its input"
    print("PASS: test_apply_cost_does_not_mutate_input")


def test_equity_curve_and_drawdown_on_known_path():
    # Equity path (starting 100k, $500/R): +2R=+1000 -> 101000, +1R=+500 -> 101500,
    # -1R=-500 -> 101000, -1R=-500 -> 100500. Peak=101500, trough=100500 -> DD=$1000/101500.
    trades = _fixed_trades()
    equity = equity_curve_from_trades(trades, risk_per_trade_usd=500, starting_equity=100_000)
    assert list(equity.round(0)) == [101000, 101500, 101000, 100500]
    dd = max_drawdown(equity)
    assert abs(dd.max_drawdown_usd - 1000) < 1e-6
    assert abs(dd.max_drawdown_pct - (1000 / 101500)) < 1e-9
    print("PASS: test_equity_curve_and_drawdown_on_known_path")


def test_calmar_ratio_is_inf_with_zero_drawdown():
    # Strictly increasing equity, exactly 1 year apart -> CAGR=21%, zero
    # drawdown -> Calmar is undefined-on-the-upside, represented as +inf.
    equity = pd.Series([100_000.0, 121_000.0],
                        index=pd.to_datetime(["2026-01-01", "2027-01-01"], utc=True))
    assert calmar_ratio(equity) == float("inf")
    print("PASS: test_calmar_ratio_is_inf_with_zero_drawdown")


def test_calmar_ratio_is_negative_for_a_net_losing_path():
    trades = _fixed_trades()
    equity = equity_curve_from_trades(trades, risk_per_trade_usd=500, starting_equity=100_000)
    calmar = calmar_ratio(equity)
    assert calmar < 0, "equity ends lower than it starts in this fixture -- Calmar must be negative"
    print("PASS: test_calmar_ratio_is_negative_for_a_net_losing_path")


def test_sharpe_is_nan_for_zero_variance():
    equity = pd.Series([100_000.0] * 10,
                        index=pd.date_range("2026-01-01", periods=10, freq="1D", tz="UTC"))
    sharpe = sharpe_from_equity_curve(equity, periods_per_year=365)
    assert np.isnan(sharpe), "flat equity (no variance) must not divide by zero silently"
    print("PASS: test_sharpe_is_nan_for_zero_variance")


def test_train_test_split_is_time_based_not_random():
    trades = _fixed_trades()
    in_s, out_s = train_test_split_by_date(trades, "2026-01-15")
    assert len(in_s) == 2 and len(out_s) == 2
    assert in_s["entry_time"].max() < out_s["entry_time"].min(), \
        "in-sample must be strictly earlier than out-of-sample, never interleaved"
    print("PASS: test_train_test_split_is_time_based_not_random")


def test_walk_forward_correlation_handles_too_few_folds_gracefully():
    trades = _fixed_trades()  # only 4 trades -- asking for 6 folds should degrade gracefully
    result = walk_forward_correlation(trades, n_folds=6)
    assert "note" in result or not np.isnan(result["lag1_correlation"])
    print("PASS: test_walk_forward_correlation_handles_too_few_folds_gracefully")


def test_mae_mfe_distribution_separates_winners_and_losers():
    trades = _fixed_trades()
    dist = mae_mfe_distribution(trades)
    winners_mae_mean = dist.query("group=='winners' and metric=='mae_r'")["mean"].iloc[0]
    losers_mae_mean = dist.query("group=='losers' and metric=='mae_r'")["mean"].iloc[0]
    assert winners_mae_mean < losers_mae_mean, \
        "in this fixture winners have much smaller MAE than losers by construction"
    print("PASS: test_mae_mfe_distribution_separates_winners_and_losers")


if __name__ == "__main__":
    test_trade_stats_matches_hand_calc()
    test_trade_stats_empty_trades_does_not_crash()
    test_cost_sensitivity_monotonically_worsens_expectancy()
    test_apply_cost_does_not_mutate_input()
    test_equity_curve_and_drawdown_on_known_path()
    test_calmar_ratio_is_inf_with_zero_drawdown()
    test_calmar_ratio_is_negative_for_a_net_losing_path()
    test_sharpe_is_nan_for_zero_variance()
    test_train_test_split_is_time_based_not_random()
    test_walk_forward_correlation_handles_too_few_folds_gracefully()
    test_mae_mfe_distribution_separates_winners_and_losers()
    print("\nAll measurement tests passed.")
