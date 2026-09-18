"""
Renders a strategy's cumulative equity curve (full backtest period),
colored by in-sample vs out-of-sample, plus a drawdown subplot. Reads
whatever trade journal a run_backtest_*.py driver already wrote.

Run: python plot_equity_curve.py <journal_path> <output_png> <title> [starting_equity] [risk_per_trade] [pnl_col]

starting_equity/risk_per_trade default to the BTC-only variant's $100k/$500
convention; the pairs variant passes its own $550,000 CAP for both (pnl_r
there is already fraction-of-CAP, net of cost -- see two_leg_backtest.py).
"""
from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import pandas as pd

from costs import ACCOUNT_EQUITY_USD, RISK_PER_TRADE_USD
from measurement import calmar_ratio, max_drawdown, sharpe_from_equity_curve, trade_stats

OOS_SPLIT_DATE = "2025-09-14"


def load_trades(journal_path: str) -> pd.DataFrame:
    rows = []
    with open(journal_path) as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    return df.sort_values("exit_time").reset_index(drop=True)


def main(journal_path: str, output_path: str, title: str, pnl_col: str = "pnl_r_after_cost",
         starting_equity: float = ACCOUNT_EQUITY_USD, risk_per_trade: float = RISK_PER_TRADE_USD):
    trades = load_trades(journal_path)
    split = pd.Timestamp(OOS_SPLIT_DATE, tz="UTC")

    equity = starting_equity + (trades[pnl_col] * risk_per_trade).cumsum()
    equity.index = trades["exit_time"]

    is_mask = trades["exit_time"] < split
    is_equity = equity[is_mask.values]
    oos_equity = equity[~is_mask.values]

    if len(is_equity) and len(oos_equity):
        oos_equity = pd.concat([is_equity.iloc[[-1]], oos_equity])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    start_point = pd.Series([starting_equity], index=[trades["entry_time"].min()])
    is_plot = pd.concat([start_point, is_equity]) if len(is_equity) else start_point

    ax1.plot(is_plot.index, is_plot.values, color="#2563eb", linewidth=1.6, label=f"In-sample (n={is_mask.sum()})")
    if len(oos_equity):
        ax1.plot(oos_equity.index, oos_equity.values, color="#dc2626", linewidth=1.6,
                  label=f"Out-of-sample (n={(~is_mask).sum()})")
    ax1.axvline(split, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.axhline(starting_equity, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel(f"Equity ($, starting ${starting_equity:,.0f})")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    running_peak = equity.cummax()
    dd_pct = (equity - running_peak) / running_peak * 100
    ax2.fill_between(dd_pct.index, dd_pct.values, 0, color="#dc2626", alpha=0.4)
    ax2.axvline(split, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Exit time")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Wrote {output_path}")

    def block(sub_trades, label):
        stats = trade_stats(sub_trades, pnl_col=pnl_col)
        eq = starting_equity + (sub_trades[pnl_col] * risk_per_trade).cumsum()
        eq.index = sub_trades["exit_time"]
        dd = max_drawdown(eq) if len(eq) >= 2 else None
        sharpe = sharpe_from_equity_curve(eq) if len(eq) >= 2 else float("nan")
        calmar = calmar_ratio(eq) if len(eq) >= 2 else float("nan")
        print(f"\n--- {label} ---")
        print(f"  n_trades: {stats.n_trades}  hit_rate: {stats.hit_rate:.3f}  "
              f"payoff: {stats.payoff_ratio:.3f}  expectancy_r: {stats.expectancy_r:.4f}")
        print(f"  sharpe: {sharpe:.3f}  calmar: {calmar:.3f}  "
              f"max_dd: {dd.max_drawdown_pct:.4f}" if dd else "  n/a")

    block(trades, "FULL")
    block(trades[is_mask], "IN-SAMPLE")
    block(trades[~is_mask], "OUT-OF-SAMPLE")


if __name__ == "__main__":
    journal_path = sys.argv[1] if len(sys.argv) > 1 else "journals/btc_only_backtest_journal.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "equity_curve_btc_only.png"
    title = sys.argv[3] if len(sys.argv) > 3 else "BTC-only variant: cumulative equity, IS vs OOS (after cost)"
    starting_equity = float(sys.argv[4]) if len(sys.argv) > 4 else ACCOUNT_EQUITY_USD
    risk_per_trade = float(sys.argv[5]) if len(sys.argv) > 5 else RISK_PER_TRADE_USD
    pnl_col = sys.argv[6] if len(sys.argv) > 6 else "pnl_r_after_cost"
    main(journal_path, output_path, title, pnl_col, starting_equity, risk_per_trade)
