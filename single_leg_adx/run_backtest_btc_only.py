"""
BTC-only variant: fetch real BTCUSDT + BNBUSDT futures data, run the
BTC/BNB stat-arb spread strategy (strategy_btc_only.py, traded on the BTC
leg only) through the backtest engine, apply the honest-measurement
checklist, and write a trade journal + summary.

Run: python run_backtest_btc_only.py (from any directory)
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest_engine import BacktestConfig, run_backtest
from costs import ACCOUNT_EQUITY_USD, COST_BASIS, RISK_PER_TRADE_USD, compute_cost_r
from data_fetch import fetch_klines
from measurement import (apply_cost, calmar_ratio, cost_sensitivity, equity_curve_from_trades,
                          mae_mfe_distribution, max_drawdown, sharpe_from_equity_curve,
                          train_test_split_by_date, trade_stats, walk_forward_correlation)
from strategy_btc_only import DEFAULT_PARAMS, build_signals

SYMBOL_TRADED = "BTCUSDT"
SYMBOL_SIGNAL = "BNBUSDT"
INTERVAL = "15m"
START = "2020-01-01"
END = "2026-09-14"  # explicit, not "today" -- matches the committed data/ cache exactly,
                    # so a clone on any future date still runs instantly, offline
MARKET = "futures"

OOS_MONTHS = 12
JOURNAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "journals",
                             "btc_only_backtest_journal.jsonl")
SUMMARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_only_summary.json")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_btc = fetch_klines(symbol=SYMBOL_TRADED, interval=INTERVAL, start=START, end=END, market=MARKET)
    df_bnb = fetch_klines(symbol=SYMBOL_SIGNAL, interval=INTERVAL, start=START, end=END, market=MARKET)
    return df_btc, df_bnb


def run_full_backtest(df_btc: pd.DataFrame, df_bnb: pd.DataFrame) -> pd.DataFrame:
    sig = build_signals(df_btc, df_bnb, DEFAULT_PARAMS)
    common_idx = df_btc.index.intersection(df_bnb.index)
    btc = df_btc.loc[common_idx]
    cfg = BacktestConfig(entry_fill="next_open", allow_overlapping=False,
                          max_holding_bars=DEFAULT_PARAMS.max_hold_bars)
    trades = run_backtest(
        btc, sig.entry_signal, sig.stop_r_price, sig.direction, cfg=cfg,
        exit_signal_long=sig.exit_signal_long, exit_signal_short=sig.exit_signal_short,
    )
    trades["cost_r"] = compute_cost_r(trades)
    trades = apply_cost(trades, trades["cost_r"], pnl_col="pnl_r")
    return trades


def _stats_block(trades: pd.DataFrame, label: str) -> dict:
    raw = trade_stats(trades, pnl_col="pnl_r")
    after_cost = trade_stats(trades, pnl_col="pnl_r_after_cost")
    equity = equity_curve_from_trades(trades, RISK_PER_TRADE_USD, ACCOUNT_EQUITY_USD,
                                       pnl_col="pnl_r_after_cost")
    dd = max_drawdown(equity) if len(equity) >= 2 else None
    sharpe = sharpe_from_equity_curve(equity) if len(equity) >= 2 else float("nan")
    calmar = calmar_ratio(equity) if len(equity) >= 2 else float("nan")
    return {
        "label": label,
        "n_trades": raw.n_trades,
        "hit_rate_gross": raw.hit_rate,
        "hit_rate_after_cost": after_cost.hit_rate,
        "expectancy_r_gross": raw.expectancy_r,
        "expectancy_r_after_cost": after_cost.expectancy_r,
        "net_return_per_trade_usd_after_cost": after_cost.expectancy_r * RISK_PER_TRADE_USD,
        "payoff_ratio_after_cost": after_cost.payoff_ratio,
        "avg_win_r_after_cost": after_cost.avg_win_r,
        "avg_loss_r_after_cost": after_cost.avg_loss_r,
        "sharpe_365d_after_cost": sharpe,
        "calmar_after_cost": calmar,
        "max_drawdown_pct_after_cost": dd.max_drawdown_pct if dd else float("nan"),
        "max_drawdown_duration_bars_after_cost": dd.duration_bars if dd else float("nan"),
        "ending_equity_usd": equity.iloc[-1] if len(equity) else float("nan"),
    }


def write_journal(trades: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for _, t in trades.iterrows():
            row = {
                "event": "backtest_trade",
                "entry_time": t["entry_time"].isoformat(),
                "exit_time": t["exit_time"].isoformat(),
                "direction": "LONG" if t["direction"] == 1 else "SHORT",
                "entry_price": round(float(t["entry_price"]), 2),
                "exit_price": round(float(t["exit_price"]), 2),
                "r_price": round(float(t["r_price"]), 4),
                "pnl_r": round(float(t["pnl_r"]), 4),
                "pnl_r_after_cost": round(float(t["pnl_r_after_cost"]), 4),
                "mae_r": round(float(t["mae_r"]), 4),
                "mfe_r": round(float(t["mfe_r"]), 4),
                "bars_held": int(t["bars_held"]),
                "exit_reason": t["exit_reason"],
                "is_closed": bool(t["is_closed"]),
            }
            f.write(json.dumps(row) + "\n")


def main():
    print(f"Fetching {SYMBOL_TRADED} (traded) + {SYMBOL_SIGNAL} (signal only) {INTERVAL} {MARKET} klines from {START}...")
    df_btc, df_bnb = load_data()
    print(f"BTC: {len(df_btc)} bars, {df_btc.index[0]} -> {df_btc.index[-1]}")
    print(f"BNB: {len(df_bnb)} bars, {df_bnb.index[0]} -> {df_bnb.index[-1]}")

    trades = run_full_backtest(df_btc, df_bnb)
    print(f"\n{len(trades)} total trades generated. Cost basis: {COST_BASIS}")
    if len(trades) == 0:
        print("No trades -- nothing further to measure.")
        return

    data_end = df_btc.index.intersection(df_bnb.index)[-1]
    split_date = data_end - pd.DateOffset(months=OOS_MONTHS)
    split_date_str = split_date.strftime("%Y-%m-%d")
    is_trades, oos_trades = train_test_split_by_date(trades, split_date_str)
    print(f"In-sample: entries before {split_date.date()} ({len(is_trades)} trades)")
    print(f"Out-of-sample: entries on/after {split_date.date()} ({len(oos_trades)} trades)")

    blocks = [_stats_block(trades, "full"), _stats_block(is_trades, "in_sample"),
              _stats_block(oos_trades, "out_of_sample")]
    for b in blocks:
        print(f"\n--- {b['label']} ---")
        for k, v in b.items():
            if k != "label":
                print(f"  {k}: {v}")

    costs = cost_sensitivity(trades, base_cost_r=trades["cost_r"], pnl_col="pnl_r")
    print(f"\nCost sensitivity (zero / base({COST_BASIS}, mean {trades['cost_r'].mean():.4f}R) / 2x):")
    print(costs[["scenario", "n_trades", "expectancy_r", "payoff_ratio"]].to_string(index=False))

    wf = walk_forward_correlation(
        trades, n_folds=6,
        metric_fn=lambda t: trade_stats(t, pnl_col="pnl_r_after_cost").expectancy_r,
    )
    print(f"\nWalk-forward (6 folds, after-cost expectancy_r): {wf}")

    dist = mae_mfe_distribution(trades, pnl_col="pnl_r_after_cost")
    print("\nMAE/MFE distribution, winners vs losers (after cost):")
    print(dist.to_string(index=False))

    write_journal(trades, JOURNAL_PATH)
    print(f"\nWrote {len(trades)}-trade journal to {os.path.abspath(JOURNAL_PATH)}")

    summary = {
        "strategy": "btc_only",
        "params": {k: v for k, v in DEFAULT_PARAMS.__dict__.items()},
        "cost_basis": COST_BASIS,
        "mean_cost_r": float(trades["cost_r"].mean()),
        "oos_split_date": str(split_date.date()),
        "blocks": blocks,
        "cost_sensitivity": costs.to_dict(orient="records"),
        "walk_forward": {k: v for k, v in wf.items() if k != "fold_edges"} | {
            "fold_edges": [[str(a), str(b)] for a, b in wf.get("fold_edges", [])]
        },
        "mae_mfe_distribution": dist.to_dict(orient="records"),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote summary to {os.path.abspath(SUMMARY_PATH)}")


if __name__ == "__main__":
    main()
