"""
Two-leg pairs variant: full honest-measurement run on the locked config
(strategy_pairs.py::DEFAULT_PARAMS). VIP3-tier Binance fees + funding are
now applied INSIDE run_pairs_backtest itself (see two_leg_backtest.py), not
bolted on afterward here -- 'pnl_r' coming out of the engine is already
net of fees and funding, expressed as a fraction of the pair's capital
allocation (CAP), not a per-trade risk-dollar R multiple.

Run: python run_backtest_pairs.py (from any directory)
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_fetch import fetch_funding_rates, fetch_klines
from measurement import (calmar_ratio, cost_sensitivity, equity_curve_from_trades, mae_mfe_distribution,
                          max_drawdown, sharpe_from_equity_curve, train_test_split_by_date, trade_stats,
                          walk_forward_correlation)
from strategy_pairs import DEFAULT_PARAMS, build_signals
from two_leg_backtest import CAP, PairsBacktestConfig, run_pairs_backtest

SYMBOL_BTC = "BTCUSDT"
SYMBOL_BNB = "BNBUSDT"
INTERVAL = "15m"
START = "2020-01-01"
END = "2026-09-14"
MARKET = "futures"

OOS_MONTHS = 12
JOURNAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "journals",
                             "pairs_backtest_journal.jsonl")
SUMMARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairs_summary.json")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_btc = fetch_klines(symbol=SYMBOL_BTC, interval=INTERVAL, start=START, end=END, market=MARKET)
    df_bnb = fetch_klines(symbol=SYMBOL_BNB, interval=INTERVAL, start=START, end=END, market=MARKET)
    funding_btc = fetch_funding_rates(symbol=SYMBOL_BTC, start=START, end=END)
    funding_bnb = fetch_funding_rates(symbol=SYMBOL_BNB, start=START, end=END)
    return df_btc, df_bnb, funding_btc, funding_bnb


def run_full_backtest(df_btc: pd.DataFrame, df_bnb: pd.DataFrame,
                       funding_btc: pd.DataFrame, funding_bnb: pd.DataFrame,
                       fee_multiplier: float = 1.0, funding_multiplier: float = 1.0) -> pd.DataFrame:
    sig = build_signals(df_btc, df_bnb, DEFAULT_PARAMS)
    cfg = PairsBacktestConfig(cap_usd=CAP, max_loss_pct_of_cap=DEFAULT_PARAMS.max_loss_pct_of_cap,
                               max_hold_bars=DEFAULT_PARAMS.max_hold_bars,
                               fee_multiplier=fee_multiplier, funding_multiplier=funding_multiplier)
    trades = run_pairs_backtest(df_btc, df_bnb, sig.entry_signal, sig.btc_direction, sig.beta,
                                 sig.exit_signal_long, sig.exit_signal_short, cfg,
                                 funding_df_btc=funding_btc, funding_df_bnb=funding_bnb)
    return trades


def _stats_block(trades: pd.DataFrame, label: str) -> dict:
    stats = trade_stats(trades, pnl_col="pnl_r")
    equity = equity_curve_from_trades(trades, risk_per_trade_usd=CAP, starting_equity=CAP, pnl_col="pnl_r")
    dd = max_drawdown(equity) if len(equity) >= 2 else None
    sharpe = sharpe_from_equity_curve(equity) if len(equity) >= 2 else float("nan")
    calmar = calmar_ratio(equity) if len(equity) >= 2 else float("nan")
    return {
        "label": label, "n_trades": stats.n_trades,
        "hit_rate": stats.hit_rate, "expectancy_pct_of_cap": stats.expectancy_r,
        "net_return_per_trade_usd": stats.expectancy_r * CAP,
        "payoff_ratio": stats.payoff_ratio,
        "avg_win_pct_of_cap": stats.avg_win_r, "avg_loss_pct_of_cap": stats.avg_loss_r,
        "sharpe_365d": sharpe, "calmar": calmar,
        "max_drawdown_pct_after_cost": dd.max_drawdown_pct if dd else float("nan"),
        "max_drawdown_duration_bars": dd.duration_bars if dd else float("nan"),
        "ending_equity_usd": equity.iloc[-1] if len(equity) else float("nan"),
    }


def write_journal(trades: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for _, t in trades.iterrows():
            row = {
                "event": "backtest_trade",
                "entry_time": t["entry_time"].isoformat(), "exit_time": t["exit_time"].isoformat(),
                "beta_entry": round(float(t["beta_entry"]), 6),
                "btc_direction": "LONG" if t["btc_direction"] == 1 else "SHORT",
                "btc_entry_price": round(float(t["btc_entry_price"]), 2),
                "btc_exit_price": round(float(t["btc_exit_price"]), 2),
                "qty_btc": round(float(t["qty_btc"]), 6),
                "bnb_entry_price": round(float(t["bnb_entry_price"]), 4),
                "bnb_exit_price": round(float(t["bnb_exit_price"]), 4),
                "qty_bnb": round(float(t["qty_bnb"]), 4),
                "pnl_usd_gross": round(float(t["pnl_usd_gross"]), 2),
                "fee_usd": round(float(t["fee_usd"]), 2), "funding_usd": round(float(t["funding_usd"]), 2),
                "pnl_usd_net": round(float(t["pnl_usd_net"]), 2), "pnl_r": round(float(t["pnl_r"]), 6),
                "mae_r": round(float(t["mae_r"]), 6), "mfe_r": round(float(t["mfe_r"]), 6),
                "bars_held": int(t["bars_held"]), "exit_reason": t["exit_reason"], "is_closed": bool(t["is_closed"]),
            }
            f.write(json.dumps(row) + "\n")


def main():
    print(f"Fetching {SYMBOL_BTC} + {SYMBOL_BNB} {INTERVAL} {MARKET} klines + funding history from {START}...")
    df_btc, df_bnb, funding_btc, funding_bnb = load_data()
    trades = run_full_backtest(df_btc, df_bnb, funding_btc, funding_bnb)
    print(f"\n{len(trades)} total trades generated. Cost basis: VIP3 Binance fees + funding, both legs, CAP=${CAP:,.0f}")

    data_end = df_btc.index.intersection(df_bnb.index)[-1]
    split_date = data_end - pd.DateOffset(months=OOS_MONTHS)
    split_date_str = split_date.strftime("%Y-%m-%d")
    is_trades, oos_trades = train_test_split_by_date(trades, split_date_str)
    print(f"In-sample: entries before {split_date.date()} ({len(is_trades)} trades)")
    print(f"Out-of-sample: entries on/after {split_date.date()} ({len(oos_trades)} trades)")

    blocks = [_stats_block(trades, "full"), _stats_block(is_trades, "in_sample"), _stats_block(oos_trades, "out_of_sample")]
    for b in blocks:
        print(f"\n--- {b['label']} ---")
        for k, v in b.items():
            if k != "label":
                print(f"  {k}: {v}")

    # Cost sensitivity: re-run the whole backtest at 0x / 1x / 2x fee+funding
    # multiplier (not a post-hoc subtraction -- costs affect exit timing via
    # the stop, so they have to be applied inside the simulation itself).
    cost_rows = []
    for label, mult in [("zero_cost", 0.0), ("base_cost", 1.0), ("2x_cost", 2.0)]:
        t = run_full_backtest(df_btc, df_bnb, funding_btc, funding_bnb, fee_multiplier=mult, funding_multiplier=mult)
        stats = trade_stats(t, pnl_col="pnl_r")
        cost_rows.append({"scenario": label, **stats.to_dict()})
    costs_df = pd.DataFrame(cost_rows)
    print(f"\nCost sensitivity (0x / 1x / 2x fee+funding multiplier):")
    print(costs_df[["scenario", "n_trades", "expectancy_r", "payoff_ratio"]].to_string(index=False))

    wf = walk_forward_correlation(trades, n_folds=6, metric_fn=lambda t: trade_stats(t, pnl_col="pnl_r").expectancy_r)
    print(f"\nWalk-forward (6 folds, expectancy_r): {wf}")

    dist = mae_mfe_distribution(trades, pnl_col="pnl_r")
    print("\nMAE/MFE distribution, winners vs losers:")
    print(dist.to_string(index=False))

    write_journal(trades, JOURNAL_PATH)
    print(f"\nWrote {len(trades)}-trade journal to {os.path.abspath(JOURNAL_PATH)}")

    summary = {
        "strategy": "pairs_two_leg", "cap_usd": CAP,
        "params": {k: v for k, v in DEFAULT_PARAMS.__dict__.items()},
        "cost_basis": "binance_vip3_plus_funding",
        "oos_split_date": str(split_date.date()), "blocks": blocks,
        "cost_sensitivity": costs_df.to_dict(orient="records"),
        "walk_forward": {k: v for k, v in wf.items() if k != "fold_edges"} | {
            "fold_edges": [[str(a), str(b)] for a, b in wf.get("fold_edges", [])]},
        "mae_mfe_distribution": dist.to_dict(orient="records"),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote summary to {os.path.abspath(SUMMARY_PATH)}")


if __name__ == "__main__":
    main()
