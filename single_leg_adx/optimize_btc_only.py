"""
BTC-only variant: dedicated re-optimization for the ADX regime filter (see
strategy_btc_only.py and graveyard.md for why ADX, not a different regime
construction this project has also worked with -- kept out of this public
module on purpose, see README). Same 2-stage discipline as optimize_pairs.py:
staged, in-sample selection throughout, out-of-sample held out until the
final confirmation run, ranked by min(IS Sharpe, OOS Sharpe).

Stage 1: cartesian grid over the three threshold parameters (low_q, high_q,
mid_q), with the ADX gate opened wide and the ATR stop set very loose, to
isolate threshold structure from regime/stop confounds.

Stage 2: cross the top-5 stage-1 threshold combos against adx_threshold,
atr_stop_multiple, beta_window, pct_lookback, and max_hold_bars. adx_window
and atr_window are both fixed at their standard defaults (14), not
searched. stop_mode is fixed at "atr" (won the head-to-head comparison
against a fixed-percentage stop -- see graveyard.md -- not re-litigated
here).

Run: python optimize_btc_only.py (from any directory)
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest_engine import BacktestConfig, run_backtest
from costs import compute_cost_r
from data_fetch import fetch_klines
from measurement import apply_cost, equity_curve_from_trades, sharpe_from_equity_curve, train_test_split_by_date

HERE = os.path.dirname(os.path.abspath(__file__))
from strategy_btc_only import StatArbBtcParams, build_signals

OOS_SPLIT_DATE = "2025-09-14"  # same convention as the pairs variant: OOS = most recent 12 months
ACCOUNT_EQUITY_USD = 100_000.0
RISK_PER_TRADE_USD = 500.0


def _score(params: StatArbBtcParams, df_btc: pd.DataFrame, df_bnb: pd.DataFrame) -> dict:
    sig = build_signals(df_btc, df_bnb, params)
    common_idx = df_btc.index.intersection(df_bnb.index)
    btc = df_btc.loc[common_idx]
    cfg = BacktestConfig(entry_fill="next_open", allow_overlapping=False, max_holding_bars=params.max_hold_bars)
    trades = run_backtest(btc, sig.entry_signal, sig.stop_r_price, sig.direction, cfg=cfg,
                           exit_signal_long=sig.exit_signal_long, exit_signal_short=sig.exit_signal_short)
    if len(trades) < 30:
        return {"n_trades": len(trades), "is_sharpe": float("nan"), "oos_sharpe": float("nan"), "min_sharpe": float("nan")}

    trades["cost_r"] = compute_cost_r(trades)
    trades = apply_cost(trades, trades["cost_r"], pnl_col="pnl_r")
    is_trades, oos_trades = train_test_split_by_date(trades, OOS_SPLIT_DATE)
    if len(is_trades) < 30 or len(oos_trades) < 10:
        return {"n_trades": len(trades), "is_sharpe": float("nan"), "oos_sharpe": float("nan"), "min_sharpe": float("nan")}

    is_eq = equity_curve_from_trades(is_trades, RISK_PER_TRADE_USD, ACCOUNT_EQUITY_USD, pnl_col="pnl_r_after_cost")
    oos_eq = equity_curve_from_trades(oos_trades, RISK_PER_TRADE_USD, ACCOUNT_EQUITY_USD, pnl_col="pnl_r_after_cost")
    is_sharpe = sharpe_from_equity_curve(is_eq)
    oos_sharpe = sharpe_from_equity_curve(oos_eq)
    min_sharpe = min(is_sharpe, oos_sharpe) if pd.notna(is_sharpe) and pd.notna(oos_sharpe) else float("nan")
    return {"n_trades": len(trades), "n_is": len(is_trades), "n_oos": len(oos_trades),
            "is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe, "min_sharpe": min_sharpe}


def stage1(df_btc, df_bnb) -> pd.DataFrame:
    low_qs = [0.01, 0.025, 0.05, 0.08, 0.10, 0.20, 0.30]
    high_qs = [0.70, 0.80, 0.90, 0.92, 0.95, 0.975, 0.99]
    mid_qs = [0.40, 0.45, 0.50, 0.55, 0.60]

    rows = []
    t0 = time.time()
    combos = list(itertools.product(low_qs, high_qs, mid_qs))
    for i, (low_q, high_q, mid_q) in enumerate(combos):
        if high_q - mid_q < 0.05 or mid_q - low_q < 0.05:
            continue
        params = StatArbBtcParams(low_q=low_q, high_q=high_q, mid_q=mid_q,
                                   adx_threshold=100.0, atr_stop_multiple=20.0, max_hold_bars=300)
        scored = _score(params, df_btc, df_bnb)
        rows.append({"low_q": low_q, "high_q": high_q, "mid_q": mid_q, **scored})
        if (i + 1) % 20 == 0:
            print(f"  stage1 {i + 1}/{len(combos)} ({time.time() - t0:.0f}s elapsed)", flush=True)

    return pd.DataFrame(rows)


def stage2(df_btc, df_bnb, top_thresholds: list[tuple[float, float, float]]) -> pd.DataFrame:
    adx_thresholds = [15.0, 20.0, 25.0, 30.0, 35.0]
    atr_stop_multiples = [6.0, 10.0, 15.0, 20.0]
    beta_windows = [100, 200, 400, 500]
    pct_lookbacks = [750, 1500, 2500]
    max_holds = [150, 300]

    rows = []
    t0 = time.time()
    combos = list(itertools.product(top_thresholds, adx_thresholds, atr_stop_multiples,
                                     beta_windows, pct_lookbacks, max_holds))
    print(f"stage2: {len(combos)} combos", flush=True)
    for i, ((low_q, high_q, mid_q), adx_thres, atr_mult, bw, pl, mh) in enumerate(combos):
        params = StatArbBtcParams(low_q=low_q, high_q=high_q, mid_q=mid_q,
                                   adx_threshold=adx_thres, atr_stop_multiple=atr_mult,
                                   beta_window=bw, pct_lookback=pl, max_hold_bars=mh)
        scored = _score(params, df_btc, df_bnb)
        rows.append({"low_q": low_q, "high_q": high_q, "mid_q": mid_q, "adx_threshold": adx_thres,
                      "atr_stop_multiple": atr_mult, "beta_window": bw, "pct_lookback": pl,
                      "max_hold_bars": mh, **scored})
        if (i + 1) % 200 == 0:
            print(f"  stage2 {i + 1}/{len(combos)} ({time.time() - t0:.0f}s elapsed)", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_btc = fetch_klines("BTCUSDT", "15m", "2020-01-01", "2026-09-14", market="futures")
    df_bnb = fetch_klines("BNBUSDT", "15m", "2020-01-01", "2026-09-14", market="futures")

    print("=== Stage 1: threshold grid ===", flush=True)
    s1 = stage1(df_btc, df_bnb)
    s1.to_csv(os.path.join(HERE, "optimize_btc_only_stage1.csv"), index=False)
    s1_valid = s1.dropna(subset=["min_sharpe"])
    print(f"stage1: {len(s1)} total, {len(s1_valid)} with enough trades, "
          f"{(s1_valid['min_sharpe'] > 0).sum()} min_sharpe>0", flush=True)
    top5 = s1_valid.sort_values("min_sharpe", ascending=False).head(5)
    print("\nTop 5 stage1 threshold combos:\n", top5[["low_q", "high_q", "mid_q", "n_trades", "is_sharpe", "oos_sharpe", "min_sharpe"]], flush=True)

    top_thresholds = list(top5[["low_q", "high_q", "mid_q"]].itertuples(index=False, name=None))

    print("\n=== Stage 2: full cross ===", flush=True)
    s2 = stage2(df_btc, df_bnb, top_thresholds)
    s2.to_csv(os.path.join(HERE, "optimize_btc_only_stage2.csv"), index=False)
    s2_valid = s2.dropna(subset=["min_sharpe"])
    print(f"stage2: {len(s2)} total, {len(s2_valid)} with enough trades, "
          f"{(s2_valid['min_sharpe'] > 0).sum()} min_sharpe>0", flush=True)
    best = s2_valid.sort_values("min_sharpe", ascending=False).head(10)
    print("\nTop 10 stage2 combos:\n", best.to_string(), flush=True)
