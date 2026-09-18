"""
Two-leg pairs variant: dedicated re-optimization, rebuilt on top of the
corrected beta-hedge sizing + combined-P&L stop (see two_leg_backtest.py)
and the ADX regime filter (see strategy_pairs.py and graveyard.md for why
ADX, not a different regime construction this project has also worked
with -- kept out of this public module on purpose, see README).

Every prior pairs search was computed under either the old, incorrect
sizing model or a different regime filter -- this is a full restart, not a
refinement of any earlier result.

Stage 1: cartesian grid over the three threshold parameters (low_q, high_q,
mid_q), with the ADX gate opened wide (a high threshold, effectively no
regime filtering) and the stop set very loose, to isolate threshold
structure from regime/stop confounds.

Stage 2: cross the top-5 stage-1 threshold combos against adx_threshold,
max_loss_pct_of_cap, beta_window, pct_lookback, and max_hold_bars.
adx_window is fixed at the standard default (14), not searched.

Run: python optimize_pairs.py (from any directory)
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_fetch import fetch_funding_rates, fetch_klines
from measurement import equity_curve_from_trades, sharpe_from_equity_curve, train_test_split_by_date
from strategy_pairs import PairsParams, build_signals
from two_leg_backtest import CAP, PairsBacktestConfig, run_pairs_backtest

HERE = os.path.dirname(os.path.abspath(__file__))

OOS_SPLIT_DATE = "2025-09-14"  # same convention as the BTC-only variant: OOS = most recent 12 months


def _score(params: PairsParams, df_btc: pd.DataFrame, df_bnb: pd.DataFrame,
           funding_df_btc: pd.DataFrame, funding_df_bnb: pd.DataFrame) -> dict:
    sig = build_signals(df_btc, df_bnb, params)
    cfg = PairsBacktestConfig(cap_usd=CAP, max_loss_pct_of_cap=params.max_loss_pct_of_cap,
                               max_hold_bars=params.max_hold_bars)
    trades = run_pairs_backtest(df_btc, df_bnb, sig.entry_signal, sig.btc_direction, sig.beta,
                                 sig.exit_signal_long, sig.exit_signal_short, cfg,
                                 funding_df_btc=funding_df_btc, funding_df_bnb=funding_df_bnb)
    if len(trades) < 30:  # not enough trades to trust a Sharpe estimate at all
        return {"n_trades": len(trades), "is_sharpe": float("nan"), "oos_sharpe": float("nan"), "min_sharpe": float("nan")}

    is_trades, oos_trades = train_test_split_by_date(trades, OOS_SPLIT_DATE)
    if len(is_trades) < 30 or len(oos_trades) < 10:
        return {"n_trades": len(trades), "is_sharpe": float("nan"), "oos_sharpe": float("nan"), "min_sharpe": float("nan")}

    is_eq = equity_curve_from_trades(is_trades, risk_per_trade_usd=CAP, starting_equity=CAP, pnl_col="pnl_r")
    oos_eq = equity_curve_from_trades(oos_trades, risk_per_trade_usd=CAP, starting_equity=CAP, pnl_col="pnl_r")
    is_sharpe = sharpe_from_equity_curve(is_eq)
    oos_sharpe = sharpe_from_equity_curve(oos_eq)
    min_sharpe = min(is_sharpe, oos_sharpe) if pd.notna(is_sharpe) and pd.notna(oos_sharpe) else float("nan")
    return {"n_trades": len(trades), "n_is": len(is_trades), "n_oos": len(oos_trades),
            "is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe, "min_sharpe": min_sharpe}


def stage1(df_btc, df_bnb, funding_df_btc, funding_df_bnb) -> pd.DataFrame:
    low_qs = [0.01, 0.025, 0.05, 0.08, 0.10, 0.20, 0.30]
    high_qs = [0.70, 0.80, 0.90, 0.92, 0.95, 0.975, 0.99]
    mid_qs = [0.40, 0.45, 0.50, 0.55, 0.60]

    rows = []
    t0 = time.time()
    combos = list(itertools.product(low_qs, high_qs, mid_qs))
    for i, (low_q, high_q, mid_q) in enumerate(combos):
        if high_q - mid_q < 0.05 or mid_q - low_q < 0.05:
            continue  # degenerate: entry/exit thresholds too close together
        params = PairsParams(low_q=low_q, high_q=high_q, mid_q=mid_q,
                              adx_threshold=100.0,   # wide open (ADX never exceeds ~100), isolate thresholds
                              max_loss_pct_of_cap=0.20)  # very loose, isolate thresholds from the stop
        scored = _score(params, df_btc, df_bnb, funding_df_btc, funding_df_bnb)
        rows.append({"low_q": low_q, "high_q": high_q, "mid_q": mid_q, **scored})
        if (i + 1) % 20 == 0:
            print(f"  stage1 {i + 1}/{len(combos)} ({time.time() - t0:.0f}s elapsed)", flush=True)

    return pd.DataFrame(rows)


def stage2(df_btc, df_bnb, funding_df_btc, funding_df_bnb,
           top_thresholds: list[tuple[float, float, float]]) -> pd.DataFrame:
    adx_thresholds = [15.0, 20.0, 25.0, 30.0, 35.0]
    stop_pcts = [0.01, 0.02, 0.03, 0.05]
    beta_windows = [100, 200, 400, 500]
    pct_lookbacks = [750, 1500, 2500]
    max_holds = [150, 300]

    rows = []
    t0 = time.time()
    combos = list(itertools.product(top_thresholds, adx_thresholds, stop_pcts,
                                     beta_windows, pct_lookbacks, max_holds))
    print(f"stage2: {len(combos)} combos", flush=True)
    for i, ((low_q, high_q, mid_q), adx_thres, stop_pct, bw, pl, mh) in enumerate(combos):
        params = PairsParams(low_q=low_q, high_q=high_q, mid_q=mid_q,
                              adx_threshold=adx_thres,
                              max_loss_pct_of_cap=stop_pct,
                              beta_window=bw, pct_lookback=pl, max_hold_bars=mh)
        scored = _score(params, df_btc, df_bnb, funding_df_btc, funding_df_bnb)
        rows.append({"low_q": low_q, "high_q": high_q, "mid_q": mid_q, "adx_threshold": adx_thres,
                      "max_loss_pct_of_cap": stop_pct,
                      "beta_window": bw, "pct_lookback": pl, "max_hold_bars": mh, **scored})
        if (i + 1) % 200 == 0:
            print(f"  stage2 {i + 1}/{len(combos)} ({time.time() - t0:.0f}s elapsed)", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_btc = fetch_klines("BTCUSDT", "15m", "2020-01-01", "2026-09-14", market="futures")
    df_bnb = fetch_klines("BNBUSDT", "15m", "2020-01-01", "2026-09-14", market="futures")
    funding_df_btc = fetch_funding_rates("BTCUSDT", "2020-01-01", "2026-09-14")
    funding_df_bnb = fetch_funding_rates("BNBUSDT", "2020-01-01", "2026-09-14")

    print("=== Stage 1: threshold grid ===", flush=True)
    s1 = stage1(df_btc, df_bnb, funding_df_btc, funding_df_bnb)
    s1.to_csv(os.path.join(HERE, "optimize_pairs_stage1.csv"), index=False)
    s1_valid = s1.dropna(subset=["min_sharpe"])
    print(f"stage1: {len(s1)} total, {len(s1_valid)} with enough trades, "
          f"{(s1_valid['min_sharpe'] > 0).sum()} min_sharpe>0", flush=True)
    top5 = s1_valid.sort_values("min_sharpe", ascending=False).head(5)
    print("\nTop 5 stage1 threshold combos:\n", top5[["low_q", "high_q", "mid_q", "n_trades", "is_sharpe", "oos_sharpe", "min_sharpe"]], flush=True)

    top_thresholds = list(top5[["low_q", "high_q", "mid_q"]].itertuples(index=False, name=None))

    print("\n=== Stage 2: full cross ===", flush=True)
    s2 = stage2(df_btc, df_bnb, funding_df_btc, funding_df_bnb, top_thresholds)
    s2.to_csv(os.path.join(HERE, "optimize_pairs_stage2.csv"), index=False)
    s2_valid = s2.dropna(subset=["min_sharpe"])
    print(f"stage2: {len(s2)} total, {len(s2_valid)} with enough trades, "
          f"{(s2_valid['min_sharpe'] > 0).sum()} min_sharpe>0", flush=True)
    best = s2_valid.sort_values("min_sharpe", ascending=False).head(10)
    print("\nTop 10 stage2 combos:\n", best.to_string(), flush=True)
