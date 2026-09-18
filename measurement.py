"""
Honest-measurement suite -- strategy-agnostic, operating entirely on the
trade-level DataFrame backtest_engine.run_backtest() produces (or anything
with the same columns, including a real trade journal). Built ahead of
picking the Part B strategy so tomorrow is signal design + interpretation,
not plumbing.

Every function here takes data in, returns numbers/DataFrames out -- nothing
is hardcoded to a specific strategy's parameters. Covers the "honest
measurement" checklist from the test almost line for line:
  - number of trades, hit rate, avg win/loss, payoff ratio, expectancy in R
  - Sharpe (state the annualisation), max drawdown, drawdown duration
  - a cost assumption and how results change if you double it
  - in-sample/out-of-sample split
  - walk-forward correlation test
  - MAE/MFE distributions of winners vs. losers
  - parameter sensitivity sweep
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Core trade stats
# ---------------------------------------------------------------------------
@dataclass
class TradeStats:
    n_trades: int
    hit_rate: float
    avg_win_r: float
    avg_loss_r: float          # negative number, e.g. -0.8
    payoff_ratio: float        # avg_win_r / abs(avg_loss_r)
    expectancy_r: float        # hit_rate*avg_win_r + (1-hit_rate)*avg_loss_r

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def trade_stats(trades: pd.DataFrame, pnl_col: str = "pnl_r") -> TradeStats:
    """trades must have a numeric pnl column (in R by default -- pass
    pnl_col='pnl_r_after_cost' etc. to reuse this on a cost-adjusted copy)."""
    closed = trades[trades.get("is_closed", True)] if "is_closed" in trades.columns else trades
    n = len(closed)
    if n == 0:
        return TradeStats(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    wins = closed[closed[pnl_col] > 0][pnl_col]
    losses = closed[closed[pnl_col] <= 0][pnl_col]
    hit_rate = len(wins) / n
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else float("inf")
    expectancy = hit_rate * avg_win + (1 - hit_rate) * avg_loss
    return TradeStats(n, hit_rate, avg_win, avg_loss, payoff, expectancy)


# ---------------------------------------------------------------------------
# 2. Equity curve, Sharpe, drawdown
# ---------------------------------------------------------------------------
def equity_curve_from_trades(trades: pd.DataFrame, risk_per_trade_usd: float,
                              starting_equity: float, pnl_col: str = "pnl_r") -> pd.Series:
    """Converts a stream of per-trade R outcomes into a running-equity time
    series, indexed at each trade's EXIT time (fixed-dollar-risk-per-trade
    convention, matching Part A's own risk_per_trade_pct sizing -- this is
    what ties the backtest's equity curve to the same risk unit the live
    trade manager uses)."""
    closed = trades[trades.get("is_closed", True)].sort_values("exit_time") if "is_closed" in trades.columns else trades.sort_values("exit_time")
    pnl_usd = closed[pnl_col] * risk_per_trade_usd
    equity = starting_equity + pnl_usd.cumsum()
    equity.index = closed["exit_time"]
    return equity


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    """Plain annualized Sharpe from a return series (e.g. daily % returns
    resampled from an equity curve). periods_per_year must be stated
    explicitly by the caller -- 252 for daily, 365 for calendar-day crypto
    (BTC trades every day, not just "trading days"), 52 for weekly, etc.
    No risk-free-rate subtraction (assumed ~0, standard simplification for
    a short BTC backtest -- documented here rather than silently assumed)."""
    if returns.std(ddof=1) == 0 or len(returns) < 2:
        return float("nan")
    return (returns.mean() / returns.std(ddof=1)) * np.sqrt(periods_per_year)


def sharpe_from_equity_curve(equity: pd.Series, resample_rule: str = "1D",
                              periods_per_year: float = 365) -> float:
    """Resamples an (irregularly-timed, trade-exit-indexed) equity curve to
    a regular frequency before computing Sharpe -- resampling first matters
    because trade exits are NOT evenly spaced in time, and computing returns
    directly on them would implicitly (and wrongly) treat every trade as one
    time unit regardless of how many days it actually spanned."""
    daily = equity.resample(resample_rule).last().ffill()
    rets = daily.pct_change().dropna()
    return sharpe_ratio(rets, periods_per_year)


@dataclass
class DrawdownResult:
    max_drawdown_pct: float     # positive number, e.g. 0.18 = 18% peak-to-trough
    max_drawdown_usd: float
    duration_bars: int          # length of the longest underwater period, in equity-curve steps
    peak_time: object
    trough_time: object


def max_drawdown(equity: pd.Series) -> DrawdownResult:
    running_peak = equity.cummax()
    drawdown_usd = equity - running_peak
    drawdown_pct = drawdown_usd / running_peak
    trough_pos = drawdown_pct.values.argmin()
    trough_time = equity.index[trough_pos]
    peak_time = running_peak.index[: trough_pos + 1][
        (equity.iloc[: trough_pos + 1] == running_peak.iloc[: trough_pos + 1]).values
    ][-1]

    underwater = drawdown_usd < 0
    # Longest consecutive run of True in `underwater`.
    max_run = cur_run = 0
    for v in underwater:
        cur_run = cur_run + 1 if v else 0
        max_run = max(max_run, cur_run)

    return DrawdownResult(
        max_drawdown_pct=abs(drawdown_pct.min()),
        max_drawdown_usd=abs(drawdown_usd.min()),
        duration_bars=max_run,
        peak_time=peak_time, trough_time=trough_time,
    )


def cagr_from_equity(equity: pd.Series, days_per_year: float = 365) -> float:
    """Compound annual growth rate over the equity curve's actual elapsed
    CALENDAR span (not bar count or trade count), since trade exits are
    irregularly spaced in time -- same reasoning as
    sharpe_from_equity_curve's resample-first approach. Extracted from
    calmar_ratio (which needs the same number) so a return-focused ranking
    doesn't have to recompute drawdown just to get CAGR."""
    if len(equity) < 2:
        return float("nan")
    start, end = equity.iloc[0], equity.iloc[-1]
    elapsed_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
    if elapsed_days <= 0 or start <= 0 or end <= 0:
        # end<=0 means the account was wiped out somewhere along this curve --
        # CAGR is undefined (a negative base to a fractional power is complex,
        # not "a very bad return"), not just numerically inconvenient.
        return float("nan")
    years = elapsed_days / days_per_year
    return (end / start) ** (1 / years) - 1


def calmar_ratio(equity: pd.Series, days_per_year: float = 365) -> float:
    """Annualized return / max drawdown (%) -- a drawdown-aware companion to
    Sharpe (Sharpe penalizes volatility symmetrically; Calmar only cares
    about the worst peak-to-trough)."""
    cagr = cagr_from_equity(equity, days_per_year)
    if pd.isna(cagr):
        return float("nan")
    dd = max_drawdown(equity).max_drawdown_pct
    if dd == 0:
        return float("inf") if cagr > 0 else float("nan")
    return cagr / dd


def trades_per_week(trades: pd.DataFrame, time_col: str = "entry_time") -> float:
    """Simple frequency metric: n_trades / elapsed weeks spanned by the
    trades' own entry timestamps -- NOT the full backtest window, so a
    sparse sub-period (e.g. an OOS block) reports its own real cadence
    rather than being diluted by the full history's span."""
    if len(trades) < 2:
        return float("nan")
    elapsed_days = (trades[time_col].max() - trades[time_col].min()).total_seconds() / 86400
    if elapsed_days <= 0:
        return float("nan")
    return len(trades) / (elapsed_days / 7)


# ---------------------------------------------------------------------------
# 3. Cost sensitivity
# ---------------------------------------------------------------------------
def apply_cost(trades: pd.DataFrame, cost_r: float | pd.Series, pnl_col: str = "pnl_r") -> pd.DataFrame:
    """Subtracts a per-trade cost (fees + slippage), expressed in R, from
    every trade's pnl. cost_r may be a flat scalar (every trade costed the
    same) or a Series aligned to trades.index (e.g. cost that varies with
    each trade's own r_price -- see costs.compute_cost_r).
    Returns a copy with a new '{pnl_col}_after_cost' column -- never mutates
    the input, so the same trades DataFrame can be re-costed at multiple
    assumptions without re-running the backtest."""
    out = trades.copy()
    out[f"{pnl_col}_after_cost"] = out[pnl_col] - cost_r
    return out


def cost_sensitivity(trades: pd.DataFrame, base_cost_r: float | pd.Series, pnl_col: str = "pnl_r") -> pd.DataFrame:
    """Required item: 'a cost assumption... and how results change if you
    double it.' Returns a small comparison table: zero cost, base cost, 2x
    cost -- each row is a full TradeStats so the whole picture (not just
    expectancy) is visible at each cost level. base_cost_r may be a scalar
    or a per-trade Series (its MEAN is what's reported in the 'cost_r'
    summary column either way, for a single comparable number per row; the
    full Series -- not just its mean -- is still what's actually subtracted
    from each trade via apply_cost)."""
    mean_cost = base_cost_r.mean() if isinstance(base_cost_r, pd.Series) else base_cost_r
    rows = []
    for label, cost, cost_label in [("zero_cost", 0.0, 0.0), ("base_cost", base_cost_r, mean_cost),
                                     ("2x_cost", 2 * base_cost_r, 2 * mean_cost)]:
        costed = apply_cost(trades, cost, pnl_col)
        stats = trade_stats(costed, pnl_col=f"{pnl_col}_after_cost")
        rows.append({"scenario": label, "cost_r": cost_label, **stats.to_dict()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. In-sample / out-of-sample split
# ---------------------------------------------------------------------------
def train_test_split_by_date(trades: pd.DataFrame, split_date, time_col: str = "entry_time"
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple, honest time-based split -- no shuffling (this is a time series;
    a random split would leak future information into 'in-sample')."""
    in_sample = trades[trades[time_col] < pd.Timestamp(split_date, tz="UTC")]
    out_sample = trades[trades[time_col] >= pd.Timestamp(split_date, tz="UTC")]
    return in_sample, out_sample


# ---------------------------------------------------------------------------
# 5. Walk-forward correlation test
# ---------------------------------------------------------------------------
def walk_forward_correlation(trades: pd.DataFrame, n_folds: int = 6,
                              metric_fn: Callable[[pd.DataFrame], float] = None,
                              time_col: str = "entry_time") -> dict:
    """Splits the trade history into n_folds SEQUENTIAL (never shuffled)
    folds, computes metric_fn on each fold, and reports the lag-1
    correlation between consecutive folds' metric values -- a strategy whose
    edge is real should show some positive persistence fold-to-fold; a
    strategy that's pure curve-fit noise typically won't. metric_fn defaults
    to expectancy_r; pass e.g. `lambda t: trade_stats(t).hit_rate` to test a
    different metric instead.

    Returns {'fold_values': [...], 'lag1_correlation': float, 'fold_edges': [...]}.
    Needs n_folds >= 3 to say anything meaningful (2 folds gives one pair,
    not really a "correlation").
    """
    if metric_fn is None:
        metric_fn = lambda t: trade_stats(t).expectancy_r
    sorted_trades = trades.sort_values(time_col).reset_index(drop=True)
    fold_indices = np.array_split(sorted_trades.index, n_folds)
    fold_values, fold_edges = [], []
    for fold_idx in fold_indices:
        fold = sorted_trades.loc[fold_idx]
        if len(fold) == 0:
            continue
        fold_values.append(metric_fn(fold))
        fold_edges.append((fold[time_col].min(), fold[time_col].max()))

    if len(fold_values) < 3:
        return {"fold_values": fold_values, "fold_edges": fold_edges,
                "lag1_correlation": float("nan"),
                "note": "fewer than 3 non-empty folds -- not enough points for a meaningful correlation"}

    series = pd.Series(fold_values)
    lag1_corr = series.corr(series.shift(1))
    return {"fold_values": fold_values, "fold_edges": fold_edges, "lag1_correlation": lag1_corr}


# ---------------------------------------------------------------------------
# 6. MAE / MFE distributions, winners vs. losers
# ---------------------------------------------------------------------------
def mae_mfe_distribution(trades: pd.DataFrame, pnl_col: str = "pnl_r") -> pd.DataFrame:
    """Summary stats of MAE/MFE (in R) split by winners vs. losers. Directly
    feeds Part A's bonus item ('derive the MAE threshold from data') --
    e.g. a soft-close threshold between the losers' median MAE and the
    winners' 75th-percentile MAE would cut most losers early while rarely
    touching a trade that was going to work anyway."""
    closed = trades[trades.get("is_closed", True)] if "is_closed" in trades.columns else trades
    winners = closed[closed[pnl_col] > 0]
    losers = closed[closed[pnl_col] <= 0]
    rows = []
    for label, group in [("winners", winners), ("losers", losers)]:
        for metric in ("mae_r", "mfe_r"):
            if len(group) == 0:
                continue
            s = group[metric]
            rows.append({
                "group": label, "metric": metric, "n": len(s),
                "mean": s.mean(), "median": s.median(),
                "p25": s.quantile(0.25), "p75": s.quantile(0.75), "p90": s.quantile(0.90),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Parameter sensitivity sweep
# ---------------------------------------------------------------------------
def parameter_sensitivity(param_grid: list[dict], run_fn: Callable[[dict], pd.DataFrame],
                           pnl_col: str = "pnl_r") -> pd.DataFrame:
    """param_grid: list of parameter dicts to try, e.g.
        [{"lookback": 10}, {"lookback": 20}, {"lookback": 30}]
    run_fn: a function(params) -> trades DataFrame (typically a thin wrapper
        around run_backtest with those params plugged in to the signal
        generation). This function doesn't know or care what the params
        MEAN -- it just runs each one and tabulates the result, so it's
        reusable for whatever strategy gets picked. Required item: 'A
        strategy that only works at one setting is a curve fit' -- this is
        what makes that check mechanical instead of eyeballed."""
    rows = []
    for params in param_grid:
        trades = run_fn(params)
        stats = trade_stats(trades, pnl_col=pnl_col)
        rows.append({**params, **stats.to_dict()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Smoke test with fabricated trades -- no network, no real strategy.
    rng = np.random.default_rng(42)
    n = 60
    times = pd.date_range("2026-01-01", periods=n, freq="3D", tz="UTC")
    pnl_r = rng.normal(loc=0.15, scale=1.0, size=n)  # slight positive edge, noisy
    trades = pd.DataFrame({
        "entry_time": times, "exit_time": times + pd.Timedelta(hours=6),
        "pnl_r": pnl_r,
        "mae_r": np.abs(rng.normal(0.5, 0.3, n)),
        "mfe_r": np.abs(rng.normal(0.8, 0.4, n)),
        "is_closed": True,
    })

    stats = trade_stats(trades)
    print("TradeStats:", stats)
    assert stats.n_trades == n
    assert 0 <= stats.hit_rate <= 1

    equity = equity_curve_from_trades(trades, risk_per_trade_usd=500, starting_equity=100_000)
    assert equity.iloc[-1] > 0
    sharpe = sharpe_from_equity_curve(equity, periods_per_year=365)
    print("Sharpe (365 annualization):", sharpe)

    dd = max_drawdown(equity)
    print("Drawdown:", dd)
    assert dd.max_drawdown_pct >= 0

    costs = cost_sensitivity(trades, base_cost_r=0.05)
    print("\nCost sensitivity:\n", costs[["scenario", "cost_r", "n_trades", "expectancy_r"]])
    assert costs.loc[costs.scenario == "2x_cost", "expectancy_r"].iloc[0] < \
           costs.loc[costs.scenario == "base_cost", "expectancy_r"].iloc[0]

    in_s, out_s = train_test_split_by_date(trades, "2026-04-01")
    assert len(in_s) + len(out_s) == n

    wf = walk_forward_correlation(trades, n_folds=6)
    print("\nWalk-forward:", wf)

    dist = mae_mfe_distribution(trades)
    print("\nMAE/MFE distribution:\n", dist)
    assert set(dist["group"]) <= {"winners", "losers"}

    sweep = parameter_sensitivity(
        [{"noise_seed": s} for s in range(3)],
        run_fn=lambda p: trades,  # dummy: same trades every time for the smoke test
    )
    assert len(sweep) == 3

    print("\nPASS: measurement.py smoke test")
