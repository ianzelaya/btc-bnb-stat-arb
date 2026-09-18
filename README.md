# BTC/BNB statistical arbitrage — research repo

A research-process showcase: a BTC/BNB statistical-arbitrage signal, built
and validated two ways — trading BTC only (via CME Micro Bitcoin futures /
IBKR), and trading both legs as a real hedged pair (Binance-only, since no
other major venue offers BNB derivatives). Every result here is produced by
the same honest-measurement discipline: in-sample/out-of-sample splits,
walk-forward folds, cost-sensitivity checks, and an explicit record of what
was tried and rejected along the way (`graveyard.md`), not just the winning
numbers.

## Real filter vs. ADX filter — read this first

Each of the two variants (single-leg, two-leg) appears in **two folders**:

- **`single_leg_real/` and `both_legs_real/`** run this research's actual
  regime filter — a momentum filter developed and refined across multiple
  strategies, not specific to this repo. Its code and the underlying
  trade-level journal aren't published here, since it's proprietary. These
  two folders contain only the honest-measurement **results** (headline
  stats, cost sensitivity, walk-forward, MAE/MFE, the equity curve) and are
  not runnable — there's nothing to run, since the filter itself is absent.
- **`single_leg_adx/` and `both_legs_adx/`** are fully public and runnable,
  using a standard ADX (Average Directional Index) regime filter in place
  of the real one. Every number in these two folders can be independently
  reproduced by anyone who clones this repo — instructions below.

The two results sit close to each other (see the comparison table below),
which is itself a finding worth stating plainly: most of the edge lives in
the hedge-ratio/spread mechanism and the entry/exit thresholds, not in the
specific choice of regime filter. The real filter is a genuine refinement
on top of that, not the whole story — which is exactly why it's reasonable
to publish a fully working, fully verifiable variant that doesn't depend
on it at all.

`graveyard.md` (research history, including failed approaches) and all
shared infrastructure (`indicators.py`, `backtest_engine.py`,
`measurement.py`, `costs.py`, `data_fetch.py`, `tests/`) are identical
across all four folders — they document and power the shared mechanism,
not the regime filter specifically.

## The thesis

BTC and BNB are both large, liquid crypto assets with a persistent, high
correlation — and there's a structural reason to expect that correlation
to hold, not just a historical accident. Both are widely-held instruments
whose price action is dominated by the same broad factor: overall crypto
risk appetite, liquidity conditions, and macro sentiment toward the asset
class as a whole. Most of BNB's and BTC's day-to-day variance comes from
that shared factor, not from anything specific to either token.

But BNB also carries a real idiosyncratic component BTC doesn't share: it's
Binance's own exchange token, so its price is additionally driven by
factors specific to Binance as a platform — trading volume and fee-burn
dynamics, exchange-specific news and incentive programs, BNB's periodic
token burns, and the general health of the Binance venue itself. These
drivers have nothing to do with BTC.

That combination — a strong, structural common factor plus episodic,
BNB-specific noise on top of it — is exactly what a pairs trade wants. A
rolling hedge ratio removes the common factor; what's left (the spread) is
mostly the idiosyncratic component. Absent a genuine structural break (a
real, business-continuity-level threat to Binance itself, not routine
exchange news), that idiosyncratic component should be temporary: an
event-driven move away from BNB's typical relationship to BTC, which fades
as the shock passes and price re-aligns. That's the mean-reversion bet this
strategy is actually making — and also exactly its main structural risk
(see `graveyard.md` and the risk discussion baked into the sizing/stop
design below): if that idiosyncratic driver ever becomes a permanent
repricing instead of a temporary shock, the position has no way to
distinguish "spread is extreme and about to revert" from "the relationship
just broke."

## Strategy components

**Signal.** A rolling OLS hedge ratio (`beta`, `cov(BNB,BTC)/var(BTC)`)
defines a spread, `BNB_close - beta * BTC_close`. The spread's own rolling
empirical percentile rank is the entry/exit signal: unusually low (bottom
`low_q`) enters long BNB / short BTC, betting the spread reverts up;
unusually high (top `high_q`) enters the opposite side. Exit is a
percentile crossback through `mid_q`, or a hard time-stop
(`max_hold_bars`), whichever comes first.

**Regime filter.** Entries are only allowed when the regime filter says
BTC isn't in a strong one-sided trend, since the hedge ratio — and the
whole mean-reversion premise — is least reliable exactly when BTC itself
is moving violently in one direction. The `_adx/` folders use ADX (Wilder's
trend-strength indicator, unsigned, 0-100) with a ceiling threshold; the
`_real/` folders' results were produced with a different, undisclosed
momentum filter (see above).

**Sizing and risk — genuinely different between the two variants.**
Single-leg execution sizes off a fixed risk-dollar convention ($500/trade
on a $100k account, matching this research's Part A convention) with an
ATR-based stop (validated against a fixed-percentage alternative — ATR won,
see `graveyard.md`). The two-leg variant is sized completely differently:
both legs are sized against a fixed capital allocation (`CAP = $550,000`)
with `qty_btc = beta_at_entry * qty_bnb` — the unit-count relationship that
makes combined P&L actually track the spread (not a dollar-notional split,
which was an earlier, incorrect draft — see `graveyard.md`). Its stop is a
maximum combined-P&L loss as a fraction of that capital, not an ATR
distance — an ATR-based, single-leg-price stop doesn't make sense for a
hedged pair (it can force-close a position that's net flat or profitable
because one leg alone crossed a price level; see `graveyard.md` for the
concrete case this was caught on).

**Costs.** Single-leg assumes IBKR's flat per-contract CME Micro Bitcoin
futures fee (an explicitly labeled estimate, not confirmed to the cent —
see `costs.py`). Two-leg assumes Binance VIP3-tier fees (no BNB discount)
plus funding across every ~8h settlement a trade spans, applied per leg
since sizing isn't equal-notional (funding doesn't net between legs) — see
`costs.py` for both.

**Measurement.** Every result in this repo (real and ADX alike) goes
through the same checklist: trade stats and Sharpe (stated annualization),
max drawdown, a stated cost assumption plus 2x-cost sensitivity, an
in-sample/out-of-sample split, 6-fold walk-forward correlation, and
MAE/MFE distributions split by winners/losers — see `measurement.py`.

## Results comparison

| | single_leg_real | single_leg_adx | both_legs_real | both_legs_adx |
|---|---|---|---|---|
| Trades (full) | 883 | 1,270 | 560 | 681 |
| Hit rate | 48.4% | 49.1% | 52.0% | 43.2% |
| Payoff ratio | 1.32 | 1.29 | 1.44 | 1.95 |
| Sharpe — full | 0.83 | 0.85 | 1.22 | 1.11 |
| Sharpe — IS | 0.75 | 0.81 | 1.23 | 1.15 |
| Sharpe — OOS | 1.26 | 1.09 | 1.82 | 1.38 |
| Max drawdown | 5.2% | 5.5% | 20.2% | 24.3% |

Single-leg uses a $100k/0.5%-risk account convention; two-leg uses a
$550,000 capital sleeve — dollar returns aren't comparable across that
divide, but Sharpe/drawdown are. Full detail (cost sensitivity,
walk-forward, MAE/MFE) is in each folder's own summary — `*_summary.json`
for the `_adx/` folders, `README.md` for the `_real/` folders.

## How to run the ADX variants

```bash
pip install -r requirements.txt
```

Historical BTC/BNB futures klines are already cached under `data/`
(2020-01-01 through 2026-09-14), so the commands below run offline unless
you extend the date range. Every script below can be run from any
directory — paths resolve relative to the script's own location, not your
shell's current directory.

**Run the offline test suite** (58 pure-logic tests, no network):
```bash
pytest tests/
```

**Single-leg (BTC-only) variant:**
```bash
python single_leg_adx/run_backtest_btc_only.py
```
Writes `single_leg_adx/journals/btc_only_backtest_journal.jsonl` and
`single_leg_adx/btc_only_summary.json`. To regenerate the equity curve:
```bash
python plot_equity_curve.py single_leg_adx/journals/btc_only_backtest_journal.jsonl single_leg_adx/equity_curve_btc_only.png "BTC-only variant (ADX): cumulative equity, IS vs OOS"
```

**Two-leg pairs variant:**
```bash
python both_legs_adx/run_backtest_pairs.py
```
Writes `both_legs_adx/journals/pairs_backtest_journal.jsonl` and
`both_legs_adx/pairs_summary.json`. To regenerate the equity curve:
```bash
python plot_equity_curve.py both_legs_adx/journals/pairs_backtest_journal.jsonl both_legs_adx/equity_curve_pairs.png "BTC/BNB pairs (ADX): cumulative equity, IS vs OOS" 550000 550000 pnl_r
```

**Re-running the parameter searches** (each takes roughly 15-40 minutes,
fetches fresh data on a first run if the cache doesn't cover the requested
date range):
```bash
python single_leg_adx/optimize_btc_only.py
python both_legs_adx/optimize_pairs.py
```
Both are staged (cartesian threshold sweep, then a full cross against
regime/risk/lookback parameters), selecting by `min(in-sample Sharpe,
out-of-sample Sharpe)` throughout — see each script's own docstring for the
exact grid and the reasoning behind it.

## Repo layout

```
backtest_engine.py     -- single-instrument vectorized backtest engine
indicators.py          -- shared indicator primitives (ATR, RSI, ADX, beta, percentile rank)
measurement.py         -- honest-measurement toolkit (Sharpe, drawdown, walk-forward, MAE/MFE, cost sensitivity)
costs.py                -- cost models (IBKR flat fee, Binance retail/VIP3, funding)
data_fetch.py            -- Binance klines + funding-rate fetch/cache
plot_equity_curve.py     -- equity curve + drawdown chart, either variant
data/                    -- cached BTC/BNB klines + funding history (2020-2026)
tests/                   -- offline test suite (58 tests, no network)
graveyard.md              -- research history: what was tried and rejected, and why
single_leg_adx/           -- BTC-only variant, public ADX regime filter, fully runnable
single_leg_real/          -- BTC-only variant, real regime filter, results only
both_legs_adx/             -- two-leg pairs variant, public ADX regime filter, fully runnable
both_legs_real/            -- two-leg pairs variant, real regime filter, results only
```
