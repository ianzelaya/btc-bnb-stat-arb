# The graveyard

What follows is a record of the approaches that were tried and rejected
along the way to the strategies in this repo, not just the winners. A
result is only as credible as the failures visible around it -- without
this, there's no way to tell a genuine finding from one config that got
lucky out of thousands tried.

Template per entry:

- **What it was:** the idea, the parameter, the twist.
- **Why it seemed promising:** the thesis before testing it.
- **What actually happened:** the number that killed it.
- **Why, in hindsight:** the mechanism, if one could be identified.

---

## Naively reusing one leg's tuned parameters on the other

- **What it was:** the BTC/BNB spread signal is one construction (rolling
  hedge ratio, percentile-rank entries, a BTC-only regime filter) that can
  be executed two ways -- trading only the BTC leg (`strategy_btc_only.py`,
  this venue's only option since it can't access BNB derivatives), or both
  legs as a real pair (`strategy_pairs.py`). The first attempt at the
  BTC-only variant simply reused a parameter set that had already been
  tuned and confirmed working for the two-leg construction, isolating just
  the BTC side of those same trades.
- **Why it seemed promising:** it's the identical signal, just measuring
  one leg's P&L instead of both -- looked like a small, safe simplification
  rather than a real change.
- **What actually happened:** a 635-run one-at-a-time sweep anchored at
  those parameters found zero in-sample-positive combinations. Reconstructing
  both legs' P&L on the same trades explained why: at those settings, the
  BNB leg alone had a 71.8% hit rate (+0.254%/trade) while the BTC leg alone
  had a 13.6% hit rate (-0.054%/trade), on identical entries and exits --
  the edge was real (confirming the reconstruction had no bug), but it lived
  almost entirely in the BNB leg's reversion, not in BTC's counter-move.
- **Why, in hindsight:** a one-at-a-time sweep anchored at an existing
  point can only find small local improvements around that point -- it
  can't discover a genuinely different region of the search space. A proper
  staged search (cartesian over the threshold trio first, stop and regime
  band opened wide to remove confounds, then crossing the winning
  thresholds against regime/stop/beta/lookback) found a completely
  different, robust region for BTC-only execution (`low_q`/`high_q` 0.02/
  0.70 vs. the two-leg variant's own thresholds, `beta_window=200` vs. a
  materially different value) -- 1,288 of 1,584 combinations in that final
  stage were positive in-sample AND every one of those was also positive
  out-of-sample. Lesson: when the same signal moves to a different
  execution configuration, the working parameter region can be an entirely
  different region, not a small perturbation of the old one.

## Choosing a stop-loss for single-leg execution: ATR vs. a fixed percentage

- **What it was:** the original spread construction has no stop-loss at
  all. Adding one to the BTC-only variant, two mechanisms were built and
  swept head-to-head: an ATR-multiple stop (adapts to BTC's own recent
  volatility) and a fixed-percentage stop (a constant distance regardless
  of volatility regime).
- **Why it seemed promising:** a fixed-percentage stop is simpler to reason
  about and explain, and there was no a priori reason to assume a
  volatility-adaptive stop would necessarily win for this signal
  specifically -- worth actually testing instead of assuming.
- **What actually happened:** across the sweep, fixed-percentage stops
  averaged a Sharpe of -0.67 (65 candidates), ATR-based stops averaged
  -0.07 (525 candidates) -- and the best individual candidate from each
  mode told the same story (ATR's best: Sharpe 0.89; fixed-percentage's
  best: Sharpe 0.72, but with a badly negative out-of-sample Sharpe next to
  it, unlike ATR's best candidate). ATR won on both the average and the
  best case.
- **Why, in hindsight:** BTC's realized volatility isn't remotely stable
  across a 2020-2026 sample -- a fixed-percentage stop that's sensible in a
  quiet regime is either far too tight (stopping out normal noise) or far
  too loose (barely a stop at all) once the regime shifts, while an
  ATR-based stop rescales itself with the regime automatically. The same
  underlying lesson resurfaces later for the two-leg variant's own stop --
  see below -- in the opposite direction: an ATR-based stop is the right
  choice for a single instrument, but the wrong one for a hedged pair.

## Chasing higher trade frequency on the single-leg signal

- **What it was:** the locked BTC-only config trades a modest few times a
  week. Explicit follow-on question: could the same signal trade
  meaningfully more often, for higher annualized return, without giving up
  the edge? Four staged attempts, always selecting in-sample and confirming
  out-of-sample, never selecting on out-of-sample data directly.
- **Why it seemed promising:** the parameter-porting entry above already
  showed this signal's working region isn't always a small perturbation of
  an existing anchor point -- worth checking whether the same held across
  trade frequency and timeframe, not just across execution configuration.
- **What actually happened:** genuinely mixed. Forcing higher frequency at
  the same (15-minute) bar resolution roughly doubled trade count but
  collapsed in-sample Sharpe from the best unfiltered candidate's 0.80 down
  to 0.42, and annualized return fell too -- loosening thresholds diluted
  the edge faster than the added volume compensated. Moving to a finer
  (5-minute) timeframe while holding the hedge-ratio/lookback windows at a
  time-equivalent scale was categorically worse (only 1.4% of candidates
  were positive both in- and out-of-sample, vs. 42.1% at 15 minutes).
  Letting those lookback windows float freely instead of holding them
  time-equivalent, then re-sweeping the full threshold grid around the new
  lookback point, finally found a real, robust higher-frequency region at 5
  minutes -- but every genuinely robust candidate in that region traded at
  a lower Sharpe than the already-locked 15-minute config, a real tradeoff
  (more frequency and higher raw return, for a rougher ride per unit of
  return) rather than a clean upgrade.
- **Why, in hindsight:** the same lesson as the parameter-porting entry,
  one level deeper -- a change along any dimension (execution
  configuration, timeframe, target frequency) can require every other
  parameter to move with it, not just the one being explored directly.
  Holding the lookback windows fixed while only changing timeframe would
  have wrongly concluded "5-minute bars don't work here"; only letting them
  re-adapt, and then re-sweeping thresholds around the new point, revealed
  the real higher-frequency region -- it just wasn't the original lookback
  scaled by the timeframe ratio.

## The first two-leg search was cost-blind

- **What it was:** the first attempt at re-optimizing the two-leg
  (`strategy_pairs.py`) construction ranked every candidate config purely
  by raw Sharpe on pre-cost P&L (archived as `optimize_pairs_stage1_gross_v1.csv`
  / `optimize_pairs_stage2_gross_v1.csv` / `optimize_pairs_gross_v1.log`).
- **Why it seemed promising:** it's the most direct read of "which
  parameters produce the best trading edge," and ranking by gross Sharpe is
  a completely standard first move.
- **What actually happened:** the winning candidate looked strong gross
  (Sharpe 1.84 in-sample) but traded very frequently at a tight stop
  (2,158 trades, a 6x-ATR-equivalent stop distance in the search space at
  the time). Its real edge (0.043R gross, in the search's original risk
  convention) was smaller than its own mean per-trade cost (0.061R) --
  net expectancy went NEGATIVE once realistic exchange costs were applied,
  despite the strong gross number.
- **Why, in hindsight:** a tighter stop mechanically shrinks the risk unit
  a trade is sized against, which inflates cost as a fraction of that risk
  unit -- exactly the kind of interaction a gross-Sharpe-only objective
  can't see, because it never looks at cost at all during the search, only
  after picking a winner. The fix was mechanical, not conceptual: bake the
  cost model directly into the objective the search ranks candidates by,
  so a config can't win by finding volume it can't actually afford to
  trade.

## The sizing behind the two-leg search wasn't actually a hedge

- **What it was:** even after the cost-blind objective above was fixed,
  the two-leg backtest sized each leg independently -- equal risk dollars
  split 50/50, each leg's own quantity set by its own ATR-based stop
  distance, with no reference to the hedge ratio the entry/exit signal
  itself is built on.
- **Why it seemed promising:** it's the natural generalization of treating
  each leg as its own risk-managed position, and "split risk evenly across
  both legs of a pair" sounds like a reasonable default in isolation.
- **What actually happened:** working through exactly how the two legs are
  supposed to offset each other showed this sizing had no relationship to
  the hedge ratio at all. The position that actually makes combined P&L
  track the spread (rather than the shared price factor both legs are
  exposed to) requires the BTC leg's quantity to equal the hedge ratio
  times the BNB leg's quantity -- a unit/coin-count relationship, not a
  dollar-notional one. The equal-risk-dollar convention satisfied neither.
  Confirmed directly (see `tests/test_two_leg_backtest.py`): under the
  corrected sizing, combined gross P&L for a single trade is provably
  identical to the hedge ratio times the spread's own price move, exactly
  as the construction requires; the old sizing had no such property.
- **Why, in hindsight:** "risk-managed independently" and "hedged as a
  pair" are different design goals that happen to look similar on the
  surface -- the first optimizes each leg's own risk in isolation, the
  second requires the two legs' sizes to be locked together by the same
  relationship that defines the trading signal itself. Every result
  produced under the old sizing had to be discarded once this was caught,
  not adjusted -- the entire two-leg backtest engine needed rebuilding
  around capital-allocated, hedge-ratio-derived sizing instead.

## An ATR-based, price-level stop doesn't fit a hedged spread trade

- **What it was:** carried over from the single-leg construction (where it
  won the head-to-head comparison above), the two-leg backtest's stop was
  originally an ATR multiple on the BTC leg's own price, checked against
  that leg's own high/low.
- **Why it seemed promising:** it had already been validated as the right
  choice for single-instrument execution, and reusing a proven mechanism
  seemed lower-risk than designing a new one from scratch.
- **What actually happened:** two separate problems, both confirmed
  directly rather than just argued for (see `tests/test_two_leg_backtest.py`).
  First, a price-level stop on one leg alone can force-close a position
  that is net flat or even profitable overall -- constructed a case where
  the BTC leg alone moved sharply enough to have crossed any reasonable
  stop, while the BNB leg moved in exactly the proportion the hedge ratio
  predicts, leaving combined P&L within a dollar of zero; the old
  mechanism would have closed that trade anyway, because it never looked
  at combined P&L at all. Second, sizing a hedged pair off a fixed capital
  allocation (rather than a fixed risk-dollar target) means the ATR
  distance alone, not position size, ends up determining dollar
  loss-at-stop -- and that swung by roughly 10x across this project's own
  price history purely from BTC's volatility regime changing, not from any
  deliberate risk choice.
- **Why, in hindsight:** a stop that looks at one leg's price level is
  answering "how far has BTC moved," but what a hedged pair is actually
  risking is "how far has the spread moved" -- those are the same question
  for a single instrument and genuinely different questions for a hedged
  one. The stop was rebuilt to check combined dollar P&L against a
  fraction of the pair's allocated capital instead -- tied to the thing
  actually being risked, and stable across price regimes by construction
  rather than by coincidence.
