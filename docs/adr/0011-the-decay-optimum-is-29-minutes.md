# ADR 0011 — The decayed-popularity optimum is ~29 minutes, and it is interior

**Status:** accepted · **Date:** 2026-09-14

## Context

F1 says of the half-life: *"Sweep it and report the curve -- it is a free ablation."* The
manual's own code defaults to 3 days and notes that "3 days is already long" for news. Swept on
dev, GAUC is **monotone across a 150× range** and the manual's default is the worst point on it:

| half-life | minutes | GAUC | NDCG@10 | MRR | headroom used |
|---|---|---|---|---|---|
| 0.02d | 29 | **0.5424** | **0.3241** | **0.2829** | 9.20% |
| 0.05d | 72 | 0.5415 | 0.3162 | 0.2711 | 9.01% |
| 0.1d | 144 | 0.5379 | 0.3150 | 0.2703 | 8.23% |
| 0.25d | 360 | 0.5333 | 0.3132 | 0.2687 | 7.23% |
| 0.5d | 720 | 0.5305 | 0.3117 | 0.2674 | 6.62% |
| 1d | 1440 | 0.5283 | 0.3103 | 0.2662 | 6.14% |
| 2d | 2880 | 0.5268 | 0.3095 | 0.2658 | 5.82% |
| 3d | 4320 | 0.5265 | 0.3096 | 0.2659 | 5.75% |
| recency (limit) | — | 0.5418 | 0.3196 | 0.2762 | 9.07% |

`gauc_ceiling` is 0.9607 and `flat_impressions` is 5,757 at **every** row including the limit,
so the half-life changes only the ordering and never the reach. The curve is comparable end to
end rather than being partly an artefact of ties.

### Why the grid could not find the answer

Each step down improved GAUC, and the gain was *accelerating*: the NDCG step from 0.05d to
0.02d (+0.0079) exceeds the entire span from 0.25d to 3d (−0.0036). The obvious move is to keep
halving. It does not work. As the half-life shrinks, `exp(-lambda * age)` collapses the sum onto
whichever click is most recent, so the model converges on a *different* model — rank by the
most recent knowable click, with no popularity in it at all.

And `exp(-x)` underflows to zero around `x > 745`. With ages up to ~14 days that bites below

```
half_life < ln2 * 14 / 745 ~= 0.013 days  (~19 minutes)
```

Past there, clicks that should weigh 1e-300 weigh exactly nothing, items tie that should not,
and a rising GAUC could be arithmetic rather than signal. 0.02d is the last safe point — which
is why `flat_impressions` is unchanged there — and 0.01d is already over the line.

## Decision

1. **Compute the limit instead of approaching it.** `score_most_recent` ranks by
   `1 / (1 + age_days)` of the item's most recent knowable click. Any strictly decreasing
   function of age gives the identical ranking; this one is bounded in (0, 1], keeps the
   "cold scores 0.0" convention, and needs no sentinel for an item nobody clicked — which a
   `-age` score would, and every sentinel is a number some real row can reach.
2. **Settle the comparison with a paired test, not point estimates.** `make compare` scores both
   models in one pass and bootstraps the per-user difference.
3. **`HALF_LIVES` spans the optimum**: `0.02 0.05 0.1 0.25 1 3`.
4. **`recency` is a baseline in its own right** and stays in `docs/results.md`. It is a real
   news baseline, not scaffolding for this measurement.

## The result

```
decayed_popularity@0.02  minus  recency
per-user NDCG@10 difference: +0.00453
95% CI: [+0.00380, +0.00519]  over 50,000 paired users
SIGNIFICANT
```

**The optimum is interior.** A 29-minute half-life beats the limit it is converging on. The
interpretation is specific: at 29 minutes the model is nearly pure recency, but a *second*
recent click still adds weight, and pure recency cannot tell "two clicks in the last hour" from
"one". That sliver of accumulated popularity survives all the way down.

The paired interval is **3.5× narrower** than either marginal (0.00139 against 0.00480 and
0.00500) and excludes zero where the marginals overlap heavily. Predicted before the run that it
would span zero, by reading the marginal widths — which is exactly the unpaired reasoning
`paired_bootstrap` exists to defeat. Recorded because the mistake was made while writing the
tool that prevents it.

## Consequences — accepted downsides

1. **A 29-minute half-life is a strange number to report**, and it will read as overfitting to a
   reviewer who has not seen the curve. The curve and the paired CI are the answer; the curve is
   monotone over 150×, not a spike at one grid point.
2. **It is tuned on dev, which is the evaluation split.** With MIND's official split there is no
   third partition to tune on, so this is an honest ablation rather than a held-out selection,
   and a lift of this size should not be quoted as a held-out result.
3. **The optimum's location is corpus-specific and probably horizon-specific.** It says
   something about MIND-small's one-week window, not about news in general.
4. **The exact optimum between 0.013d and 0.02d is unmeasured**, bounded below by underflow and
   above by the grid. Not pursued: the interval is narrower than the effect being discussed.

## Alternatives considered

- **Keep halving the grid.** Rejected: 0.01d is already past the underflow threshold, so the
  curve would stop being trustworthy exactly where it gets interesting.
- **Report 0.02d and recency as tied.** Rejected by measurement — that was the expected outcome
  and the paired test refuted it.
- **Use `float128` or log-space accumulation to push the grid lower.** Rejected: it would
  measure to more decimal places the approach to a limit that is now computed exactly.
- **Keep the manual's 3-day default.** Rejected: it is the worst point on the measured curve,
  0.0159 GAUC below the optimum — larger than the gap between most-popular and decayed
  popularity at 3 days.
