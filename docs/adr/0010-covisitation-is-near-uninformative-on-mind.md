# ADR 0010 — Co-visitation is near-uninformative on MIND, and that is the F2 result

**Status:** accepted · **Date:** 2026-09-14

## Context

F2 calls item-item co-visitation "the workhorse of e-commerce retrieval, cheap enough to be a
permanent member of your candidate sources in Part I rather than only a baseline." Built to
the gate in ADR 0009 — pairs from a user's clicks across impressions, never within one, matrix
from train alone — it scored, on dev:

| | GAUC | NDCG@10 | MRR | ceiling | headroom used |
|---|---|---|---|---|---|
| co-visitation, 1h window | 0.5007 | 0.2778 | 0.2348 | 0.5443 | 1.6% |
| co-visitation, 24h window | 0.5016 | 0.2884 | 0.2483 | 0.5598 | 2.7% |
| random | 0.5007 | 0.2855 | 0.2449 | 1.0000 | 0.1% |

At 1h: identical GAUC to random, and NDCG *below* it. That pairing is not a weak result, it is
a silent one, and it took a measurement rather than a metric to see why.

### What was measured

`make coverage` sweeps the pairing window and reports reach rather than quality:

| window | edges | slates with any signal | GAUC ceiling | headroom |
|---|---|---|---|---|
| 1h | 46,149 | 8.86% | 0.5443 | 0.0443 |
| 6h | 100,986 | 11.38% | 0.5569 | 0.0569 |
| 24h | 150,140 | **11.96%** | **0.5598** | 0.0598 |
| 72h | 203,315 | 11.47% | 0.5574 | 0.0574 |

A slate where every candidate ties contributes exactly 0.5 to GAUC whatever the scores are, so
a model that reaches a fraction `s` of slates cannot exceed `0.5 * (1 - s) + s`. That ceiling,
less 0.5, is the entire budget available to a perfect ranker at that setting.

Three readings follow, and they are separable.

**The window is not the problem.** 1h → 24h quadruples the edges and moves slate reach from
8.86% to 11.96%. The ceiling moves 0.544 → 0.560. That is a plateau, not a mistuned parameter.
The 1h default was the manual's number and it was genuinely too small for this corpus — only
15.7% of train's 106,965 inter-impression transitions fall inside an hour (`make gap`) — but
correcting it does not change the conclusion.

**The model does not spend the budget it has.** At 1h it scored 0.5007 against a ceiling of
0.5443: `(0.5007 - 0.5) / 0.0443 ≈ 1.6%` of available headroom. Confined to only the slates it
could see, GAUC was ≈ 0.508 — barely above chance where it had anything to say at all.

**When it fires, it is usually wrong.** 0.72% of rows scored against 8.86% of slates, over
2,740,998 rows in 73,152 slates, is ≈ 3.0 scored candidates in an average 37-item slate. If the
clicked item is not one of those three, the model lifts three non-clicks above it. That is why
NDCG@10 lands below random while GAUC sits exactly on it.

**Widening to 24h moves the ranking metrics across random, and GAUC not at all.** The card now
in `docs/results.md` is the 24h one: NDCG@10 0.2884 against random's 0.2855, MRR 0.2483 against
0.2449, Recall@10 0.5246 against 0.5223 — marginally ahead on all three, where the 1h card was
behind on all three. GAUC stayed put at 0.5016.

That split is worth reading carefully rather than as an improvement. GAUC is a within-slate
comparison and is capped by the flat slates; NDCG and MRR are not capped the same way, because
a slate where every score ties still produces a ranking, and that ranking is resolved by row
order. So the 1h card's sub-random NDCG was partly an artefact of tie-breaking, and widening the
window traded some of it back. **The model is ahead of random by roughly one part in a thousand
while using 2.7% of its budget. Neither window supports a claim that co-visitation works here.**

**72h is worse than 24h despite 35% more edges.** The likely mechanism is `top_k = 50`
displacement: long-gap pairs accumulate on globally popular items and crowd specific neighbours
out of each item's list — popularity contamination inside the matrix. Not confirmed; see below.

### Why the corpus does this

Co-visitation needs dense repeat behaviour per user. MIND-small is one week, and `make gap`
measured a **median 765-minute gap** between a user's consecutive impressions: people check the
news once or twice a day rather than browsing in sittings. The whole corpus offers 106,965
inter-impression transitions to build a matrix from. That is not a tuning shortfall, it is the
shape of the data.

## Decision

1. **`COVISIT_MAX_GAP` defaults to 86400** (24h), chosen from the reach curve rather than from
   the manual. The point is not that it performs well there; it is that the window can no
   longer be the thing to blame.
2. **Record co-visitation as a measured null result and keep it in `docs/results.md`.** The row
   stays with the other baselines, scored by the same harness over the same rows.
3. **`make coverage` stays in the repo** as the tool that distinguishes "ranked badly" from
   "had nothing to rank with". It found this; a report card did not.

## Consequences — accepted downsides

1. **A row in the results table that reads as a failure.** Without this ADR beside it, "GAUC
   0.50" invites the reading that the implementation is broken. The tests in
   `models/tests/test_covisit.py` pin the arithmetic that says otherwise.
2. **The `top_k` displacement hypothesis is unconfirmed.** A probe at `top_k=200` for the 24h
   and 72h windows would settle it. Deliberately not run: it would refine the explanation of a
   plateau, not move the plateau.
3. **Part I inherits a candidate source that contributes almost nothing on this corpus.** Worth
   knowing before it is wired into a funnel as though it were the e-commerce workhorse.
4. **The conclusion is about MIND-small, not about co-visitation.** On a denser log — sessions
   with several clicks, repeat visits within the hour — the same code would likely behave as
   the manual describes. Nothing here licenses the claim that co-visitation is a weak method.

## Alternatives considered

- **Widen the window further.** Rejected on the measurement: reach plateaus by 24h and falls by
  72h. The curve is in the table above rather than in an assertion.
- **Report only the best window and drop the curve.** Rejected: the curve is the evidence that
  the null result is a property of the corpus rather than of one arbitrary setting.
- **Drop co-visitation from the table entirely.** Rejected. A baseline removed because it lost
  is how a results table becomes a sales document; and Part I needs to know this before
  treating it as a candidate source.
- **Build the matrix over train plus dev.** Rejected under ADR 0009: the label being predicted
  would be inside the edge that scores it, and the `<` on the context protects the lookup key
  while the looked-up value came from the future.
- **A time-indexed matrix**, `W_T` built only from pairs observed before `T` and advanced as
  `T` advances — structurally what `asof.py` already does for the hourly CTR series. This is
  the correct way to use dev-week pairs without leaking, and it would raise reach legitimately.
  Deferred, not rejected: it is a materially larger build than F2 specifies, and it belongs
  with Part I's candidate sources rather than with the baselines.
