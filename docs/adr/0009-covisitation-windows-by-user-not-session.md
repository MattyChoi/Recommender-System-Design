# ADR 0009 — Co-visitation windows by user and time gap, not by `session_id`

**Status:** accepted · **Date:** 2026-09-14

## Context

F2 specifies item-item co-visitation, with a signature that takes sessions:

> `def build_covisitation(sessions, max_gap_seconds=3600, top_k=50):`
> `    for events in sessions:                        # sorted by ts`

The step's gate is right and is not in dispute: pairs must come from **clicks within a
user's behaviour**, never from items co-occurring in one impression. MIND's impression lists
were chosen by Microsoft's recommender, so counting within-impression co-occurrence measures
the incumbent system rather than user intent. And the matrix must be built from **train
only**, because a co-visitation matrix computed over the whole corpus leaks inside the model,
where no split test can see it.

What is in dispute is the partition. `session_id` is cut by `sessionize.py` at
`session.gap_minutes` and written to `silver/impressions` — and it survives into
`gold/training_examples`, because `attach_point_in_time_features` only ever adds columns to
its labels frame and the as-of joins never project the label side down. Verified from the
parquet footers, it is column four of both splits:

```
impression_id  user_id  user_idx  session_id  item_id  item_idx  clicked  slot  ts  ...
```

So anyone building F2 from gold — the source every other model in the harness is fitted on —
finds `session_id` already in the frame they are holding. Using it looks free, and it is the
obvious move twice over.

**Measuring the corpus showed it is not free, and that the session is the wrong unit here.**
`make gap` over train's 106,965 inter-impression transitions (33,617 users with two or more
impressions):

| statistic | value |
|---|---|
| median gap between a user's consecutive impressions | **765 minutes (12.75 h)** |
| p25 | 128 minutes |
| transitions kept in-session at `gap_minutes: 30` | **9.9%** |
| share/min under 5m → past 4h | 0.44% → 0.03%, smooth decay, **no elbow** |

MIND users check the news once or twice a day; they do not browse in sittings. At 30 minutes
roughly 90% of transitions are session boundaries, so sessions are overwhelmingly singletons,
and cross-impression click pairs — the only pairs the gate permits — come from the ~10,600
intra-session transitions that remain. Over a 65,238-item catalogue that is not a matrix.

Raising the threshold buys pairs at the cost of the word. At 1,440 minutes 72.3% of
transitions stay in-session, but `session_id` then means "same day" for every other consumer
of that column, and the distribution offers no principled value to raise it *to* — the decay
is smooth, so any threshold is arbitrary.

## Decision

**Do not partition co-visitation by `session_id`.** `build_covisitation` windows by
`user_id` ordered by `ts`, and bounds pairs with two explicit caps of its own:

- `max_gap_seconds` — the real elapsed time between two clicks, which the manual's decay term
  already uses, now also doing the work of the boundary.
- a **rank-distance cap** — the Spark equivalent of the manual's `events[i + 1 : i + 30]`
  slice, without which one heavy user's click run explodes into O(n²) pairs.

Pairs are formed **across impressions only**. Within one impression this is forced as well as
forbidden: `to_events` posexplodes the impression list and broadcasts that impression's single
`time` to every row, so all rows in an impression share a `ts`, every intra-impression gap is
zero, and the decay weight and the forward/backward asymmetry both degenerate.

`session.gap_minutes` stays at 30 for its other consumers, now documented with this
measurement rather than a TODO. The `session_id` column stays in gold: it is cheap, it is
correct for what it claims to be, and Part H's sequence features may yet want it. It is
simply not what co-visitation windows on.

**The matrix is fitted on `gold/training_examples/train`**, not on silver, so every model in
the results table is fitted on identical rows. The cost is measured and small — gold drops
the warm-up impressions, where neither the item nor its category had a closed feature bucket
yet:

| split | silver rows | gold rows | dropped |
|---|---|---|---|
| train | 5,843,444 | 5,837,182 | 6,262 (**0.11%**) |
| dev | 2,740,998 | 2,740,998 | 0 |

0.11% of the earliest train clicks is a smaller price than fitting one baseline on a
different row set than the rest of the table.

## Consequences — accepted downsides

1. **A visible departure from F2's signature.** The `sessions` argument is gone. Anyone
   diffing against the guide will notice; that is what this ADR is for.
2. **`max_gap_seconds` is now a hyperparameter with no default from the data.** The gap
   distribution has no knee to read one off. It must be swept and reported as a curve, the
   way F1 swept the half-life — a lone tuned value here would be a number someone picked.
3. **The window can now cross a session boundary**, by construction. Two clicks 40 minutes
   apart are a pair under a 1-hour cap even though `sessionize` calls them separate sessions.
   Given the measured distribution this is the point rather than a defect, but it does mean
   co-visitation and any future session-scoped feature see different neighbourhoods, and the
   two must not be described as using "the same sessions".
4. **The rank-distance cap makes the matrix order-dependent at the tail.** A user with a long
   run of clicks contributes only pairs within the cap, so the pairs that survive depend on
   where the run is truncated. This is the same trade the manual's slice makes; it is
   recorded here so it is not rediscovered as a bug.

## Alternatives considered

- **Fit the matrix on silver instead of gold**, recovering the 6,262 warm-up rows. Rejected:
  it buys 0.11% more train clicks at the cost of making co-visitation the one model in the
  results table fitted on a different row set. When a baseline later beats or loses to it by
  a point, nobody should have to ask whether the row sets differed.
- **Partition by `session_id` at `gap_minutes: 30`.** Rejected: ~10,600 candidate pairs over
  65,238 items is too sparse to be a model, and the sparsity would be invisible in the
  metrics — it would read as "co-visitation is weak on news" rather than "the window was
  wrong".
- **Raise `gap_minutes` to 1,440 and keep the session partition.** Rejected: it silently
  redefines `session_id` as "day" for every other consumer, requires a full silver rebuild,
  and picks a threshold the distribution does not support.
- **Add a second, co-visitation-specific session cut in silver.** Rejected: two `session_id`
  columns in one table is a naming problem that outlives whatever it solves, and the cut adds
  nothing a direct gap cap does not already do.
- **Use MIND's `history` column for pairs.** Rejected under ADR 0008: it is an unordered,
  untimestamped snapshot whose cut-off is not stated, so pairs drawn from it may encode
  dev-week clicks. That is exactly the leak F2's gate warns about, in the one place no split
  test would catch it.
