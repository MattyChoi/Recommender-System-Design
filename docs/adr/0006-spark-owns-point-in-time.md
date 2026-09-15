# ADR 0006 — Spark owns point-in-time; Feast owns online serving

**Status:** accepted · **Date:** 2026-09-12

## Context

Two systems in this repo can perform a point-in-time feature join, and only one of them
does.

`data_pipeline/features/asof.py` implements the offline join directly in Spark: union
labels with features, order by `(_ts, _is_label)` over
`rowsBetween(unboundedPreceding, 0)`, and carry the last non-null value forward. Four
joins — item, user, user×category, category. It is the join D4's leakage gate is written
against, and the one that produced the measured 6,262 rows dropped as unknowable.

Feast also has an offline store (`DaskOfflineStore`, via `type: dask`) whose
`get_historical_features` does the same job. **It is never called.** Feast is used only to
materialise the gold series into Redis — 1,110,343 keys — for online reads.

The framing "two implementations of point-in-time correctness, one of them untested" is
the obvious reading and it is wrong. At **serving** time there is no point-in-time
problem: the request timestamp *is* now, and the freshest materialised value is correct by
construction. Reconstructing what was knowable at a past instant is a problem that exists
only offline. So one PIT implementation is the right number, not a shortfall.

What the split actually creates is a **freshness** seam, not a correctness one. Training
reads "the last bucket that closed before `ts`". Serving reads "the most recent bucket
materialised, subject to the FeatureView TTL". Those differ by materialisation lag, and
that difference is what D4 measures.

Both paths read the **same** gold Parquet. One producer, two readers.

## Decision

**Spark's `asof_join` is the only offline point-in-time join. Feast is the online store
and nothing else.** `get_historical_features` is not used for training, now or later.

Three reasons, in order of weight:

1. **Feast's TTL is a hard lookback bound on its offline join.** `item_stats` has
   `ttl=2h`, so a label with no item bucket in the preceding two hours would come back
   null. Our as-of join looks back unboundedly and then handles absence explicitly —
   category-prior imputation, `has_*_features` flags, and the 6,262-row drop. The two
   would not produce the same training set, and ours is the one that is measured.
2. **Scale.** The dask offline store is a single-machine pandas-family engine. The label
   set is millions of rows joined against four feature series; this is what Spark is for.
3. **Training would gain a dependency on the registry.** The registry is recompiled by
   every `feast apply` and bakes in whatever `storage.backend` was set at the time.
   Training sets should not be reproducible only relative to a mutable blob.

## Consequences — accepted downsides

1. **The Feast offline store is dead surface.** It is configured, it works, and nothing
   exercises it. A reader may reasonably ask why a feature store is present with half of
   it unused.
2. **Nothing automatically proves the two readers agree.** D4's skew report is the proof,
   and it is blocked on Part M, so the seam is currently unmeasured rather than
   measured-and-small.
3. **The feature list exists twice** — the constants in `asof.py` and the `Field`s in
   `definition.py`. `test_feature_store_schema.py` covers the shape; it does not cover
   someone adding a feature to training and forgetting the FeatureView.
4. **Context features are declared once but still computed twice.** `hour_of_day` and
   `day_of_week` now have a single written definition — the `context_features` on-demand
   view — and `test_context_features.py` drives it and Spark over the same instants and
   asserts they agree. Whether Part M's Go service calls that transformation or
   reimplements it against the spec is still open; until that is settled, agreement is
   enforced by a test rather than by construction.

## Alternatives considered

- **Use `get_historical_features` for training.** Rejected for the three reasons above.
  The strongest argument for it — one code path, so no seam — is weakened by the fact that
  the seam is between offline and *online* reads, which this would not remove.
- **Drop Feast, have Spark write Redis directly.** Simpler, and honest about what is
  actually used. Rejected: it discards the PushSource path that Part Q's Flink job needs,
  the TTL and entity-key semantics, and the FeatureService as a serving contract.
- **Make the two agree by construction, by materialising from the training join's
  output.** Rejected as circular: it would serve features stamped at training time rather
  than the freshest available, which is the wrong trade at request time.
