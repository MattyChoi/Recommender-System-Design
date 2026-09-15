# ADR 0004 — MIND and PySpark

**Status:** accepted · **Date:** 2026-08-30 · **Supersedes:** [ADR 0001](0001-dataset-and-stack.md)

## Context

ADR 0001 accepted MovieLens-25M as the sole dataset and named four downsides we agreed to
live with. It also named the condition under which to revisit: once the ingest layer was
written against a schema contract rather than against MovieLens column names, switching
would be cheap. We are revisiting before that ingest layer exists, which makes the switch
cheaper still — no consumer code has been written yet.

Separately, ADR 0001 left the processing engine unstated, and the working assumption was
Polars on the grounds that MovieLens-25M fits in memory. Tier 2 is partly a claim about
data processing at scale, and Polars does not support that claim.

## Decision

Use **MIND** (Microsoft News Dataset) as the dataset, and **PySpark** as the processing
engine. Evaluate primarily on MIND's official temporal train/dev/test split.

## What this resolves

Three of ADR 0001's four accepted downsides disappear:

1. **Impression logs are real.** MIND records what was shown and not clicked. Negatives are
   observed rather than sampled, so calibration stops being semi-fictional and the
   position-bias correction has ground truth to correct against. This was the single
   largest weakness in ADR 0001 and it is gone outright.
2. **Catalog size clears the ANN threshold.** ~160K articles sits inside the guide's
   100K–2M target, so the HNSW index is load-bearing rather than decorative. ADR 0002 can
   now be decided on real numbers.
4. **Overexposure is gone.** MIND is rarely used for portfolio projects, so the dataset no
   longer works against the project.

Consequence 3 only half-resolves. Click / no-click on a shown item is a genuine funnel, but
there is no second stage equivalent to cart-or-purchase, so the MMoE multi-task story still
requires constructed tasks rather than observed ones.

The Polars-versus-Spark gap closes as a side effect: the pipeline now runs on the engine
Tier 2 is claiming credit for.

## New consequences — accepted

1. **News items go stale in hours.** Item cold-start is the normal case, not the exception.
   This shifts weight onto content features (category, subcategory, title, abstract — all
   of which MIND provides) and away from ID embeddings. It is a materially different
   modelling problem than MovieLens implied, and the README must say so.
2. **A JVM enters the toolchain.** Spark 4.x requires Java 17 or 21. `pip install pyspark`
   alone yields an import that succeeds and a SparkSession that dies, so the requirement
   has to be documented in `.env.example` and the README.
3. **The dev loop gets slower.** Spark's startup cost and shuffle overhead make iteration
   worse than in-memory Polars. Mitigated by developing against MIND-small and running
   final numbers on MIND-large.
4. **Domain mismatch with the project's name.** MIND is news; the project is titled
   "Movie Recommendation System". The engineering is domain-agnostic, but the naming is
   now inconsistent and one of the two should change. See "Open" below.
5. **String IDs require a persisted mapping.** MIND uses `U12345` / `N45678`. Embedding
   tables need contiguous integers, and an index derived independently at training and
   serving time will diverge — a training/serving skew bug that is hard to see. The
   mapping tables are built once in bronze and persisted.

## Alternatives considered

- **Amazon Reviews 2023, `Movies_and_TV`.** Keeps the movie framing and offers greater
  scale with rich metadata. Rejected because it has no impression logs, which is the
  property we are switching in order to obtain.
- **Stay on MovieLens, add one Spark job.** Would have closed the engine gap while leaving
  the dataset weaknesses in place. Rejected as the smaller half of the problem.
- **Polars with MIND.** Viable — MIND-small fits in memory — but forfeits the scale claim
  and would need rewriting for MIND-large anyway.

## Open

Whether to rename the project to reflect a news domain, or to accept the mismatch and
explain it. Decide before the README is written, since the README is the artifact a
reviewer reads first.
