# ADR 0001 — Dataset and stack

**Status:** superseded by [ADR 0004](0004-mind-and-pyspark.md) · **Date:** 2026-08-30

> Superseded on 2026-08-30. The dataset moved to MIND and the processing engine to
> PySpark. Consequences 1, 2 and 4 below no longer apply. The text is kept unedited
> as the record of what was decided and why — see ADR 0004 for what replaced it.

## Context

The project targets Tier 2 scope: multi-source retrieval, DCNv2 + MMoE ranking, a feature
store with point-in-time joins, Go gRPC serving, and a full evaluation harness. Tier 2
normally pairs that scope with MIND or Amazon Reviews 2023 at real scale.

## Decision

Use **MovieLens-25M as the only dataset**, for both the development loop and final numbers.
Stack per the guide's §1 defaults: Python 3.13, PyTorch, LightGBM, FAISS, Feast, Kafka,
Redis, Go serving, Docker Compose + kind, GitHub Actions, Prometheus/Grafana.

## Consequences — accepted downsides

These are real and should be stated in the README rather than discovered by an interviewer:

1. **No impression logs.** MovieLens gives positives only, so every negative is sampled
   rather than observed. Offline calibration is therefore semi-fictional, and the
   position-bias correction has no ground truth to correct against. Mitigation: be explicit
   about the sampling distribution, apply logQ correction, and describe what would change
   with real impressions.
2. **62K items is below the scale where ANN is necessary.** The guide targets 100K–2M items
   precisely so that brute-force search is too slow to be viable. At 62K × 128 dims, exact
   search is a few milliseconds, so HNSW is a demonstration rather than a requirement. The
   recall/QPS benchmark is still worth building, but it must be labelled as such.
3. **Ratings, not a true funnel.** No click/cart/order equivalent, which weakens the
   multi-task (MMoE) story — the tasks have to be synthesised (e.g. watch vs. high-rating)
   rather than observed.
4. **Overexposure.** MovieLens is the most common recsys portfolio dataset. The engineering
   — multi-stage funnel, latency budget, skew report — has to carry the novelty, because
   the dataset will not.

## Alternatives considered

- **MIND** — has real impression logs, which fixes consequence 1 outright and is the single
  highest-value property available in a public dataset. Rejected for now on scope grounds.
- **Amazon Reviews 2023** — 48M items, fixes consequences 2 and 4. Rejected: requires Spark
  at real scale and carries heavy skew.

## Revisit when

The serving path is green end-to-end. Swapping the dataset later is comparatively cheap if
the pipeline is written against a schema rather than against MovieLens columns — so **write
the ingest layer to a contract from day one** to keep that door open.
