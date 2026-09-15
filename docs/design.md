# Design: multi-stage movie recommender

**Status:** draft, Phase 0. Written before any model, on purpose.

## 1. Problem statement

Increase session-level engagement (clicks per session) on the homepage rail **without
reducing catalog coverage@20 below 40%**. Optimising engagement alone collapses onto the
head of the catalog; the coverage floor is the guardrail that keeps the objective honest.

Primary metric: clicks per session.
Guardrail metrics: catalog coverage@20, intra-list diversity, p99 latency.

## 2. Scale assumptions

Stated explicitly even though this is a portfolio project, because the design only makes
sense against a target:

| Assumption | Value |
|---|---|
| Users | 10M |
| Items | ~160K (MIND news catalog) — inside the guide's 100K–2M ANN target, see ADR 0004 |
| Peak QPS | 500 |
| p99 end-to-end | < 100 ms |
| Retrain cadence | daily batch |
| Index refresh | hourly |

## 3. Funnel design

| Stage | Candidates in | Candidates out | Budget (p99) | Actual |
|---|---|---|---|---|
| Experiment assignment | — | — | 2 ms | `TODO` |
| Feature fetch (Redis, pipelined) | — | — | 8 ms | `TODO` |
| Retrieval fan-out (5 sources, parallel) | ~160K | 800–1200 | 25 ms | `TODO` |
| Filtering (seen, blocked, inventory, age-gate) | 800–1200 | 400–600 | 3 ms | `TODO` |
| Ranking (batch 400) | 400–600 | 100 | 35 ms | `TODO` |
| Re-ranking (MMR, caps, exploration) | 100 | 20 | 5 ms | `TODO` |
| Serialization + network | — | — | 12 ms | `TODO` |
| **Total** | | | **90 ms** | `TODO` |

Retrieval sources: two-tower + HNSW, co-visitation, trending, recently-viewed,
content-similarity. Union, dedupe, source-tagged so per-source contribution is measurable.

## 4. Data contracts

`TODO` — schemas for the event log, item catalog, and feature tables. Enforced as pytest
data contracts in `data_pipeline/tests/`, not as documentation.

## 4b. Feature groups, by refresh cadence

Grouped by how often a feature changes rather than by what it describes, because cadence is
what decides where a feature is computed and how it can leak.

| Group | Columns here | Refresh | Table |
|---|---|---|---|
| User static | *(none — MIND ships no demographics)* | daily | — |
| User dynamic | `user_impressions_24h`, `user_clicks_24h`, `user_ctr_smoothed`, `user_tenure_hours` | daily | `user_hourly_features` |
| User sequence | *(not built — see below)* | near-real-time | — |
| Item static | `category`, `subcategory`, `title` | on create | carried on silver |
| Item dynamic | `item_impressions_24h`, `item_clicks_24h`, `item_ctr_smoothed`, `item_age_hours` | hourly | `item_hourly_features` |
| Context | `hour_of_day`, `day_of_week` | request time | derived at join |
| Cross | `user_cat_affinity`, `user_cat_impressions_cum`, `user_cat_clicks_cum` | hourly | `user_category_cross_features` |

Two groups are deliberately empty. **User static** has nothing to hold: MIND ships no
demographics, which is also why `user_ctr_smoothed` shrinks toward the global rate rather
than a segment rate. **User sequence** — the "last 50 clicked items" — is not built here
because MIND's `history` is a *constant per-user snapshot*, identical across every
impression that user makes (verified at zero exceptions over 100,000 users). Used raw it
would be stale by up to six days. Topping it up with in-window clicks as of each label is
an array-valued as-of aggregation, and it belongs with the sequential model in Part H
rather than here.

**`age_hours` is a proxy, and the proxy is the interesting part.** MIND's `news.tsv`
carries no publication timestamp, so an article's age is measured from its **first
impression in the log** — the first hour anyone was shown it. That is not when it was
published. An article written at 06:00 and first surfaced at 14:00 reads as eight hours
younger than it is, and one first surfaced in the log's opening hour reads as brand new
regardless of when it was written.

This matters beyond the feature itself: age is the strongest single ranking signal on a
news corpus, and it is the mechanism behind the `age_hours` row in the skew report. A
serving layer computing age from *ingestion* time while training computes it from *first
impression* produces two different numbers for the same article, drifting apart by however
long the pipeline lags. That divergence is the one to look for first in
`docs/skew_report.md`.

## 5. Out of scope

Naming non-goals up front:

- No real-time embedding updates. Embeddings refresh on the daily retrain.
- No federated or differential-privacy layer.
- No multi-region serving; single region, single index replica.
- No cross-surface models; the homepage rail is the only surface.
- No LLM-based generative retrieval (see §25 of the guide — deliberately deferred).

## 6. Failure modes

| Failure | Behaviour |
|---|---|
| Ranker times out | Fall back to retrieval order, tagged in the response |
| ANN index stale or missing | Serve last-good index; alert on staleness > 2h |
| Brand-new user (cold start) | Popularity + content-based fallback |
| Feature store miss | Serve model defaults; count and alert on miss rate |
| One retrieval source errors | Serve the union of the rest; never fail the request |

## 7. Open questions

- `TODO` — how to weight observed non-clicks now that MIND supplies them (ADR 0004).
- `TODO` — item churn: news goes stale in hours, so how often must the index rebuild? (ADR 0002.)
