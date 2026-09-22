# ADR 0012 — Exploration over designed diversity policies

**Status:** accepted · **Date:** 2026-09-22

## Context

Part L built the policy layer over the ranker's top 100: MMR, per-category
caps, a freshness boost, epsilon-greedy exploration with logged propensities,
and a Redis-backed Bloom seen-list. Every arm shares one fitted booster, so the
table measures policy rather than model variance. Full numbers in
[ranking.md](../ranking.md#the-policy-layer-part-l).

**The ceiling was measured before any policy was built.** A re-ranker cannot
serve what retrieval never proposed, so coverage is bounded by the candidate
pool, not the catalogue: across 5,722 held-out requests the pool holds **1,270
distinct items** — 1.9% of the catalogue — of which the ranker serves 190.

k = 10, paired per user against the ranker's own order:

| config | NDCG@10 | vs ranker only | ILD | coverage | tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| ranker only | 0.1283 | — | 0.590 | 15.0% | 2.9% |
| + MMR λ=0.5 | 0.1271 | −0.0008 (n.s.) | 0.595 | **14.2%** | 2.8% |
| + MMR + caps | 0.1281 | +0.0003 (n.s.) | 0.591 | 15.1% | 2.9% |
| + freshness 24h | 0.1047 | **−0.0257** \* | 0.584 | 15.4% | 4.9% |
| + exploration ε=0.1 | 0.1277 | **−0.0005** \* | 0.590 | **27.6%** | 3.2% |
| + seen filter | 0.0883 | **−0.0227** \* | 0.592 | **42.2%** | 5.7% |

## Decision

**Ship epsilon-greedy exploration on the last two of ten slots, with the
propensity logged per slot. Do not ship MMR. Do not ship the freshness
boost.** Category caps are retained at ≤3 per subcategory as a cheap guardrail
that costs nothing measurable. The seen-list ships as a correctness
requirement, not as a diversity policy.

The comparison that decides it is coverage bought per point of NDCG given up:

| policy | coverage gained | NDCG per point |
| --- | ---: | ---: |
| exploration ε=0.1 | +12.6 pp | **0.00004** |
| seen filter | +27.2 pp | 0.00083 |
| freshness | +0.4 pp | 0.064 |

**Three orders of magnitude separate the cheapest from the dearest.** Random
beats designed here.

## Consequences

**MMR is not merely unhelpful on this corpus — its premise is false.** It
exists because a relevance-ranked list is full of near-duplicates. The ranker's
top 10 already scores an intra-list diversity of **0.590** before any policy
runs, so there is nothing to deduplicate: λ=0.9 moves nothing at all, and λ=0.5
buys 0.005 of ILD for 0.0008 of NDCG. This is a statement about the two-tower's
top-100 on a news corpus, not about MMR, and it would need re-measuring on a
catalogue with genuine redundancy.

**And MMR trades catalogue coverage for slate diversity.** λ=0.5 raises ILD and
*lowers* distinct items served, 190 → 180. It rewards distance from what is
already chosen, and across users the far items are the same globally-atypical
articles. **"Diversity" is not one quantity**, and a table reporting only ILD
would have shown this policy succeeding.

**Exploration is infrastructure, not charity.** It is the only arm whose mean
logged propensity is below 1.0, which makes it the only source of data an
unbiased off-policy estimator can consume. That is its real justification; the
coverage gain is a bonus. **The propensity must be logged at selection time or
not at all** — by the time a request is over, the candidate set and the random
draw are both gone.

**Freshness is rejected at fifty times the price, and its signal is
confounded.** On a week of news, age and popularity are nearly the same axis: a
new article has few clicks because it is new. It moves the long tail furthest
(2.9% → 4.9%) and nothing here shows that *recency specifically* was the useful
signal rather than inverse popularity.

**Every number above assumes click behaviour is unchanged by the reordering.**
MIND labels what MIND showed. When a policy blocks 11% of candidates and serves
different items, what the user would have done facing that slate is unknown.
These are "what the metric would be if behaviour were fixed" — the standard
limitation of evaluating a policy offline, binding loosely on the MMR rows and
hard on the seen filter. It is the second argument for the propensity log.

**The exchange rate is measured; whether to pay it is not a question this
answers.** Every row below the first is a cost by construction, because each
request here has exactly one relevant item and no reordering can surface a
second. Choosing a point on that curve needs an A/B test against a north-star
metric, or a bandit over the weights.

## Alternatives considered

**Ship MMR anyway, for the story.** Rejected: it would mean reporting a policy
whose measured effect on this corpus is indistinguishable from noise on
relevance and negative on coverage.

**Tune λ lower than 0.5.** The trend is already adverse — coverage falls
monotonically as λ drops — so lower λ buys more ILD and less catalogue
diversity. The knob is working; the corpus has no redundancy for it to remove.

**Drop category caps.** They change nothing measurable here (+0.0003, n.s.),
which is an argument for either choice. Retained because the constraint binds
rarely on a broad pool but binds exactly when a single subcategory floods a
slate, which is the failure it exists to prevent, and it costs one comparison
per slot.

**Treat the seen-list as the diversity instrument.** It buys the most coverage
in the table. Rejected as a framing: it is a correctness requirement whose
coverage gain is a side effect, and at ~0.00083 NDCG per point it is 20x
dearer than exploration for that purpose.
