# Evaluation protocol

Written before any model, so the metrics cannot be chosen to flatter a result.

## Split protocol

**No random split.** A random split lets the model see August to predict July, so every
number computed on one is meaningless. Per-user leave-one-out has the same flaw more
subtly: user B's held-out impression may precede user A's training impressions, so the
model still trains on a future no serving system could have observed.

Two splits exist in this project, for two different purposes.

### Primary: MIND's official train/dev boundary

Used for every headline number. It already satisfies the protocol, and that is verified
rather than assumed:

| Split | First impression | Last impression | Impressions | Rows | Users |
|---|---|---|---|---|---|
| train | 2019-11-09 00:00:19 | 2019-11-14 23:59:13 | 156,965 | 5,843,444 | 50,000 |
| dev | 2019-11-15 00:00:01 | 2019-11-15 23:58:03 | 73,152 | 2,740,998 | 50,000 |

`max(train.ts) < min(dev.ts)` — eight seconds of gap, no overlap. Nothing in
`evaluation/offline/split.py` reproduces this boundary; it is Microsoft's, and it holds.

### Secondary: a temporal carve-up of the train week, for ablations only

`temporal_split()` cuts one split's timeline into `train < t1 <= val < t2 <= test`, with
`t1`/`t2` derived from `SplitConfig.holdout_days` and snapped to midnight so they align
with the `dt` partitioning. At the default of 1 day that is val = 11-13, test = 11-14.

Splitting happens on the **impression**, never on the row. An impression torn across a
boundary would put some of the items a user was shown into training and the rest into
test, with the answer already seen — and no metric would reveal it, because every metric
here aggregates *within* an impression, so the leak hides inside the unit the numbers are
computed over.

## Cohorts — and why the slicing is not optional here

Every metric is reported three ways: overall, warm users, cold users. On this corpus that
is not a courtesy slice, it is the main result.

| Cohort (official dev split) | Count | Share |
|---|---|---|
| Users seen in train (warm) | 5,943 | 11.9% |
| Users never seen in train (cold) | 44,057 | **88.1%** |
| Articles shown in dev but never in train | 2,483 of 5,369 | 46.2% |
| Dev rows showing a never-before-seen article | 582,984 of 2,740,998 | 21.3% |

**An unsliced aggregate over dev is not a personalization result with a cold-start
footnote. It is a cold-start result with a personalization footnote.** That is a fine
thing to report and a bad thing to report by accident, which is what `cohort_summary()`
exists to prevent.

### Three different cold-start numbers, and which one to quote

Item cold-start has three defensible denominators, and they differ by two orders of
magnitude. Quoting the wrong one badly misstates the problem.

| Question | Answer | Denominator |
|---|---|---|
| What share of dev's *catalogue* is new? | **46.2%** | 2,483 of 5,369 distinct articles |
| What share of dev *impressions* show an article absent from train? | **21.3%** | 582,984 of 2,740,998 rows |
| What share of dev impressions have **no usable feature history at the moment they occur**? | **0.31%** | measured from `has_item_features` |

The first two are cross-split questions: was this article in the training week? The third
is the point-in-time question the model actually faces, and it is much smaller for a
reason worth understanding. The gold feature series spans the whole timeline rather than
being built per split, so an article first shown at 08:00 on dev's day already has a
closed bucket by 09:00 — every impression of it after that reads real statistics. Being
new to the corpus and being cold *at request time* are only the same thing for an
article's first hour.

So the reading is: **the cold tail is wide but thin.** Nearly half the catalogue is new,
and almost none of the traffic is. Popular articles dominate impressions, so cold-start
handling moves aggregate metrics very little — and matters almost entirely for catalogue
coverage and the long tail, which is where the re-ranking policy in Part L lives. A
headline NDCG that improves because cold items got better is implausible on these
numbers; a coverage figure that improves is exactly what to expect.

The catalogue figure is still the right one to quote for the *modelling* decision: 46.2%
of the articles a model must be able to represent have no interaction history when they
launch, which is the argument for a content-weighted item tower over one leaning on ID
embeddings (see ADR 0005). The 0.31% is the right one to quote for what cold-start costs
the online metrics.

Train reads 0.90% on the third measure, and 100% of train items are "cold" at least once
— every article has a first impression, and `has_item_features` is per row, not per item.
That is worth remembering before reading the flag as an item property.

### The eligibility filter, and why it defaults to off

`temporal_split(min_user_impressions=...)` can restrict the test set to users with enough
training history to be personalizable — the guide's guard against "measuring cold-start
and calling it personalization." How much it costs depends entirely on the corpus, so it
is measured rather than assumed:

| Threshold | Train-week ablation | Official train/dev |
|---|---|---|
| 0 (off) | 100% of test users | 100% |
| 1 | 70.9% | 11.9% |
| 3 | 31.1% | 7.1% |
| 5 | 13.8% | 4.1% |

At the config default of 3, applying it to the official split would discard **93% of
dev**. The filter assumes users recur across the boundary; MIND's dev week is 88% new
users. So it defaults to off, and is meaningful only on the ablation path.

## Metrics

Implemented in `evaluation/offline/metrics.py` as plain NumPy over per-impression lists.
The lists are tiny, so a Spark UDF would spend longer in serialisation than in arithmetic
and would be far harder to test.

### Which pool a metric is computed over

This is the distinction that makes a number comparable or meaningless, and it is not
visible in the number itself. `recall_at_k` over the ~5 items in one impression and
`recall_at_k` over 65,238 catalogue items are different measurements sharing a name.

| | Ranking | Retrieval |
|---|---|---|
| Pool | the items shown together in one impression | the whole catalogue |
| Question | given the slate MSN already chose, can you order it? | can you find the clicked item at all? |
| Entry point | `protocols.evaluate_ranking` | `protocols.evaluate_retrieval` |
| Comparable to | published MIND results | nothing published; internal only |

`evaluate_retrieval` **raises** when any request supplies fewer than `k` candidates.
Scoring one positive against ~100 sampled negatives is biased and does not rank-correlate
with full ranking (Krichene & Rendle, 2020); it also produces a perfectly reasonable
looking number with no hint in the output that it happened. The guard is that stance made
executable rather than documented.

### The metrics

| Metric | Scope | Undefined when |
|---|---|---|
| `gauc` | ranking — impression-weighted per-slate AUC. **The headline.** | every slate is degenerate |
| `reciprocal_rank` / MRR | ranking — rank of the first click. Reported because MIND's leaderboard does | no click |
| `ndcg_at_k` | ranking — full ordering, ideal DCG from this slate's own clicks | no click |
| `recall_at_k` | either — the pool decides which | no click |
| `catalog_coverage` | retrieval — distinct items served / addressable catalogue | — |
| `novelty` | retrieval — mean `-log2(p)` against **train-only** popularity | empty slate |
| `intra_list_diversity` | re-ranking — 1 − mean pairwise cosine of **content** vectors | fewer than 2 known vectors |
| `gini` | monitoring — exposure concentration; Part O watches it rise | no exposure at all |

Report GAUC prominently: ranking only ever happens within a request, so a global AUC
rewards cross-request separability no user ever experiences.

### Undefined is not zero

A slate with no click has no correct ordering to have found, so every per-impression
metric returns `NaN` rather than `0.0`, and the aggregations carry the denominator with
them:

```python
Aggregate(mean=0.64, scored=12_431, skipped=27_902)
```

Scoring a clickless slate zero would make the mean a function of how many such slates the
current filter happens to leave in — so two runs under different filters would stop being
comparable while both looking fine.

**Measured, rather than assumed:** on MIND-small dev every one of the 73,152 impressions
carries a click, so `skipped` is zero for the overall slice. The convention still earns its
place — a cohort mask can cut a slate down to all-clicks or all-misses, and `warm_item`
skipped 13,390 slates on the random run — but the dataset itself logs no clickless page
views. Earlier drafts of this file asserted the opposite; it was never checked.

## Statistical rigor

Implemented in `evaluation/offline/stats.py`. Two decisions, both of which change whether a
reported win is real.

### The resampling unit is the user, not the impression

One user generates many impressions and their scores are correlated. Resampling impressions
treats dependent observations as independent, inflating the effective sample size by roughly
the impressions per user and narrowing the interval by roughly its square root. The point
estimate is unaffected, which is what makes the error hard to notice — only the uncertainty
is wrong, and nothing in the output looks out of place.

This is not left to discipline: `bootstrap_ci(scores, user_ids)` takes the user ids and does
the per-user reduction itself, so passing impression-level scores is not an available
mistake. `bootstrap_ci_by_impression` exists **only** to reproduce the contrast, and its
`n_users` field is deliberately set to the impression count — the lie, made legible.

Every `make eval` prints both, so the gap is demonstrated on real data rather than asserted:

```
ndcg@10 95% CI over 50,000 USERS:       [0.2851, 0.2903]  width 0.0052
ndcg@10 95% CI over 73,152 IMPRESSIONS: [0.2832, 0.2877]  width 0.0045  <- WRONG unit
```

The JSON carries both, the wrong one under a key named
`ci95_by_impression_DO_NOT_QUOTE`.

**The effect is real but mild on this corpus, and the reason matters.** Dev has 73,152
impressions across 50,000 users — **1.46 impressions per user**. The expected narrowing is
roughly √1.46 = 1.21×; the measured narrowing is 1.16×. Close enough to confirm the
mechanism, and small enough that on MIND-small dev the wrong unit would rarely flip a
conclusion.

That is a property of this dataset, not a reason to relax. The distortion scales with
impressions per user: at 20 per user — ordinary for a production log, and what
`test_stats.py` simulates — the interval narrows by more than 2×, and a null result becomes
a significant one. Quoting the measured 1.16× as evidence that the unit does not matter
would be exactly the wrong lesson. The number to remember is √(impressions per user), not
1.16.

### NaN aggregation changes the denominator

Per-slate metrics are undefined for a slate with no click, so `per_user_means` skips NaN
rather than zeroing it — and a user whose every slate was clickless has **no** score and
leaves the resample entirely. A fabricated zero would both drag the mean and tighten the
interval, which is the worst combination: it moves the estimate toward the null and makes
you more confident about it.

### Comparisons are paired

Each user is scored under both systems and the **per-user difference** is bootstrapped.
Variance across users dwarfs the effect size — some users are simply easier to serve — so an
unpaired comparison buries a 2% lift under between-user spread. `paired_bootstrap` refuses
disjoint user sets rather than silently approximating, because comparing means over
different users *is* the unpaired test it replaces.

Bootstrapped rather than a t-test: per-user NDCG and AUC differences are bounded, skewed and
spiky, so normality is not a safe assumption and buys nothing here. The percentile bootstrap
is the simple variant; BCa would correct for skew and is what a publication would want.

**Measured, on the two closest models in the table.** Decayed popularity at a 29-minute
half-life against pure recency — the same family, adjacent on the curve, and the hardest pair
to separate:

| | NDCG@10 95% CI | width |
|---|---|---|
| recency | [0.3236, 0.3284] | 0.00480 |
| decayed, hl = 0.02d | [0.3280, 0.3330] | 0.00500 |
| **paired difference** | **[+0.00380, +0.00519]** | **0.00139** |

The paired interval is **3.5× narrower** than either marginal, and it excludes zero where the
marginals overlap heavily. A +0.00453 difference sitting inside two intervals 0.005 wide looks
like noise and is not: the between-user variance inflating both marginals cancels when each
user is compared against themselves. **Marginal CI width places no bound on the paired
difference CI** — reading one off the other is the specific mistake this section exists to
prevent, and it is an easy one to make even while writing the tooling that prevents it.

Run it with `make compare BASELINE=<model> CANDIDATE=<model>`; a model with a tunable takes it
after `@`, as in `decayed_popularity@0.02`. Both sides are scored in one pass over one read, so
the pairing is by key rather than by trusting two runs to have seen the same rows.

## Leakage tests

Live as unit tests that fail CI, not as a one-time notebook check.

- `evaluation/tests/test_split.py` — window ordering, impression-level disjointness, row
  conservation, and an impression straddling a boundary staying whole.
- `data_pipeline/tests/test_contracts.py` — the as-of join's point-in-time boundary, the
  hourly feature buckets, and the bronze/news/history data contracts.

Both run without a built corpus: the contracts fall back to a committed synthetic fixture
(`data_pipeline/tests/fixtures/`), and the split tests build their frames inline.
