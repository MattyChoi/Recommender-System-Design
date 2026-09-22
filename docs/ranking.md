# Ranking

**Nothing here is comparable to [baselines.md](baselines.md), and only the
ceiling connects it to [retrieval.md](retrieval.md).** That table reorders the
~37 items MSN already chose. This page reorders the ~100 candidates **our own
retriever produced**, which is a third candidate set and a third denominator —
and the distinction is the whole point of the stage, because a ranker must be
trained on the candidate distribution it will be served.

## The setup, once

19,006 requests replayed through `two_tower + trending + covisit + content` at
100 candidates each: **1,900,600 candidate rows**. Split **by user** into 13,284
requests to fit and **5,722 held out** over 2,965 users — by user rather than by
time, because a user cut is the same unit the bootstrap resamples and makes
leakage through a repeated user impossible.

> ⚠️ **Read the source list in an arm's name carefully: it is not a blend.**
> With the default quota every one of the 100 slots goes to the **first** source,
> so the candidate set is `two_tower`'s top 100 and nothing else. The other three
> sources contribute *features* — `trending_rank`, and their presence in
> `n_sources` — at the cost of no slot. The confirmation is exact: the candidate
> ceiling over all 19,006 requests is **0.3776**, which is `two_tower` alone at
> 100 in [retrieval.md](retrieval.md#the-five-alone), to four decimals. This is
> Part I's finding carried forward — at a fixed budget no blend beat the
> two-tower alone — but it means "four sources" here describes the feature
> table, not the pool.

**Retrieval's recall over the candidate list is 0.3820, and that is the
ceiling.** A request whose clicked article was never retrieved cannot be fixed
by any ordering. Those requests carry no positive, so they teach a pairwise
objective nothing and are dropped from **training** — and kept in
**evaluation**, where they count as zero. Dropping them from both would report
the ranker's skill on the subset retrieval had already solved.

**4.6% of negatives were actually shown to the user.** The rest are assumed: a
candidate marked unobserved may still have been in the real impression, because
the gold table keeps only a prefix of each slate. The number undercounts and is
reported rather than corrected.

---

## The gate: does a neural ranker beat a tree?

| arm | NDCG@10 (per user) | paired vs `lgbm` |
|---|---:|---|
| **LightGBM `lambdarank`** | **0.1443** [0.1351, 0.1529] | — |
| MMoE | 0.1403 | **−0.0040** [−0.0079, −0.0005] \* |
| DCN v2 | 0.1399 | **−0.0044** [−0.0085, −0.0006] \* |

**The tree wins, and both neural arms lose significantly.** DCN v2 against MMoE
is +0.0004 [−0.0022, +0.0034], not resolvable.

### The noise floor, measured rather than assumed

An identical MMoE configuration was run twice and the two scorings paired:

| | NDCG@10 | paired |
|---|---:|---|
| `mmoe` | 0.1403 | — |
| `mmoe2`, same arguments | 0.1405 | **+0.0002** [−0.0006, +0.0010] — not resolvable |

**±0.0010 is what "no difference" looks like on this arm**, and every claim
above should be read against it. The tree's −0.0040 and −0.0044 are about four
times it and stand; DCN v2 against MMoE at +0.0004 is inside it and does not.
Two runs of the tree agree to four decimals — LightGBM is deterministic here,
the neural arms are not.

A same-config pair is the cheapest control this project has: it costs one rerun
and it converts "the interval excludes zero" into "the interval excludes zero
by more than the model's own run-to-run wobble".

**Read the paired column, not the gap between the point estimates.** The spread
*between users* is far larger than 0.004; the pairing cancels it by scoring
every user under both models. See [ADR 0003](adr/0003-lightgbm-baseline-before-dcnv2.md).

**Two things differ between the families, not one.** Architecture, and the
objective: the tree is fitted listwise on within-request comparisons, the neural
arms pointwise with binary cross-entropy. −0.0044 is therefore not attributable
to architecture alone, and isolating it needs a listwise loss on the neural
side, which is not built.

### The stage justifies itself, which is the prior question

| | NDCG@10, per request |
|---|---:|
| retrieval order alone | 0.0716 |
| the tree over the same candidates | **0.1283** |
| difference | **+0.0671** [+0.0588, +0.0752] \* |

Retrieval hands over candidates already sorted by score. A ranker that could not
beat that ordering would be a model, a feature pipeline and serving latency for
nothing.

---

## What the shipped model does, on four denominators

| | |
|---|---:|
| GAUC, per request, size-weighted | **0.8583** |
| — scored on | 2,186 requests |
| — skipped, no positive to order | **3,536 requests** |
| AUC, pooled over all rows | 0.8735 |
| NDCG@10, funnel, per USER | 0.1443 |
| NDCG@10, funnel, per REQUEST | 0.1283 |
| NDCG@10, retrieved requests only | 0.3358 |
| ceiling | 0.3820 |
| headroom used | 37.8% |

**GAUC and funnel NDCG are not rival estimates of one quantity.** GAUC skips a
request with no positive, correctly — there is no right order to have found —
so its denominator is the 38% of requests retrieval solved, and it is
**structurally blind to a retrieval miss** that funnel NDCG scores as zero. 62%
of the holdout is skipped. Quote GAUC for ordering quality and funnel NDCG for
the system.

**Pooled AUC sits above GAUC, as it should.** It rewards separating one
request's negatives from another request's positives, a comparison no user ever
experiences. It is on the table to be undercut, not to be quoted.

**The per-user and per-request NDCG differ by weighting, not by noise**, and no
interval brackets both: a user with forty requests outweighs a user with one in
the second column and not in the first. The headline comes from the bootstrap's
own mean, so the point estimate cannot land outside its own interval — a bug
this project shipped once and now guards against.

---

## Feature importance, and why it is a claim rather than a result

| feature | gain share |
|---|---:|
| `subcategory_idx` | **35.9%** |
| `two_tower_rank` | 20.2% |
| `content_similarity` | 12.3% |
| `trending_rank` | 9.3% |
| `prior_clicks` | 7.0% |
| `train_clicks` | 6.6% |
| `retrieval_score` | 5.1% |
| `history_length` | 2.3% |
| `category_idx` | 0.9% |
| `n_sources` | 0.4% |
| `is_cold_item` | 0.0% |

**Gain does not separate signal from flexibility.** LightGBM splits a
categorical by searching partitions of its levels, which is far more expressive
than a numeric threshold, so a 121-level column can earn a large gain by fitting
the training folds. The test is to drop it:

| arm | NDCG@10 | paired vs full |
|---|---:|---|
| full | 0.1443 | — |
| both categoricals dropped | 0.1415 | **−0.0028** [−0.0058, +0.0001] — not resolvable |

**36.8% of the gain buys no measurable NDCG.** That is the single most useful
line on this page: it is the difference between "the model uses this feature"
and "this feature is worth having", and only the ablation can tell them apart.

It is also the interaction an explicit cross layer would have been expected to
exploit — which is one reason DCN v2 had no edge to find.

### The quota arm is not a clean comparison

| arm | NDCG@10 | ceiling | paired vs full |
|---|---:|---:|---|
| full (all slots to the strongest source) | 0.1443 | 0.3820 | — |
| 70/30 split | 0.1418 | **0.3598** | −0.0025 [−0.0059, +0.0008] — not resolvable |

**The ceilings differ.** Splitting the slot budget changes which candidates
exist, so the 70/30 arm is handicapped by 0.022 of recall before the ranker
touches it. The not-resolvable difference is therefore a statement about two
different candidate sets, not about two rankers.

---

## Calibration, and a number that looks like a result and is not

Isotonic regression fitted on a **by-user half of the holdout** and reported on
the other half — 286,000 rows fitted, 286,200 reported. The split is by user
because two rows of one request are not independent: a row-wise split would put
a request's negatives in the fit and its positive in the report.

| bucket (decile) | rows | predicted | observed | gap |
|---|---:|---:|---:|---:|
| 0 – 0.000128 | 23,917 | 0.0000 | 0.0000 | +0.0000 |
| … | 31,094 | 0.0001 | 0.0000 | +0.0001 |
| … | 18,890 | 0.0003 | 0.0002 | +0.0001 |
| … | 26,875 | 0.0006 | 0.0004 | +0.0001 |
| … | 37,007 | 0.0008 | 0.0008 | +0.0000 |
| … | 24,308 | 0.0011 | 0.0010 | +0.0001 |
| … | 36,955 | 0.0015 | 0.0018 | −0.0004 |
| … | 18,453 | 0.0027 | 0.0029 | −0.0003 |
| … | 35,662 | 0.0054 | 0.0047 | +0.0007 |
| top decile, to 0.25 | 33,039 | 0.0219 | 0.0220 | −0.0001 |

| | calibrated | constant base rate | raw score |
|---|---:|---:|---:|
| ECE (row-weighted) | 0.0002 | **0.0000** | 0.3460 |
| log loss | **0.0205** | 0.0249 | — |

**The constant model is better calibrated than the calibrated model.** A
predictor that ignores every feature and returns the base rate for every row is
perfectly calibrated and perfectly useless, and it scores an ECE of exactly
zero. So **a near-zero ECE is evidence of nothing on its own**, and this page
would have claimed otherwise if the control had not been printed beside it.

**Log loss is the line with content**: 0.0205 against the constant model's
0.0249, **17.7% better**, because log loss punishes a failure to discriminate
and ECE does not.

**What the calibrated number means.** The base rate is **0.0038** — one positive
among the candidates retrieval returned, and only 38% of requests have one at
all. A calibrated 0.02 means *five times as likely as an average retrieved
candidate to be the click*, **not** "a 2% chance this user clicks this article".
Those differ by the candidate pool, and a pool is a denominator.

> ⚠️ An earlier version of this table used equal-width buckets and put 286,173
> of 286,200 rows in `[0.00, 0.10)`. The diagram collapsed to a single point
> and looked like flawless calibration. The manual asks for the rate "by
> decile" and means quantile bins; with them the calibrator's resolution is 10
> distinct buckets and the deciles track observed closely.

---

## Position bias: built, and deliberately not run

`models/ranking/position.py` implements the shallow additive position tower —
trained into the logit, dropped at serving, initialised to zero so that at step
zero the wrapper is exactly the model it wraps.

**It is not wired in, and running it here would look like it worked.** MIND's
impression lists are presented in shuffled order, so the exploded index this
project calls `slot` carries no display-order signal. Fit a position tower on it
and it fits noise — and the standard verification, *plot average predicted CTR
by position and check it is flat*, **comes out flat by construction** because
the input was noise to begin with. A correction that cannot fail its own test is
not evidence.

The module waits for the serving layer to record the slot it actually chose. The
ablation then measures a bias this project created and removed, which is a
stronger claim than correcting someone else's.

---

## Multi-task: one label, one gate

MMoE runs with `TASKS = ("click",)`. MIND records click or no-click — no dwell,
no completion, no "not interested" — so a second head would have to be
fabricated, and the multi-task result would then be an artifact of the
fabrication. A mixture of experts with one gate is the machinery without the
reason, and it is reported as such.

**The gates are measured for collapse anyway**, because a gate that always
routes to one expert is a plain MLP wearing a mixture's name. Measured on the
`mmoe` arm at 100 candidates (8 experts, maximum entropy `ln 8` = 2.079):

| | | |
|---|---:|---|
| entropy **of the mean** gate | 2.060 of 2.079 | 99.1% |
| **mean** per-row entropy | 1.762 of 2.079 | 15.2% specialised |

mean gate weights: `0.105 0.120 0.113 0.094 0.153 0.103 0.165 0.146`

Both figures moved by under 0.005 between the two runs of this arm, so the
routing is a stable property of the fit rather than of the seed.

**Those two numbers are not the same statistic and only one answers the
question.** The entropy of the *averaged* gate says the eight experts are used
evenly *across the corpus* — which a gate that ignores its input entirely would
also achieve, by spreading uniformly on every row. The mean of the *per-row*
entropies is the one that can see specialisation, and at 1.766 it says each
individual request does tilt toward a subset: **15.1% of the way from "uniform
on every row" to "one expert per row"**. Summarise-then-aggregate and
aggregate-then-summarise, giving different answers to different questions —
the same distinction that produced three other bugs in this project.

So the gates route, mildly. With one task there is nothing for them to route
*between*, so this is a liveness check on the machinery, not a multi-task
result.

The serving-score combination `p(click)^α · (1 + dwell)^β · p(not_negative)^γ`
is a **product decision** and cannot be settled offline: the weights are A/B
tested, or learned against a north-star metric by a bandit over the simplex.
Nothing on this page licenses a choice of α, β, γ.

---

## Infrastructure measured alongside

Three pieces were built here and are reported with their own denominators,
which are **not** the c100 table above.

**Distributed training.** Two Ray workers against one, on the `dcn` arm at 20
candidates and two sources: paired difference **−0.0022** [−0.0048, +0.0002],
not resolvable. No measurable accuracy cost — **not** a claim of equivalence,
since DDP over two shards runs at twice the effective batch and is a different
optimisation.

**A bug that arm found.** `torch_fit.fit` derived the standardiser, the
embedding cardinalities and the positive rate from its own rows. All three are
wrong on a shard: different scaling per rank means DDP averages gradients of
different objectives, and different cardinalities mean different embedding
shapes and a failed broadcast — or a one-worker rerun that hides it. Fitted once
in the driver now and passed to every worker.

**TorchRec.** An `EmbeddingBagCollection` backend for the categorical block,
selectable with `--embeddings torchrec`. At this scale it is **scaffolding and
is labelled as such**: 18 categories and 121 subcategories at 16 dimensions is
9 KB, and nothing about 9 KB needs a planner. Parity with `nn.Embedding` is
tested at the unit level; **no end-to-end comparison is claimed**, because the
two backends cannot be seed-matched from the command line — they leave the
global RNG in different places, so every layer built after the block differs.

> ⚠️ **One retraction.** The TorchRec arm first measured −0.0114
> [−0.0159, −0.0072] \*, which was reported as a backend cost and was not one.
> TorchRec's default initialiser is uniform on ±1/√rows — standard deviation
> **0.0129 against `nn.Embedding`'s 1.0** — so the categoricals started at 1/77
> scale beside a standardised dense block. One line of initialiser recovered the
> whole gap. A bootstrap over users cannot see a difference in initialisation,
> because it resamples users.

---

## Caveats that apply to every number here

- **The ceiling is 0.3820 and the ranker reaches 37.8% of it.** The largest
  available gain in this funnel is in retrieval, not ranking.
- **Single seed per arm unless stated.** The measured noise floor is ~0.0004 for
  the neural arms and ~0 for the tree; differences at that scale are not
  interpretable.
- **Validation is a carved tail of TRAIN**, so its users are overwhelmingly
  warm. Dev is 88% cold users and will be lower. Never quote this as a headline
  system result.
- **Retrieval scores are mildly optimistic for every request**, because both
  halves of the user split come from the window the retriever early-stopped on.
- **`prior_clicks` is a single snapshot** taken at the validation boundary:
  stale-but-safe to the right of it, and an oracle to the left. Widening the
  window without a per-request point-in-time lookup would leak, and a tree would
  find it immediately.

## Reproducing

```
make rank         RANK_ARGS="--names two_tower trending covisit content --max-candidates 100 --label lgbm"
make rank-neural  RANK_ARGS="--model dcn  --names two_tower trending covisit content --max-candidates 100 --label dcn"
make rank-neural  RANK_ARGS="--model mmoe --names two_tower trending covisit content --max-candidates 100 --label mmoe"
uv run python -m models.ranking.compare lgbm dcn
```

`--label` writes per-request scores to `evaluation/results/ranking/<label>.npz`;
`compare` refuses two runs that did not score the same requests in the same
order, because two arms over different holdouts would still pair row-for-row and
still return a confident number about nothing.
