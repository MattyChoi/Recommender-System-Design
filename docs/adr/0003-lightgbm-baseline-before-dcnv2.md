# ADR 0003 — LightGBM baseline before DCNv2

**Status:** accepted · **Date:** 2026-09-22

## Context

The ranker reorders the ~100 candidates retrieval produced. Two families were
available: a gradient-boosted tree fitted listwise with `lambdarank`, and a
neural ranker — DCN v2's explicit bounded-degree crosses, or MMoE's gated
experts — fitted pointwise with binary cross-entropy.

The decision was made in that order deliberately. The tree trains in minutes,
has almost nothing to configure wrongly, and needs no feature scaling, so it
establishes a number before any neural result can be interpreted. A neural
ranker that is not compared against it is not a result, it is an artifact.

All arms below share one candidate table: 19,006 requests replayed through
`two_tower + trending + covisit + content` at 100 candidates, 1,900,600
candidate rows, split by user into 13,284 fitting and 5,722 held-out requests.
**Retrieval's recall over that list is 0.3820, and no ranker can exceed it.**

**That source list names the feature table, not a blend.** Under the default
quota all 100 slots go to the first source, so the pool is `two_tower`'s top
100; the others contribute `trending_rank` and `n_sources` and consume no slot.
The pool's recall over all 19,006 requests is 0.3776 — `two_tower` alone at 100
in `docs/retrieval.md`, to four decimals. That is Part I's decision carried
forward, and it is why a ranking arm's name should not be read as a blend.

## Decision

**Ship the tree.** It wins, and the neural arms lose significantly.

| arm | NDCG@10 (per user) | paired vs `lgbm` |
|---|---:|---|
| **LightGBM `lambdarank`** | **0.1443** [0.1351, 0.1529] | — |
| MMoE | 0.1403 | **−0.0040** [−0.0079, −0.0005] \* |
| DCN v2 | 0.1399 | **−0.0044** [−0.0085, −0.0006] \* |

DCN v2 against MMoE is **+0.0004** [−0.0022, +0.0034], not resolvable.

**The noise floor is measured, not assumed.** An identical MMoE configuration
run twice and paired gives **+0.0002** [−0.0006, +0.0010] — so ±0.0010 is what
"no difference" looks like on this arm. The two losing margins are about four
times that and stand; DCN v2 against MMoE sits inside it and does not. The tree
is deterministic: two runs of it agree to four decimals.

**The stage itself is justified**, which is the prior question. Against the
ordering retrieval already hands over, the tree is **+0.0671**
[+0.0588, +0.0752] \* per user — 0.0716 to 0.1283 per request. A ranker that
could not beat the inherited order would be latency and complexity for nothing.

### What the winning model actually does

| | |
|---|---:|
| GAUC, per request, size-weighted | **0.8583** |
| — scored on | 2,186 requests |
| — skipped, no positive to order | 3,536 requests |
| AUC, pooled over all rows | 0.8735 |
| NDCG@10, funnel, per user | 0.1443 |
| ceiling (retrieval recall) | 0.3820 |
| headroom used | 37.8% |

**GAUC and funnel NDCG are not rival estimates of one quantity.** GAUC skips
every request with no positive — there is no correct order to have found — so
its denominator is the 38% of requests retrieval solved, and it is structurally
blind to the miss that funnel NDCG counts as zero. Pooled AUC sits *above* GAUC
because it rewards separating one request's negatives from another's positives,
which no user ever experiences. Report GAUC for ordering, funnel NDCG for the
system.

## Consequences

**One serving stack, not two.** A `.txt` booster, no GPU at inference, no
standardiser to ship beside the weights, no embedding tables. The neural path
stays in the tree (`models/ranking/{dcn,mmoe}.py`) because it is built, tested
and measured, and because a second family is cheap to re-measure when the
feature set grows.

**The confound is stated, not corrected.** Two things differ between the
families, not one: architecture *and* objective. The tree is fitted listwise on
within-request comparisons; the neural arms are fitted pointwise with BCE. So
−0.0044 is not attributable to architecture alone. Isolating it needs a listwise
loss on the neural side, which is not built.

**Training cost was not measured for this stage**, so no cost ratio is claimed
here. The tree's wall-clock advantage is obvious in use and unquantified on
paper; treat any "Nx cheaper" statement as unsupported until it is timed.

**Calibration is fitted and reported, and its headline number is a trap.**
Isotonic regression on a by-user half of the holdout gives ECE 0.0002 — against
**0.0000 for a model that ignores every feature and predicts the base rate**.
A constant predictor is perfectly calibrated and perfectly useless, so ECE alone
separates nothing. Log loss does: **0.0205 calibrated against 0.0249 constant,
17.7% better**, because log loss punishes a failure to discriminate. The
calibrated probability answers "how much more likely than an average retrieved
candidate is this the click", against a base rate of 0.0038 set by the candidate
pool — not "will this user click this article".

## Alternatives considered

**Go straight to DCN v2.** Rejected on the measurement above: it would have
shipped a model that is significantly worse than a tree that took minutes to
fit, and nothing in the run would have revealed it.

**Neural for the feature crosses.** The strongest prior for DCN v2 is that
explicit crosses beat a tree's axis-aligned boxes. Not supported here. The
largest single gain share in the tree is `subcategory_idx` at 35.9%, and
dropping both categoricals costs **−0.0028** [−0.0058, +0.0001] — **not
resolvable**. A feature carrying a third of the gain that cannot be shown to
carry a third of the value is a flexibility artifact, and it is also the
interaction a cross layer would have been expected to exploit.

**XGBoost / CatBoost.** Not tried. LightGBM's `lambdarank` with
`lambdarank_truncation_level=30` is the listwise objective this stage wants, and
a second boosting library is a different implementation of the same decision.

**Multi-task MMoE as the shipped model.** MIND records click or no-click and
nothing else. MMoE runs with `TASKS = ("click",)`, which is a mixture of experts
with one gate — the machinery without the reason. Fabricating a dwell signal to
populate a second head would make the multi-task story an artifact of the
fabrication. The gates are measured for collapse instead, and the architecture
waits for a corpus with a second label.
