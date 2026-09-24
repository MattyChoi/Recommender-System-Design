# Benchmarks

Each table records the hardware and the commit it was produced on. Sections still
marked `TODO` are not yet measured; nothing below is a placeholder number.

Retrieval quality lives in [retrieval.md](retrieval.md); ranking quality in
[baselines.md](baselines.md). This file is for the cost side: memory, sharding,
latency and throughput.

## Quality vs. baselines

`TODO` -- see [baselines.md](baselines.md) for the ranking table and
[retrieval.md](retrieval.md) for the retrieval gates.

## Per-source contribution and leave-one-out ablations

`TODO` -- Part I. The three-arm item-tower ablation (ID / content / both) in
[retrieval.md](retrieval.md) is the closest thing that exists.

---

# Embedding tables (Part G4)

**Hardware: RTX 4090, 24,564 MiB, WSL2 · torch 2.13.0+cu130 · torchrec 1.8.0 ·
fbgemm-gpu 1.8.0.** Requires `uv sync --extra sharded`.

**Two of the three tables below are DRY RUNS**, and the distinction is the most
important thing on this page. `make shard-plan` reports what
`EmbeddingShardingPlanner` *decides* on `meta` tensors: no kernel executes, no
collective runs, no throughput is observed. Those numbers are a cost model's
prediction. `make hash-bench` measures a real hash over the real catalogue but
trains nothing. Only the two-rank test at the foot of this page runs anything.

## G4c -- when the planner starts splitting

`make shard-plan` · dim 64, fp32, batch 8192, compute device `cuda`, storage
reservation fixed at 15%.

### The catalogues that exist

| topology | rows | params | sharding | shards |
| --- | ---: | ---: | --- | ---: |
| 8 x 24 GB | 65,238 | 0.02 GiB | `table_wise` | 1 |
| 8 x 24 GB | 2,000,000 | 0.48 GiB | `table_wise` | 1 |
| 8 x 80 GB | 65,238 | 0.02 GiB | `table_wise` | 1 |
| 8 x 80 GB | 2,000,000 | 0.48 GiB | `table_wise` | 1 |

### The crossover

| topology | rows | params | sharding | shards | ranks |
| --- | ---: | ---: | --- | ---: | ---: |
| 8 x 24 GB | **85,648,437** | 20.42 GiB | `row_wise` | 8 | 8 |
| 8 x 80 GB | **286,015,625** | 68.19 GiB | `row_wise` | 8 | 8 |

### The finding: this is arithmetic, not a cost model

Both boundaries land on `hbm x (1 - reservation) / (dim x 4)`:

| topology | predicted | measured | error |
| --- | ---: | ---: | ---: |
| 8 x 24 GB | 85,558,706 | 85,648,437 | 0.1% |
| 8 x 80 GB | 285,195,687 | 286,015,625 | 0.3% |

Both inside the bisection's own 1% tolerance. **TorchRec's perf model contributes
nothing to this boundary.** A table is placed whole exactly when it fits on one
device after reservation; the cost model only chooses *among* plans that already
fit. Which means the crossover is partly a number we set: `DENSE_RESERVATION =
0.15` is a constant chosen in `models/layers/sharded_embeddings.py`, not something
TorchRec discovered. Change it and the crossover moves linearly.

**Why row-wise and not column-wise.** Row-wise splits the table across ranks, so
each rank owns a slice of the rows and every lookup becomes an all-to-all. With
uniform access that balances memory and communication together, which is what the
planner's cost model prefers once table-wise stops fitting. *Access is only
uniform because nothing told it otherwise* -- a real news corpus is heavily
skewed, and TorchRec can take skew through `ParameterConstraints`. Unmodelled here.

### What this says about the project

**MIND-small is 1,312x below the crossover.** The `item_map` is 65,238 rows --
a 15.9 MiB table. The Field Manual's G4 text says "160K MIND articles"; the
measured number is less than half that, and below the 100K floor ADR 0004 claims
the corpus sits inside. Same wrong figure as `docs/design.md` lines 22 and 34.

**The manual's own 2M design target is still 43x below it.** Building for the
design target demonstrates the mechanism; it does not produce a decision. Every
2M row above is `table_wise`, one shard, one rank.

**So the honest recommendation is: do not shard this.** That is a stronger result
than a clean demo, because it says when *not* to reach for the machinery -- and it
agrees with the item-tower gate, which could not distinguish the entire ID table
from **zero recall** while measuring it at **0.0037 of cold-band recall to the
bad**. That gate reads +0.0049 overall, against a between-run sd of 0.0103; see
[retrieval.md](retrieval.md#seed-variance-and-what-it-costs-this-page).

## G4d -- the hashing trick

`make hash-bench` · blake2b over the original string ids, dim 64, fp32.
**65,238 catalogue items, of which 7,179 (11.0%) ever receive a gradient.**
Full table: **15.9 MiB**.

| buckets | vs catalogue | catalogue | expected | trained | **traffic** | saved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65,238 | 100% | 63.4% | 63.2% | **9.2%** | 6.6% | 0.0 MiB |
| 32,619 | 50% | 86.6% | 86.5% | **19.3%** | 17.5% | 8.0 MiB |
| 16,309 | 25% | 98.2% | 98.2% | **35.3%** | 37.0% | 11.9 MiB |
| 6,523 | 10% | 100.0% | 100.0% | 66.6% | 61.2% | 14.3 MiB |
| 3,261 | 5% | 100.0% | 100.0% | 89.1% | 90.7% | 15.1 MiB |
| 652 | 1% | 100.0% | 100.0% | 100.0% | 100.0% | 15.8 MiB |

Three denominators, because "the collision rate" is three different questions:

- **`catalogue`** -- every item sharing a bucket. What the manual asks for.
- **`trained`** -- of the 7,179 gradient-receiving items, those sharing with
  *another trained item*. An untrained neighbour never updates, so it cannot
  interfere.
- **`traffic`** -- probability a random training click lands on such an item.

### The finding: the headline rate overstates by sevenfold

At full width **63.4%** of catalogue rows share a bucket and only **9.2%** of
trained rows share with another trained row. 89% of this catalogue never receives
a gradient, so most of what the headline counts is two permanently-untouched rows
landing together, which costs nothing. **`trained` is the number to quote.**

### The null: traffic weighting does not matter

`traffic` tracks `trained` within a few points at every width and crosses it
twice. The reason is clear in hindsight and should have been predicted: **a
uniform hash is independent of popularity**, so a click is no more likely to land
on a colliding item than a random trained item is to be one. The metric was worth
defining -- it is what would EXPOSE a popularity interaction -- and it reports
that on this corpus there is none.

### The control

Measured `catalogue` matches the closed form `1 - (1 - 1/b)^(n-1)` to within **0.2
points at every width**. These numbers are about the technique, not about blake2b.

**Hashing into as many buckets as there are items still collides.** The 100% row
is not a control that should read 0%: a uniform hash into `n` buckets leaves
roughly `1/e` of them empty and puts the crowding elsewhere.

### At scale, and this half is arithmetic

| catalogue | full table | at 50% buckets | saved | expected collisions |
| --- | ---: | ---: | ---: | ---: |
| MIND-small | 0.02 GiB | 0.01 GiB | 0.01 GiB | 86.5% |
| 2M design target | 0.48 GiB | 0.24 GiB | 0.24 GiB | 86.5% |
| 85.6M crossover | 20.42 GiB | 10.21 GiB | 10.21 GiB | 86.5% |

**At a fixed bucket ratio the collision rate is scale-free.** Every row reads
86.5%, because `1 - (1 - 2/n)^(n-1)` tends to `1 - e^-2` for any large `n`.
Halving the table costs the same collision rate at 65 thousand items as at 85
million; only the bytes saved change. That is simultaneously the whole argument
for the technique and the whole argument against it here -- 8 MiB on a card with
24,564.

**No 2M-item corpus was hashed.** Those rows are arithmetic and a closed form.

### What is NOT measured

**The recall cost.** Nothing here trains a model with a hashed table, so nothing
here says what a collision does to Recall@100. The collision rate is an input to
that question, not an answer. The item-tower gate bounds how much it could matter
on this corpus: the entire ID table is worth +0.0049 at the most, and that delta
is inside seed noise.

## G4e -- what actually ran

`models/tests/test_sharded_distributed.py`, two gloo ranks on CPU, spawned.

| property | result |
| --- | --- |
| plan distributes across both ranks | yes -- `table_wise`, one table per rank |
| every rank receives identical embeddings | yes |
| backward updates a table | yes |

A plan is not a run. Under table-wise sharding one rank does not hold the `item`
table at all and has to receive those rows over the wire, so this is the property
that exists only with two processes: **a rank that does not hold a table still
gets the right vector back.** A broken redistribution gives each rank only what it
owns, which from inside one process is indistinguishable from working.

**Row-wise is NOT exercised.** The CPU sharder offers four sharding types where an
accelerator offers seven, and row-wise -- the split `make shard-plan` reports at
the crossover -- is one of the three it lacks. Two NCCL ranks on one 4090 is not a
workaround: NCCL does not support two ranks sharing a device. Closing that gap
needs hardware this project does not have.

### A parity result, and an architectural consequence

`models/tests/test_sharded_embeddings.py` asserts an `EmbeddingBagCollection`
returns what an `nn.Embedding` returns for the same indices -- without which "we
measured TorchRec and did not adopt it" is a comparison between two different
models. The equivalence is **conditional**: an embedding *bag* pools, and SUM over
a bag of one is the identity. The two-tower feeds one item id per row, so it holds.

Measured alongside it: **a collection's forward is all-or-nothing over its
tables.** It walks every table and indexes the input by that table's feature name,
so a batch carrying only `item_id` raises `KeyError` even when only the item
embedding is wanted. Asking for one table's output alone is not expressible.

That lands on the property this project is built around.
`TwoTower.encode_item` is a function of `item_idx` **alone**, which is what lets
the Part J index be precomputed as `encode_item(arange(1, n_items+1))`. A
two-table collection would force a dummy user feature through every item-tower
call. Two separate collections avoid it -- at the cost of the planner losing the
chance to balance both tables against each other, which is most of the reason to
use TorchRec at all. **A fourth independent argument against adoption at this
scale**, alongside the unsupported +0.0049, the 15.9 MiB, and the 1,312x.

---

# Sequence pooling (Part H3)

**Hardware: RTX 4090, 24,564 MiB, WSL2.** What the SASRec user-tower pooler costs
to train and to serve, against the mean pooler it replaces. Quality -- which is
what actually decided H3 -- is in
[retrieval.md](retrieval.md#sequence-pooling-part-h).

## Serving: per-request user encode

`uv run python -m scripts.probe_encode_latency <pooled.pt> <sasrec.pt>` ·
10,000 timed iterations, **batch size 1**, 100 warm-up calls discarded, 512 real
validation histories cycled.

| encoder | p50 ms | p90 ms | p99 ms |
| --- | ---: | ---: | ---: |
| mean pool | **0.2800** | 0.3322 | 0.4393 |
| SASRec 2x2 | **1.0150** | 1.0784 | 1.3290 |
| ratio | **3.63x** | 3.25x | 3.03x |

**Quote the p50 ratio.** The tail ratio is *smaller*, and that is an artefact
rather than a result: scheduling jitter is additive, so a fixed ~0.15 ms inflates
the smaller number proportionally more. Both encoders are under 1.5% of
`docs/design.md`'s 90 ms budget, so latency did not decide anything here.

### What the numbers exclude

User-tower forward only. **No feature fetch, no ANN search, no serialisation, no
network, no host-to-device copy** -- requests are built on the device before the
timer starts. This is a floor on request latency and not a budget; 0.28 ms
against 90 ms does not mean 89.72 ms is spare.

### The measurement has a floor and a resolution, and they are different

| | p50 | p99 |
| --- | ---: | ---: |
| floor -- empty synchronised loop | 0.0027 | 0.0038 |
| resolution -- spread over 3 repeats of one identical configuration | **0.0067** | **0.0489** |

The floor says the timer can see the work; both encoders sit 100x above it. The
**resolution** says what "no difference" looks like on this machine, and it is
the number that governs any comparison between two rows. Two figures can each be
a hundred times the floor while their difference sits inside the noise.

That is not hypothetical. **Trimming each request to its real history length was
reported as a 2.7% saving and is a null**: measured at +0.0019 ms (mean pool) and
+0.0063 ms (SASRec) against a resolution of 0.0067, and not reproducible in
magnitude across runs. At batch 1 the cost is kernel launches, not arithmetic, so
a 29-slot history costs what a 50-slot one costs. The probe now prints
`NOT resolvable` on both rather than leaving the subtraction to the reader.

**p99 is measured but not precise.** It moves 0.0489 ms across identical
repeats even at 10,000 iterations, because a p99 rests on its slowest 1% and
those are scheduling outliers. At 1,000 iterations it moved 37%.

## Training: seconds per epoch

Two budgets per encoder, so the fixed setup cost **cancels**: a wall-clock total
is `setup + epochs x rate`, and `(t6 - t3) / 3` is the rate alone. Same arm
(`both`), `--patience 99` so neither run early-stops.

| encoder | 3 epochs | 6 epochs | **s/epoch** | implied setup |
| --- | ---: | ---: | ---: | ---: |
| mean pool | 6.7s | 11.3s | **1.533** | 2.10s |
| SASRec 2x2 | 9.8s | 17.1s | **2.433** | 2.50s |

**SASRec costs 1.59x per epoch.** The two independently implied setup constants
agree to 0.4s, which is the check on the linear model -- if it were wrong they
would not.

### Why the first attempt at this was wrong

Run durations pulled straight from MLflow gave 34.4s over 10 pooled epochs and
16.9s over 6 SASRec epochs, which reads as SASRec training **faster** -- and the
8-head/6-block variant faster still. More capacity cannot cost less time, so the
numbers were not published.

Fitting the controlled runs above explains it. Predicted against observed:

| run | predicted | observed |
| --- | ---: | ---: |
| SASRec, 6 epochs | 17.1s | 16.9s |
| mean pool, 10 epochs | 17.4s | **34.4s** |

**The 34.4s pooled run is the outlier**, by a factor of two, for reasons its
record does not carry -- different flags, a contended GPU, geometry logging. The
whole "SASRec trains faster" reading came from that one number. Inverting it
gives the physically sensible answer: the encoder that is 3.63x slower to serve
is 1.59x dearer to train.

**Single runs, and that is the caveat.** Each cell above is one measurement, and
the strongest evidence on this page about training wall-clock is that it varied
2x under uncontrolled conditions. The latency figures have a measured resolution;
these do not.

## Latency: before and after optimization

`TODO` -- Part M. `docs/design.md` carries the 90 ms budget with the `Actual`
column empty, deliberately: a budget derived from measurements already taken is a
description, not a budget.

The user-encode figures above are **one component** of that budget, not a draft
of it. They exclude the feature fetch, the ANN search and the network, which are
where a 90 ms budget is actually spent.

**The harness exists; the numbers do not.** `make bench` sweeps the orchestrator
with `ghz` and writes the table below. Two properties of it are worth stating
before any number lands here, because both are ways this table is commonly
wrong rather than merely absent:

**Every request uses a different user.** ghz's default is one payload repeated,
which on this system measures the warm path and nothing else -- the same
`user_id` hits the same cached embedding in the retrieval sidecar, the same
seen-list key in Redis, and the same Feast row in the gateway. That produces a
beautiful table describing a system nobody runs. The harness templates the
request number into the id across a pool of 10,000 users.

**The sweep runs to saturation rather than to a chosen rate.** A single QPS
figure says nothing without knowing how close to the limit it was. Saturation
is flagged on two independent symptoms -- p99 past the 100 ms budget, or
throughput that stopped tracking the offered rate -- because they come apart: a
service can hold its latency and refuse work, or accept everything and get
slower. A sweep watching only one reports the knee in the wrong place.

§14.4 wants the before/after ladder (naive → batched → cached → Go → INT8), and
this project cannot produce most of those rows honestly: there is no Python
serving implementation to be the "before", and no INT8 arm. What it can produce
is the current stack's curve and a per-stage breakdown from
`recsys_stage_duration_seconds`. A ladder with invented rows would be worse
than a single measured column.

---

# The ANN index (Part J)

**Hardware: RTX 4090 host, faiss-cpu 1.15.1 · 65,238 items x 128d · 19,006
queries · nlist=1021, m=32, nbits=8 · HNSW M=32, efConstruction=200.**

Everything here is **measured**: real indexes, real searches, real clocks. The
quality side -- and one retracted finding -- is in
[retrieval.md](retrieval.md#the-ann-index-part-j).

## The sweep

Seed A, checkpoint `both-logq-n4u0-...-ab7e1d500b9bf792`. `agreement` is overlap
with exact search's top 100; `recall` is the share of clicked articles found.

| index | param | agreement | recall@100 | vs exact | p50 ms | p99 ms | QPS | MB |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| flat | exact | 1.0000 | 0.3776 | -- | 0.354 | 0.511 | 27,404 | 33.4 |
| hnsw | efSearch=16 | 0.6396 | 0.3179 | -0.0600 * | 0.013 | 0.028 | 1,041,489 | 51.2 |
| hnsw | efSearch=32 | 0.8233 | 0.3602 | -0.0183 * | 0.018 | 0.035 | 695,961 | 51.2 |
| hnsw | efSearch=64 | 0.9499 | 0.3744 | -0.0033 * | 0.024 | 0.043 | 467,588 | 51.2 |
| hnsw | efSearch=128 | 0.9866 | 0.3769 | -0.0005 | 0.036 | 0.055 | 319,588 | 51.2 |
| hnsw | efSearch=256 | 0.9960 | 0.3774 | -0.0004 | 0.072 | 0.098 | 136,265 | 51.2 |
| hnsw | efSearch=512 | 0.9997 | 0.3776 | -0.0000 | 0.162 | 0.210 | 63,395 | 51.2 |
| ivfpq | nprobe=1 | 0.1653 | 0.1776 | -0.1971 * | 0.012 | 0.023 | 34,291 | 3.3 |
| ivfpq | nprobe=4 | 0.5522 | 0.3235 | -0.0525 * | 0.017 | 0.030 | 33,202 | 3.3 |
| ivfpq | nprobe=16 | 0.8901 | 0.3772 | -0.0002 | 0.027 | 0.040 | 34,176 | 3.3 |
| ivfpq | nprobe=32 | 0.9085 | 0.3782 | +0.0008 | 0.039 | 0.058 | 35,023 | 3.3 |
| ivfpq | nprobe=64 | 0.9098 | 0.3792 | +0.0021 | 0.063 | 0.100 | 32,134 | 3.3 |
| ivfpq | nprobe=128 | 0.9099 | 0.3797 | +0.0028 * | 0.109 | 0.137 | 40,741 | 3.3 |
| ivfpq | nprobe=1021 | 0.9099 | 0.3798 | +0.0029 * | 0.863 | 0.987 | 10,009 | 3.3 |

`nprobe=1021` searches every cell, so what remains is quantisation and nothing
else. Without that row a difference from exact search cannot be told apart from
having looked at only part of the catalogue.

⚠️ **The positive rows do not reproduce on a second checkpoint** -- see
[retrieval.md](retrieval.md#two-seeds-and-the-second-one-retracts-a-finding).
They are left in because the retraction is the result.

## What the three indexes actually trade

| | wins on | costs | measured |
| --- | --- | --- | --- |
| flat | exact, no parameters, no build | linear in catalogue size | 33.4 MB, 0.51 ms p99 |
| HNSW | **latency**, ~10x QPS | **more** memory than the vectors | 51.2 MB |
| IVF-PQ | **memory**, 10x smaller | lossy, trained, no speed here | 3.3 MB, ~33k QPS flat across `nprobe` |

**IVF-PQ is not faster than brute force on this corpus.** Skipping 98% of the
catalogue does not pay when the catalogue is small enough that the lookup-table
overhead eats the saving. Only HNSW's graph walk converts to speed at this size.

**HNSW costs more memory than storing the vectors exactly.** `IndexHNSWFlat`
keeps the vectors intact and adds the graph, so 33.4 MB becomes 51.2 MB. It
approximates the *search*, not the data, which is why cranking `efSearch`
converges on exact answers.

## When exact search stops being enough

Brute force is one pass over the table: **5.6 ns per item per query** from the
0.354 ms p50 over 65,238 items.

| retrieval budget | items a linear scan covers |
| ---: | ---: |
| 1 ms | ~180,000 |
| 10 ms | **~1,800,000** |
| 50 ms | ~9,000,000 |

MIND-small sits at 65,238, **27x below the 10 ms line**. The 2M-item design
target is roughly at it. The same arithmetic as the sharding crossover, and the
same conclusion: build it, measure it, and state where it would start to matter.

## Two runs, and the parameter stability problem

Seed B, checkpoint `...-07934b9e86dd5c4c`, same everything else:

| | seed A | seed B | moved by |
| --- | ---: | ---: | ---: |
| flat recall@100 | 0.3776 | 0.3668 | 0.0108 |
| hnsw ef=128 agreement | 0.9866 | 0.9876 | 0.0010 |
| hnsw ef=128 **vs exact** | -0.0005 | **-0.0034** * | **7x** |
| ivfpq exhaustive agreement | 0.9099 | 0.9098 | 0.0001 |
| ivfpq exhaustive **vs exact** | **+0.0029** * | **-0.0021** | sign |

**Agreement is stable across checkpoints to within 0.001. What that agreement
costs in clicks is not.** So an `efSearch` chosen against one checkpoint can be
indistinguishable from exact on that one and significantly worse on the next.
An index rebuilt on a schedule against a retrained model must either re-tune its
parameters per rebuild or use an index that has none.

That is the operational argument for exact search, and it is separate from the
latency one.

## Index: recall@k vs. QPS

Measured above.

## Diversity/relevance tradeoff

The table itself is quality, so it lives in
[ranking.md](ranking.md#the-trade-off-table). Short version: MMR is a null on
this corpus, exploration buys 84% more catalogue coverage for -0.0005 NDCG, and
the seen filter buys the most coverage at 45x exploration's price.

---

# The seen-list (Part L)

**Redis 7.4.11-alpine, `redis://localhost:6379/1` · `make seen-bench` for the
arithmetic, `make rerank RERANK_ARGS="... --seen"` for the live figures.**

Feast owns db 0 on the same instance. Sharing a keyspace with a feature store
means one `FLUSHDB` during a materialisation takes every seen-list with it.

## Sizing, and the saving that is not what it looks like

Target rates against measured, at three capacities. The measured rate is
consistently a little above target because the double-hashing scheme is not the
independent-hash assumption the closed form uses; bit load lands at ~50%, which
is the optimum, so the sizing itself is right.

| held | target | measured | bits | payload B | load |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.010 | 0.0135 | 480 | 60 | 52.3% |
| 200 | 0.010 | 0.0090 | 1,918 | 240 | 50.6% |
| 1,000 | 0.010 | 0.0110 | 9,586 | 1,198 | 52.1% |

Bits per item is `-log2(p) / ln 2`, **a function of the error rate alone** --
1% costs 9.6 bits per item however many items a user holds. On paper that makes
the saving over exact 8-byte ids a constant 85% at every capacity: you buy an
error rate, not a capacity.

**That property is arithmetic and does not survive deployment.** Measured with
`MEMORY USAGE` against a live key: a 1,918-bit filter is 240 B of payload and
**504 B in Redis** -- 2.1x. The extra ~264 B is key name, object header, SDS
header and dict entry, and it is per KEY, so it does not scale with capacity:

| held | payload B | + overhead | exact B | real saving |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 60 | 324 | 400 | **19.0%** |
| 200 | 240 | 504 | 1,600 | 68.5% |
| 1,000 | 1,198 | 1,462 | 8,000 | 81.7% |

**The saving stops being constant and starts growing with capacity.** At a
50-item seen-list the filter barely pays for itself against a plain set. The
structure earns its place at hundreds of items per user, not tens -- and a
bits-only estimate claims 85% at every row, wrong at exactly the sizes where
someone might reasonably reach for the simpler thing.

## Overfilling is a cliff, not a slope

Sized for 100 items at a 1% target, then given more:

| held | measured rate | bit load |
| ---: | ---: | ---: |
| 100 | 1.1% | 51% |
| 200 | 13.7% | 75% |
| 400 | 65.0% | 94% |
| 1,000 | **100.0%** | **100%** |

At ten times capacity every bit is set and the filter answers "seen" to every
candidate. **A seen-filter that hides everything does not return a slightly
worse slate; it returns an empty one** -- and the one-sided guarantee still
holds while being worth nothing.

So capacity is a number to monitor rather than to set once, and **bit load is
the alarm, not the error rate**: it is observable per user and it moves long
before the rate does. The selector carries the matching guard -- a block mask
that removes every candidate is ignored, because a blank page is a product
decision nobody made.

## Serving shape

**One `GET`, not `k x n` `GETBIT`s.** A request tests ~100 candidates against a
4-hash filter: 400 round trips would consume the whole re-rank budget. The
bitmap is 60-1,200 B, so fetching it whole and testing locally is one round
trip. **At serving time the cost is round trips, not bit arithmetic**, and that
inverts the obvious implementation.

**Writes are one transaction.** The `k` SETBITs per shown item plus the EXPIRE
go in a `MULTI`/`EXEC` pipeline. Two concurrent requests for one user could
otherwise interleave and leave an item's bits partially written -- which is the
one way this structure *can* produce a false negative and re-show something.

⚠️ **The TTL contradicts the one-sided guarantee, and both are load-bearing.**
"Recently shown" needs a window, or the filter grows until it saturates and
hides everything. The window is an `EXPIRE`. But an `EXPIRE` is a scheduled
reset, and the never-a-false-negative promise holds only while the filter is
never cleared. **The mechanism that bounds memory is the mechanism that
reintroduces the failure users notice.** The trade is unavoidable; what a design
chooses is where to put it -- a long TTL re-shows rarely and costs memory, a
short one is cheap and re-shows sooner. It is a product decision about how long
"recently" means, and it should be written down as one rather than inherited
from a default.

## What it cost in the funnel

Replayed sequentially over the 5,722 held-out requests, capacity 200 at a 1%
target: **blocked 62,730 candidates, 10.96% of all rows**, at a 35.5% bit load
on a sampled user.

| | value |
| --- | ---: |
| NDCG@10 | 0.0883 (from 0.1283) |
| paired vs ranker only | **-0.0227** [-0.0259, -0.0195] \* |
| coverage of the pool | 15.0% -> **42.2%** |
| distinct items served | 190 -> **536** |
| long-tail share | 2.9% -> **5.7%** |

Sequential, not batched, because a seen-list is a function of what the user was
served *earlier* -- every other arm in the trade-off table is order-independent
and this one is not.

**It buys the most coverage in the table and is the wrong instrument for buying
coverage**: ~0.00083 NDCG per point, against exploration's ~0.00004. It is a
correctness requirement whose coverage gain is a side effect, and it belongs in
the table for its cost rather than as a rival to MMR.

---

# Serving (Part M)

## The retrieval fallback, and an 11x that is not about the algorithm

ADR 0013 puts the FAISS index behind a Python sidecar and keeps exact search in
the Go process as rung 2 of the degradation ladder. `go test ./internal/index
-bench BenchmarkSearch` measures that rung. AMD Ryzen 9 7900X, `k=100`, one
query at a time:

| shape | table | ns/op | per item per query | scanned |
| --- | ---: | ---: | ---: | ---: |
| 65,000 x 128 (MIND-small) | 33.3 MB | **3,923,312** | 60.4 ns | 8,483 MB/s |
| 160,000 x 128 (design target) | 81.9 MB | 9,508,814 | 59.4 ns | 8,615 MB/s |

Two allocations and 832 B per query, which is the k-sized result pair and is
not worth optimising.

**Set against the same algorithm in FAISS on the same corpus and the same box,
this is 11x slower.** The Part J sweep measured `IndexFlatIP` at **0.354 ms
p50** — and that row is directly comparable, because `benchmark_index.py`
measures latency at batch size 1 on purpose ("a batched sweep divided by the
batch size is throughput wearing a latency label"). 5.4 ns per item per query
there, 60.4 ns here.

**The gap is SIMD, not algorithm.** Both do one pass over the table computing
one inner product per row. 8.32M multiply-adds in 3.92 ms is 2.1 G FMA/s, which
is about half of what one core retires with a *scalar* FMA per cycle — and Go's
compiler does not auto-vectorise, so the inner loop is scalar. FAISS compiles to
AVX2/AVX-512 and does 8-16 lanes per instruction. The ratio lands where the lane
count says it should.

⚠️ **An earlier reading of this called the scan memory-bandwidth-bound, on the
evidence that MB/s is flat across a 2.5x change in table size. That inference
was wrong.** Every byte of the table feeds exactly one multiply-add, so bytes
per second and FLOP per second are proportional by construction — flat MB/s
shows only that cost is linear in table size, which a compute-bound scan does
too. At 8.5 GB/s against ~2.1 G FMA/s on a scalar loop, the arithmetic is the
binding constraint, not the memory bus.

### What it means for the ladder

**Latency: rung 2 holds.** `docs/design.md` budgets 25 ms for the whole
five-source fan-out. 3.9 ms fits, with the caveat that at the 160K design target
it is 9.5 ms — 38% of the retrieval budget consumed by one degraded source.

**Capacity: rung 2 is not free.** At the design doc's 500 peak QPS, 3.92 ms of
CPU per request is **~2.0 cores of scan alone**, and the guide's own Kubernetes
manifest (§15) sets `limits: {cpu: "2"}`. A sidecar outage at peak would spend
the entire pod budget on the fallback, leaving nothing for the blend, the
re-ranker, serialisation or the Triton call. So rung 2 is a fallback for a
degraded *service*, not a capacity-neutral one, and the honest operational
statement is that a sidecar outage at peak needs load-shedding rather than
transparent failover.

**What is NOT measured:** the sidecar hop itself. Everything above is the
fallback path. The primary path — gRPC to Python, `encode_user`, HNSW search —
has no number yet, and until `ghz` produces one, the claim that the sidecar fits
the 25 ms budget is a design intention rather than a result.
