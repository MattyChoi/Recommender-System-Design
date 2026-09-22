# Benchmarks

Each table records the hardware and the commit it was produced on. Sections still
marked `TODO` are not yet measured; nothing below is a placeholder number.

Retrieval quality lives in [results.md](results.md) under **Part G -- retrieval**.
This file is for the cost side: memory, sharding, latency and throughput.

## Quality vs. baselines

`TODO` -- see [results.md](results.md). Part F's ranking table and Part G's three
retrieval gates are both there.

## Per-source contribution and leave-one-out ablations

`TODO` -- Part I. G1's three-arm ablation (ID / content / both) is in
[results.md](results.md) and is the closest thing that exists.

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
agrees with G1's gate, which measured the entire ID table as worth **+0.0049
recall** while costing **0.0037 of cold-band recall**.

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
that question, not an answer. G1's gate bounds how much it could matter on this
corpus: the entire ID table is worth +0.0049.

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
scale**, alongside the +0.0049, the 15.9 MiB, and the 1,312x.

## Latency: before and after optimization

`TODO` -- Part M. `docs/design.md` carries the 90 ms budget with the `Actual`
column empty, deliberately: a budget derived from measurements already taken is a
description, not a budget.

## Index: recall@k vs. QPS

`TODO` -- Part J.

## Diversity/relevance tradeoff

`TODO` -- Part L.
