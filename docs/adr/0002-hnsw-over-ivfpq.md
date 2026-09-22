# ADR 0002 — HNSW for the serving index, at efSearch=512

**Status:** accepted · **Date:** 2026-09-22 · **Supersedes the stub of the same
number, whose title promised a decision that had not been measured.**

## Context

Part J built all three candidate indexes over the item tower's 65,238 x 128
output and swept them: exact (`IndexFlatIP`), a navigable graph
(`IndexHNSWFlat`, M=32, efConstruction=200) and a clustered compressed index
(`IndexIVFPQ`, nlist=1021, m=32, nbits=8). Full tables in
[benchmarks.md](../benchmarks.md#the-ann-index-part-j).

Two quantities are both called recall and are not the same. **`agreement`** is
overlap with exact search's top 100 -- a property of the index. **`recall`** is
the share of clicked articles found -- a property of the system. An index can
drop candidates nobody was going to click.

| index | param | agreement | p50 ms | p99 ms | QPS | MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| flat | exact | 1.0000 | 0.354 | 0.511 | 27,404 | 33.4 |
| hnsw | efSearch=128 | 0.9866 | 0.036 | 0.055 | 319,588 | 51.2 |
| **hnsw** | **efSearch=512** | **0.9997** | **0.162** | **0.210** | **63,395** | **51.2** |
| ivfpq | nprobe=1021 | 0.9099 | 0.863 | 0.987 | 10,009 | 3.3 |

## Decision

**Serve HNSW at `efSearch=512`. Keep exact search as the offline reference
index**, which is what every quality number in this project is measured
against.

**This overrides Part J's own corpus-level recommendation, deliberately, and
the reason is the design target rather than the measurement.** On MIND-small,
exact search wins outright: 33.4 MB, 0.51 ms p99, no build step, no parameters
to tune, exact by construction. A linear scan costs **5.6 ns per item per
query**, so on a 10 ms retrieval budget it stays viable to roughly **1.8
million items** — 27x this corpus, and slightly above the 2M design target.
Shipping exact here and revisiting at 1.8M would be the right call for this
catalogue and the wrong shape for a serving path meant to demonstrate one.

**`efSearch=512`, not the 128 an ANN benchmark would pick.** 128 is where the
throughput headline lives — 319,588 QPS, a 10x gain — and it is also where the
recall cost stops being stable across checkpoints: **-0.0005 (not resolvable)
on seed A and -0.0034 (significant) on seed B.** 512 agrees with exact search
on **99.97%** of candidates, costs -0.0000 and -0.0013 on the same two seeds,
and still runs **2.3x the throughput of exact search** at a 0.21 ms p99. The
throughput this project can actually use is bounded by the ranker and the
feature fetch, not by the index, so trading 5x of unused QPS for near-exact
agreement is the cheap side of the trade.

## Consequences

**An obligation, not just a footprint: the parameter must be re-tuned on every
index rebuild.** This is Part J's sharpest operational finding. Agreement is
reproducible across checkpoints to within 0.001 at every setting, but **what
that agreement costs in clicks is not** — it moved 7x for HNSW at efSearch=128
between two seeds and reversed sign for IVF-PQ. An index rebuilt nightly
against a retrained model therefore needs a per-rebuild validation gate that
measures recall against the exact index on a held-out probe set, not a
parameter chosen once and inherited. `efSearch=512` is chosen partly because it
has the most margin before that instability bites.

**More memory than storing the vectors exactly.** 51.2 MB against 33.4 MB.
`IndexHNSWFlat` keeps the vectors intact and adds the graph, so it approximates
the *search* and not the data — which is exactly why raising `efSearch`
converges on exact answers, and why the graph is not a compression technique.

**IVF-PQ is rejected on measurement, not on principle.** It is 10x smaller
(3.3 MB) and buys no speed at all here: ~33k QPS flat across every `nprobe`,
because skipping 98% of a small catalogue does not pay once the lookup-table
overhead is counted. Its quality result was also the one Part J retracted —
significant and positive on one checkpoint, insignificant and negative on the
next.

**Every offline number in this project remains an exact-search number**, and
that must stay true. Measuring quality through the serving index would fold the
index's error into every model comparison, which is how an ANN parameter ends
up looking like a modelling result.

**The serving path inherits a build step and a version.** `index_version` is in
the response contract (`serving/proto/recsys.proto`) for this reason: an
approximate index that was rebuilt is a different index, and a response that
cannot name which one answered it cannot be attributed after the fact.

## Alternatives considered

**Exact search (`IndexFlatIP`).** Wins on this corpus on every axis that
matters and is what Part J recommended. Retained as the offline reference and
as the fallback if the serving index fails to load. Revisit as the serving
choice if the catalogue stays below ~1M items.

**HNSW at efSearch=128.** The throughput-optimal point, rejected for the
measured seed instability above. If a future workload is genuinely
QPS-bound, this is the setting to revisit — with the per-rebuild gate in place
first.

**IVF-PQ.** Rejected: no speed gain at this scale, a lossy trained index, and
the one arm whose quality finding did not reproduce.

**ScaNN.** Not evaluated. FAISS covers the three structural families that
matter here (exact, graph, quantised) and adding a fourth library would not
change the shape of the answer.
