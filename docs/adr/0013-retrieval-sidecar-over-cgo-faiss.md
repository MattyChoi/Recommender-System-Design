# ADR 0013 — A Python retrieval sidecar, not FAISS in the Go binary

**Status:** accepted · **Date:** 2026-09-22

## Context

Part M puts a Go orchestrator in the request path. ADR 0002 ships an HNSW
index, built by `indexing/build_index.py` with FAISS. Those two facts do not
compose on their own: **a FAISS index is a C++ artifact with a Python API, and
the Go process cannot read one.** The guide is silent on the join — §11 builds
the index in Python, §11.2 describes the lifecycle ("serving nodes hot-reload")
without naming what does the reading, and §14.2 shows retrievers only as an
interface:

```go
got, err := src.Retrieve(srcCtx, req.UserId, userFeats, history)
```

The same gap covers the **user tower**. Retrieval is `encode_user` then a
nearest-neighbour lookup, and Go cannot run a PyTorch forward pass either. Any
answer for the index has to answer for the encoder at the same time, because
they are two halves of one operation.

## Decision

**A Python retrieval service, called over gRPC, owning both the two-tower user
encoder and the FAISS index.** It is a retriever like any other in the fan-out:
its own address, its own deadline, its own entry in `degraded_sources`.

Scope is deliberately narrow — **vector search only**. Trending,
co-visitation and recently-viewed are precomputed lists that Go reads directly
from Redis. Putting them behind Python would add a language boundary and a
process to three sources whose entire implementation is a key lookup, and would
make one crash take out all of retrieval instead of one source of five.

## Alternatives rejected

**cgo bindings to libfaiss.** Removes the hop and keeps one process. Rejected
on two grounds. It puts a C++ toolchain and a shared library into the serving
binary's build, which is a real operational cost for a service whose entire
pitch is that it is a small static Go binary. More importantly it does not
solve the user tower: a cgo-linked FAISS index still needs a query vector, so
either PyTorch also gets linked in — which is not a thing one does — or a
second remote call appears anyway and the hop was never avoided.

**FAISS behind Triton's Python backend.** Genuinely close, and its argument is
good: Triton is already a dependency, so Go would talk to exactly one
model-serving address for both retrieval and ranking. Rejected because it
couples the index lifecycle to Triton's model-repository reload, and §11.2's
lifecycle — versioned artifacts, a promotion gate, an atomic pointer swap,
hot-reload — is a deliberate piece of this project that would have to be
re-expressed in Triton's terms. Worth revisiting if the sidecar's operational
cost turns out to exceed that.

**Exact search in Go, no index at all.** At ~160K items × 64 dims, one query is
~10M multiply-adds, which plausibly fits the 25 ms retrieval budget. Rejected
as the *primary* path, for a reason that is about design rather than speed:
every retriever in §14.2 has a budget, can be dropped, and is reported as
degraded, and **an in-process matmul cannot time out**. Making the strongest
source incapable of degrading would hollow out the fan-out architecture that
Part I measured and §10.3 specifies. It would also retire the ANN index that
ADR 0002 and the §11 recall/QPS benchmark are about, on latency grounds that
were never the reason the index was chosen.

## Consequences

**A hop appears in the budget.** `docs/design.md` budgets 25 ms for the whole
five-source parallel fan-out. A local gRPC round trip plus an HNSW search has
to fit inside that, and the `ghz` table (§14.4) is what will say whether it
does. This ADR is not claiming it does; that number is not yet measured.

**A degradation ladder, and it is the point.** §14.4 already prescribes caching
user embeddings in Redis with a short TTL, to keep a tower forward pass off the
hot path. That cache also makes the sidecar's failure survivable:

1. Sidecar answers — HNSW over the promoted index.
2. Sidecar unreachable, user embedding cached — Go runs exact search
   in-process over the item-vector artifact (`serving/go/internal/index`).
   Correct results, more CPU, no ANN.
3. Neither — the source reports degraded and the pipeline's popularity
   fallback serves the slate.

**Rung 2 is now measured: 3.9 ms per query at MIND-small's 65K x 128, 9.5 ms at
the 160K design target** ([benchmarks.md](../benchmarks.md#the-retrieval-fallback-and-an-11x-that-is-not-about-the-algorithm)).
That fits the 25 ms retrieval budget, so the rung holds on latency. It does not
hold on capacity: at 500 peak QPS it is ~2.0 cores of scan, against the `cpu:
"2"` pod limit in the guide's own manifest. A sidecar outage at peak needs
load-shedding, not transparent failover — and this ADR is where that is written
down rather than discovered.

It is also 11x the cost of the SAME algorithm in FAISS (0.354 ms p50, Part J,
also batch-1), because Go's compiler does not auto-vectorise and FAISS compiles
to AVX. Worth knowing before anyone reads 3.9 ms as "what exact search costs".

Rung 2 is why `internal/index` exists. ADR 0002 already named exact search as
the load-failure fallback and `HealthResponse.index_kind` already reports
`"hnsw"` or `"flat"`; this gives that field something real to say. **It is a
fallback and not a baseline** — a server sitting on rung 2 is serving correct
results without the index, and the whole point of reporting `index_kind` is
that this is otherwise invisible in every system metric.

**Two processes to operate, and a second parity surface.** The sidecar and the
Go fallback must agree on what the top-k is, or the degraded path silently
serves a different slate than the healthy one. That is the fourth
cross-language parity surface in this project, after the blend, the re-ranker
and the feature vector. It is owed, not paid.
