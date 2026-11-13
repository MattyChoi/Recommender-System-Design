# ADR 0013 — Static linking via musl, at 61.9 MB

**Status:** accepted · **Date:** 2026-11-13

## Context

The guide's N1 specifies the standard Go container recipe: `CGO_ENABLED=0`,
`golang:1.23-alpine`, a `distroless/static-debian12` runtime, ~20 MB. **That
recipe cannot build this binary.**

The impression log uses `confluent-kafka-go`, which wraps librdkafka — C. So
cgo is mandatory, and `CGO_ENABLED=0` does not compile. `go.mod` also says
`go 1.27.1`, past the pinned toolchain. Adopting Kafka therefore forced a choice
the guide does not present, and this is the first direct, unavoidable cost of
that decision.

Two options actually exist:

| | runtime base | libc in the image | DNS |
| --- | --- | --- | --- |
| **static via musl** | `distroless/static` | none | works |
| static via glibc | `distroless/static` | none | **broken** |
| dynamically linked | `distroless/base` | glibc, to be patched | works |

**The middle row is why this is an ADR rather than a build flag.** A statically
linked **glibc** binary cannot resolve hostnames: `getaddrinfo` loads NSS
modules through `dlopen` at runtime, and a static binary has no dynamic loader
to do it with. The failure mode is specific and ugly — IP addresses work,
names do not — and every backend this service talks to is addressed by name:
`features`, `retrieval`, `triton`, `kafka`, `schema-registry` under compose, and
`recsys-features`, `recsys-minio`, `prometheus` under Kubernetes, where Service
DNS *is* the addressing mechanism.

**musl resolves names in-tree**, parsing `/etc/resolv.conf` itself with no
`dlopen`, so a static musl binary keeps working. confluent-kafka-go ships a
prebuilt static musl librdkafka for exactly this combination.

⚠️ **This row is rejected on a documented property of glibc, not on a
measurement taken here.** Unlike ADR 0002's rejections, no static-glibc image
was built and probed. The property is well established and the cost of
confirming it is a broken deployment, so it was taken on authority.

## Decision

**Build static against musl, and keep `distroless/static`.**

```dockerfile
FROM golang:1.27-alpine AS build
RUN apk add --no-cache gcc musl-dev
RUN CGO_ENABLED=1 GOOS=linux \
    go build -tags musl -trimpath \
      -ldflags='-s -w -extldflags "-static"' -o /out/server ./cmd/server
RUN ! ldd /out/server 2>/dev/null | grep -q '=>' \
    || { echo "FAIL: dynamically linked -- check -extldflags"; exit 1; }

FROM gcr.io/distroless/static-debian12:nonroot
```

**`-tags musl` is not optional**, and not cosmetic: it is how
confluent-kafka-go selects its static musl librdkafka rather than expecting a
system one.

**The `ldd` guard is the enforcement.** Everything above it is intent;
that line is the only thing that makes it true. Without it, a future dependency
pulling in dynamic linkage produces an image that builds cleanly, passes every
test that runs outside a container, and then fails at container start on a base
image with no libc — in other words, the failure moves from CI to production.

## Consequences

**The image is 61.9 MB, and the guide's ~20 MB should not be quoted for it.**
Static librdkafka bundles OpenSSL, zlib, zstd, lz4 and SASL. The estimate
before measuring was ~25 MB; the measurement was 61.9 MB, so the estimate was
wrong by 2.5x and is recorded here rather than quietly dropped. The number is
the honest price of the Kafka decision, not a regression to be optimised away.

**An obligation, not just a footprint: the `ldd` guard must survive every
refactor of this Dockerfile.** It is the one line with no visible purpose on a
green build, which makes it the one most likely to be deleted as noise.

**The build now needs a C toolchain**, so it cannot use the cgo-free fast path,
and `apk add gcc musl-dev` is on the critical path of every uncached image
build. The repo already pays this elsewhere: `make go-race` guards for a C
compiler and prints install instructions per distro
(`make/quality.mk`), because the race detector is cgo instrumentation.

**Every future cgo dependency must have a musl build.** This narrows the
dependency set in a way that will not be obvious at the point of choosing a
library, and the symptom will be a link error deep in a vendored C project.

**No shell, no libc, no package manager in the runtime.** That is the security
win and it is also the debugging cost: nothing can be inspected from inside a
running container. The substitutes are deliberate — the Prometheus endpoint on
`:9102`, the three gRPC health names (ADR-adjacent; see
`infra/k8s/README.md`), and `-reflection` for ad-hoc gRPC calls, default off.

**The runtime user is uid 65532 and that leaks into callers.** Anything mounting
host files into this image must make them readable by that uid —
`tests/test_serving_smoke.py` has to pass `--user $(id -u)` because pytest's
temporary directory is `0700`, and `infra/k8s/recsys-api.yaml` names the uid
explicitly rather than inheriting it, so the Pod stays correct if the base
image's default ever moves.

## Alternatives considered

**`CGO_ENABLED=0`, as the guide specifies.** Not a trade-off, simply
impossible while the impression log uses confluent-kafka-go.

**A pure-Go Kafka client — `franz-go` or `segmentio/kafka-go`.** The serious
alternative, and the only one that would restore the guide's recipe and its
image size. Rejected for now rather than on merit: confluent-kafka-go gives the
Confluent wire format — the magic byte, the schema id, and the **varint
message-index prefix** that a plain protobuf reader chokes on — together with
Schema Registry integration, and that framing is the thing the whole
`BACKWARD_TRANSITIVE` story rests on. `franz-go` does have a schema registry
package, so this is reopenable. **Revisit if the C toolchain or the 61.9 MB
becomes a real constraint**, and expect the wire format to be the work rather
than the client swap.

**Dynamically linked onto `distroless/base`.** The simplest build: no
`-extldflags`, no musl tag, no `ldd` guard. Rejected because it puts a glibc
back in the image, and a libc in the image is a libc to patch and a CVE feed to
track — which is the main reason `distroless/static` exists at all. Worth
revisiting only if a dependency arrives with no musl build, at which point this
becomes the fallback rather than a preference.

**Alpine as the runtime base.** Keeps `apk` and a shell for debugging, at the
cost of shipping a package manager and a shell into production. Rejected on the
same reasoning as the previous entry, more strongly.
