# Vendored Triton protobufs

**Third-party, fetched not written.** These come from
[triton-inference-server/common](https://github.com/triton-inference-server/common),
and nothing in this repository edits them. `make triton-proto` downloads them;
`make proto` generates the Go stubs alongside this project's own contracts.

```
make triton-proto           # fetch at TRITON_PROTO_REF
make proto                  # regenerate stubs for every .proto
```

## Why they are vendored rather than fetched at build time

A build that reaches the network to define its own wire format is a build that
can produce two different binaries from one commit. Vendoring makes the
generated client a function of the checked-in tree, which is the same reason
`serving/testdata/*.json` is committed rather than regenerated in CI.

## The version has to be pinned, and the default is not a pin

`TRITON_PROTO_REF` defaults to `main`, which is a moving target and is **wrong
for anything real**. Set it to the tag matching the Triton server actually being
run — the container tag and the proto ref must agree, because a client generated
from a newer `grpc_service.proto` can send fields an older server ignores
silently. That failure looks like a model that quietly disregards a setting,
which is considerably worse than a connection error.

```
make triton-proto TRITON_PROTO_REF=r24.08
```

## Why the `M` flags exist

**Neither proto declares `option go_package`** -- checked, not assumed. A
vendored schema generated for many languages usually omits it, because it cannot
know one language's directory layout. Without it `protoc-gen-go` has no import
path to emit and refuses, so `make proto` supplies one per file:

```
--go_opt=Mgrpc_service.proto=<module>/internal/tritonpb
--go_opt=Mmodel_config.proto=<module>/internal/tritonpb
```

Go stubs only. Nothing in Python talks to Triton: the offline path scores with
torch directly, and the serving path is the Go orchestrator.

## Why gRPC rather than REST

The first client here spoke Triton's HTTP/REST v2 API and the reasoning was
proportion: one request carries ~100 x 11 float32, about 4 KB, and JSON encoding
of 4 KB is not where a 90 ms budget goes.

gRPC replaces it for reasons that are about the connection rather than the
payload:

- **Persistent HTTP/2 connections.** The REST client opened a connection per
  request under Go's default transport unless carefully pooled; gRPC multiplexes
  over one, which removes a TCP and TLS handshake from the hot path.
- **No float-to-text round trip.** JSON encodes every float as decimal text and
  parses it back. That is both slower and lossy at the last bit, and this
  project has already been bitten once by float width changing an argmax.
- **Deadlines travel with the call.** A Go context deadline becomes a gRPC
  deadline the server honours, instead of a client-side timeout that abandons a
  request the server keeps working on.
- **Typed messages.** A misspelled field in a JSON body is ignored; in a
  generated struct it does not compile.

The cost is this directory: a vendored third-party schema and the build step
that consumes it.
