"""``make retrieval-serve`` -- run the retrieval sidecar.

    uv run python -m serving.retrieval --checkpoint data/checkpoints/<run>.pt

Everything that can be wrong lives in service.py, where it is testable without
a socket. This file opens a port and nothing else.
"""

from __future__ import annotations

import argparse
import signal
from concurrent import futures
from pathlib import Path
from typing import Any

import torch

from common.pb import retrieval_pb2_grpc
from models.retrieval.dataloader.dataset import CONTENT_VARIANTS
from serving.retrieval import cache as cache_module
from serving.retrieval.service import (
    DEFAULT_ARTIFACTS,
    DEFAULT_EF_SEARCH,
    DEFAULT_POINTER,
    RetrievalServicer,
    load,
)

#: gRPC handler threads. The tower forward pass is serialised by a lock inside
#: the servicer, so this bounds concurrent FAISS searches -- which do release
#: the GIL and do parallelise.
DEFAULT_WORKERS = 8

DEFAULT_PORT = 50052


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The retrieval sidecar (ADR 0013).")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="A .pt from models.retrieval.train."
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--variant", choices=CONTENT_VARIANTS, default=CONTENT_VARIANTS[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    parser.add_argument(
        "--redis-url",
        default=cache_module.DEFAULT_URL,
        help="User-embedding cache. Empty string disables it.",
    )
    parser.add_argument("--cache-ttl", type=int, default=cache_module.DEFAULT_TTL)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Intra-op threads per forward pass. One by default: the tower runs "
        "on a batch of ONE user, so intra-op parallelism buys little and "
        "competes with the FAISS searches running alongside it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import grpc

    args = build_parser().parse_args(argv)
    torch.set_num_threads(args.torch_threads)

    loaded = load(
        checkpoint=args.checkpoint,
        artifacts=args.artifacts,
        pointer=args.pointer,
        variant=args.variant,
        ef_search=args.ef_search,
    )

    embeddings: Any = None
    if args.redis_url:
        try:
            embeddings = cache_module.UserEmbeddingCache(
                cache_module.connect(args.redis_url), loaded.tower_dim, args.cache_ttl
            )
        except RuntimeError as exc:
            # Starting without the cache is correct: it is an optimisation, and
            # refusing to serve because an optimisation is down converts a
            # latency problem into an outage. Said loudly, because it ALSO
            # removes rung 2 of ADR 0013's ladder -- with nothing writing
            # embeddings to Redis, a later outage of this service drops the
            # orchestrator straight to popularity.
            print(f"WARNING: no user-embedding cache ({exc})")
            print("         rung 2 of the degradation ladder is unavailable while this holds")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    retrieval_pb2_grpc.add_RetrievalServicer_to_server(  # type: ignore[no-untyped-call]  # generated, untyped
        RetrievalServicer(loaded, embeddings), server
    )
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()

    print(
        f"retrieval sidecar on :{args.port} · {loaded.kind} index {loaded.version} · "
        f"{loaded.n_items:,} items x {loaded.dim}d · efSearch {loaded.ef_search}"
    )
    if loaded.dim != loaded.tower_dim:
        print(
            f"  WARNING: the tower emits {loaded.tower_dim}d against a {loaded.dim}d index; "
            "Health will report not-ready"
        )

    # SIGTERM is what a container runtime sends. Without this the process dies
    # mid-request on every deploy; grace lets in-flight searches finish.
    def stop(*_: object) -> None:
        server.stop(grace=5).wait()

    signal.signal(signal.SIGTERM, stop)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
