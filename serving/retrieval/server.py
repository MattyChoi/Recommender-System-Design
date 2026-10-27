"""One retrieval process: load the tower and index, bind the port, serve.

Its own module, not `__main__`, so `--processes` can spawn it: see
serving/processes.py. Each process loads its own copy of the index and tower.
"""

from __future__ import annotations

import argparse
import signal
from concurrent import futures
from types import FrameType
from typing import Any

import faiss
import grpc
import torch

from common.pb import retrieval_pb2_grpc
from serving.processes import EXCLUSIVE_PORT
from serving.retrieval.cache import UserEmbeddingCache, connect
from serving.retrieval.index import load_index
from serving.retrieval.service import RetrievalServicer

GRACE_SECONDS = 10.0


def serve(args: argparse.Namespace) -> None:
    # ONE thread each for torch and FAISS, with all the concurrency coming from
    # the gRPC pool. Both libraries default to using every core for intra-op
    # parallelism, which against `workers` request threads oversubscribes the
    # box by workers x cores.
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)

    loaded = load_index(args.checkpoint, args.artifacts, args.pointer, args.ef_search)

    embeddings_cache: Any = None
    if args.redis_url:
        try:
            embeddings_cache = UserEmbeddingCache(
                connect(args.redis_url), loaded.index.d, args.cache_ttl
            )
        except RuntimeError as exc:
            print(f"WARNING: no user-embedding cache ({exc})")

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers), options=EXCLUSIVE_PORT
    )
    retrieval_pb2_grpc.add_RetrievalServicer_to_server(
        RetrievalServicer(loaded, embeddings_cache), server
    )
    server.add_insecure_port(f"[::]:{args.port}")

    print(
        f"retrieval sidecar on :{args.port} | index {loaded.kind} "
        f"{loaded.version} efSearch={loaded.ef_search} "
        f"| tower {args.checkpoint.name}"
    )

    def drain(signum: int, frame: FrameType | None) -> None:
        print("retrieval sidecar draining")
        server.stop(GRACE_SECONDS)

    signal.signal(signal.SIGTERM, drain)
    signal.signal(signal.SIGINT, drain)

    server.start()
    server.wait_for_termination()
