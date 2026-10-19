"""The FAISS + user-tower sidecar, on :50052."""

from __future__ import annotations

import argparse
import os
import signal
from concurrent import futures
from pathlib import Path
from types import FrameType

import faiss
import grpc
import torch
from dotenv import load_dotenv

from common.pb import retrieval_pb2_grpc
from serving.retrieval.index import load_index
from serving.retrieval.service import RetrievalServicer

load_dotenv()
DEFAULT_PORT = int(os.getenv("RETRIEVAL_PORT", 50052))
DEFAULT_WORKERS = 8

DEFAULT_EF_SEARCH = 512

# MUST be the checkpoint the promoted index was built from -- see the print
# below, and the same warning on INDEX_CHECKPOINT in make/serving_artifacts.mk.
DEFAULT_CHECKPOINT = Path("data/checkpoints/both-logq-n4u0-b8192e10lr0.001-ab7e1d500b9bf792.pt")

GRACE_SECONDS = 10.0


def serve(
    port: int,
    workers: int,
    checkpoint: Path,
    artifacts: Path,
    pointer: Path,
    ef_search: int,
) -> None:
    # ONE thread each for torch and FAISS, with all the concurrency coming from
    # the gRPC pool. Both libraries default to using every core for intra-op
    # parallelism, which against `workers` request threads oversubscribes the
    # box by workers x cores.
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)

    loaded = load_index(checkpoint, artifacts, pointer, ef_search)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    retrieval_pb2_grpc.add_RetrievalServicer_to_server(RetrievalServicer(loaded), server)
    server.add_insecure_port(f"[::]:{port}")

    print(
        f"retrieval sidecar on :{port} | index {loaded.kind} "
        f"{loaded.version} efSearch={loaded.ef_search} "
        f"| tower {checkpoint.name}"
    )

    def drain(signum: int, frame: FrameType | None) -> None:
        print("retrieval sidecar draining")
        server.stop(GRACE_SECONDS)

    signal.signal(signal.SIGTERM, drain)
    signal.signal(signal.SIGINT, drain)

    server.start()
    server.wait_for_termination()


def main() -> int:
    parser = argparse.ArgumentParser(description="The FAISS + user-tower sidecar.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--artifacts", type=Path, default=Path("data/index"))
    parser.add_argument("--pointer", type=Path, default=Path("data/index/CURRENT"))
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    args = parser.parse_args()

    serve(
        args.port,
        args.workers,
        args.checkpoint,
        args.artifacts,
        args.pointer,
        args.ef_search,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
