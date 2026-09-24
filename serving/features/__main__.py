"""``make features-serve`` -- run the feature gateway.

    uv run python -m serving.features

Everything that can be wrong lives in service.py, where it is testable without
a socket or a Redis. This file opens a port.
"""

from __future__ import annotations

import argparse
import signal
from concurrent import futures
from pathlib import Path

from common.pb import features_pb2_grpc
from serving.features.service import FEATURE_REPO, FeaturesServicer, load

DEFAULT_PORT = 50053

#: Feast's online read is I/O to Redis, so handlers spend their time waiting
#: rather than holding the GIL. Higher than the retrieval sidecar's pool, which
#: is bounded by CPU-bound searches.
DEFAULT_WORKERS = 32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The feature gateway (Feast behind gRPC).")
    parser.add_argument("--repo", type=Path, default=FEATURE_REPO)
    parser.add_argument("--item-map", type=Path, default=Path("serving/artifacts/item_map.json"))
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser


def main(argv: list[str] | None = None) -> int:
    import grpc

    args = build_parser().parse_args(argv)
    loaded = load(repo=args.repo, item_map=args.item_map)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    features_pb2_grpc.add_FeaturesServicer_to_server(  # type: ignore[no-untyped-call]  # generated, untyped
        FeaturesServicer(loaded), server
    )
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()

    print(
        f"feature gateway on :{args.port} · item map {loaded.items.version or '(unversioned)'} "
        f"· {len(loaded.items.mapping):,} items"
    )

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
