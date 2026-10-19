from __future__ import annotations

import argparse
import os
import signal
from concurrent import futures
from types import FrameType

import grpc
from dotenv import load_dotenv

from common.pb import features_pb2_grpc
from serving.features.service import FeaturesServicer
from serving.features.store import Store

load_dotenv()

DEFAULT_PORT = int(os.getenv("FEATURES_PORT", 50053))
DEFAULT_WORKERS = 8
GRACE_SECONDS = 10.0


def serve(port: int, workers: int) -> None:
    store = Store()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    features_pb2_grpc.add_FeaturesServicer_to_server(FeaturesServicer(store), server)
    server.add_insecure_port(f"[::]:{port}")
    print(f"features gateway on :{port}")

    def drain(signum: int, frame: FrameType | None) -> None:
        print("features gateway draining")
        server.stop(GRACE_SECONDS)

    signal.signal(signal.SIGTERM, drain)
    signal.signal(signal.SIGINT, drain)

    server.start()
    server.wait_for_termination()


def main() -> int:
    parser = argparse.ArgumentParser(description="The Feast gateway sidecar.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    serve(args.port, args.workers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
