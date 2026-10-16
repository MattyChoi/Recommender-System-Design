from __future__ import annotations

import argparse
import os
import signal
from concurrent import futures

import grpc
from dotenv import load_dotenv

from common.pb import features_pb2_grpc
from serving.features.service import FeaturesServicer
from serving.features.store import Store

load_dotenv()
DEFAULT_PORT = int(os.getenv("FEATURES_REDIS_PORT", 50053))
DEFAULT_WORKERS = 8


def serve(port: int, workers: int) -> None:
    store = Store()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    features_pb2_grpc.add_FeaturesServicer_to_server(FeaturesServicer(store), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"features gateway on :{port}")

    def stop(*_: object) -> None:
        server.stop(grace=5).wait()

    signal.signal(signal.SIGTERM, stop)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="The Feast gateway sidecar.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    serve(args.port, args.workers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
