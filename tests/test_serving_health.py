"""The sidecars' grpc.health.v1 contract, which the k8s probes depend on.

SERVING while up, NOT_SERVING once a drain starts -- and NOT_SERVING while the
port still answers, which is the whole point: readiness must fail before the
in-flight requests are cut, or a rollout drops them.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent import futures

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

from serving.health import add_health, serving


@pytest.fixture
def server() -> Iterator[tuple[grpc.Server, object, int]]:
    srv = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = add_health(srv)
    port = srv.add_insecure_port("127.0.0.1:0")
    srv.start()
    serving(servicer)
    yield srv, servicer, port
    srv.stop(None)


def check(port: int) -> int:
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        return stub.Check(health_pb2.HealthCheckRequest(service=""), timeout=3).status


def test_serving_once_started(server: tuple[grpc.Server, object, int]) -> None:
    _, _, port = server
    assert check(port) == health_pb2.HealthCheckResponse.SERVING


def test_not_serving_during_drain_while_port_still_answers(
    server: tuple[grpc.Server, object, int],
) -> None:
    _, servicer, port = server
    servicer.enter_graceful_shutdown()  # what both sidecars' SIGTERM handler does first
    assert check(port) == health_pb2.HealthCheckResponse.NOT_SERVING
