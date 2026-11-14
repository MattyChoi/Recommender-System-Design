"""One typed grpc.health.v1 Check, shared by the tests that probe a server.

Why a helper rather than ``HealthStub(channel).Check(...)`` inline: the
generated HealthStub sets Check/List/Watch inside ``__init__``, but typeshed's
stub for grpcio-health-checking (which Pylance bundles) declares the class with
no methods at all -- so every inline call is an editor error, and to mypy it
is ``Any``. This is the one place that works around that, and it hands back a
real HealthCheckResponse.
"""

from __future__ import annotations

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


def check(channel: grpc.Channel, service: str = "", timeout: float = 5) -> int:
    """The status of ``service`` ("" is the server as a whole), as the enum int."""
    stub = health_pb2_grpc.HealthStub(channel)
    response: health_pb2.HealthCheckResponse = stub.Check(  # pyright: ignore[reportAttributeAccessIssue]
        health_pb2.HealthCheckRequest(service=service), timeout=timeout
    )
    return int(response.status)  # mypy sees grpc_health as untyped (pyproject override)
