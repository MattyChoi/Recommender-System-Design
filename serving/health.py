"""The standard gRPC health service, for the sidecars.

Each sidecar already has its own ``Health`` RPC, but nothing outside this repo
can call it. ``grpc.health.v1`` is the one Kubernetes' native ``grpc:`` probes,
grpc-health-probe and load balancers speak, and it is what the Go orchestrator
serves too -- so every process in the system is probed the same way.

Only the overall service ``""`` is reported. It flips to SERVING once the port
is bound and to NOT_SERVING the moment a drain starts, so a readiness probe
pulls the pod from its Service BEFORE in-flight requests are cut.
"""

from __future__ import annotations

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc


def add_health(server: grpc.Server) -> health.HealthServicer:
    """Register the health service. Call :func:`serving` after ``server.start()``."""
    servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(servicer, server)
    return servicer


def serving(servicer: health.HealthServicer) -> None:
    """Report SERVING. Separate from :func:`add_health`, so it follows ``start()``."""
    servicer.set("", health_pb2.HealthCheckResponse.SERVING)
