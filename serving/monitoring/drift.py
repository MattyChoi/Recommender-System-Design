"""``python -m serving.monitoring.drift`` -- the live feature-drift monitor.

Consumes ``impression_context`` -- the features the orchestrator ACTUALLY
SERVED -- keeps a rolling window of them, and exposes the Population Stability
Index of each against the bins training was fitted on. Those bins come from
``serving/artifacts/drift_baseline.json`` (``make drift-baseline``) rather than
being recomputed, because recomputing them is a Spark read of gold and Spark has
no business on the serving path.

The same statistic as ``scripts/writeups/skew_report.py``, through the same
``common.drift`` functions, over the same training selection. That is the whole
point: a gauge that pages at a value the report calls OK would be worse than no
gauge. The report is the batch view; this is the continuous one.

The arithmetic is in ``window.py``, which imports nothing from the ``serving``
extra. This file is the wiring: Kafka in, gauges out, nothing worth a test.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from confluent_kafka import Consumer, KafkaError
from confluent_kafka.schema_registry.protobuf import ProtobufDeserializer
from confluent_kafka.serialization import MessageField, SerializationContext
from prometheus_client import Gauge, start_http_server

from common.config import load_settings
from common.pb import impressions_pb2
from serving.monitoring.window import (
    EXPECTED_CHURN,
    MIN_ROWS,
    WINDOW_ROWS,
    Window,
    load_baseline,
)

TOPIC = "impression_context"

# Records consumed between gauge refreshes. Re-binning per record would be
# wasted work, and a 15s scrape sees every refresh regardless.
REFRESH_EVERY = 200

_log = logging.getLogger("drift")

PSI = Gauge(
    "recsys_feature_psi",
    "Population Stability Index of served user features against training's bins.",
    ["feature", "alerting"],
)
ROWS = Gauge(
    "recsys_drift_window_rows",
    f"Rows in the rolling window. Below {MIN_ROWS} no PSI is reported at all.",
)
COVERAGE = Gauge(
    "recsys_drift_user_found_rate",
    "Share of non-degraded requests that had a stored user row. NaN when none were eligible.",
)

# Why an empty window is empty. Without these, zero rows has three causes that
# look identical on the endpoint -- no records consumed, records whose feature
# fetch had already failed, and records served against a user the store does
# not hold -- and telling them apart needed a Kafka client.
SEEN = Gauge("recsys_drift_records_seen", "Records consumed from the topic since start.")
FOUND = Gauge("recsys_drift_records_found", "Of those, how many carried a stored user row.")
DEGRADED = Gauge(
    "recsys_drift_records_degraded", "Of those, how many had already lost their feature fetch."
)
MALFORMED = Gauge(
    "recsys_drift_records_malformed", "Of those, how many had the wrong feature width."
)


def bind(window: Window) -> None:
    """Point the cheap gauges at the window, evaluated at SCRAPE time.

    ``set_function`` rather than ``publish``: each of these is one attribute
    read, so there is no reason for them to be as stale as the PSI refresh, and
    at scrape time they are always current -- including on a topic that has gone
    quiet, which is precisely when someone is asking why the window is empty.

    The PSI stays on the refresh path because it re-bins the whole window.
    """
    ROWS.set_function(lambda: len(window.rows))
    COVERAGE.set_function(window.coverage)
    SEEN.set_function(lambda: window.requests)
    FOUND.set_function(lambda: window.found)
    DEGRADED.set_function(lambda: window.degraded)
    MALFORMED.set_function(lambda: window.malformed)


def publish(window: Window) -> None:
    """Recompute the PSI gauges. The rest are bound to the window directly."""
    for name, value in window.drift().items():
        # The churn set rides on a LABEL rather than suppressing the series: a
        # feature expected to move is still worth seeing, it just must not page.
        # The alert selects alerting="true".
        PSI.labels(feature=name, alerting=str(name not in EXPECTED_CHURN).lower()).set(value)


def consume(window: Window, brokers: str, group: str, poll: float) -> None:
    """Tail the topic forever, refreshing the gauges as records arrive."""
    # No SchemaRegistryClient: ProtobufDeserializer uses the embedded id only to
    # skip Confluent's framing, so this keeps working against a topic whose
    # registry is unreachable -- which is when you most want the monitor up.
    # Same reasoning as scripts/writeups/decode_events.py.
    deserializer_cls = cast(Any, ProtobufDeserializer)
    deserialize = deserializer_cls(
        impressions_pb2.ImpressionContext, {"use.deprecated.format": False}
    )
    ctx = SerializationContext(TOPIC, MessageField.VALUE)

    consumer = Consumer(
        {
            "bootstrap.servers": brokers,
            "group.id": group,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe([TOPIC])

    since_refresh = 0
    try:
        while True:
            message = consumer.poll(timeout=poll)
            if message is None:
                # Nothing to refresh. The PSI cannot have moved -- the deque
                # only evicts on append -- and every other gauge is bound to
                # the window and read at scrape time.
                continue

            error = message.error()
            if error is not None:
                if error.code() != KafkaError._PARTITION_EOF:
                    _log.warning("kafka: %s", error)
                continue

            record = deserialize(message.value(), ctx)
            if record is None:
                continue  # a tombstone has no payload

            window.observe(
                user_found=record.user_found,
                degraded=list(record.degraded),
                feats=list(record.user_feats),
            )

            since_refresh += 1
            if since_refresh >= REFRESH_EVERY:
                publish(window)
                since_refresh = 0
    finally:
        consumer.close()


def main(argv: Sequence[str] | None = None) -> int:
    settings = load_settings()

    parser = argparse.ArgumentParser(description="Live feature drift against training's bins.")
    parser.add_argument(
        "--baseline", type=Path, default=Path("serving/artifacts/drift_baseline.json")
    )
    parser.add_argument("--port", type=int, default=9103, help="NOT 9102: that is the orchestrator")
    parser.add_argument("--bootstrap-servers", default=settings.replay.bootstrap_servers)
    parser.add_argument("--group", default=f"drift-monitor-{os.getenv('HOSTNAME', 'local')}")
    parser.add_argument("--window", type=int, default=WINDOW_ROWS)
    parser.add_argument("--poll", type=float, default=5.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    baseline = load_baseline(args.baseline)
    window = Window(baseline, args.window)
    bind(window)

    # Serving BEFORE the first record, so a monitor that cannot reach Kafka
    # still scrapes -- as an endpoint reporting zero rows, which is a legible
    # state. An endpoint that only appears once the pipeline works tells you
    # nothing on the day the pipeline does not.
    start_http_server(args.port)
    publish(window)
    _log.info(
        "drift monitor on :%d, %d features, window %d, judging above %d rows",
        args.port,
        len(baseline),
        args.window,
        MIN_ROWS,
    )

    consume(window, args.bootstrap_servers, args.group, args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
