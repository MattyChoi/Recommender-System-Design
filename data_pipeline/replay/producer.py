"""Replay bronze onto Kafka as if it were happening now.

    uv run python -m data_pipeline.replay.producer --split train --dry-run --limit 5

Three things depend on this harness, and each one wants something different
from it:

* Later Flink jobs needs a topic that behaves like a live one -- event-time
  ordered in the main, with a realistic tail of late arrivals.
* Later parity test needs the same window to replay IDENTICALLY every time,
  or comparing a streamed aggregate against a batch one proves nothing.
* Later Lambda-versus-Kappa answer needs backfill-by-replay to be real
  rather than aspirational.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

import pyarrow.parquet as pq

from common.config import load_settings
from common.utils import SPLITS
from data_pipeline.replay.pacing import LatenessInjector, sleep_seconds
from data_pipeline.replay.records import BRONZE_COLUMNS, event_ts_ms, to_record

# Rows materialised out of Arrow at a time. The sorted table stays in Arrow's
# compact columnar form; only this many rows exist as Python dicts at once.
_BATCH_ROWS = 65_536


class Sink(Protocol):
    """Where records go. Narrow on purpose, so tests can substitute a list."""

    def send(self, key: str, value: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class DryRunSink:
    """Prints JSON lines instead of producing. Needs no broker and no client."""

    def __init__(self) -> None:
        self.count = 0

    def send(self, key: str, value: dict[str, Any]) -> None:
        print(f"{key}\t{json.dumps(value)}")
        self.count += 1

    def close(self) -> None:
        print(f"# dry run: {self.count} records")


class KafkaSink:
    """A thin wrapper over confluent-kafka's Producer.

    The import is deferred to construction so ``--dry-run`` works on a machine
    where the client is not installed -- which is the case in CI, where the
    tests exercise the pure logic and never open a socket.
    """

    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        from confluent_kafka import Producer

        self.topic = topic
        # linger.ms waits to batch records; without it, sends records one at a time
        self._producer = Producer({"bootstrap.servers": bootstrap_servers, "linger.ms": 20})

    def send(self, key: str, value: dict[str, Any]) -> None:
        # Kafka guarantees order WITHIN a partition only, and the partition is
        # chosen by hashing the key. Keying on user_id is what keeps one user's
        # events in sequence
        self._producer.produce(self.topic, key=key.encode(), value=json.dumps(value).encode())
        # Serve delivery callbacks and apply backpressure. Skip this and the
        # internal queue grows until the client raises BufferError.
        self._producer.poll(0)

    def close(self) -> None:
        self._producer.flush()


def iter_bronze(
    bronze_root: Path, split: str, limit: int | None = None
) -> Iterator[dict[str, Any]]:
    """Yield bronze rows for one split in event-time order.

    Args:
        bronze_root: ``settings.paths.bronze``.
        split: ``train`` or ``dev``.
        limit: Stop after this many rows. For smoke tests and demos.

    Yields:
        Row dicts carrying exactly :data:`BRONZE_COLUMNS`.
    """
    root = bronze_root / "events" / split
    partitions = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("dt="))
    # An unpartitioned layer is still readable; only the ordering argument
    # changes, and sorting the single table covers it.
    targets = partitions or [root]

    emitted = 0
    for part in targets:
        table = pq.read_table(part, columns=list(BRONZE_COLUMNS))
        table = table.sort_by([("ts", "ascending")])
        for batch in table.to_batches(max_chunksize=_BATCH_ROWS):
            for row in batch.to_pylist():
                yield row
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def replay(
    rows: Iterator[dict[str, Any]],
    sink: Sink,
    injector: LatenessInjector,
    speed: float,
) -> int:
    """Drive rows through the injector into the sink, pacing as configured.

    Returns:
        The number of records sent.
    """
    sent = 0
    # Pacing follows INGEST time, not event time: the injector emits in arrival
    # order, and arrival is what a consumer's clock actually sees.
    prev_ingest_ms: int | None = None

    def emit(ingest_ms: int, row: dict[str, Any]) -> None:
        nonlocal sent, prev_ingest_ms
        pause = sleep_seconds(prev_ingest_ms, ingest_ms, speed)
        if pause > 0:
            time.sleep(pause)
        record = to_record(row, ingest_ms)
        sink.send(record["user_id"], record)
        prev_ingest_ms = ingest_ms
        sent += 1

    for row in rows:
        for ingest_ms, held in injector.push(event_ts_ms(row["ts"]), row):
            emit(ingest_ms, held)
    # Without this the topic loses up to one buffer of records, and because
    # they are the delayed ones the loss falls entirely on the late-arrival
    # path -- the exact behaviour the harness exists to exercise.
    for ingest_ms, held in injector.drain():
        emit(ingest_ms, held)
    return sent


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        A process exit code.
    """
    settings = load_settings()
    cfg = settings.replay

    parser = argparse.ArgumentParser(description="Replay bronze events onto Kafka.")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--topic", default=cfg.topic)
    parser.add_argument("--bootstrap-servers", default=cfg.bootstrap_servers)
    parser.add_argument(
        "--speed",
        type=float,
        default=cfg.speed,
        help="Event-time seconds per wall-clock second. 0 replays unthrottled.",
    )
    parser.add_argument("--max-lateness-seconds", type=int, default=cfg.max_lateness_seconds)
    parser.add_argument("--late-fraction", type=float, default=cfg.late_fraction)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N records.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print records instead of producing."
    )
    args = parser.parse_args(argv)

    if not (settings.paths.bronze / "events" / args.split / "_SUCCESS").is_file():
        print(f"error: bronze not built for {args.split!r}. Run `make bronze` first.")
        return 1

    sink: Sink = DryRunSink() if args.dry_run else KafkaSink(args.bootstrap_servers, args.topic)
    injector = LatenessInjector(
        max_lateness_ms=args.max_lateness_seconds * 1000,
        fraction=args.late_fraction,
        seed=args.seed,
    )

    if not args.dry_run:
        print(
            f"{args.split} -> {args.bootstrap_servers}/{args.topic} "
            f"at {args.speed}x, {args.late_fraction:.1%} late by up to "
            f"{args.max_lateness_seconds}s, seed {args.seed}"
        )
    try:
        rows = iter_bronze(settings.paths.bronze, args.split, args.limit)
        sent = replay(rows, sink, injector, args.speed)
    finally:
        sink.close()

    if not args.dry_run:
        print(f"{sent} records produced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
