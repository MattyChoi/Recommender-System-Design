"""Wall-clock pacing and out-of-order arrival, as pure functions.

The lateness injector is the reason this harness exists at all. A replay that
emits events in perfect event-time order is a replay that never exercises a
watermark, never drops a late record, and never tells you whether the streaming
job is correct -- it only tells you it runs. The later Flink DDL declares
``WATERMARK FOR ts AS ts - INTERVAL '30' SECOND``; ``max_lateness_seconds``
here is the other half of that pair and the two must be changed together.
"""

from __future__ import annotations

import heapq
import random
from typing import Any


def drift_seconds(origin_ms: int | None, event_ms: int, speed: float, elapsed: float) -> float:
    """How long to pause so this record lands on schedule.

    The schedule is ABSOLUTE, anchored on the first record of the run, not
    incremental from the previous one. That distinction is the whole function.

    Anchoring on the origin makes the schedule self-correcting. Fall behind and
    the next target is already in the past, so the pause is zero and the run
    closes the gap on its own.

    Args:
        origin_ms: Timestamp of the first record emitted, or None before one
            has been.
        event_ms: Timestamp of the record about to be emitted.
        speed: Event-time seconds per wall-clock second. ``3600`` replays an
            hour of event time per second; ``0`` or less is unthrottled, which
            is what backfills and tests want and a live demo does not.
        elapsed: Wall-clock seconds since the first record was emitted.

    Returns:
        Seconds to sleep. Zero when already on or behind schedule.
    """
    if origin_ms is None or speed <= 0:
        return 0.0
    target = (event_ms - origin_ms) / 1000.0 / speed
    return max(0.0, target - elapsed)


class LatenessInjector:
    """Delays a fraction of records, and re-emits everything in ingest order.

    Records go in ordered by event time and come out ordered by *arrival*,
    which is what a consumer actually sees. Determinism is a hard requirement,
    not a nicety: The later batch/stream parity test compares a replayed window
    against a batch scan of the same window, and that comparison means nothing
    if two replays of identical input disagree.
    """

    def __init__(self, max_lateness_ms: int, fraction: float, seed: int = 0) -> None:
        """
        Args:
            max_lateness_ms: Upper bound on injected delay. Must not exceed the
                consumer's allowed lateness, or records are dropped rather than
                handled late.
            fraction: Share of records to delay, in [0, 1]. Zero makes this a
                pass-through and the output identical to the input.
            seed: Fixes the randomness so results can be reproducable
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        if max_lateness_ms < 0:
            raise ValueError(f"max_lateness_ms must be >= 0, got {max_lateness_ms}")

        self.max_lateness_ms = max_lateness_ms
        self.fraction = fraction
        self._rng = random.Random(seed)
        # (ingest_ms, sequence, payload).
        self._buffer: list[tuple[int, int, Any]] = []
        self._seq = 0  # sequence number to break ties in the buffer, making ties FIFO

    def push(self, event_ms: int, payload: Any) -> list[tuple[int, Any]]:
        """Admit one record and return every record now safe to emit.

        Safety is the same argument a watermark makes. Input arrives in event
        order, and no record can be ingested before it happened, so once a
        record with event time ``E`` has been read, nothing still unread can
        have an ingest time below ``E``. Everything buffered at or under ``E``
        is therefore final and can be released in ingest order.

        Returns:
            ``(ingest_ms, payload)`` pairs, in non-decreasing ingest order.
            Often empty; occasionally several at once.
        """
        delay = 0
        if self.max_lateness_ms > 0 and self._rng.random() < self.fraction:
            delay = self._rng.randint(1, self.max_lateness_ms)

        heapq.heappush(self._buffer, (event_ms + delay, self._seq, payload))
        self._seq += 1

        ready: list[tuple[int, Any]] = []
        while self._buffer and self._buffer[0][0] <= event_ms:
            ingest_ms, _, item = heapq.heappop(self._buffer)
            ready.append((ingest_ms, item))
        return ready

    def drain(self) -> list[tuple[int, Any]]:
        """Flush whatever is still held back, at end of stream.

        Forgetting this silently truncates the topic by up to one buffer's
        worth of records -- and because they are the *late* ones, the loss is
        biased toward exactly the cases the harness exists to test.
        """
        ready: list[tuple[int, Any]] = []
        while self._buffer:
            ingest_ms, _, item = heapq.heappop(self._buffer)
            ready.append((ingest_ms, item))
        return ready
