"""The replay harness (data_pipeline/replay).

No broker, no JVM, no built pipeline. The producer was split so that every
decision worth testing lives in a pure function, and these run in milliseconds
as a result.

The property that matters most here is DETERMINISM. Part P3 compares a
streamed aggregate against a batch scan of the same window; if two replays of
identical input can disagree, a red parity test tells you nothing about
whether the streaming logic is correct.
"""

from __future__ import annotations

import itertools
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from data_pipeline.replay.pacing import LatenessInjector, sleep_seconds
from data_pipeline.replay.producer import DryRunSink, replay
from data_pipeline.replay.records import WIRE_FIELDS, to_record

T0 = datetime(2019, 11, 14, 9, 0, 0)
PROTO = Path("serving/proto/events.proto")


def _rows(n: int, step_seconds: int = 60) -> list[dict[str, Any]]:
    """n bronze-shaped rows, one per step, in event-time order."""
    return [
        {
            "impression_id": 1000 + i,
            "user_id": f"U{i % 3}",
            "item_id": f"N{i}",
            "clicked": i % 5 == 0,
            "slot": i % 4,
            "ts": T0 + timedelta(seconds=i * step_seconds),
        }
        for i in range(n)
    ]


def _event_id(row: dict[str, Any]) -> str:
    return f"{row['impression_id']}:{row['item_id']}"


def _event_ms(row: dict[str, Any]) -> int:
    return int(row["ts"].timestamp() * 1000)


def _drive(rows: list[dict[str, Any]], injector: LatenessInjector) -> list[tuple[int, str]]:
    """Run rows through the injector, returning (ingest_ms, event_id) in order."""
    out: list[tuple[int, str]] = []
    for row in rows:
        for ingest_ms, held in injector.push(_event_ms(row), row):
            out.append((ingest_ms, _event_id(held)))
    for ingest_ms, held in injector.drain():
        out.append((ingest_ms, _event_id(held)))
    return out


# --- the wire contract -----------------------------------------------------


def test_wire_fields_match_the_proto() -> None:
    """The .proto stays the single definition of the field set.

    JSON on the wire means nothing enforces the schema at runtime, so the
    agreement has to be checked here instead. Add a field to the proto without
    teaching the producer to emit it and this goes red -- which is the alarm
    you want, because the Flink DDL is written against the proto.
    """
    if not PROTO.exists():
        pytest.skip("run from the repository root")
    declared = re.findall(r"^\s+\w+\s+(\w+)\s*=\s*\d+;", PROTO.read_text(), flags=re.M)
    assert tuple(declared) == WIRE_FIELDS


def test_a_record_carries_every_field_even_when_empty() -> None:
    """Absent fields are emitted at their zero values, not omitted.

    A consumer that branches on whether a key exists is a consumer that breaks
    the day the field starts arriving. Emitting "" and 0 keeps the shape
    stable across the whole life of the dataset.
    """
    record = to_record(_rows(1)[0], ingest_ms=0)

    assert tuple(record) == WIRE_FIELDS
    assert record["session_id"] == ""  # bronze is pre-sessionization
    assert record["dwell_ms"] == 0  # MIND ships no dwell
    assert record["propensity"] == 0.0  # MIND logs no propensity
    assert record["request_id"] == "1000"  # impression_id IS the request


def test_event_type_follows_the_label() -> None:
    """MIND collapses impression and click into one labelled row."""
    shown, clicked = _rows(6)[1], _rows(6)[5]
    assert to_record(shown, 0)["event_type"] == "impression"
    assert to_record(clicked, 0)["event_type"] == "click"


# --- lateness --------------------------------------------------------------


def test_zero_lateness_is_a_pass_through() -> None:
    """With nothing delayed, output order is input order and ingest == event."""
    rows = _rows(20)
    got = _drive(rows, LatenessInjector(max_lateness_ms=0, fraction=0.0))

    assert [event_id for _, event_id in got] == [_event_id(r) for r in rows]
    assert [ingest for ingest, _ in got] == [_event_ms(r) for r in rows]


def test_injected_delay_never_exceeds_the_bound() -> None:
    """The bound is the contract with the consumer's watermark.

    Part P's Flink table allows 30 seconds of lateness. A record delayed past
    that is DROPPED, not handled late -- so an injector that can overshoot its
    own maximum would show up downstream as silent data loss.
    """
    max_ms = 5_000
    rows = _rows(200, step_seconds=1)
    injector = LatenessInjector(max_lateness_ms=max_ms, fraction=1.0, seed=7)

    by_event = {_event_id(r): _event_ms(r) for r in rows}
    for ingest_ms, event_id in _drive(rows, injector):
        delay = ingest_ms - by_event[event_id]
        assert 0 <= delay <= max_ms, f"{event_id} delayed {delay}ms"


def test_lateness_actually_reorders_the_stream() -> None:
    """A harness that never produces an inversion never tests a watermark.

    This is the test that would catch the injector degrading into a
    pass-through -- a fraction silently read as 0, a bound rounded to nothing.
    Without it the suite would stay green while the harness stopped doing the
    one thing it exists for.
    """
    rows = _rows(200, step_seconds=1)
    injector = LatenessInjector(max_lateness_ms=10_000, fraction=0.3, seed=11)

    by_event = {_event_id(r): _event_ms(r) for r in rows}
    event_times = [by_event[event_id] for _, event_id in _drive(rows, injector)]

    inversions = sum(a > b for a, b in itertools.pairwise(event_times))
    assert inversions > 0, "no out-of-order arrivals: the watermark path is untested"


def test_nothing_is_lost_or_duplicated() -> None:
    """drain() is easy to forget, and forgetting it loses only LATE records.

    That makes the bias the dangerous part: the missing rows would be exactly
    the ones the harness exists to produce, so the topic would look fine.
    """
    rows = _rows(150, step_seconds=1)
    got = _drive(rows, LatenessInjector(max_lateness_ms=8_000, fraction=0.5, seed=3))

    assert len(got) == len(rows)
    assert {event_id for _, event_id in got} == {_event_id(r) for r in rows}


def test_arrivals_come_out_in_ingest_order() -> None:
    """The buffer's whole job: emit by arrival, not by occurrence."""
    rows = _rows(150, step_seconds=1)
    ingest = [i for i, _ in _drive(rows, LatenessInjector(8_000, 0.4, seed=5))]

    assert ingest == sorted(ingest)


def test_the_same_seed_replays_identically() -> None:
    """Without this, P3's batch/stream parity comparison proves nothing."""
    rows = _rows(120, step_seconds=1)
    first = _drive(rows, LatenessInjector(6_000, 0.25, seed=42))
    second = _drive(rows, LatenessInjector(6_000, 0.25, seed=42))
    different = _drive(rows, LatenessInjector(6_000, 0.25, seed=43))

    assert first == second
    assert first != different, "the seed is not reaching the draws"


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_an_impossible_fraction_is_rejected(fraction: float) -> None:
    """Fail at construction, not silently at 2 a.m. in a replay."""
    with pytest.raises(ValueError, match="fraction"):
        LatenessInjector(max_lateness_ms=1_000, fraction=fraction)


# --- pacing ----------------------------------------------------------------


def test_pacing_scales_with_speed() -> None:
    """speed is a compression factor: 3600 replays an hour per wall second."""
    hour_ms = 3_600_000
    assert sleep_seconds(0, hour_ms, speed=3600.0) == pytest.approx(1.0)
    assert sleep_seconds(0, hour_ms, speed=7200.0) == pytest.approx(0.5)


def test_unthrottled_and_first_record_never_sleep() -> None:
    """speed <= 0 is the backfill and test path; it must not pause at all."""
    assert sleep_seconds(0, 3_600_000, speed=0.0) == 0.0
    assert sleep_seconds(None, 3_600_000, speed=3600.0) == 0.0


def test_pacing_never_returns_a_negative_pause() -> None:
    """Ingest order is monotone, but a clamp here is cheaper than a bug there."""
    assert sleep_seconds(10_000, 5_000, speed=1.0) == 0.0


# --- end to end, still without a broker ------------------------------------


def test_replay_sends_every_row_keyed_by_user(capsys: pytest.CaptureFixture[str]) -> None:
    """Keying on user_id is what preserves per-user order across partitions.

    Kafka orders within a partition only, and the partition is chosen by
    hashing the key. Key on item_id instead and every sequence model in the
    project is reading a shuffled history.
    """
    rows = _rows(30)
    sink = DryRunSink()

    sent = replay(iter(rows), sink, LatenessInjector(2_000, 0.2, seed=1), speed=0.0)

    assert sent == len(rows)
    lines = capsys.readouterr().out.splitlines()
    keys = [line.split("\t", 1)[0] for line in lines if "\t" in line]
    assert set(keys) == {"U0", "U1", "U2"}
