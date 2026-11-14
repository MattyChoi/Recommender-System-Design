"""The replay harness (data_pipeline/replay).

No broker, no JVM, no built pipeline. The producer was split so that every
decision worth testing lives in a pure function, and these run in milliseconds
as a result.

The property that matters most here is DETERMINISM. The replay-parity check
compares a streamed aggregate against a batch scan of the same window; if two
replays of identical input can disagree, a red parity test tells you nothing
about whether the streaming logic is correct.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta
from typing import Any

import pytest

from common.pb import events_pb2
from data_pipeline.replay.pacing import LatenessInjector, drift_seconds
from data_pipeline.replay.producer import DryRunSink, replay
from data_pipeline.replay.records import KNOWN_EMPTY, to_record

T0 = datetime(2019, 11, 14, 9, 0, 0)


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


def test_every_declared_field_is_assigned_or_declared_empty() -> None:
    """The mapper cannot silently miss a field added to the proto.

    This is the gap protobuf does not close. A new field in events.proto that
    to_record never sets arrives at its default and is indistinguishable from
    real data -- a propensity of 0.0 on every record, say. Accounting for every
    declared field either as assigned or as deliberately empty makes adding one
    a decision rather than an omission.
    """
    row = _rows(6)[1]
    # A row whose assigned values are ALL non-default, and ingest_ms != 0,
    # because ListFields reports only fields that are SET -- and a proto3
    # scalar set to its zero value is not set. With _rows(6)[0] the slot is 0,
    # position would vanish from `assigned`, and this test would fail for a
    # reason that has nothing to do with its hypothesis.
    assert row["slot"] != 0, "pick a row whose position is non-default"

    record = to_record(row, ingest_ms=1)
    declared = {field.name for field in events_pb2.InteractionEvent.DESCRIPTOR.fields}
    assigned = {field.name for field, _ in record.ListFields()}

    unaccounted = declared - assigned - set(KNOWN_EMPTY)
    assert not unaccounted, (
        f"fields declared in events.proto but neither assigned by to_record nor "
        f"listed in KNOWN_EMPTY: {sorted(unaccounted)}"
    )
    # And the inverse, so KNOWN_EMPTY cannot keep naming a field that was
    # removed from the proto or has since been mapped.
    assert not (set(KNOWN_EMPTY) & assigned)
    assert set(KNOWN_EMPTY) <= declared


def test_a_parsed_record_exposes_every_field_at_its_default() -> None:
    """Absent fields are readable, not missing.

    The JSON version emitted "" and 0 so a consumer never had to branch on
    whether a key exists. Protobuf writes nothing for them and the GENERATED
    READER materialises the defaults -- the same contract, held by the schema
    instead of by the producer.

    Asserted through a serialize/parse round trip rather than on the constructed
    message, because it is the READER's view the contract is about, and only the
    round trip proves the bytes carry it.
    """
    wire = to_record(_rows(1)[0], ingest_ms=0).SerializeToString()
    parsed = events_pb2.InteractionEvent()
    parsed.ParseFromString(wire)

    # Read, not HasField: HasField raises on a proto3 scalar without explicit
    # presence, which is all five of these.
    for name in KNOWN_EMPTY:
        assert not getattr(parsed, name), name

    assert parsed.request_id == "1000"  # impression_id IS the request


def test_event_type_follows_the_label() -> None:
    """MIND collapses impression and click into one labelled row."""
    shown, clicked = _rows(6)[1], _rows(6)[5]
    assert to_record(shown, 0).event_type == "impression"
    assert to_record(clicked, 0).event_type == "click"


# --- lateness --------------------------------------------------------------


def test_zero_lateness_is_a_pass_through() -> None:
    """With nothing delayed, output order is input order and ingest == event."""
    rows = _rows(20)
    got = _drive(rows, LatenessInjector(max_lateness_ms=0, fraction=0.0))

    assert [event_id for _, event_id in got] == [_event_id(r) for r in rows]
    assert [ingest for ingest, _ in got] == [_event_ms(r) for r in rows]


def test_injected_delay_never_exceeds_the_bound() -> None:
    """The bound is the contract with the consumer's watermark.

    The Flink table allows 30 seconds of lateness. A record delayed past
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
    """Without this, the batch/stream parity comparison proves nothing."""
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
    assert drift_seconds(0, hour_ms, speed=3600.0, elapsed=0.0) == pytest.approx(1.0)
    assert drift_seconds(0, hour_ms, speed=7200.0, elapsed=0.0) == pytest.approx(0.5)


def test_unthrottled_and_the_first_record_never_sleep() -> None:
    """speed <= 0 is the backfill and test path; it must not pause at all."""
    assert drift_seconds(0, 3_600_000, speed=0.0, elapsed=0.0) == 0.0
    assert drift_seconds(None, 3_600_000, speed=3600.0, elapsed=0.0) == 0.0


def test_the_schedule_is_absolute_not_incremental() -> None:
    """The property an incremental pause cannot have, and the reason for the rewrite.

    Targets are measured from the ORIGIN, so per-record overhead cannot
    accumulate: a record whose slot has already passed sleeps zero, and the
    records after it are still measured against the original schedule rather
    than against the delay. Compute each pause as (this - previous) / speed
    instead and every millisecond of overhead is permanent, so the replay
    slowly falls behind and the arrival PATTERN -- the thing the harness
    exists to reproduce -- stretches with it.
    """
    speed = 3600.0  # one wall second per event hour

    # Two hours in, but five wall seconds have already elapsed: behind
    # schedule, so do not sleep.
    assert drift_seconds(0, 7_200_000, speed, elapsed=5.0) == 0.0

    # Ten hours in, same five seconds elapsed. An incremental scheme would
    # sleep the full eight seconds from the previous record and stay five
    # seconds late forever; anchored, the target is 10s and the pause is what
    # remains of it.
    assert drift_seconds(0, 36_000_000, speed, elapsed=5.0) == pytest.approx(5.0)


def test_pacing_never_returns_a_negative_pause() -> None:
    """Ingest order is monotone, but a clamp here is cheaper than a bug there."""
    assert drift_seconds(10_000, 5_000, speed=1.0, elapsed=0.0) == 0.0


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
