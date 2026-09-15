"""One bronze row, as it appears on the Kafka topic.

The field names below are taken verbatim from ``recsys.v1.InteractionEvent`` in
``serving/proto/events.proto``. JSON is what goes on the wire -- Part P's Flink
table declares ``'format'='json'``, and a topic you can read with
``kafka-console-consumer`` is worth more during a demo than the bytes saved by
protobuf. The proto stays the single definition of the field set;
``test_replay.py`` asserts the two never drift apart.

One trap if this is ever switched to protobuf's own JSON printer: that emits
lowerCamelCase (``eventTsMs``), not the snake_case written in the .proto. The
Flink DDL would have to change with it.

MIND is an impression log and nothing more, so five of the thirteen fields have
no source. They are emitted anyway, at their proto3 zero values, because a
consumer that has to branch on whether a key exists is a consumer that will
break the day the field starts arriving. Each one is documented below with the
reason it is empty -- that reason is the interesting half.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

# Columns this module needs out of bronze. Bronze is the raw event contract, so
# it carries no session_id, no category, and none of the integer id columns.
# That is deliberate: user_idx and item_idx come from a mapping built over the
# WHOLE corpus, and putting them on a stream would cause feature leakage
BRONZE_COLUMNS: tuple[str, ...] = (
    "impression_id",
    "user_id",
    "item_id",
    "clicked",
    "slot",
    "ts",
)

# Proto field order. The wire is JSON, so order is cosmetic -- but keeping it
# aligned with the .proto makes the two files diffable by eye.
WIRE_FIELDS: tuple[str, ...] = (
    "event_id",
    "user_id",
    "item_id",
    "event_type",
    "event_ts_ms",
    "ingest_ts_ms",
    "session_id",
    "position",
    "request_id",
    "dwell_ms",
    "propensity",
    "device",
    "surface",
)


def event_ts_ms(ts: datetime) -> int:
    """Milliseconds since the epoch, which is what the proto declares."""
    return int(ts.timestamp() * 1000)


def to_record(row: dict[str, Any], ingest_ms: int) -> dict[str, Any]:
    """Map one bronze row onto the wire schema.

    Args:
        row: A bronze row carrying every column in :data:`BRONZE_COLUMNS`.
        ingest_ms: When the pipeline LEARNED about the event. Supplied by the
            caller rather than computed here, because it is the lateness
            injector that decides it

    Returns:
        A dict whose keys are exactly :data:`WIRE_FIELDS`.
    """
    occurred = event_ts_ms(row["ts"])
    return {
        # Deterministic, not a uuid4. Two replays of the same bronze partition
        # must produce byte-identical event_ids or the parity test cannot
        # deduplicate. (impression_id, item_id) is unique per the bronze
        # contract test, so it is a legitimate primary key.
        "event_id": f"{row['impression_id']}:{row['item_id']}",
        "user_id": row["user_id"],
        "item_id": row["item_id"],
        "event_type": "click" if row["clicked"] else "impression",
        "event_ts_ms": occurred,
        "ingest_ts_ms": ingest_ms,
        # EMPTY: bronze is pre-sessionization by design (see BRONZE_COLUMNS).
        # Deriving sessions from the stream is the consumer's job; handing it a
        # batch-computed answer would undercut the whole parity exercise.
        "session_id": "",
        "position": int(row["slot"]),
        "request_id": str(row["impression_id"]),
        # EMPTY: MIND ships no dwell time. Its absence is why this project
        # cannot model watch-through or engagement depth, only click.
        "dwell_ms": 0,
        # EMPTY: MIND logs no propensity, and the policy that produced these
        # impressions is undocumented. This is THE reason off-policy evaluation
        # in Part O has to lean on estimated propensities rather than logged
        # ones -- worth being able to say out loud.
        "propensity": 0.0,
        # EMPTY: no device column in the dataset.
        "device": "",
        # EMPTY: every MIND impression is the same surface (the MSN feed).
        # Filling in a constant would invent a distinction the data does not
        # make, and any model that split on it would be learning noise.
        "surface": "",
    }
