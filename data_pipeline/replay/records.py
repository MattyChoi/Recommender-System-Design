"""One bronze row, as it appears on the Kafka topic.

Protobuf on the wire with Confluent Schema Registry framing. The message type is
``recsys.v1.InteractionEvent``, generated into ``common/pb/events_pb2.py`` by
``make proto``, so this module no longer *describes* the field set -- it
constructs it, and a keyword that does not name a declared field is a
``ValueError`` at construction rather than a key nobody reads.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from common.pb import events_pb2

# Columns this module needs out of bronze. Bronze is the raw event contract, so
# it carries no session_id, no category, and none of the integer id columns.
# That is deliberate: user_idx and item_idx come from a mapping built over the
# WHOLE corpus, and putting them on a stream would cause feature leakage.
BRONZE_COLUMNS: tuple[str, ...] = (
    "impression_id",
    "user_id",
    "item_id",
    "clicked",
    "slot",
    "ts",
)

# The fields InteractionEvent declares that MIND cannot fill, and why. This
# replaces WIRE_FIELDS: the proto is now the field list, so what is worth
# writing down is not the names but the DECISIONS -- and
# test_every_declared_field_is_assigned_or_declared_empty turns the dict into
# a gate, so a field added to the proto has to be either mapped here or
# explicitly declared unfillable.
KNOWN_EMPTY: dict[str, str] = {
    "session_id": (
        "bronze is pre-sessionization by design (see BRONZE_COLUMNS). Deriving "
        "sessions from the stream is the consumer's job; handing it a "
        "batch-computed answer would undercut the whole parity exercise."
    ),
    "dwell_ms": (
        "MIND ships no dwell time. Its absence is why this project cannot model "
        "watch-through or engagement depth, only click."
    ),
    "propensity": (
        "MIND logs no propensity, and the policy that produced these impressions "
        "is undocumented. This is THE reason off-policy evaluation has to lean on "
        "estimated propensities rather than logged ones -- worth being able to "
        "say out loud. The SERVING producer does log it; this one cannot."
    ),
    "device": "no device column in the dataset.",
    "surface": (
        "every MIND impression is the same surface (the MSN feed). Filling in a "
        "constant would invent a distinction the data does not make, and any "
        "model that split on it would be learning noise."
    ),
}


def event_ts_ms(ts: datetime) -> int:
    """Milliseconds since the epoch, which is what the proto declares."""
    return int(ts.timestamp() * 1000)


def to_record(row: dict[str, Any], ingest_ms: int) -> events_pb2.InteractionEvent:
    """Map one bronze row onto the wire schema.

    Args:
        row: A bronze row carrying every column in :data:`BRONZE_COLUMNS`.
        ingest_ms: When the pipeline LEARNED about the event. Supplied by the
            caller rather than computed here, because it is the lateness
            injector that decides it.

    Returns:
        An ``InteractionEvent``. The five fields in :data:`KNOWN_EMPTY` are left
        unset, which is indistinguishable on the wire from being set to their
        defaults -- and identical to a reader.
    """
    return events_pb2.InteractionEvent(
        # Deterministic, not a uuid4. Two replays of the same bronze partition
        # must produce byte-identical event_ids or the parity test cannot
        # deduplicate. (impression_id, item_id) is unique per the bronze
        # contract test, so it is a legitimate primary key.
        event_id=f"{row['impression_id']}:{row['item_id']}",
        user_id=row["user_id"],
        item_id=row["item_id"],
        event_type="click" if row["clicked"] else "impression",
        event_ts_ms=event_ts_ms(row["ts"]),
        ingest_ts_ms=ingest_ms,
        position=int(row["slot"]),
        request_id=str(row["impression_id"]),
    )
