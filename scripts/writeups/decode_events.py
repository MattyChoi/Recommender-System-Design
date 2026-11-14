"""Decode a protobuf Kafka topic to readable JSON.

    make consume TOPIC=impressions N=5

A CONSUMER rather than a stdin filter, for the reason in the step above: there
is no reliable way to get framed binary through a pipe from
``kafka-console-consumer``.

``ProtobufDeserializer`` handles the Confluent framing -- the magic byte, the
four-byte schema id, and the varint message-index array -- so nothing here
parses bytes by hand.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from typing import Any, cast

from confluent_kafka import TIMESTAMP_CREATE_TIME, Consumer, KafkaError
from confluent_kafka.schema_registry.protobuf import ProtobufDeserializer
from confluent_kafka.serialization import MessageField, SerializationContext
from google.protobuf import json_format
from google.protobuf.message import Message

from common.config import load_settings
from common.pb import events_pb2, impressions_pb2

# One message type per topic, because there is one subject per topic under the
# default TopicNameStrategy. A topic carrying two types would need
# RecordNameStrategy and a different lookup than this.
TOPIC_TYPES: dict[str, type[Message]] = {
    "impressions": events_pb2.InteractionEvent,
    "impression_context": impressions_pb2.ImpressionContext,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        A process exit code.
    """
    cfg = load_settings().replay

    parser = argparse.ArgumentParser(description="Decode a protobuf Kafka topic.")
    parser.add_argument("--topic", required=True, choices=sorted(TOPIC_TYPES))
    parser.add_argument("-n", "--count", type=int, default=5)
    parser.add_argument("--bootstrap-servers", default=cfg.bootstrap_servers)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    # NO SchemaRegistryClient. ProtobufDeserializer takes the message type and
    # uses the embedded id only to skip the framing, so this works against a
    # topic whose registry is unreachable -- which is exactly the moment you
    # most want to read it. The cost is that it trusts the local type to match;
    # `make schemas-check` is what makes that trust reasonable.
    # The library aliases ``__init__ = __init_impl`` (a name-mangled method), which
    # type checkers can't follow, and it annotates ``message_type`` as an
    # instance rather than a class -- hence the Any-typed constructor.
    deserializer_cls = cast(Any, ProtobufDeserializer)
    deserialize = deserializer_cls(TOPIC_TYPES[args.topic], {"use.deprecated.format": False})
    ctx = SerializationContext(args.topic, MessageField.VALUE)

    consumer = Consumer(
        {
            "bootstrap.servers": args.bootstrap_servers,
            # A FRESH group every run, so repeated reads start from the
            # beginning instead of resuming an offset a previous debugging
            # session committed. A fixed group.id here is the reason a
            # decoder "stops working" on the second invocation.
            "group.id": f"decode-{os.getpid()}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([args.topic])

    seen = 0
    try:
        while seen < args.count:
            message = consumer.poll(timeout=args.timeout)
            if message is None:
                print(f"# timed out after {seen} records")
                break
            error = message.error()
            if error is not None:
                # EOF is not an error here: it means this partition is
                # exhausted, and another may not be.
                if error.code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"# error: {error}")
                return 1

            # The library annotates __call__ as returning bytes; it actually
            # returns an instance of the message type it was built with.
            record = cast(Message | None, deserialize(message.value(), ctx))
            if record is None:
                # A tombstone (null value) has no payload to decode.
                continue
            timestamp_type, timestamp = message.timestamp()
            # The RECORD timestamp printed beside the payload, and its TYPE.
            # event-time-versus-produce-time is the one §5.4 check with no
            # error attached anywhere, and a CreateTime that disagrees with
            # event_ts_ms is the symptom. NOT_AVAILABLE means the producer
            # never set one.
            kind = "create" if timestamp_type == TIMESTAMP_CREATE_TIME else "append/none"
            raw_key = message.key()
            key = raw_key.decode() if raw_key is not None else ""
            print(
                f"{timestamp}\t{kind}\t{key}\t"
                + json_format.MessageToJson(
                    record,
                    preserving_proto_field_name=True,
                    always_print_fields_with_no_presence=True,
                    indent=None,
                )
            )
            seen += 1
    finally:
        # close() commits nothing (auto-commit is off) but it does leave the
        # group cleanly, which stops a rebalance timeout from delaying the
        # next run by the session timeout.
        consumer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
