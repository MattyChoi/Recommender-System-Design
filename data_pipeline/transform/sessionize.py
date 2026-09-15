"""Group a user's impressions into sessions by inactivity gap

A session is COARSER than MIND's ``impression_id``. That column already groups
the items shown together on one page view; a session groups several page views
into one sitting, split wherever the user goes quiet.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f


def sessionize(events: DataFrame, gap_minutes: int) -> DataFrame:
    """Attach a ``session_id`` to every row.

    The mechanism is a running sum over a 0/1 boundary flag: each gap larger
    than the threshold increments a counter, and every row between two
    boundaries inherits the same index.

    Args:
        events: Rows carrying ``user_id`` and ``ts``.
        gap_minutes: Inactivity gap that ends a session.

    Returns:
        ``events`` with one additional column, ``session_id``, formatted as
        ``<user_id>#<n>``. Row count is unchanged.
    """
    # ORDERED window: lag() is meaningless without one, and the ordering must
    # be `ts`. impression_id is NOT chronological in MIND, so ordering by it
    # would produce sessions from a shuffled timeline.
    ordered = Window.partitionBy("user_id").orderBy("ts")

    return (
        events.withColumn("prev_ts", f.lag("ts").over(ordered))
        .withColumn(
            "new_session",
            (
                f.col("prev_ts").isNull()
                | (f.unix_timestamp("ts") - f.unix_timestamp("prev_ts") > gap_minutes * 60)
            ).cast("int"),
        )
        .withColumn("session_idx", f.sum("new_session").over(ordered))
        # concat_ws rather than arithmetic on the index: the id stays readable
        # in logs and cannot collide between users.
        .withColumn("session_id", f.concat_ws("#", "user_id", "session_idx"))
        .drop("prev_ts", "new_session", "session_idx")
    )
