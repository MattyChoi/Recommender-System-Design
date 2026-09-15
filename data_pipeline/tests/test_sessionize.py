"""Sessionization behaviour (guide 5.2).

These tests build their own frames, so they run without ``data/`` and without
a built pipeline. They pin the two decisions encoded in ``sessionize``:

    A session ends when the gap to the PREVIOUS ROW exceeds the threshold,
    and the comparison is strict, so a gap of exactly ``gap_minutes``
    continues the session rather than ending it.

Session length therefore has no upper bound -- a user clicking every 25
minutes for six hours is one session. That is the web-analytics convention
and it is deliberate, but if a cap is ever added it is a SECOND boundary
condition, not an edit to this one, and these tests should keep passing.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from data_pipeline.transform.sessionize import sessionize

T0 = datetime(2019, 11, 14, 9, 0, 0)
GAP = 30

_SCHEMA = "user_id string, ts timestamp"


def _events(spark: SparkSession, rows: list[tuple[str, datetime]]) -> DataFrame:
    return spark.createDataFrame(rows, _SCHEMA)


def _ids(df: DataFrame) -> list[str]:
    """Session ids in chronological order, one per input row."""
    return [r["session_id"] for r in df.orderBy("user_id", "ts").collect()]


def test_activity_within_the_gap_stays_one_session(spark: SparkSession) -> None:
    """Quiet stretches shorter than the threshold do not split a sitting.

    The last gap is EXACTLY the threshold. It must not split, because the
    comparison in sessionize is ``>`` and not ``>=``; if someone loosens it,
    this is the test that goes red.
    """
    events = _events(
        spark,
        [
            ("U1", T0),
            ("U1", T0 + timedelta(minutes=12)),
            ("U1", T0 + timedelta(minutes=20)),
            ("U1", T0 + timedelta(minutes=50)),  # exactly 30 min after the last
        ],
    )

    got = sessionize(events, gap_minutes=GAP)

    # Row count is part of the contract: sessionize annotates, it never filters.
    assert got.count() == events.count()
    assert got.select("session_id").distinct().count() == 1
    assert _ids(got) == ["U1#1"] * 4


def test_a_gap_past_the_threshold_starts_a_new_session(spark: SparkSession) -> None:
    """The core split, and that the index increments rather than toggling."""
    events = _events(
        spark,
        [
            ("U1", T0),
            ("U1", T0 + timedelta(minutes=20)),
            ("U1", T0 + timedelta(minutes=125)),  # 105 min later: splits
            ("U1", T0 + timedelta(minutes=150)),
            ("U1", T0 + timedelta(minutes=195)),  # 45 min later: splits again
        ],
    )

    got = sessionize(events, gap_minutes=GAP)

    assert _ids(got) == ["U1#1", "U1#1", "U1#2", "U1#2", "U1#3"]


def test_the_first_row_of_a_user_is_always_a_boundary(spark: SparkSession) -> None:
    """The regression test for a bug that fails SILENTLY.

    A user's first row has no predecessor, so ``prev_ts`` is null and the gap
    comparison evaluates to NULL rather than True. Drop the ``isNull()``
    branch from the boundary flag and the running sum propagates that null
    through the entire partition: every session_id in the table comes back
    null, no exception is raised, and nothing downstream notices until a join
    on session_id quietly returns zero rows.

    Indices are 1-based for the same reason -- the first prefix sum is
    already 1.
    """
    events = _events(spark, [("U1", T0), ("U1", T0 + timedelta(minutes=5))])

    got = sessionize(events, gap_minutes=GAP)

    assert got.filter(f.col("session_id").isNull()).count() == 0
    assert _ids(got)[0] == "U1#1"


def test_sessions_are_scoped_to_one_user(spark: SparkSession) -> None:
    """Two users interleaved in time must not see each other's gaps.

    The rows alternate between users, so without ``partitionBy("user_id")``
    the lag() would reach across users, the gaps would collapse to a few
    minutes, and everything would land in one session. The user prefix on the
    id is what makes ``#1`` unambiguous once several users are present.
    """
    events = _events(
        spark,
        [
            ("U1", T0),
            ("U2", T0 + timedelta(minutes=5)),
            ("U1", T0 + timedelta(minutes=90)),  # 90 min after U1's own last row
            ("U2", T0 + timedelta(minutes=95)),  # 90 min after U2's own last row
        ],
    )

    got = sessionize(events, gap_minutes=GAP)

    assert _ids(got) == ["U1#1", "U1#2", "U2#1", "U2#2"]
