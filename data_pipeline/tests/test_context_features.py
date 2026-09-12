"""The context-feature ODFV must agree with Spark, not with a description of Spark.

hour_of_day and day_of_week are computed twice: by Spark when training examples
are built, and by the on-demand view when they are served. Nothing structural
keeps the two in step, and both conventions at play are ones where the Python
default is WRONG:

* Spark reads timestamps with the session timezone pinned to UTC. A naive
  local-time conversion shifts every hour by the developer's offset -- and
  unlike the fixture-timezone bug this project already hit, it would not show up
  in any offline metric.
* Spark's dayofweek is 1=Sunday..7=Saturday. Python's isoweekday is
  1=Monday..7=Sunday. Off by one, wrapping, and plausible-looking either way.

So the test drives BOTH implementations over the same instants and compares
them, rather than asserting the Python side against restated rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from data_pipeline.features.recsys_store.feature_repo.definition import context_features

# Every hour across a full week from a Sunday, so all 24 hours and all 7 days
# appear, including the Sunday/Monday boundary where the two numbering schemes
# disagree most visibly.
_START = datetime(2019, 11, 10, 0, 0, 0, tzinfo=UTC)  # a Sunday
_INSTANTS = [_START + timedelta(hours=n) for n in range(24 * 7)]


@pytest.fixture
def spark_answer(spark: SparkSession) -> list[tuple[int, int]]:
    """What build_training_examples would attach, from the same expressions."""
    rows = [(moment.replace(tzinfo=None),) for moment in _INSTANTS]
    frame = (
        spark.createDataFrame(rows, "ts timestamp")
        .withColumn("hour_of_day", f.hour("ts"))
        .withColumn("day_of_week", f.dayofweek("ts"))
        .orderBy("ts")
    )
    return [(row["hour_of_day"], row["day_of_week"]) for row in frame.collect()]


def _run_odfv(request_ts: Sequence[object]) -> list[tuple[int, int]]:
    """Call the transform exactly as Feast's python mode does."""
    ft = context_features.feature_transformation
    if not ft:
        return []
    got = ft.udf({"request_ts": request_ts})
    return list(zip(got["hour_of_day"], got["day_of_week"], strict=True))


@pytest.fixture
def odfv_answer() -> list[tuple[int, int]]:
    """What the on-demand view serves, driven the way FEAST drives it.

    Feast passes datetimes for a UnixTimestamp field -- its schema inference
    calls the transform with ``ValueType.UNIX_TIMESTAMP: [_utc_now()]``. An
    earlier version of this fixture passed epoch ints, a type Feast never sends:
    the test was green while `feast apply` could not run the view at all. Drive
    it the way the caller does.
    """
    return _run_odfv(list(_INSTANTS))


def test_the_two_implementations_agree_over_a_full_week(
    spark_answer: list[tuple[int, int]], odfv_answer: list[tuple[int, int]]
) -> None:
    assert odfv_answer == spark_answer


def test_the_week_starts_on_sunday_as_spark_numbers_it(
    odfv_answer: list[tuple[int, int]],
) -> None:
    """Pinned separately, because agreement alone would not catch both sides
    being wrong together if the Spark expression were ever changed."""
    assert odfv_answer[0][1] == 1  # the first instant is a Sunday
    assert odfv_answer[24][1] == 2  # Monday


def test_every_hour_and_every_day_is_covered(
    odfv_answer: list[tuple[int, int]],
) -> None:
    assert {hour for hour, _ in odfv_answer} == set(range(24))
    assert {day for _, day in odfv_answer} == set(range(1, 8))


def test_the_service_carries_the_context_view() -> None:
    """Serving must request these rather than invent them."""
    from data_pipeline.features.recsys_store.feature_repo.definition import ranker_v1

    served = {projection.name for projection in ranker_v1.feature_view_projections}

    assert "context_features" in served


def test_epoch_numbers_are_accepted_too(
    odfv_answer: list[tuple[int, int]],
) -> None:
    """The field is named ..._ts, so a caller passing epochs is being reasonable."""
    assert _run_odfv([moment.timestamp() for moment in _INSTANTS]) == odfv_answer


def test_a_naive_datetime_is_read_as_utc_not_as_local_time(
    odfv_answer: list[tuple[int, int]],
) -> None:
    """The bug this would hide is invisible offline.

    astimezone() on a naive value applies the machine's offset, so every hour
    would shift by however far the developer sits from UTC -- correct on a CI
    runner pinned to UTC, wrong on a laptop, and identical in every metric.
    """
    naive = [moment.replace(tzinfo=None) for moment in _INSTANTS]

    assert _run_odfv(naive) == odfv_answer
