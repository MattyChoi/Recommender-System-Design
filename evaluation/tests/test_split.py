"""The temporal split protocol

Every frame here is built in the test, so none of this needs a built corpus.
That matters more than usual: this is the file that decides whether any number
the project ever reports is trustworthy, so it must run on every push rather
than only where the dataset happens to exist.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from common.config import SplitConfig
from evaluation.offline.split import (
    cohort_summary,
    label_cohorts,
    split_boundaries,
    temporal_split,
)

T1 = datetime(2019, 11, 13)
T2 = datetime(2019, 11, 14)
_SCHEMA = "impression_id long, user_id string, item_id string, ts timestamp"
_Row = tuple[int, str, str, datetime]


def _impressions(spark: SparkSession, rows: list[_Row]) -> DataFrame:
    return spark.createDataFrame(rows, _SCHEMA)


def _one(impression_id: int, user: str, when: datetime, items: int = 2) -> list[_Row]:
    """All of one impression's rows, sharing a timestamp as MIND's do."""
    return [(impression_id, user, f"N{impression_id}{i}", when) for i in range(items)]


@pytest.fixture
def timeline(spark: SparkSession) -> DataFrame:
    """Three users across the train, val and test windows."""
    rows: list[_Row] = []
    rows += _one(1, "U1", datetime(2019, 11, 10))
    rows += _one(2, "U1", datetime(2019, 11, 11))
    rows += _one(3, "U1", datetime(2019, 11, 12))
    rows += _one(4, "U2", datetime(2019, 11, 12, 9))
    rows += _one(5, "U1", datetime(2019, 11, 13, 9))  # val
    rows += _one(6, "U2", datetime(2019, 11, 13, 10))  # val
    rows += _one(7, "U1", datetime(2019, 11, 14, 9))  # test, warm
    rows += _one(8, "U2", datetime(2019, 11, 14, 10))  # test, one train impression
    rows += _one(9, "U3", datetime(2019, 11, 14, 11))  # test, never seen in train
    return _impressions(spark, rows)


# --- the boundary itself ---------------------------------------------------


def test_the_windows_are_strictly_ordered_in_time(timeline: DataFrame) -> None:
    """The property that makes every downstream number mean anything.

    A random split lets the model see August to predict July. This is the
    assertion that says it cannot.
    """
    train, val, test = temporal_split(timeline, T1, T2)

    assert train.agg(f.max("ts")).collect()[0][0] < T1
    assert val.agg(f.min("ts")).collect()[0][0] >= T1
    assert val.agg(f.max("ts")).collect()[0][0] < T2
    assert test.agg(f.min("ts")).collect()[0][0] >= T2


def test_no_impression_lands_in_two_windows(timeline: DataFrame) -> None:
    """Disjointness on the IMPRESSION, which is the unit metrics aggregate over."""
    train, val, test = temporal_split(timeline, T1, T2)
    ids = [df.select("impression_id").distinct() for df in (train, val, test)]

    assert ids[0].intersect(ids[1]).count() == 0
    assert ids[1].intersect(ids[2]).count() == 0
    assert ids[0].intersect(ids[2]).count() == 0


def test_every_row_is_assigned_exactly_once(timeline: DataFrame) -> None:
    """No row invented, none dropped, while the eligibility filter is off.

    Catches the boundary written with two ``>=`` or two ``<``, which duplicates
    or loses a whole window without touching any other assertion here.
    """
    train, val, test = temporal_split(timeline, T1, T2)
    assert train.count() + val.count() + test.count() == timeline.count()


def test_an_impression_straddling_a_boundary_stays_whole(spark: SparkSession) -> None:
    """The leak the whole design is arranged around, and the reason for _anchored.

    One impression whose rows fall either side of t1. Split row-wise -- as the
    guide's own train filter does -- and it is torn in half: the model trains
    on some of the items a user was shown and is then asked about the rest,
    having already seen the answer. No metric reveals it, because every metric
    aggregates WITHIN an impression, so the leak hides inside the unit the
    numbers are computed over.

    Assigning on the impression's earliest timestamp puts it wholly in train.
    """
    rows = [
        (1, "U1", "NA", T1 - timedelta(minutes=1)),
        (1, "U1", "NB", T1 + timedelta(minutes=1)),
        (1, "U1", "NC", T1 + timedelta(minutes=2)),
    ]
    train, val, test = temporal_split(_impressions(spark, rows), T1, T2)

    assert train.count() == 3, "the impression was torn across the boundary"
    assert val.count() == 0
    assert test.count() == 0


def test_boundaries_out_of_order_are_rejected(timeline: DataFrame) -> None:
    """Fail at the call, not as an empty validation set discovered much later."""
    with pytest.raises(ValueError, match="ordered"):
        temporal_split(timeline, T2, T1)


# --- the eligibility filter ------------------------------------------------


def test_the_eligibility_filter_is_off_at_zero(timeline: DataFrame) -> None:
    """Zero must mean "no filter", not "at least zero", or the official-split
    path silently keeps only users who happen to recur."""
    _, _, unfiltered = temporal_split(timeline, T1, T2, min_user_impressions=0)
    assert {r[0] for r in unfiltered.select("user_id").distinct().collect()} == {"U1", "U2", "U3"}


def test_the_eligibility_filter_counts_train_impressions_not_rows(
    timeline: DataFrame,
) -> None:
    """U1 has three train impressions, U2 one, U3 none.

    Counting rows instead of distinct impressions would pass every user here --
    each impression carries two rows, so U2's single impression would look like
    two and clear a threshold of two it should fail.
    """
    _, _, test = temporal_split(timeline, T1, T2, min_user_impressions=3)
    assert {r[0] for r in test.select("user_id").distinct().collect()} == {"U1"}

    _, _, test_low = temporal_split(timeline, T1, T2, min_user_impressions=1)
    assert {r[0] for r in test_low.select("user_id").distinct().collect()} == {"U1", "U2"}


# --- deriving the boundaries ----------------------------------------------


def test_boundaries_are_derived_from_the_data_and_land_on_midnight(
    timeline: DataFrame,
) -> None:
    """Day-aligned so the cut can use dt partition pruning, and be eyeballed."""
    t1, t2 = split_boundaries(timeline, SplitConfig(holdout_days=1))
    assert (t1, t2) == (datetime(2019, 11, 13), datetime(2019, 11, 14))

    t1_wide, t2_wide = split_boundaries(timeline, SplitConfig(holdout_days=2))
    assert (t1_wide, t2_wide) == (datetime(2019, 11, 11), datetime(2019, 11, 13))


def test_a_holdout_that_would_leave_no_training_data_is_rejected(
    timeline: DataFrame,
) -> None:
    """MIND's train week is six days, so three-day windows consume it.

    Returning an empty train set instead would surface much later as a
    confusing model failure rather than a configuration error.
    """
    with pytest.raises(ValueError, match="no training data"):
        split_boundaries(timeline, SplitConfig(holdout_days=4))


# --- cohorts ---------------------------------------------------------------


def test_cohorts_are_labelled_against_train_alone(timeline: DataFrame) -> None:
    """U3 never appears before t2, so every U3 row is cold; U1's are warm.

    Defining cold against train+val would let a hyperparameter-selection window
    change which rows count as cold, so the cohort boundary would depend on a
    fitted quantity and two runs would stop being comparable.
    """
    train, _, test = temporal_split(timeline, T1, T2)
    labelled = label_cohorts(test, train)

    cold = {r["user_id"] for r in labelled.filter(f.col("is_cold_user")).collect()}
    warm = {r["user_id"] for r in labelled.filter(~f.col("is_cold_user")).collect()}
    assert cold == {"U3"}
    assert warm == {"U1", "U2"}
    # Every item id is unique to its impression here, so nothing in test was
    # seen in train -- item cold-start is the norm on a news corpus.
    assert labelled.filter(~f.col("is_cold_item")).count() == 0


def test_labelling_preserves_the_row_count(timeline: DataFrame) -> None:
    """label_cohorts annotates; a left join that fans out would inflate metrics."""
    train, _, test = temporal_split(timeline, T1, T2)
    assert label_cohorts(test, train).count() == test.count()


def test_cohort_summary_counts_impressions_and_users(timeline: DataFrame) -> None:
    """The gate's evidence: report the slices rather than assuming them."""
    train, _, test = temporal_split(timeline, T1, T2)
    summary = {
        (r["is_cold_user"], r["is_cold_item"]): (r["rows"], r["impressions"], r["users"])
        for r in cohort_summary(label_cohorts(test, train)).collect()
    }
    assert summary[(True, True)] == (2, 1, 1)  # U3's single cold impression
    assert summary[(False, True)] == (4, 2, 2)  # U1 and U2, one impression each
