"""Point-in-time feature attachment and cold-start handling (guide D2).

Frames are built inline, so none of this needs a built corpus. The cold-start
assertions matter more than they look on this dataset: 46.2% of the articles
shown in dev were never shown in train, so the cold path is the common path.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pyspark.sql import DataFrame, SparkSession

from data_pipeline.features.gold import attach_point_in_time_features

T0 = datetime(2019, 11, 14, 12, 0, 0)

_LABELS = (
    "impression_id long, user_id string, item_id string, category string, "
    "clicked boolean, ts timestamp"
)
_USERS = (
    "user_id string, feature_ts timestamp, user_impressions_24h long, "
    "user_clicks_24h long, user_ctr_smoothed double, user_tenure_hours double"
)
_SERIES = (
    "item_id string, category string, feature_ts timestamp, "
    "item_impressions_1h long, item_clicks_1h long, "
    "item_impressions_24h long, item_clicks_24h long, "
    "item_ctr_smoothed double, cat_expanding_ctr double, item_age_hours double"
)


@pytest.fixture
def series(spark: SparkSession) -> DataFrame:
    """One warm item with history, plus a category prior available all along."""
    return spark.createDataFrame(
        [
            # WARM: a closed bucket an hour before the label.
            ("N1", "sports", T0 - timedelta(hours=1), 40, 2, 400, 20, 0.05, 0.03, 6.0),
            # A later bucket the label must NOT see.
            ("N1", "sports", T0 + timedelta(hours=1), 40, 30, 400, 300, 0.75, 0.03, 8.0),
            # The category series carries the prior for items with no history.
            ("N9", "sports", T0 - timedelta(hours=2), 10, 0, 100, 3, 0.03, 0.03, 1.0),
        ],
        _SERIES,
    )


@pytest.fixture
def users(spark: SparkSession) -> DataFrame:
    """U1 has a closed bucket an hour before the label; nobody else does."""
    return spark.createDataFrame([("U1", T0 - timedelta(hours=1), 30, 3, 0.09, 12.0)], _USERS)


@pytest.fixture
def labels(spark: SparkSession) -> DataFrame:
    """One warm item and one the series has never seen, at the same instant."""
    return spark.createDataFrame(
        [
            (1, "U1", "N1", "sports", True, T0),
            (1, "U1", "N2", "sports", False, T0),  # COLD: no bucket anywhere
        ],
        _LABELS,
    )


def _rows(got: DataFrame) -> dict[str, dict[str, object]]:
    return {r["item_id"]: r.asDict() for r in got.collect()}


def test_a_warm_item_reads_only_its_past(
    labels: DataFrame, series: DataFrame, users: DataFrame
) -> None:
    """The 0.75 bucket closes an hour AFTER the label and must stay invisible."""
    got = _rows(attach_point_in_time_features(labels, series, users))["N1"]

    assert got["item_ctr_smoothed"] == pytest.approx(0.05), "read a feature from the future"
    assert got["item_impressions_24h"] == 400
    assert got["has_item_features"] is True


def test_a_cold_item_gets_the_category_prior_not_a_zero_rate(
    labels: DataFrame, series: DataFrame, users: DataFrame
) -> None:
    """The distinction the guide is specific about.

    N2 has no feature bucket anywhere, so every item column comes back null.
    Zero-filling the RATE would tell the ranker this article has a measured
    click-through of zero -- a claim about evidence, not a missing value. The
    prior arrives through the category timeline, which is why it is joined
    separately: a prior read off N2's own (null) row would be null too.
    """
    got = _rows(attach_point_in_time_features(labels, series, users))["N2"]

    assert got["item_ctr_smoothed"] == pytest.approx(0.03), "cold item was zero-filled"
    assert got["has_item_features"] is False


def test_cold_item_counts_are_zero_not_null(
    labels: DataFrame, series: DataFrame, users: DataFrame
) -> None:
    """Counts are the opposite case: zero is the honest answer.

    No bucket had closed, so nothing was knowable -- genuinely zero prior
    impressions, and an age of zero at first sight. Leaving them null would
    push the imputation decision onto every downstream consumer.
    """
    got = _rows(attach_point_in_time_features(labels, series, users))["N2"]

    assert got["item_impressions_24h"] == 0
    assert got["item_clicks_24h"] == 0
    assert got["item_age_hours"] == pytest.approx(0.0)


def test_the_joins_never_multiply_labels(
    labels: DataFrame, series: DataFrame, users: DataFrame
) -> None:
    """Two rows in, two rows out -- every label here has a reachable prior.

    The category series has one row per (category, hour) and the label set is
    one row per (impression, item). Drop the distinct() and `sports` matches
    three series rows, silently tripling every label in the category.
    """
    assert attach_point_in_time_features(labels, series, users).count() == labels.count()


def test_context_features_come_from_the_label_timestamp(
    labels: DataFrame, series: DataFrame, users: DataFrame
) -> None:
    """Cheap, and the only D1 group available without another table."""
    got = _rows(attach_point_in_time_features(labels, series, users))["N1"]

    assert got["hour_of_day"] == 12
    assert got["day_of_week"] == 5  # Spark's dayofweek: Sunday = 1, so Thursday = 5


def test_the_label_keeps_its_own_category(
    labels: DataFrame, series: DataFrame, users: DataFrame
) -> None:
    """Silver's category is static metadata and must survive the join.

    asof_join refuses a column collision outright, so `category` is excluded
    from the item feature list on purpose. Were it attached instead, a cold
    item would lose the very column its prior is keyed on.
    """
    got = _rows(attach_point_in_time_features(labels, series, users))

    assert got["N1"]["category"] == "sports"
    assert got["N2"]["category"] == "sports"


def test_a_label_with_nothing_knowable_is_dropped(spark: SparkSession) -> None:
    """The corpus's opening hour, where no bucket has closed anywhere.

    Measured on train: 6,257 of 6,262 such rows precede the first closed bucket
    in the WHOLE corpus, so neither the item nor the category fallback has
    anything to offer -- and a global third tier would be null there too. Such
    a label teaches a model nothing except the imputation rule, so it is
    dropped rather than carried as a null every consumer must remember to
    check.
    """
    series = spark.createDataFrame(
        [("N1", "sports", T0 + timedelta(hours=1), 10, 1, 100, 5, 0.05, 0.04, 3.0)], _SERIES
    )
    labels = spark.createDataFrame(
        [
            (1, "U1", "N1", "sports", False, T0),  # before every bucket: unknowable
            (2, "U1", "N1", "sports", True, T0 + timedelta(hours=2)),  # after one: kept
        ],
        _LABELS,
    )

    users = spark.createDataFrame([], _USERS)
    got = attach_point_in_time_features(labels, series, users)

    assert got.count() == 1, "the unknowable label was not dropped"
    assert got.collect()[0]["impression_id"] == 2


def test_a_user_with_history_gets_their_own_rate(
    labels: DataFrame, series: DataFrame, users: DataFrame
) -> None:
    """The second as-of join, on its own timeline.

    U1 has a bucket an hour before the label, so the user columns are populated
    and prefixed -- unprefixed, `impressions_24h` would collide between the
    user and item series and asof_join would raise.
    """
    got = _rows(attach_point_in_time_features(labels, series, users))["N1"]

    assert got["user_ctr_smoothed"] == pytest.approx(0.09)
    assert got["user_impressions_24h"] == 30
    assert got["has_user_features"] is True


def test_a_users_first_impression_is_kept_with_a_null_rate(
    spark: SparkSession, series: DataFrame
) -> None:
    """Not dropped and not imputed, unlike the item rate.

    The row still carries real item features, so it is a usable example --
    and dropping it would delete every user's first impression, which is
    exactly the cold-start cohort the evaluation slices on. The rate stays
    null and has_user_features records why; the counts are zero-filled,
    because zero really is how much this user had done.
    """
    labels = spark.createDataFrame([(1, "U9", "N1", "sports", True, T0)], _LABELS)
    users = spark.createDataFrame([], _USERS)

    got = _rows(attach_point_in_time_features(labels, series, users))["N1"]

    assert got["has_user_features"] is False
    assert got["user_ctr_smoothed"] is None
    assert got["user_impressions_24h"] == 0
