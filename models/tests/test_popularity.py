"""Popularity baselines: point-in-time discipline, and the cold-item blind spot.

The two failures worth catching are both invisible in a metric. Fitting on
train plus dev makes the baseline look STRONGER, and decaying against a single
global clock makes every impression score alike -- neither produces an error, a
warning, or an implausible number.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pyspark.sql import DataFrame, SparkSession

from models.retrieval.popularity import (
    click_counts,
    score_decayed_popular,
    score_most_popular,
)

T0 = datetime(2019, 11, 14, 12, 0, 0)

_TRAIN = "item_id string, ts timestamp, clicked boolean"
_LABELS = "item_id string, impression_id long, ts timestamp"


@pytest.fixture
def train(spark: SparkSession) -> DataFrame:
    """N1 clicked three times, N2 once, N3 shown but never clicked."""
    return spark.createDataFrame(
        [
            ("N1", T0 - timedelta(days=6), True),
            ("N1", T0 - timedelta(days=3), True),
            ("N1", T0 - timedelta(hours=1), True),
            ("N2", T0 - timedelta(days=6), True),
            ("N3", T0 - timedelta(days=1), False),
        ],
        _TRAIN,
    )


class TestMostPopular:
    def test_clicks_not_impressions(self, spark: SparkSession, train: DataFrame) -> None:
        """N3 was shown but never clicked, so it has no popularity.

        Counting impressions instead would rank items by how often MICROSOFT's
        recommender chose to show them -- a measure of the incumbent system, not
        of what users wanted.
        """
        counts = {r["item_id"]: r["clicks"] for r in click_counts(train).collect()}

        assert counts == {"N1": 3, "N2": 1}

    def test_a_cold_item_scores_zero_not_null(self, spark: SparkSession, train: DataFrame) -> None:
        """Structural blindness, kept visible rather than patched.

        On dev, 32% of slates have a cold clicked item. This baseline cannot
        rank them, and the cold-item cohort is where that must show up.
        """
        labels = spark.createDataFrame([("N99", 1, T0)], _LABELS)

        got = score_most_popular(labels, train).collect()[0]

        assert got["score"] == 0.0

    def test_more_clicks_ranks_higher(self, spark: SparkSession, train: DataFrame) -> None:
        labels = spark.createDataFrame([("N1", 1, T0), ("N2", 1, T0)], _LABELS)

        scores = {r["item_id"]: r["score"] for r in score_most_popular(labels, train).collect()}

        assert scores["N1"] > scores["N2"]


class TestDecayedPopular:
    def test_recency_outweighs_raw_count(self, spark: SparkSession, train: DataFrame) -> None:
        """The whole reason this baseline exists on a news corpus."""
        labels = spark.createDataFrame([("N1", 1, T0), ("N2", 1, T0)], _LABELS)

        scores = {
            r["item_id"]: r["score"]
            for r in score_decayed_popular(labels, train, half_life_days=1.0).collect()
        }

        # N1's most recent click is an hour old; N2's only click is six days old.
        assert scores["N1"] > scores["N2"]

    def test_a_click_after_the_label_does_not_count(self, spark: SparkSession) -> None:
        """THE point-in-time rule, at the baseline layer.

        A click that happened after the impression was not knowable when the
        impression was served. Including it is the same leak the as-of join
        exists to prevent, committed one layer up -- and it would make this
        baseline look better, which is the direction nobody checks.
        """
        train = spark.createDataFrame([("N1", T0 + timedelta(hours=1), True)], _TRAIN)
        labels = spark.createDataFrame([("N1", 1, T0)], _LABELS)

        got = score_decayed_popular(labels, train).collect()[0]

        assert got["score"] == 0.0

    def test_each_label_is_decayed_from_its_own_instant(self, spark: SparkSession) -> None:
        """Not from one global `now`.

        Two impressions of the same item at different times must score
        differently: the later one sits further from the click. A single global
        clock would give them identical scores, silently.
        """
        train = spark.createDataFrame([("N1", T0 - timedelta(days=1), True)], _TRAIN)
        labels = spark.createDataFrame([("N1", 1, T0), ("N1", 2, T0 + timedelta(days=4))], _LABELS)

        scores = {
            r["impression_id"]: r["score"]
            for r in score_decayed_popular(labels, train, half_life_days=1.0).collect()
        }

        assert scores[1] > scores[2]

    def test_half_life_halves_the_weight(self, spark: SparkSession) -> None:
        """A click one half-life old is worth exactly half a fresh one."""
        train = spark.createDataFrame([("N1", T0 - timedelta(days=2), True)], _TRAIN)
        labels = spark.createDataFrame([("N1", 1, T0)], _LABELS)

        got = score_decayed_popular(labels, train, half_life_days=2.0).collect()[0]

        assert got["score"] == pytest.approx(0.5, abs=1e-9)

    def test_row_count_is_preserved(self, spark: SparkSession, train: DataFrame) -> None:
        """The scorer aggregates over a join; losing or duplicating label rows
        would corrupt every downstream metric silently."""
        labels = spark.createDataFrame([("N1", 1, T0), ("N2", 1, T0), ("N99", 2, T0)], _LABELS)

        assert score_decayed_popular(labels, train).count() == 3

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_a_nonpositive_half_life_is_refused(
        self, spark: SparkSession, train: DataFrame, bad: float
    ) -> None:
        labels = spark.createDataFrame([("N1", 1, T0)], _LABELS)

        with pytest.raises(ValueError, match="must be positive"):
            score_decayed_popular(labels, train, half_life_days=bad)
