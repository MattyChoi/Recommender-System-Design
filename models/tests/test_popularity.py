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

from models.retrieval.baselines.popularity import (
    click_counts,
    score_decayed_popular,
    score_most_popular,
    score_most_recent,
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


class TestMostRecent:
    """The half-life -> 0 limit, computed rather than approached."""

    @pytest.fixture
    def recency_train(self, spark: SparkSession) -> DataFrame:
        """N2 is five times more popular than N1, and six days staler.

        Built so popularity and recency DISAGREE. A fixture where the same item
        wins under both would pass whichever model was wired in.
        """
        return spark.createDataFrame(
            [
                ("N1", T0 - timedelta(hours=1), True),
                *[("N2", T0 - timedelta(days=6), True) for _ in range(5)],
                ("N3", T0 - timedelta(days=1), False),
            ],
            _TRAIN,
        )

    def _scores(self, frame: DataFrame) -> dict[str, float]:
        return {r["item_id"]: r["score"] for r in frame.collect()}

    def test_recency_beats_popularity_on_the_same_rows(
        self, spark: SparkSession, recency_train: DataFrame
    ) -> None:
        """N2 has five clicks to N1's one, and still ranks below it."""
        labels = spark.createDataFrame(
            [("N1", 1, T0 + timedelta(hours=1)), ("N2", 1, T0 + timedelta(hours=1))],
            _LABELS,
        )

        recent = self._scores(score_most_recent(labels, recency_train))
        popular = self._scores(score_most_popular(labels, recency_train))

        assert recent["N1"] > recent["N2"]
        assert popular["N2"] > popular["N1"]

    def test_it_agrees_with_a_very_short_half_life(
        self, spark: SparkSession, recency_train: DataFrame
    ) -> None:
        """The claim that this IS the limit, as a test rather than a docstring.

        0.02 days is the shortest half-life the exponential survives: below
        roughly 0.013 days, exp() underflows for two-week-old clicks and the
        decayed model stops being comparable. If these two ever disagree in
        ORDER, one of them is wrong about what the limit is.
        """
        labels = spark.createDataFrame(
            [
                ("N1", 1, T0 + timedelta(hours=1)),
                ("N2", 1, T0 + timedelta(hours=1)),
                ("N3", 1, T0 + timedelta(hours=1)),
            ],
            _LABELS,
        )

        recent = self._scores(score_most_recent(labels, recency_train))
        decayed = self._scores(score_decayed_popular(labels, recency_train, half_life_days=0.02))

        by_recency = sorted(recent, key=lambda i: -recent[i])
        by_decay = sorted(decayed, key=lambda i: -decayed[i])

        assert by_recency == by_decay == ["N1", "N2", "N3"]

    def test_a_click_after_the_label_does_not_count(self, spark: SparkSession) -> None:
        """The label is scored at an instant BEFORE the only click there is."""
        train = spark.createDataFrame([("N1", T0, True)], _TRAIN)
        labels = spark.createDataFrame([("N1", 1, T0 - timedelta(hours=1))], _LABELS)

        assert self._scores(score_most_recent(labels, train))["N1"] == 0.0

    def test_an_unclicked_item_scores_zero_not_null(
        self, spark: SparkSession, recency_train: DataFrame
    ) -> None:
        """N3 was shown and never clicked; the same honest zero the others give."""
        labels = spark.createDataFrame([("N3", 1, T0 + timedelta(hours=1))], _LABELS)

        assert self._scores(score_most_recent(labels, recency_train))["N3"] == 0.0

    def test_a_cold_item_never_outranks_a_stale_one(
        self, spark: SparkSession, recency_train: DataFrame
    ) -> None:
        """1/(1+age) is strictly positive, so any real click beats no click.

        This is why the score is not -age: a negative-age score needs a sentinel
        for "never clicked", and every sentinel is a number some real row can
        reach.
        """
        labels = spark.createDataFrame(
            [("N2", 1, T0 + timedelta(hours=1)), ("N3", 1, T0 + timedelta(hours=1))],
            _LABELS,
        )
        got = self._scores(score_most_recent(labels, recency_train))

        assert got["N2"] > got["N3"] == 0.0

    def test_the_row_count_is_preserved(
        self, spark: SparkSession, recency_train: DataFrame
    ) -> None:
        """An item with several clicks must collapse to one row, not several."""
        labels = spark.createDataFrame(
            [("N2", 1, T0 + timedelta(hours=1)), ("N2", 2, T0 + timedelta(hours=2))],
            _LABELS,
        )

        assert score_most_recent(labels, recency_train).count() == 2
