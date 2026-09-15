"""ALS: the cold-user wall, and the row-deletion trap that hides it.

Spark's ALS offers ``coldStartStrategy="drop"``, which silently removes label
rows whose user or item was unseen. On MIND that would delete most of the split
and leave a card whose metrics look respectable because the hard rows are gone
-- and nothing on the card would say the denominator moved. The same bug
shipped in the first ``score_covisit`` and was caught by a row count.

These tests fix a tiny factorisation (rank 2, few iterations) so the arithmetic
is about structure rather than convergence: which rows survive, which score
zero, and whether an unseen user can be reached at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame, SparkSession

from models.retrieval.als import fit_als, score_als, score_als_item

T0 = datetime(2019, 11, 14, 12, 0, 0)

_EVENTS = (
    "user_id string, item_id string, user_idx int, item_idx int, "
    "impression_id long, clicked boolean, ts timestamp"
)


def _rows(
    spark: SparkSession, rows: list[tuple[str, str, int, int, int, bool, datetime]]
) -> DataFrame:
    return spark.createDataFrame(rows, _EVENTS)


@pytest.fixture
def train(spark: SparkSession) -> DataFrame:
    """Two warm users over three warm items, plus an item nobody clicked."""
    return _rows(
        spark,
        [
            ("U1", "N1", 1, 1, 1, True, T0 - timedelta(hours=3)),
            ("U1", "N2", 1, 2, 2, True, T0 - timedelta(hours=2)),
            ("U2", "N2", 2, 2, 3, True, T0 - timedelta(hours=2)),
            ("U2", "N3", 2, 3, 4, True, T0 - timedelta(hours=1)),
            ("U1", "N9", 1, 9, 1, False, T0 - timedelta(hours=3)),
        ],
    )


@pytest.fixture
def model(train: DataFrame) -> ALSModel:
    return fit_als(train, rank=2, max_iter=3, seed=0)


class TestFit:
    def test_the_cold_start_strategy_is_never_drop(self, model: ALSModel) -> None:
        """The one setting that would corrupt the denominator silently.

        Asserted on the fitted model rather than trusted from the call site,
        because a future edit to fit_als is exactly what this guards.
        """
        assert model.getColdStartStrategy() == "nan"

    def test_the_oov_bucket_is_not_factorised(self, spark: SparkSession) -> None:
        """Index 0 is "unknown", not an entity.

        Learn a vector for it and every unmapped id becomes similar to every
        other unmapped id -- a similarity the data never showed.
        """
        train = _rows(
            spark,
            [
                ("U1", "N1", 1, 1, 1, True, T0),
                ("U?", "N?", 0, 0, 2, True, T0),
            ],
        )
        fitted = fit_als(train, rank=2, max_iter=3, seed=0)

        assert 0 not in {r["id"] for r in fitted.userFactors.collect()}
        assert 0 not in {r["id"] for r in fitted.itemFactors.collect()}

    def test_clicks_not_impressions_are_the_signal(self, model: ALSModel) -> None:
        """N9 was shown and never clicked, so it has no factor at all."""
        assert 9 not in {r["id"] for r in model.itemFactors.collect()}


class TestScoreAls:
    def test_an_unseen_user_scores_zero_and_keeps_its_row(
        self, spark: SparkSession, train: DataFrame, model: ALSModel
    ) -> None:
        """The cold-user wall, and proof the row survived it.

        Under coldStartStrategy="drop" this row would simply not be in the
        output, the card would be computed over a smaller split, and every
        metric on it would be measuring a different population.
        """
        labels = _rows(
            spark,
            [
                ("U1", "N1", 1, 1, 10, False, T0 + timedelta(hours=1)),
                ("U99", "N1", 99, 1, 10, False, T0 + timedelta(hours=1)),
            ],
        )
        got = {r["user_id"]: r["score"] for r in score_als(labels, train, model).collect()}

        assert len(got) == 2
        assert got["U99"] == 0.0

    def test_nan_predictions_become_zero_not_nan(
        self, spark: SparkSession, train: DataFrame, model: ALSModel
    ) -> None:
        """NaN must not survive into the metrics.

        coalesce does not catch it -- NaN is not null -- and a NaN score poisons
        every comparison it enters, because NaN > x and NaN < x are both false.
        A slate carrying one would silently lose comparisons rather than error.
        """
        labels = _rows(spark, [("U99", "N99", 99, 99, 10, False, T0 + timedelta(hours=1))])
        scores = [r["score"] for r in score_als(labels, train, model).collect()]

        assert scores == [0.0]
        assert not any(s != s for s in scores)

    def test_the_row_count_is_preserved(
        self, spark: SparkSession, train: DataFrame, model: ALSModel
    ) -> None:
        labels = _rows(
            spark,
            [
                ("U1", "N1", 1, 1, 10, False, T0 + timedelta(hours=1)),
                ("U1", "N2", 1, 2, 10, True, T0 + timedelta(hours=1)),
                ("U99", "N3", 99, 3, 11, False, T0 + timedelta(hours=1)),
            ],
        )

        assert score_als(labels, train, model).count() == 3


class TestScoreAlsItem:
    def test_it_reaches_a_user_absent_from_train(
        self, spark: SparkSession, train: DataFrame, model: ALSModel
    ) -> None:
        """The whole reason this formulation exists.

        U99 never appears in train, so score_als gives it nothing. Here its own
        earlier click supplies the profile, and a warm candidate scores.
        """
        labels = _rows(
            spark,
            [
                ("U99", "N1", 99, 1, 20, True, T0 + timedelta(hours=1)),
                ("U99", "N2", 99, 2, 21, False, T0 + timedelta(hours=2)),
            ],
        )

        blind = {r["impression_id"]: r["score"] for r in score_als(labels, train, model).collect()}
        reaching = {
            r["impression_id"]: r["score"] for r in score_als_item(labels, train, model).collect()
        }

        assert blind[21] == 0.0
        assert reaching[21] != 0.0

    def test_only_clicks_before_the_label_contribute(
        self, spark: SparkSession, train: DataFrame, model: ALSModel
    ) -> None:
        """Same candidate, same user, two instants -- one before its own click."""
        labels = _rows(
            spark,
            [
                ("U99", "N1", 99, 1, 30, True, T0 + timedelta(hours=1)),
                ("U99", "N2", 99, 2, 31, False, T0 - timedelta(hours=9)),
                ("U99", "N2", 99, 2, 32, False, T0 + timedelta(hours=2)),
            ],
        )
        scored = score_als_item(labels, train, model).collect()
        got = {r["impression_id"]: r["score"] for r in scored}

        assert got[31] == 0.0
        assert got[32] != 0.0

    def test_a_cold_item_scores_zero(
        self, spark: SparkSession, train: DataFrame, model: ALSModel
    ) -> None:
        """No interactions in train, no factor, nothing to compare against.

        This wall is real rather than a quirk of the formulation, and it is why
        content similarity is the baseline that matters for cold items.
        """
        labels = _rows(
            spark,
            [
                ("U1", "N1", 1, 1, 40, True, T0 + timedelta(hours=1)),
                ("U1", "NEW", 1, 777, 41, False, T0 + timedelta(hours=2)),
            ],
        )
        scored = score_als_item(labels, train, model).collect()
        got = {r["impression_id"]: r["score"] for r in scored}

        assert got[41] == 0.0

    def test_the_row_count_is_preserved(
        self, spark: SparkSession, train: DataFrame, model: ALSModel
    ) -> None:
        """Two joins and a group-by. A lost row changes every denominator."""
        labels = _rows(
            spark,
            [
                ("U1", "N1", 1, 1, 50, False, T0 + timedelta(hours=1)),
                ("U1", "N2", 1, 2, 50, True, T0 + timedelta(hours=1)),
                ("U99", "N3", 99, 3, 51, False, T0 - timedelta(days=9)),
                ("U2", "N3", 2, 3, 52, False, T0 + timedelta(hours=1)),
            ],
        )

        assert score_als_item(labels, train, model).count() == 4
