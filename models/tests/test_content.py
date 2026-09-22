"""Content similarity: the cold-item wall, and the two ways a lexical score lies.

This is the only baseline that can score an item train has never seen, so the
test that matters most is the one showing it does. The two failure modes are
quieter: a stopword-heavy title matching everything, and a profile built from
clicks the request could not have known about.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pyspark.sql import DataFrame, SparkSession

from models.retrieval.baselines.content import score_content, term_weights, tokenize

T0 = datetime(2019, 11, 14, 12, 0, 0)

_EVENTS = "user_id string, item_id string, impression_id long, clicked boolean, ts timestamp"
_TITLES = "item_id string, title string"


@pytest.fixture
def catalogue(spark: SparkSession) -> DataFrame:
    """Two election stories, one recipe, and a cold election story."""
    return spark.createDataFrame(
        [
            ("N1", "Senate election results announced"),
            ("N2", "Election turnout breaks record"),
            ("N3", "The best pancake recipe"),
            ("NEW", "Election recount ordered in three states"),
        ],
        _TITLES,
    )


@pytest.fixture
def train(spark: SparkSession) -> DataFrame:
    """U1 clicked an election story. N1 is the only item train has seen clicked."""
    return spark.createDataFrame(
        [("U1", "N1", 1, True, T0 - timedelta(hours=2))],
        _EVENTS,
    )


def _scores(frame: DataFrame) -> dict[int, float]:
    return {r["impression_id"]: r["score"] for r in frame.collect()}


class TestTokenize:
    def test_stopwords_and_single_characters_go(
        self, spark: SparkSession, catalogue: DataFrame
    ) -> None:
        """ "The" would otherwise match every title to every other."""
        terms = {r["term"] for r in tokenize(catalogue).collect()}

        assert "the" not in terms
        assert "in" not in terms
        assert "election" in terms

    def test_a_null_title_yields_no_terms(self, spark: SparkSession) -> None:
        """Cold items sometimes arrive without metadata; that must not raise."""
        frame = spark.createDataFrame([("N5", None)], _TITLES)

        assert tokenize(frame).count() == 0


class TestTermWeights:
    def test_a_term_in_every_title_weighs_zero(self, spark: SparkSession) -> None:
        """log(N/df) is unsmoothed on purpose: a word all articles use
        distinguishes nothing, and should contribute nothing."""
        frame = spark.createDataFrame(
            [("N1", "election monday"), ("N2", "election tuesday")], _TITLES
        )
        weights, _ = term_weights(frame)
        got = {(r["item_id"], r["term"]): r["weight"] for r in weights.collect()}

        assert got[("N1", "election")] == 0.0
        assert got[("N1", "monday")] > 0.0


class TestScoreContent:
    def test_it_scores_an_item_train_has_never_seen(
        self, spark: SparkSession, train: DataFrame, catalogue: DataFrame
    ) -> None:
        """The reason this baseline exists.

        NEW appears in no training interaction, so popularity, co-visitation
        and ALS all score it 0.0 by construction. It shares "election" with what
        U1 read, and that is reachable from the title alone.
        """
        labels = spark.createDataFrame(
            [("U1", "NEW", 10, False, T0 + timedelta(hours=1))],
            _EVENTS,
        )

        assert _scores(score_content(labels, train, catalogue))[10] > 0.0

    def test_an_unrelated_title_scores_below_a_related_one(
        self, spark: SparkSession, train: DataFrame, catalogue: DataFrame
    ) -> None:
        """A recipe is not an election story, and the vectors should say so."""
        labels = spark.createDataFrame(
            [
                ("U1", "N2", 20, False, T0 + timedelta(hours=1)),
                ("U1", "N3", 21, False, T0 + timedelta(hours=1)),
            ],
            _EVENTS,
        )
        got = _scores(score_content(labels, train, catalogue))

        assert got[20] > got[21]

    def test_only_clicks_before_the_label_build_the_profile(
        self, spark: SparkSession, train: DataFrame, catalogue: DataFrame
    ) -> None:
        """Same user, same candidate, scored either side of the only click.

        MIND gives one timestamp per impression, so `<` rather than `<=` is also
        what keeps a slate from scoring itself.
        """
        labels = spark.createDataFrame(
            [
                ("U1", "N2", 30, False, T0 - timedelta(hours=9)),
                ("U1", "N2", 31, False, T0 + timedelta(hours=1)),
            ],
            _EVENTS,
        )
        got = _scores(score_content(labels, train, catalogue))

        assert got[30] == 0.0
        assert got[31] > 0.0

    def test_a_user_with_no_prior_click_scores_zero_not_null(
        self, spark: SparkSession, train: DataFrame, catalogue: DataFrame
    ) -> None:
        """Cosine against an empty profile is undefined; 0.0 is the honest read."""
        labels = spark.createDataFrame(
            [("U99", "N2", 40, False, T0 + timedelta(hours=1))],
            _EVENTS,
        )

        assert _scores(score_content(labels, train, catalogue))[40] == 0.0

    def test_the_score_is_a_cosine_and_stays_within_one(
        self, spark: SparkSession, train: DataFrame, catalogue: DataFrame
    ) -> None:
        """Normalisation is what stops a long headline outscoring a relevant one.

        An unnormalised dot product rewards titles for having more words, which
        on a news corpus means rewarding whichever desk writes longest.
        """
        labels = spark.createDataFrame(
            [
                ("U1", "N1", 50, False, T0 + timedelta(hours=1)),
                ("U1", "NEW", 51, False, T0 + timedelta(hours=1)),
            ],
            _EVENTS,
        )
        got = _scores(score_content(labels, train, catalogue))

        assert all(0.0 <= value <= 1.0 + 1e-9 for value in got.values())
        # N1 against a profile built from N1 itself is the same vector.
        assert got[50] == pytest.approx(1.0)

    def test_the_row_count_is_preserved(
        self, spark: SparkSession, train: DataFrame, catalogue: DataFrame
    ) -> None:
        """Four joins, one group-by, and an inner join in the middle.

        The dot product is computed over shared terms only -- an inner join that
        drops rows with nothing in common -- and the count is restored by
        left-joining back onto labels. If that left join ever became inner,
        every slate would silently shrink to its matching candidates.
        """
        labels = spark.createDataFrame(
            [
                ("U1", "N2", 60, False, T0 + timedelta(hours=1)),
                ("U1", "N3", 60, True, T0 + timedelta(hours=1)),
                ("U99", "N3", 61, False, T0 + timedelta(hours=1)),
            ],
            _EVENTS,
        )

        assert score_content(labels, train, catalogue).count() == 3
