"""Co-visitation: the three ways this model leaks, none of which move a metric.

Pairing inside an impression measures Microsoft's recommender rather than
users. Building the matrix over train plus dev puts the label being predicted
into the edge that scores it. Using ``<=`` instead of ``<`` lets a user's clicks
in the impression under evaluation score that impression -- and because every
row of a MIND impression shares one timestamp, that is not a boundary case, it
is every slate.

All three make the model look BETTER. None raises, warns, or produces an
implausible number, so each one is tested against an arithmetic value that only
the correct implementation can produce.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest
from pyspark.sql import DataFrame, SparkSession

from models.retrieval.baselines.covisit import (
    BACKWARD_DISCOUNT,
    build_covisitation,
    click_stream,
    score_covisit,
)

T0 = datetime(2019, 11, 14, 12, 0, 0)

_EVENTS = "user_id string, item_id string, impression_id long, ts timestamp, clicked boolean"


def weight(minutes: float) -> float:
    """The edge weight the implementation must produce for a gap, computed independently.

    Written out rather than imported so a change to the decay has to be made
    twice, deliberately, instead of silently agreeing with itself.
    """
    return 1.0 / math.log2(2.0 + minutes)


# A 30-minute gap weighs exactly 1/log2(32) = 0.2, which makes the asymmetry and
# accumulation assertions readable rather than a wall of float noise.
W30 = weight(30)
W60 = weight(60)


@pytest.fixture
def train(spark: SparkSession) -> DataFrame:
    """One user, three clicks in three impressions, 30 minutes apart.

    N9 is shown and not clicked, so anything counting impressions rather than
    clicks picks it up.
    """
    return spark.createDataFrame(
        [
            ("U1", "N1", 1, T0 - timedelta(minutes=60), True),
            ("U1", "N9", 1, T0 - timedelta(minutes=60), False),
            ("U1", "N2", 2, T0 - timedelta(minutes=30), True),
            ("U1", "N3", 3, T0, True),
        ],
        _EVENTS,
    )


def edges(matrix: DataFrame) -> dict[tuple[str, str], float]:
    return {(r["item_id"], r["related_item_id"]): r["weight"] for r in matrix.collect()}


def scores(frame: DataFrame) -> dict[int, float]:
    return {r["impression_id"]: r["score"] for r in frame.collect()}


class TestMatrix:
    def test_pairs_never_come_from_one_impression(self, spark: SparkSession) -> None:
        """Two clicks in the same slate are not a co-visitation.

        Those items were placed together by MIND's incumbent recommender, not
        chosen together by the user. Counting them trains the model to imitate
        MSN. They are also degenerate: one impression carries one timestamp, so
        the gap is zero and the decay and asymmetry both collapse.
        """
        same_slate = spark.createDataFrame(
            [
                ("U2", "N4", 10, T0, True),
                ("U2", "N5", 10, T0, True),
            ],
            _EVENTS,
        )

        assert edges(build_covisitation(same_slate)) == {}

    def test_the_forward_edge_is_stronger_than_the_backward(self, train: DataFrame) -> None:
        """ "Clicked A then B" is not the same signal as "clicked B then A"."""
        got = edges(build_covisitation(train))

        assert got[("N1", "N2")] == pytest.approx(W30)
        assert got[("N2", "N1")] == pytest.approx(W30 * BACKWARD_DISCOUNT)

    def test_an_unclicked_item_never_appears(self, train: DataFrame) -> None:
        """N9 was shown, never clicked; impression counts would drag it in."""
        got = edges(build_covisitation(train))

        assert not any("N9" in pair for pair in got)

    def test_a_pair_beyond_the_time_window_is_dropped(self, train: DataFrame) -> None:
        """N1 to N3 spans 60 minutes and must fall outside a 45-minute window.

        The 30-minute pairs survive, so this isolates the window rather than
        just emptying the matrix.
        """
        got = edges(build_covisitation(train, max_gap_seconds=45 * 60))

        assert ("N1", "N3") not in got
        assert got[("N1", "N2")] == pytest.approx(W30)
        assert got[("N2", "N3")] == pytest.approx(W30)

    def test_the_rank_cap_bounds_the_pair_explosion(self, train: DataFrame) -> None:
        """At a cap of one, only clicks adjacent in the stream pair up.

        N1 to N3 is two clicks apart and goes, even though 60 minutes is inside
        the default time window -- which is the point: the cap is a compute
        guard that binds independently of the modelling parameter.
        """
        got = edges(build_covisitation(train, max_rank_gap=1))

        assert ("N1", "N3") not in got
        assert ("N1", "N2") in got
        assert ("N2", "N3") in got

    def test_weights_accumulate_across_users(self, spark: SparkSession, train: DataFrame) -> None:
        """The same pair seen twice weighs twice. A matrix that overwrites ranks by
        last-writer rather than by evidence."""
        second_user = spark.createDataFrame(
            [
                ("U7", "N1", 20, T0 - timedelta(minutes=60), True),
                ("U7", "N2", 21, T0 - timedelta(minutes=30), True),
            ],
            _EVENTS,
        )
        got = edges(build_covisitation(train.unionByName(second_user)))

        assert got[("N1", "N2")] == pytest.approx(2 * W30)

    def test_top_k_keeps_the_strongest_neighbour(self, train: DataFrame) -> None:
        """N1 neighbours N2 at 30 minutes and N3 at 60; only N2 survives k=1."""
        got = edges(build_covisitation(train, top_k=1))

        assert got[("N1", "N2")] == pytest.approx(W30)
        assert ("N1", "N3") not in got

    def test_the_click_stream_ranks_in_time_order(self, train: DataFrame) -> None:
        """Ranks must follow ts. MIND's impression ids are not chronological, so
        ordering by them would rank a shuffled timeline."""
        ranked = {r["item_id"]: r["rank"] for r in click_stream(train).collect()}

        assert ranked == {"N1": 1, "N2": 2, "N3": 3}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_gap_seconds": 0},
            {"max_gap_seconds": -1},
            {"max_rank_gap": 0},
            {"top_k": 0},
        ],
    )
    def test_a_non_positive_bound_is_refused(
        self, train: DataFrame, kwargs: dict[str, int]
    ) -> None:
        """An empty matrix reads downstream as "co-visitation does not work on
        news". Refusing the configuration is the only way that cannot happen."""
        with pytest.raises(ValueError):
            build_covisitation(train, **kwargs)


class TestScoring:
    def _labels(self, spark: SparkSession, rows: list[tuple[object, ...]]) -> DataFrame:
        return spark.createDataFrame(rows, _EVENTS)

    def test_only_clicks_before_the_label_score(
        self, spark: SparkSession, train: DataFrame
    ) -> None:
        """The same user and the same candidate, scored at two instants.

        At T0+2h the user's N1 and N2 clicks are knowable and support N3. Ninety
        minutes before T0 none of them have happened. A model reading the user's
        clicks without regard to when they occurred cannot tell these apart.
        """
        labels = self._labels(
            spark,
            [
                ("U1", "N3", 100, T0 - timedelta(minutes=90), False),
                ("U1", "N3", 101, T0 + timedelta(hours=2), False),
            ],
        )
        got = scores(score_covisit(labels, train))

        assert got[100] == pytest.approx(0.0)
        assert got[101] == pytest.approx(W60 + W30)

    def test_a_click_at_the_label_instant_does_not_count(
        self, spark: SparkSession, train: DataFrame
    ) -> None:
        """``<`` not ``<=``, and on MIND the difference is every slate.

        The label sits at exactly T0-30min, the instant U1 clicked N2. Only the
        N1 click precedes it, so N3 scores W60 alone. Under ``<=`` the N2 click
        would join in and the score would be W60 + W30 -- and since every row of
        an impression shares one timestamp, that is the impression under
        evaluation scoring itself.
        """
        labels = self._labels(spark, [("U1", "N3", 102, T0 - timedelta(minutes=30), False)])
        got = scores(score_covisit(labels, train))

        assert got[102] == pytest.approx(W60)
        assert got[102] != pytest.approx(W60 + W30)

    def test_the_matrix_ignores_pairs_that_exist_only_in_the_labels(
        self, spark: SparkSession, train: DataFrame
    ) -> None:
        """Model parameters come from train alone.

        U8 clicks N6 then N7 in the evaluation split. That pair is nowhere in
        train, so N7 has no edge to score it with -- however knowable those
        clicks were at label time. Building the matrix over train plus the split
        would put the label's own click into the edge that ranks it.
        """
        labels = self._labels(
            spark,
            [
                ("U8", "N6", 110, T0 + timedelta(minutes=1), True),
                ("U8", "N7", 111, T0 + timedelta(minutes=31), True),
            ],
        )
        got = scores(score_covisit(labels, train))

        assert got[111] == pytest.approx(0.0)

    def test_context_clicks_are_not_double_counted(
        self, spark: SparkSession, train: DataFrame
    ) -> None:
        """A click reached through both train and the labels is one event.

        Counting it twice doubles every score and moves no ranking, so GAUC and
        NDCG stay exactly where they were while the numbers are wrong.
        """
        labels = train.unionByName(
            self._labels(spark, [("U1", "N3", 103, T0 + timedelta(hours=2), False)])
        )
        got = scores(score_covisit(labels, train))

        assert got[103] == pytest.approx(W60 + W30)

    def test_a_user_with_no_prior_clicks_scores_zero_not_null(
        self, spark: SparkSession, train: DataFrame
    ) -> None:
        """Being unable to score a cold user is the finding, not a gap to patch."""
        labels = self._labels(spark, [("U99", "N3", 104, T0 + timedelta(hours=2), False)])
        got = scores(score_covisit(labels, train))

        assert got[104] == pytest.approx(0.0)

    def test_the_row_count_is_preserved(self, spark: SparkSession, train: DataFrame) -> None:
        """Scoring fans out through two joins. A row lost or duplicated here
        changes the denominator of every metric on the card."""
        labels = self._labels(
            spark,
            [
                ("U1", "N1", 105, T0 + timedelta(hours=2), False),
                ("U1", "N2", 105, T0 + timedelta(hours=2), False),
                ("U1", "N3", 105, T0 + timedelta(hours=2), True),
                # A user with clicks, none of which precede the label. This is
                # the row the first implementation lost: the left join matched,
                # every matched row then failed the time test, and the label
                # left the split entirely.
                ("U1", "N3", 107, T0 - timedelta(minutes=90), False),
                # A user with no clicks at all -- a different path through the
                # same join, and the only one the left join protects by itself.
                ("U99", "N3", 106, T0 + timedelta(hours=2), False),
            ],
        )

        assert score_covisit(labels, train).count() == 5

    def test_a_train_only_context_starves_a_cold_user(
        self, spark: SparkSession, train: DataFrame
    ) -> None:
        """The ablation behind the default, as a test rather than a claim.

        U8's clicks are in the split, not in train. With the default context
        they are knowable and score; restricted to train the user has no history
        at all. On dev that is 87.8% of impressions, which is the difference
        between a number and a dead baseline.
        """
        labels = self._labels(
            spark,
            [
                ("U8", "N1", 120, T0 - timedelta(minutes=61), True),
                ("U8", "N2", 121, T0 - timedelta(minutes=31), True),
                ("U8", "N3", 122, T0 + timedelta(hours=2), False),
            ],
        )

        with_split = scores(score_covisit(labels, train))
        train_only = scores(score_covisit(labels, train, context=train))

        assert with_split[122] > 0.0
        assert train_only[122] == pytest.approx(0.0)
