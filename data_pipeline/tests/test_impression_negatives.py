"""Slate negatives, and the property §9.4's sketch does not have.

The one that matters is determinism. ``F.shuffle(F.collect_list(...))`` is
non-reproducible twice over -- ``shuffle`` takes no seed, and ``collect_list``
returns whatever order the partitions produced -- and these rows are TRAINING
input, so two runs of the same configuration would disagree by more than noise
and G3's bottom table row could not be cited. The hash order is what makes
``test_the_order_survives_a_reshuffle`` possible at all.

The second is that a slate with no unclicked item produces no row rather than
an empty one. The loader's left join is what turns that into an all-padding
request, and an inner join there would silently delete training examples.
"""

from __future__ import annotations

import pytest
from pyspark.sql import DataFrame, Row, SparkSession

from data_pipeline.features.impression_negatives import slate_negatives


def _slate(spark: SparkSession, rows: list[tuple[int, int, bool]]) -> DataFrame:
    return spark.createDataFrame([Row(impression_id=i, item_idx=k, clicked=c) for i, k, c in rows])


def _negatives(frame: DataFrame) -> dict[int, list[int]]:
    return {row["impression_id"]: list(row["neg_idx"]) for row in frame.collect()}


class TestWhatIsKept:
    def test_only_unclicked_items_become_negatives(self, spark: SparkSession) -> None:
        got = _negatives(
            slate_negatives(_slate(spark, [(1, 10, True), (1, 11, False), (1, 12, False)]))
        )

        assert sorted(got[1]) == [11, 12]

    def test_a_fully_clicked_slate_produces_no_row(self, spark: SparkSession) -> None:
        """Not an empty array -- no row. The loader left-joins, so the request
        still trains, on in-batch negatives alone."""
        got = _negatives(slate_negatives(_slate(spark, [(1, 10, True), (2, 20, False)])))

        assert 1 not in got
        assert got[2] == [20]

    def test_the_cap_binds(self, spark: SparkSession) -> None:
        rows = [(1, k, False) for k in range(10, 20)]
        got = _negatives(slate_negatives(_slate(spark, rows), max_negs=3))

        assert len(got[1]) == 3

    def test_slates_do_not_borrow_from_each_other(self, spark: SparkSession) -> None:
        """The grain is the impression. A groupBy on the wrong key would hand one
        request another's slate and nothing downstream would notice."""
        got = _negatives(
            slate_negatives(_slate(spark, [(1, 11, False), (1, 12, False), (2, 21, False)]))
        )

        assert sorted(got[1]) == [11, 12]
        assert got[2] == [21]

    def test_neg_len_matches_what_was_kept(self, spark: SparkSession) -> None:
        rows = [(1, k, False) for k in range(10, 20)]
        written = slate_negatives(_slate(spark, rows), max_negs=4).collect()[0]

        assert written["neg_len"] == len(written["neg_idx"]) == 4


class TestDeterminism:
    def test_the_order_survives_a_reshuffle(self, spark: SparkSession) -> None:
        """The whole reason the order is a hash rather than F.shuffle.

        Repartitioning changes the order rows reach the aggregate in, which is
        exactly what ``collect_list`` would otherwise expose.
        """
        rows = [(1, k, False) for k in range(10, 30)]
        frame = _slate(spark, rows)

        first = _negatives(slate_negatives(frame.repartition(1), max_negs=5))
        second = _negatives(slate_negatives(frame.repartition(7), max_negs=5))

        assert first == second

    def test_row_order_does_not_change_the_result(self, spark: SparkSession) -> None:
        rows = [(1, k, False) for k in range(10, 30)]

        forward = _negatives(slate_negatives(_slate(spark, rows), max_negs=5))
        backward = _negatives(slate_negatives(_slate(spark, list(reversed(rows))), max_negs=5))

        assert forward == backward

    def test_the_sample_is_not_the_lowest_indices(self, spark: SparkSession) -> None:
        """Ordering by item_idx would be deterministic too, and would bias every
        slate toward whichever articles were ingested first -- a systematic
        negative sample dressed as a random one."""
        rows = [(1, k, False) for k in range(100, 160)]
        got = _negatives(slate_negatives(_slate(spark, rows), max_negs=5))

        assert set(got[1]) != set(range(100, 105))

    def test_a_prefix_is_a_subset_of_a_longer_prefix(self, spark: SparkSession) -> None:
        """What lets the loader sweep max_negs without rebuilding gold: taking
        k of the stored array must mean the same k every time, and the same k a
        larger cap would have started with."""
        rows = [(1, k, False) for k in range(10, 40)]
        frame = _slate(spark, rows)

        short = _negatives(slate_negatives(frame, max_negs=3))
        long = _negatives(slate_negatives(frame, max_negs=9))

        assert short[1] == long[1][:3]


class TestTheFalseNegativeCase:
    def test_an_index_that_is_both_clicked_and_not_is_still_emitted(
        self, spark: SparkSession
    ) -> None:
        """Documented rather than defended against, because the guard is
        downstream and this is the honest place to say so.

        The filter is row-wise, so an index appearing both clicked and unclicked
        inside one impression comes out as a negative for its own positive --
        the false-negative case ``sampled_softmax_loss``'s duplicate mask exists
        for. Deduplicating here instead would be the wrong layer: the loss has to
        handle a repeat across the whole batch regardless, since two users
        clicking the same article in one batch produces exactly the same
        collision from entirely valid rows.

        It also cannot arise from this corpus. MIND lists each article once per
        impression, and C2 measured zero OOV across 8,584,442 silver rows, so
        nothing collapses two articles onto one index.
        """
        got = _negatives(slate_negatives(_slate(spark, [(1, 10, True), (1, 10, False)])))

        assert got[1] == [10]


@pytest.mark.parametrize("max_negs", [1, 4, 20])
def test_the_cap_never_exceeds_the_slate(spark: SparkSession, max_negs: int) -> None:
    rows = [(1, k, False) for k in range(10, 15)]
    got = _negatives(slate_negatives(_slate(spark, rows), max_negs=max_negs))

    assert len(got[1]) == min(max_negs, 5)
