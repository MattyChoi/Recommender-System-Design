"""The policy layer, and the six ways it could reorder a slate wrongly in silence.

**Ordering.** Every policy here changes WHICH items are served and in WHAT
order, and both are invisible in a shape. A slate of ten items is a slate of
ten items whether or not MMR ever looked at a vector, so each knob is tested at
its null position as well as its active one -- a lever with no measurable
off-state is a lever nobody can trust.

**Per-request slicing.** ``slate_per_request`` holds four parallel arrays and
must cut all of them at the same boundary. Handing one request's scores to
another request's categories produces a full, plausible slate.

**Propensity.** It cannot be recomputed after the fact, so if it is wrong at
serving time it is wrong forever, and every off-policy estimate built on it is
quietly biased. The arithmetic is pinned here rather than trusted.

**Metric order.** ``score_slates`` must score the slate's OWN order. Handing the
model's scores back to the NDCG function re-sorts the slate and silently undoes
the policy -- the table would then show every arm scoring identically to the
ranker and read as "MMR costs nothing".

**The Bloom filter's error direction.** The one-sided guarantee is the entire
argument for using it. A filter that never returns True would also have no
false negatives, so the control matters as much as the property.

**Sizing arithmetic**, and what happens past capacity -- which is not the
graceful degradation this file first claimed. At ten times capacity the filter
saturates and hides every candidate.

Run just this file: ``uv run pytest models/tests/test_reranking.py -q``
"""

from __future__ import annotations

import numpy as np
import pytest

from models.ranking.dataset import RankingRows
from models.reranking.evaluate import score_slates
from models.reranking.headroom import LONG_TAIL_EDGE, top_k_items
from models.reranking.policies import (
    DETERMINISTIC,
    freshness_multiplier,
    normalise,
    select,
    slate_per_request,
)
from models.reranking.seen import BloomFilter, measure_false_positives, sizing

ITEMS = np.array([10, 11, 12, 13], dtype=np.int64)


def orthogonal(n: int, dim: int = 4) -> np.ndarray:
    """``n`` unit vectors that are pairwise orthogonal, so similarity is 0 or 1."""
    return np.eye(n, dim, dtype=np.float32)


class TestMmr:
    def test_lambda_one_is_exactly_the_ranker(self) -> None:
        """The null position of the knob. Without this, a broken MMR that always
        returned the ranker's order would pass every other test here."""
        scores = np.array([1.0, 0.9, 0.8, 0.7])
        vectors = orthogonal(4)

        with_mmr = select(scores, ITEMS, k=3, vectors=vectors, lambda_=1.0)
        without = select(scores, ITEMS, k=3)

        assert with_mmr.items.tolist() == without.items.tolist() == [10, 11, 12]

    def test_a_near_duplicate_is_demoted(self) -> None:
        """The active position, on the case MMR exists for: the second-best
        candidate is a copy of the best, and the third is unlike either."""
        vectors = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],  # 0 and 1 identical
            dtype=np.float32,
        )
        scores = np.array([1.0, 0.9, 0.8])
        items = np.array([10, 11, 12], dtype=np.int64)

        greedy = select(scores, items, k=2, vectors=vectors, lambda_=1.0)
        diverse = select(scores, items, k=2, vectors=vectors, lambda_=0.5)

        assert greedy.items.tolist() == [10, 11]
        assert diverse.items.tolist() == [10, 12]

    def test_similarity_is_to_the_whole_chosen_set_not_the_last_pick(self) -> None:
        """A running MAXIMUM, not the similarity to the most recent choice. With
        the latter, a candidate identical to slot 0 becomes eligible again as
        soon as slot 1 is unlike it."""
        vectors = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],  # 2 is a copy of 0
            dtype=np.float32,
        )
        scores = np.array([1.0, 0.9, 0.85])
        items = np.array([10, 11, 12], dtype=np.int64)

        slate = select(scores, items, k=3, vectors=vectors, lambda_=0.5)

        # 12 is still served -- k forces it -- but it must come last.
        assert slate.items.tolist() == [10, 11, 12]


class TestCategoryCaps:
    def test_a_cap_forces_a_lower_scoring_category(self) -> None:
        scores = np.array([3.0, 2.0, 1.0])
        categories = np.array([0, 0, 1], dtype=np.int64)
        items = np.array([10, 11, 12], dtype=np.int64)

        capped = select(scores, items, k=2, categories=categories, cap=1)
        uncapped = select(scores, items, k=2)

        assert capped.items.tolist() == [10, 12]
        assert uncapped.items.tolist() == [10, 11]

    def test_an_unsatisfiable_cap_fills_the_slate_anyway(self) -> None:
        """**The decision, pinned.** Every candidate shares one category and the
        cap is 1, so the constraint cannot be honoured past slot 0. A short
        slate would be a product decision nobody made -- a blank slot on the
        page -- so the cap yields instead."""
        scores = np.array([3.0, 2.0, 1.0])
        categories = np.zeros(3, dtype=np.int64)
        items = np.array([10, 11, 12], dtype=np.int64)

        slate = select(scores, items, k=3, categories=categories, cap=1)

        assert len(slate.items) == 3

    def test_capping_inside_the_loop_keeps_the_slate_full(self) -> None:
        """Why caps are not a filter applied after MMR. Capping afterwards
        discards the offending pick and returns k-1 items; capping inside
        promotes the best FEASIBLE candidate and returns k."""
        scores = np.array([3.0, 2.0, 1.0, 0.5])
        categories = np.array([0, 0, 1, 1], dtype=np.int64)

        slate = select(scores, ITEMS, k=3, categories=categories, cap=1)
        after_the_fact = [item for item in [10, 11, 12] if item != 11]

        assert len(slate.items) == 3
        assert len(after_the_fact) == 2


class TestExplorationAndPropensity:
    def test_a_deterministic_slate_logs_one_everywhere(self) -> None:
        slate = select(np.array([3.0, 2.0, 1.0, 0.5]), ITEMS, k=3)

        assert slate.propensity.tolist() == [DETERMINISTIC] * 3

    def test_only_the_last_slots_explore(self) -> None:
        """Last rather than first: an explored item in slot 0 costs the most
        relevance, and the point is evidence at the cheapest slot still seen."""
        slate = select(
            np.array([3.0, 2.0, 1.0, 0.5]),
            ITEMS,
            k=3,
            epsilon=1.0,
            explore_slots=1,
            rng=np.random.default_rng(0),
        )

        assert slate.propensity[0] == DETERMINISTIC
        assert slate.propensity[1] == DETERMINISTIC
        assert slate.propensity[2] < DETERMINISTIC

    def test_the_always_random_propensity_is_one_over_what_was_available(self) -> None:
        """At epsilon 1 the slot is a uniform draw from what is left: two of the
        four candidates are already placed, so the denominator is 2, not 4.
        Logging 1/4 here would bias every downstream estimate by 2x."""
        slate = select(
            np.array([3.0, 2.0, 1.0, 0.5]),
            ITEMS,
            k=3,
            epsilon=1.0,
            explore_slots=1,
            rng=np.random.default_rng(0),
        )

        assert slate.propensity[2] == pytest.approx(0.5)

    def test_the_greedy_branch_counts_both_paths_to_the_same_item(self) -> None:
        """When the coin says "be greedy", that item could ALSO have come from
        the random draw. Logging only ``1 - epsilon`` understates the propensity
        of exactly the item most likely to be logged."""
        rng = np.random.default_rng(12345)
        slate = select(
            np.array([3.0, 2.0, 1.0, 0.5]),
            ITEMS,
            k=2,
            epsilon=0.5,
            explore_slots=1,
            rng=rng,
        )

        # Three candidates remain at slot 1, so either branch lands on a value
        # built from both terms; neither is a bare 0.5.
        assert slate.propensity[1] in (
            pytest.approx(0.5 + 0.5 / 3),
            pytest.approx(0.5 / 3 + 0.5),
            pytest.approx(0.5 / 3),
        )
        assert slate.propensity[1] > 0.0

    def test_exploration_without_a_generator_is_refused(self) -> None:
        """A run that explores and cannot be reproduced is a run whose numbers
        cannot be checked."""
        with pytest.raises(ValueError, match="rng"):
            select(np.array([1.0, 0.5]), ITEMS[:2], k=1, epsilon=0.1, explore_slots=1)

    def test_epsilon_without_slots_is_refused(self) -> None:
        with pytest.raises(ValueError, match="explores nothing"):
            select(
                np.array([1.0, 0.5]),
                ITEMS[:2],
                k=1,
                epsilon=0.1,
                rng=np.random.default_rng(0),
            )

    def test_the_same_seed_gives_the_same_slate(self) -> None:
        def run() -> list[int]:
            slate = select(
                np.array([3.0, 2.0, 1.0, 0.5]),
                ITEMS,
                k=3,
                epsilon=0.9,
                explore_slots=2,
                rng=np.random.default_rng(7),
            )
            return [int(item) for item in slate.items]

        assert run() == run()


class TestPerRequestSlicing:
    def test_each_request_uses_its_own_categories(self) -> None:
        """**The bug this file exists to prevent.** The first version forwarded
        per-candidate arrays unsliced, so request two would have been judged
        against request one's categories. The fixture is built so that mistake
        changes the answer rather than raising."""
        scores = np.array([3.0, 2.0, 1.0, 3.0, 2.0, 1.0])
        items = np.array([10, 11, 12, 20, 21, 22], dtype=np.int64)
        categories = np.array([0, 0, 1, 0, 1, 1], dtype=np.int64)

        slates = slate_per_request(scores, items, [3, 3], k=2, categories=categories, cap=1)

        # Request one: slot 0 takes category 0, so slot 1 must take item 12.
        assert slates[0].items.tolist() == [10, 12]
        # Request two: only item 20 is category 0, so slot 1 takes the better of
        # the two category-1 items -- 21, NOT 22. Using request one's categories
        # here would have forced 22.
        assert slates[1].items.tolist() == [20, 21]

    def test_row_indices_are_shifted_into_the_flat_arrays(self) -> None:
        """The rows come back local to each block and must be usable against the
        arrays the caller passed in, or a label lookup reads another request's
        rows."""
        scores = np.array([3.0, 2.0, 1.0, 3.0, 2.0, 1.0])
        items = np.array([10, 11, 12, 20, 21, 22], dtype=np.int64)

        slates = slate_per_request(scores, items, [3, 3], k=2)

        assert items[slates[1].rows].tolist() == slates[1].items.tolist()


class TestFreshness:
    def test_a_new_item_is_unchanged_and_one_half_life_halves_it(self) -> None:
        got = freshness_multiplier(np.array([0.0, 24.0, 48.0]), half_life_hours=24.0)

        assert got.tolist() == pytest.approx([1.0, 0.5, 0.25])

    def test_a_non_positive_half_life_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            freshness_multiplier(np.array([1.0]), half_life_hours=0.0)


class TestNormalise:
    def test_the_reserved_zero_row_survives(self) -> None:
        """Row 0 of the content table is the OOV vector and is exactly zero.
        Dividing it by its norm gives NaNs that propagate into every similarity
        in any slate containing it."""
        vectors = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)

        got = normalise(vectors)

        assert got[0].tolist() == [0.0, 0.0]
        assert got[1].tolist() == pytest.approx([0.6, 0.8])
        assert not np.isnan(got).any()


def rows_for(labels: list[int], items: list[int]) -> RankingRows:
    """A one-request :class:`RankingRows` carrying only what scoring reads."""
    n = len(labels)
    return RankingRows(
        names=("retrieval_score", "train_clicks"),
        features=np.zeros((n, 2), dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray([n], dtype=np.int64),
        items=np.asarray(items, dtype=np.int64),
        request=np.zeros(n, dtype=np.int64),
        user_ids=np.asarray([1], dtype=np.int64),
        observed=np.zeros(n, dtype=bool),
        found=np.asarray([bool(sum(labels))], dtype=bool),
    )


class TestScoreSlates:
    def test_the_slate_order_decides_the_ndcg_not_the_model_score(self) -> None:
        """**The load-bearing test for the whole table.** If NDCG is computed
        from the ranker's scores rather than from the slate's order, every
        policy row scores exactly what the ranker scored and the table reports
        that diversity is free."""
        rows = rows_for([0, 1], [10, 11])
        vectors = np.zeros((12, 2), dtype=np.float32)
        vectors[10] = [1.0, 0.0]
        vectors[11] = [0.0, 1.0]
        slates = slate_per_request(np.array([1.0, 0.5]), rows.items, [2], k=2)

        arm = score_slates(
            slates, rows, vectors, np.zeros(12, dtype=np.int64), pool_size=2, k=2, name="x"
        )

        # The positive sits in slot 2, so NDCG is 1/log2(3), not 1.0.
        assert arm.ndcg == pytest.approx(1.0 / np.log2(3.0))

    def test_a_request_retrieval_missed_scores_zero_rather_than_nan(self) -> None:
        """``ndcg_at_k`` returns NaN when nothing was clicked, which is right for
        it and wrong here: the funnel counts a retrieval miss as a zero, and a
        NaN would silently drop the request from the mean instead."""
        rows = rows_for([0, 0], [10, 11])
        vectors = np.zeros((12, 2), dtype=np.float32)
        slates = slate_per_request(np.array([1.0, 0.5]), rows.items, [2], k=2)

        arm = score_slates(
            slates, rows, vectors, np.zeros(12, dtype=np.int64), pool_size=2, k=2, name="x"
        )

        assert arm.ndcg == 0.0

    def test_the_tail_share_counts_impressions_not_distinct_items(self) -> None:
        """The two denominators that were printed under one heading in the first
        headroom run. Here the share is over SERVED SLOTS."""
        rows = rows_for([0, 1], [10, 11])
        vectors = np.zeros((12, 2), dtype=np.float32)
        train_clicks = np.zeros(12, dtype=np.int64)
        train_clicks[10] = LONG_TAIL_EDGE + 5  # head
        train_clicks[11] = 1  # tail
        slates = slate_per_request(np.array([1.0, 0.5]), rows.items, [2], k=2)

        arm = score_slates(slates, rows, vectors, train_clicks, pool_size=2, k=2, name="x")

        assert arm.tail_share == pytest.approx(0.5)
        assert arm.distinct_items == 2


class TestTopKItems:
    def test_it_takes_the_top_k_of_each_request_separately(self) -> None:
        items = np.array([10, 11, 12, 20, 21, 22], dtype=np.int64)
        rows = RankingRows(
            names=("retrieval_score",),
            features=np.zeros((6, 1), dtype=np.float32),
            labels=np.zeros(6, dtype=np.int64),
            groups=np.asarray([3, 3], dtype=np.int64),
            items=items,
            request=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
            user_ids=np.asarray([1, 2], dtype=np.int64),
            observed=np.zeros(6, dtype=bool),
            found=np.asarray([False, False], dtype=bool),
        )
        scores = np.array([1.0, 3.0, 2.0, 2.0, 1.0, 3.0])

        got = top_k_items(scores, rows, k=2)

        assert got.tolist() == [11, 12, 22, 20]


class TestTheBloomFilter:
    def test_it_never_denies_an_item_it_holds(self) -> None:
        """**The one-sided guarantee, and the entire argument for the structure.**
        A false negative re-shows an item the user just saw, which is the failure
        users notice. It must be impossible, not merely rare."""
        bits, hashes = sizing(capacity=200, false_positive_rate=0.01)
        filter_ = BloomFilter(bits=bits, hashes=hashes)
        held = list(range(200))
        for item in held:
            filter_.add(item)

        assert all(item in filter_ for item in held)

    def test_false_positives_actually_occur(self) -> None:
        """The control. A filter whose every answer is True also has no false
        negatives, so the test above alone proves nothing about correctness."""
        filter_ = BloomFilter(bits=64, hashes=3)
        for item in range(40):
            filter_.add(item)

        intruders = sum(1 for item in range(1000, 2000) if item in filter_)

        assert intruders > 0
        assert filter_.load < 1.0

    def test_a_fresh_filter_holds_nothing(self) -> None:
        filter_ = BloomFilter(*sizing(capacity=100, false_positive_rate=0.01))

        assert 42 not in filter_
        assert filter_.load == 0.0

    def test_two_filters_agree_on_positions(self) -> None:
        """A named digest, never ``hash()``: the serving side is a different
        process, and a per-process salt would make the same user's filter answer
        differently there."""
        first = BloomFilter(bits=1024, hashes=4)
        second = BloomFilter(bits=1024, hashes=4)
        first.add(99)
        second.add(99)

        assert 99 in second
        assert first.load == second.load

    def test_a_tighter_target_costs_more_bits(self) -> None:
        loose, _ = sizing(capacity=100, false_positive_rate=0.1)
        tight, _ = sizing(capacity=100, false_positive_rate=0.001)

        assert tight > loose

    @pytest.mark.parametrize("rate", [0.0, 1.0, -0.5])
    def test_an_impossible_rate_is_refused(self, rate: float) -> None:
        with pytest.raises(ValueError, match="false_positive_rate"):
            sizing(capacity=100, false_positive_rate=rate)

    def test_a_non_positive_capacity_is_refused(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            sizing(capacity=0, false_positive_rate=0.01)

    def test_the_measured_rate_is_near_the_target_at_capacity(self) -> None:
        """The formula assumes ideal independent hashes. This says the double
        hashing actually used lands near it, rather than trusting the algebra."""
        got = measure_false_positives(
            seen=list(range(500)),
            probes=list(range(10_000, 20_000)),
            capacity=500,
            target_rate=0.01,
        )

        assert got["measured_rate"] < 0.05
        assert got["items_held"] == 500.0

    def test_mild_overfilling_degrades_the_rate(self) -> None:
        honest = measure_false_positives(
            seen=list(range(100)),
            probes=list(range(10_000, 20_000)),
            capacity=100,
            target_rate=0.01,
        )
        overfilled = measure_false_positives(
            seen=list(range(200)),
            probes=list(range(10_000, 20_000)),
            capacity=100,
            target_rate=0.01,
        )

        assert overfilled["measured_rate"] > honest["measured_rate"]
        assert overfilled["load"] > honest["load"]

    def test_severe_overfilling_saturates_and_hides_everything(self) -> None:
        """**Not a graceful degradation, and the benchmark is what said so.**

        At ten times its capacity every bit is set and the filter answers "seen"
        to every candidate. A seen-filter that hides everything does not return
        a slightly worse slate -- it returns an empty one, and the one-sided
        guarantee still holds while being worth nothing.

        Pinned because this module's own prose called it a quiet degradation
        until ``seen_bench`` measured 100%. Load is the alarm to monitor: it is
        observable per user and moves long before the rate does.
        """
        got = measure_false_positives(
            seen=list(range(1000)),
            probes=list(range(10_000, 20_000)),
            capacity=100,
            target_rate=0.01,
        )

        assert got["measured_rate"] == pytest.approx(1.0)
        assert got["load"] == pytest.approx(1.0)
