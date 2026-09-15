"""Metric behaviour at the ends and at the awkward middle.

Every assertion here is a value computed by hand or reasoned from the
definition, not a golden number captured from a previous run. A golden number
records what the code did; these record what it should do.
"""

from __future__ import annotations

import math
import time

import pytest

from evaluation.offline.metrics import (
    Aggregate,
    aggregate,
    catalog_coverage,
    gauc,
    gini,
    group_slates,
    impression_auc,
    intra_list_diversity,
    ndcg_at_k,
    novelty,
    recall_at_k,
    reciprocal_rank,
)


class TestImpressionAuc:
    def test_perfect_ranking(self) -> None:
        assert impression_auc([1, 0, 0], [0.9, 0.2, 0.1]) == 1.0

    def test_inverted_ranking(self) -> None:
        assert impression_auc([1, 0, 0], [0.1, 0.8, 0.9]) == 0.0

    def test_all_ties_is_a_coin_flip(self) -> None:
        assert impression_auc([1, 0, 0], [0.5, 0.5, 0.5]) == 0.5

    @pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1]])
    def test_degenerate_is_nan_not_zero(self, labels: list[int]) -> None:
        """The distinction the whole harness rests on.

        An impression with no click is not a failed ranking -- there is no
        correct order to have missed. Scoring it 0.0 makes the mean a function
        of how many such impressions the current filter leaves in.
        """
        assert math.isnan(impression_auc(labels, [0.9, 0.5, 0.1]))

    def test_one_tie_at_the_boundary(self) -> None:
        """Positive tied with one negative, above the other: 0.5 + 1.0 over 2."""
        assert impression_auc([1, 0, 0], [0.5, 0.5, 0.1]) == 0.75


class TestGrouping:
    def test_grouping_is_one_pass_not_one_per_slate(self) -> None:
        """A complexity guard, because correctness tests cannot see this.

        The natural implementation rescans every row for every slate, which is
        O(rows x slates). It is invisible on a dozen rows and does not finish on
        MIND's dev split. 4,000 rows in 2,000 slates is quadratic enough to hang
        noticeably if the single pass is ever lost, and trivial if it is not.
        """
        n = 2_000
        preds = list(range(2 * n))
        labels = [1, 0] * n
        groups = [f"s{i // 2}" for i in range(2 * n)]

        started = time.perf_counter()
        result = gauc(preds, labels, groups)
        elapsed = time.perf_counter() - started

        assert result.scored == n
        assert elapsed < 2.0, f"gauc took {elapsed:.1f}s on {2 * n} rows"

    def test_slates_keep_first_seen_order(self) -> None:
        """Ordering is stable so a rerun cannot reshuffle which slate is which."""
        slates = group_slates([0.1, 0.2, 0.3], [0, 1, 0], ["b", "a", "b"])

        assert list(slates) == ["b", "a"]
        assert slates["b"] == ([0, 0], [0.1, 0.3])


class TestGauc:
    def test_equals_the_size_weighted_mean_of_its_parts(self) -> None:
        """gauc is defined in terms of impression_auc; this pins that it stays so."""
        preds = [0.9, 0.1, 0.8, 0.7, 0.6, 0.5]
        labels = [1, 0, 0, 1, 0, 0]
        groups = ["a", "a", "b", "b", "b", "b"]

        # a: perfect, size 2. b: the positive sits below one negative, size 4.
        expected = (1.0 * 2 + impression_auc([0, 1, 0, 0], [0.8, 0.7, 0.6, 0.5]) * 4) / 6

        assert gauc(preds, labels, groups).mean == pytest.approx(expected)

    def test_degenerate_groups_are_counted_not_scored(self) -> None:
        result = gauc([0.9, 0.1, 0.5], [1, 0, 0], ["a", "a", "b"])

        assert result == Aggregate(mean=1.0, scored=1, skipped=1)

    def test_all_degenerate_is_nan(self) -> None:
        assert math.isnan(gauc([0.9, 0.1], [0, 0], ["a", "a"]).mean)


def test_aggregate_carries_the_denominator() -> None:
    result = aggregate([1.0, float("nan"), 0.0, float("nan")])

    assert result == Aggregate(mean=0.5, scored=2, skipped=2)


class TestRecallAndNdcg:
    def test_recall_finds_both_clicks_in_the_top_two(self) -> None:
        assert recall_at_k([1, 1, 0], [0.9, 0.8, 0.1], k=2) == 1.0

    def test_recall_misses_the_click_below_k(self) -> None:
        assert recall_at_k([1, 0, 0], [0.1, 0.9, 0.8], k=2) == 0.0

    def test_ndcg_perfect_is_one(self) -> None:
        assert ndcg_at_k([1, 0, 0], [0.9, 0.5, 0.1], k=3) == 1.0

    def test_ndcg_second_position_by_hand(self) -> None:
        """One click ranked second: DCG = 1/log2(3), IDCG = 1/log2(2) = 1."""
        assert ndcg_at_k([1, 0], [0.4, 0.9], k=2) == pytest.approx(1 / math.log2(3))

    def test_ndcg_ideal_uses_this_impression_s_own_clicks(self) -> None:
        """Two clicks, both found: 1.0, not a fraction of an unreachable ideal."""
        assert ndcg_at_k([1, 1, 0], [0.9, 0.8, 0.1], k=3) == 1.0

    @pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k])
    def test_no_click_is_nan(self, metric: object) -> None:
        assert math.isnan(metric([0, 0], [0.9, 0.1], k=2))  # type: ignore[operator]


class TestReciprocalRank:
    def test_click_at_the_top_is_one(self) -> None:
        assert reciprocal_rank([1, 0, 0], [0.9, 0.5, 0.1]) == 1.0

    def test_click_in_third_place_is_a_third(self) -> None:
        assert reciprocal_rank([1, 0, 0], [0.1, 0.9, 0.5]) == pytest.approx(1 / 3)

    def test_only_the_first_click_counts(self) -> None:
        """The difference from NDCG, in one assertion.

        Two clicks, ranked first and last. NDCG is dragged down by the second;
        MRR ignores it entirely, because the question it asks is whether the
        user found something immediately.
        """
        labels, scores = [1, 0, 1], [0.9, 0.5, 0.1]

        assert reciprocal_rank(labels, scores) == 1.0
        assert ndcg_at_k(labels, scores, k=3) < 1.0

    def test_no_click_is_nan_not_zero(self) -> None:
        assert math.isnan(reciprocal_rank([0, 0], [0.9, 0.1]))

    def test_ties_resolve_stably_rather_than_by_luck(self) -> None:
        """argsort is stable, so tied scores keep input order.

        Not a claim that this is the RIGHT tie-break -- it is a claim that the
        metric is deterministic, so a rerun cannot move the number.
        """
        assert reciprocal_rank([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


class TestBeyondAccuracy:
    def test_coverage_is_against_the_catalogue_not_what_was_shown(self) -> None:
        assert catalog_coverage(["a", "a", "b"], catalogue_size=10) == 0.2

    def test_coverage_rejects_a_meaningless_denominator(self) -> None:
        with pytest.raises(ValueError):
            catalog_coverage(["a"], catalogue_size=0)

    def test_novelty_rewards_the_rare_item(self) -> None:
        popular = novelty(["a"], {"a": 0.5, "b": 0.125})
        rare = novelty(["b"], {"a": 0.5, "b": 0.125})

        assert popular == pytest.approx(1.0)
        assert rare == pytest.approx(3.0)
        assert rare > popular

    def test_an_unseen_item_is_maximally_novel_not_dropped(self) -> None:
        """Cold items ARE the novel case; dropping them removes the point."""
        assert novelty(["zzz"], {"a": 0.5, "b": 0.125}) == pytest.approx(3.0)

    def test_diversity_of_identical_vectors_is_zero(self) -> None:
        vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0]}

        assert intra_list_diversity(["a", "b"], vectors) == pytest.approx(0.0)

    def test_diversity_of_orthogonal_vectors_is_one(self) -> None:
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}

        assert intra_list_diversity(["a", "b"], vectors) == pytest.approx(1.0)

    def test_diversity_needs_two_known_vectors(self) -> None:
        assert math.isnan(intra_list_diversity(["a", "unknown"], {"a": [1.0, 0.0]}))


class TestGini:
    def test_even_exposure_is_zero(self) -> None:
        assert gini([5, 5, 5, 5]) == pytest.approx(0.0)

    def test_concentration_approaches_one(self) -> None:
        assert gini([0, 0, 0, 0, 0, 0, 0, 0, 0, 100]) == pytest.approx(0.9)

    def test_no_exposure_at_all_is_nan(self) -> None:
        assert math.isnan(gini([0, 0, 0]))
