"""The two protocols, and the guard that keeps them apart.

The mistake these prevent is not a crash. It is a plausible number computed over
the wrong pool, reported in a table, and compared against published results it
has no relationship to.
"""

from __future__ import annotations

import math

import pytest

from evaluation.offline.protocols import evaluate_ranking, evaluate_retrieval

POPULARITY = {f"N{i}": 1 / 200 for i in range(200)}


class TestRanking:
    def test_perfect_ordering_in_every_slate(self) -> None:
        result = evaluate_ranking(
            scores=[0.9, 0.1, 0.8, 0.2],
            labels=[1, 0, 1, 0],
            impression_ids=["a", "a", "b", "b"],
            k=10,
        )

        assert result.gauc.mean == 1.0
        assert result.mrr.mean == 1.0
        assert result.ndcg.mean == 1.0
        assert result.recall.mean == 1.0
        assert result.gauc.scored == 2

    def test_clickless_slates_are_skipped_not_zeroed(self) -> None:
        """The denominator moves with every filter, so it has to be reported."""
        result = evaluate_ranking(
            scores=[0.9, 0.1, 0.8, 0.2],
            labels=[1, 0, 0, 0],
            impression_ids=["a", "a", "b", "b"],
            k=10,
        )

        assert result.gauc == (1.0, 1, 1)
        assert result.ndcg.scored == 1
        assert result.ndcg.skipped == 1

    def test_slates_are_grouped_not_pooled(self) -> None:
        """Two slates, each ordered correctly within itself, but whose scores
        interleave badly across slates. Pooling them would score far below 1.0;
        grouping gives a perfect result, which is the point of GAUC."""
        result = evaluate_ranking(
            scores=[0.2, 0.1, 0.9, 0.8],
            labels=[1, 0, 1, 0],
            impression_ids=["a", "a", "b", "b"],
            k=10,
        )

        assert result.gauc.mean == 1.0


class TestRetrievalGuard:
    def test_a_sampled_pool_is_refused(self) -> None:
        """The Krichene & Rendle mistake, made loud.

        One positive against 100 sampled negatives returns a perfectly
        reasonable-looking recall that is comparable to nothing.
        """
        with pytest.raises(ValueError, match="not a sampled pool"):
            evaluate_retrieval(
                retrieved={"r1": [f"N{i}" for i in range(50)]},
                relevant={"r1": ["N3"]},
                catalogue_size=200,
                train_popularity=POPULARITY,
                k=100,
            )

    def test_a_catalogue_smaller_than_k_is_refused(self) -> None:
        with pytest.raises(ValueError, match="smaller than k"):
            evaluate_retrieval(
                retrieved={"r1": [f"N{i}" for i in range(10)]},
                relevant={"r1": ["N3"]},
                catalogue_size=10,
                train_popularity=POPULARITY,
                k=100,
            )

    def test_the_error_names_an_offending_request(self) -> None:
        """A count alone is not actionable when one caller in ten is wrong."""
        with pytest.raises(ValueError, match="r2"):
            evaluate_retrieval(
                retrieved={
                    "r1": [f"N{i}" for i in range(100)],
                    "r2": [f"N{i}" for i in range(4)],
                },
                relevant={"r1": ["N1"], "r2": ["N2"]},
                catalogue_size=200,
                train_popularity=POPULARITY,
                k=100,
            )


class TestRetrieval:
    def test_recall_counts_clicked_items_found(self) -> None:
        result = evaluate_retrieval(
            retrieved={"r1": [f"N{i}" for i in range(100)]},
            relevant={"r1": ["N3", "N150"]},  # one in the top 100, one not
            catalogue_size=200,
            train_popularity=POPULARITY,
            k=100,
        )

        assert result.recall.mean == pytest.approx(0.5)
        assert result.catalogue_size == 200

    def test_requests_with_no_click_are_skipped(self) -> None:
        result = evaluate_retrieval(
            retrieved={
                "r1": [f"N{i}" for i in range(100)],
                "r2": [f"N{i}" for i in range(100)],
            },
            relevant={"r1": ["N3"]},
            catalogue_size=200,
            train_popularity=POPULARITY,
            k=100,
        )

        assert result.recall.scored == 1
        assert result.recall.skipped == 1

    def test_coverage_is_against_the_catalogue(self) -> None:
        result = evaluate_retrieval(
            retrieved={"r1": [f"N{i}" for i in range(100)]},
            relevant={"r1": ["N3"]},
            catalogue_size=200,
            train_popularity=POPULARITY,
            k=100,
        )

        assert result.coverage == pytest.approx(0.5)

    def test_novelty_is_finite_on_a_uniform_prior(self) -> None:
        result = evaluate_retrieval(
            retrieved={"r1": [f"N{i}" for i in range(100)]},
            relevant={"r1": ["N3"]},
            catalogue_size=200,
            train_popularity=POPULARITY,
            k=100,
        )

        # -log2(1/200) for every served item.
        assert result.novelty == pytest.approx(math.log2(200))


def test_mrr_is_reported_and_skips_clickless_slates() -> None:
    """MIND's leaderboard reports MRR, so the ranking protocol has to."""
    result = evaluate_ranking(
        scores=[0.1, 0.9, 0.8, 0.2],
        labels=[1, 0, 0, 0],
        impression_ids=["a", "a", "b", "b"],
        k=10,
    )

    # Slate a: the click is second of two. Slate b: no click, undefined.
    assert result.mrr.mean == pytest.approx(0.5)
    assert result.mrr.scored == 1
    assert result.mrr.skipped == 1
