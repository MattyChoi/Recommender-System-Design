"""The report card's shape, its slicing, and the harness sanity check.

The assertion that matters most is the last one: random scoring must produce
GAUC near 0.5. If it does not, every number this harness ever reports is
suspect, and it is far cheaper to learn that here than from a model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from evaluation.offline.report import default_cohorts, report_card


@pytest.fixture
def corpus() -> dict[str, np.ndarray]:
    """Twelve rows, four slates, three users, mixed cold-start."""
    return {
        "scores": np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5, 0.2, 0.1]),
        "labels": np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]),
        "impressions": np.array(["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2 + ["e"] * 4),
        "users": np.array(["U1"] * 4 + ["U2"] * 4 + ["U3"] * 4),
        "cold_user": np.array([False] * 8 + [True] * 4),
        "cold_item": np.array([False, True] * 6),
    }


def _card(corpus: dict[str, np.ndarray]) -> dict[str, Any]:
    return report_card(
        "test",
        corpus["scores"],
        corpus["labels"],
        corpus["impressions"],
        corpus["users"],
        default_cohorts(
            corpus["cold_user"],
            corpus["cold_item"],
            corpus["impressions"],
            corpus["labels"],
        ),
    )


class TestShape:
    def test_it_carries_provenance(self, corpus: dict[str, np.ndarray]) -> None:
        """Two files of numbers with nothing to say what changed between them
        are worse than one file."""
        card = _card(corpus)

        assert card["model"] == "test"
        assert card["split"] == "dev"
        assert card["generated"].endswith("+00:00")
        assert "git_sha" in card

    def test_all_five_cohorts_are_present(self, corpus: dict[str, np.ndarray]) -> None:
        assert set(_card(corpus)["cohorts"]) == {
            "overall",
            "warm_user",
            "cold_user",
            "warm_item",
            "cold_item",
        }

    def test_every_cohort_reports_its_denominators(self, corpus: dict[str, np.ndarray]) -> None:
        """A metric without its sample size is not comparable to the next run."""
        overall = _card(corpus)["cohorts"]["overall"]

        assert overall["rows"] == 12
        assert overall["users"] == 3
        assert (
            overall["scored_impressions"] + overall["skipped_impressions"]
            == (overall["impressions"])
        )

    def test_the_card_is_json_serialisable(self, corpus: dict[str, np.ndarray]) -> None:
        import json

        assert json.loads(json.dumps(_card(corpus)))["model"] == "test"


class TestSlicing:
    def test_user_cohorts_partition_the_rows(self, corpus: dict[str, np.ndarray]) -> None:
        cohorts = _card(corpus)["cohorts"]

        assert cohorts["warm_user"]["rows"] + cohorts["cold_user"]["rows"] == 12

    def test_item_cohorts_keep_slates_whole(self) -> None:
        """The bug this replaced: masking rows by is_cold_item cut slates in half.

        One slate of four items, one of them cold and NOT the clicked one. The
        slate is a warm-item case and must appear intact -- four rows, not the
        three that survive a row mask. Scoring a truncated slate is an easier
        problem, and on dev it inflated random NDCG@10 from 0.2855 to 0.4923.
        """
        card = report_card(
            "test",
            [0.9, 0.5, 0.3, 0.1],
            [1, 0, 0, 0],
            ["a"] * 4,
            ["U1"] * 4,
            default_cohorts([False] * 4, [False, False, True, False], ["a"] * 4, [1, 0, 0, 0]),
        )

        assert card["cohorts"]["warm_item"]["rows"] == 4
        assert card["cohorts"]["cold_item"]["rows"] == 0

    def test_a_cold_clicked_item_makes_the_whole_slate_cold(self) -> None:
        card = report_card(
            "test",
            [0.9, 0.5, 0.3, 0.1],
            [1, 0, 0, 0],
            ["a"] * 4,
            ["U1"] * 4,
            default_cohorts([False] * 4, [True, False, False, False], ["a"] * 4, [1, 0, 0, 0]),
        )

        assert card["cohorts"]["cold_item"]["rows"] == 4
        assert card["cohorts"]["warm_item"]["rows"] == 0

    def test_a_clickless_slate_joins_neither_item_cohort(self) -> None:
        """There is no target item to classify, so it belongs to neither."""
        cohorts = default_cohorts([False] * 2, [True, False], ["a", "a"], [0, 0])

        assert not cohorts["warm_item"].any()
        assert not cohorts["cold_item"].any()
        assert cohorts["overall"].all()

    def test_an_empty_cohort_is_reported_not_crashed(self) -> None:
        """Cold-item cohorts can be empty on a small ablation split."""
        card = report_card(
            "test",
            [0.9, 0.1],
            [1, 0],
            ["a", "a"],
            ["U1", "U1"],
            {"overall": [True, True], "cold_item": [False, False]},
        )

        assert card["cohorts"]["cold_item"] == {"rows": 0}

    def test_a_clickless_cohort_has_no_interval_rather_than_nan(self) -> None:
        card = report_card(
            "test", [0.9, 0.1], [0, 0], ["a", "a"], ["U1", "U1"], {"overall": [True, True]}
        )

        assert card["cohorts"]["overall"]["ci95"] is None
        assert card["cohorts"]["overall"]["ci_users"] == 0


def test_random_scoring_gives_gauc_near_a_half() -> None:
    """The harness's own sanity check.

    Not a test of a model -- a test of the ruler. Random scores carry no
    information, so any systematic departure from 0.5 means the harness is
    mis-ordering, mis-grouping, or mis-labelling something.
    """
    rng = np.random.default_rng(0)
    n_slates, per_slate = 400, 5

    impressions = np.repeat([f"i{j}" for j in range(n_slates)], per_slate)
    users = np.repeat([f"U{j % 80}" for j in range(n_slates)], per_slate)
    labels = np.zeros(n_slates * per_slate, dtype=int)
    labels[rng.integers(0, per_slate, n_slates) + np.arange(n_slates) * per_slate] = 1
    scores = rng.random(n_slates * per_slate)

    card = report_card(
        "random",
        scores,
        labels,
        impressions,
        users,
        {"overall": np.ones(n_slates * per_slate, dtype=bool)},
    )

    assert card["cohorts"]["overall"]["gauc"] == pytest.approx(0.5, abs=0.05)
