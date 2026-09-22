"""The guard that stops ``make results`` from eating a hand-written document.

``render`` returns the whole file and ``main`` truncates, so for a long time the
only thing protecting the 287 hand-written lines under ``docs/results.md`` was an
HTML comment asking a human not to run the command. This project's own rule is
that a note is not a mechanism. These tests are the mechanism.

The load-bearing one is ``test_render_survives_its_own_guard``: the tool must
accept what the tool writes. Everything else here is symmetry around that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evaluation.offline.results_table import BANNER, BANNER_WINDOW, guard, main, render

GENERATED = f"# Results\n\n{BANNER}\n\nSplit `dev`, ranking within the impression, k=10.\n"
HAND_WRITTEN = "# Results\n\nSplit `dev`, ranking within the impression, k=10.\n"


def _cards() -> list[dict[str, Any]]:
    """One card, shaped like what ``run_eval`` writes."""
    overall = {
        "gauc": 0.5243,
        "ndcg@10": 0.3097,
        "ci95": [0.3129, 0.3179],
        "mrr": 0.2670,
        "recall@10": 0.5490,
        "impressions": 73152,
        "gauc_ceiling": 0.9606,
        "coverage@10": 0.0309,
    }
    return [
        {
            "model": "popularity",
            "split": "dev",
            "k": 10,
            "git_sha": "abc1234",
            "half_life_days": None,
            "cohorts": {
                "overall": overall,
                "cold_item": {"gauc": 0.4133},
                "cold_user": {"gauc": 0.5251},
            },
        }
    ]


class TestTheGuard:
    def test_the_two_fixtures_differ_only_by_the_banner(self) -> None:
        """The precondition, asserted before anything is concluded from it.

        Both branches below are reached by files that are otherwise identical, so
        a refusal can only be attributable to the banner. Without this, a fixture
        that differed in some second way would let the pair of tests pass while
        the guard keyed on something else entirely.
        """
        assert GENERATED.replace(f"{BANNER}\n\n", "") == HAND_WRITTEN
        assert BANNER not in HAND_WRITTEN

    def test_a_missing_destination_is_allowed(self, tmp_path: Path) -> None:
        """First run on a fresh checkout has nothing to protect."""
        guard(tmp_path / "baselines.md")

    def test_a_generated_file_is_overwritten(self, tmp_path: Path) -> None:
        destination = tmp_path / "baselines.md"
        destination.write_text(GENERATED)

        guard(destination)

    def test_a_hand_written_file_is_refused(self, tmp_path: Path) -> None:
        """The regression. This is the shape ``docs/results.md`` takes after the
        generated table moves out of it."""
        destination = tmp_path / "results.md"
        destination.write_text(HAND_WRITTEN)

        with pytest.raises(SystemExit, match="generated banner"):
            guard(destination)

    def test_a_banner_quoted_deep_in_prose_does_not_authorise_deletion(
        self, tmp_path: Path
    ) -> None:
        """A document explaining this guard would otherwise become a legal
        destination by quoting the string it is documenting."""
        destination = tmp_path / "retrieval.md"
        destination.write_text(f"# Retrieval\n\n{'x' * BANNER_WINDOW}\n\n{BANNER}\n")

        with pytest.raises(SystemExit):
            guard(destination)


class TestTheWiring:
    def test_main_refuses_before_it_reads_the_cards(self, tmp_path: Path) -> None:
        """Ordering, not just presence.

        ``--results`` points at nothing, so if the guard ran second this would
        fail with the cards error instead. The cheap check that prevents data
        loss goes before the expensive one that only prevents an empty table.
        """
        destination = tmp_path / "results.md"
        destination.write_text(HAND_WRITTEN)

        with pytest.raises(SystemExit, match="generated banner"):
            main(["--results", str(tmp_path / "nowhere"), "--out", str(destination)])

    def test_the_refused_file_is_left_byte_for_byte(self, tmp_path: Path) -> None:
        """The property the guard actually exists for. A tool that raises after
        truncating has still destroyed the file."""
        destination = tmp_path / "results.md"
        destination.write_text(HAND_WRITTEN)

        with pytest.raises(SystemExit):
            main(["--results", str(tmp_path / "nowhere"), "--out", str(destination)])

        assert destination.read_text() == HAND_WRITTEN


def test_render_survives_its_own_guard(tmp_path: Path) -> None:
    """Round trip: the tool accepts what the tool writes.

    This is what keeps the banner honest. ``render`` and ``guard`` share the one
    constant, so they cannot disagree on the text -- but they can still disagree
    on its POSITION, and ``BANNER_WINDOW`` is what makes position matter. Insert
    a long preamble above the banner and this test is the one that notices.
    """
    destination = tmp_path / "baselines.md"
    destination.write_text(render(_cards()))

    guard(destination)
