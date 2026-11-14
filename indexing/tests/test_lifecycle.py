"""The promotion gate, and the two ways it is usually built wrong.

**Gating on the wrong metric.** Agreement with exact search and click-recall
come apart: measured here, an index 1.3% off exact lost 0.05% of clicks, and a
coarser one 9% off exact found MORE. A gate on agreement blocks good indexes and
passes bad ones whose errors land on items nobody clicks.

**Failing open.** A gate that promotes when it cannot evaluate is not a gate.
The live index already works; the thing to design against is replacing it with a
regression, never missing a refresh.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from indexing.lifecycle import (
    RECALL_TOLERANCE,
    current,
    expired,
    gate,
    promote,
    version_label,
)


class TestTheGate:
    def test_an_improvement_promotes(self) -> None:
        assert gate(0.3800, 0.3776).allowed

    def test_a_drop_inside_tolerance_promotes(self) -> None:
        assert gate(0.3776 - RECALL_TOLERANCE / 2, 0.3776).allowed

    def test_a_drop_past_tolerance_aborts(self) -> None:
        decision = gate(0.3776 - RECALL_TOLERANCE * 2, 0.3776)

        assert not decision.allowed
        assert "tolerance" in decision.reason

    def test_an_undefined_recall_aborts(self) -> None:
        """Fail closed: nothing measured means nothing promoted."""
        assert not gate(float("nan"), 0.3776).allowed

    def test_the_first_index_promotes_with_nothing_to_compare(self) -> None:
        decision = gate(0.3776, None)

        assert decision.allowed
        assert "first index" in decision.reason

    def test_the_reason_is_given_on_success_too(self) -> None:
        """A gate that only explains itself when it fails leaves no evidence."""
        assert gate(0.3800, 0.3776).reason

    def test_the_tolerance_is_wider_than_training_noise(self) -> None:
        """Otherwise the gate rejects rebuilds for having a different seed.

        Run-to-run spread on the embeddings that feed this index is ~0.0073 at
        one sigma; a relative 1% of a 0.38 recall would be 0.0038, well inside
        it, so the gate would fire on noise and every rebuild would need a human.
        """
        assert RECALL_TOLERANCE >= 0.005


class TestVersions:
    def test_the_label_is_hourly_and_sorts_chronologically(self) -> None:
        early = version_label(datetime(2019, 11, 9, 2))
        late = version_label(datetime(2019, 11, 15, 2))

        assert early == "v=2019-11-09T02:00Z"
        assert early < late, "lexicographic order must be chronological order"

    def test_two_builds_in_one_hour_share_a_label(self) -> None:
        """Known and accepted: the second overwrites the first."""
        assert version_label(datetime(2019, 11, 9, 2, 5)) == version_label(
            datetime(2019, 11, 9, 2, 55)
        )

    def test_retention_keeps_the_newest(self) -> None:
        versions = [f"v=2019-11-0{day}T02:00Z" for day in range(1, 7)]

        assert expired(versions, keep=3) == versions[:3]

    def test_retention_never_deletes_what_it_should_keep(self) -> None:
        assert expired(["v=a"], keep=3) == []


class TestPointer:
    def test_a_promotion_is_readable_afterwards(self, tmp_path: Path) -> None:
        pointer = tmp_path / "CURRENT"

        promote(pointer, "v=2019-11-15T02:00Z")

        assert current(pointer) == "v=2019-11-15T02:00Z"

    def test_nothing_promoted_reads_as_none(self, tmp_path: Path) -> None:
        assert current(tmp_path / "CURRENT") is None

    def test_a_second_promotion_replaces_the_first(self, tmp_path: Path) -> None:
        pointer = tmp_path / "CURRENT"
        promote(pointer, "v=1")
        promote(pointer, "v=2")

        assert current(pointer) == "v=2"

    def test_no_temporary_file_survives_the_swap(self, tmp_path: Path) -> None:
        """The staged write is renamed, not copied and left behind."""
        pointer = tmp_path / "CURRENT"
        promote(pointer, "v=1")

        assert [path.name for path in tmp_path.iterdir()] == ["CURRENT"]
