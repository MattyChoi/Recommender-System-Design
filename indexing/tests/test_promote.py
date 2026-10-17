"""The gate's effect on the pointer, without Airflow, Spark or a GPU.

These used to be DAG-shape assertions -- "the swap depends on the gate" -- which
tested the wiring that expressed the property rather than the property. The
composition moved into :func:`indexing.pipeline.rebuild_and_promote`, so the
branch is now directly assertable: a refused candidate must leave serving
pointing where it was.

``probe`` is passed in, which is the seam that makes this cheap. Everything else
is real: real FAISS indexes, really written to disk, really reloaded and
searched. The only thing faked is the encoder, because encoding is the part that
needs a catalogue and a checkpoint.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from indexing.lifecycle import Promotion, current, version_label
from indexing.pipeline import INDEX_FILENAME, Outcome, Probe, rebuild_and_promote

HOUR_ONE = datetime(2019, 11, 15, 2, tzinfo=UTC)
HOUR_TWO = datetime(2019, 11, 15, 3, tzinfo=UTC)


@pytest.fixture
def probe() -> Probe:
    """Four orthogonal unit vectors, and two requests with known answers.

    Orthogonal so the nearest neighbour is never ambiguous: position ``p`` holds
    item ``p + 1``, so a query equal to row 2 must come back as item 3. Unit norm
    because ``build_flat`` asserts it -- inner product is cosine only when the
    vectors are normalised, and an index built from unnormalised rows ranks by
    magnitude while looking perfectly healthy.
    """
    vectors = np.eye(4, dtype="float32")
    return Probe(
        vectors=vectors,
        queries=vectors[[2, 0]],
        clicked=np.asarray([3, 1], dtype="int64"),
    )


def _rebuild(root: Path, probe: Probe, moment: datetime) -> Outcome:
    return rebuild_and_promote(
        checkpoint=Path("unused-because-probe-is-supplied.pt"),
        artifacts=root,
        pointer=root / "CURRENT",
        kind="flat",
        moment=moment,
        probe=probe,
    )


class TestTheFirstIndex:
    def test_it_promotes_with_nothing_live(self, tmp_path: Path, probe: Probe) -> None:
        """The one case where an unmeasured comparison may promote."""
        outcome = _rebuild(tmp_path, probe, HOUR_ONE)

        assert outcome.promoted is True
        assert outcome.live_recall is None
        assert current(tmp_path / "CURRENT") == version_label(HOUR_ONE)

    def test_the_index_is_written_under_its_own_version(self, tmp_path: Path, probe: Probe) -> None:
        """Never in place: a rollback target has to still exist."""
        _rebuild(tmp_path, probe, HOUR_ONE)

        assert (tmp_path / version_label(HOUR_ONE) / INDEX_FILENAME).is_file()

    def test_recall_is_measured_rather_than_assumed(self, tmp_path: Path, probe: Probe) -> None:
        """Orthogonal vectors and exact search: both answers are findable."""
        outcome = _rebuild(tmp_path, probe, HOUR_ONE)

        assert outcome.candidate_recall == pytest.approx(1.0)


class TestARefusedGate:
    @pytest.fixture(autouse=True)
    def _live(self, tmp_path: Path, probe: Probe) -> None:
        """Something serving, so the refusal has a pointer to leave alone."""
        _rebuild(tmp_path, probe, HOUR_ONE)

    @pytest.fixture
    def refuse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Refuse whatever the next candidate scores.

        The gate itself is tested in ``test_lifecycle.py`` against recall pairs.
        What is tested here is the BRANCH: given a refusal, what happens to the
        pointer. Driving that through real recalls would need two indexes that
        genuinely disagree on four vectors, which measures FAISS rather than
        this function.
        """
        monkeypatch.setattr(
            "indexing.pipeline.gate",
            lambda candidate, live, tolerance: Promotion(False, "refused by the test"),
        )

    def test_the_pointer_is_left_alone(self, tmp_path: Path, probe: Probe, refuse: None) -> None:
        """Fail-closed: a bad index is worse than a stale one."""
        outcome = _rebuild(tmp_path, probe, HOUR_TWO)

        assert outcome.promoted is False
        assert current(tmp_path / "CURRENT") == version_label(HOUR_ONE)

    def test_the_candidate_stays_on_disk(self, tmp_path: Path, probe: Probe, refuse: None) -> None:
        """Kept for inspection. It is why retention must not run on a refusal."""
        _rebuild(tmp_path, probe, HOUR_TWO)

        assert (tmp_path / version_label(HOUR_TWO) / INDEX_FILENAME).is_file()

    def test_retention_does_not_run(self, tmp_path: Path, probe: Probe, refuse: None) -> None:
        """A refused build is the NEWEST label, so retiring here could count it
        among the kept versions and drop a good rollback target instead.
        """
        outcome = _rebuild(tmp_path, probe, HOUR_TWO)

        assert outcome.retired == []


class TestASecondPromotion:
    def test_an_equal_index_still_promotes(self, tmp_path: Path, probe: Probe) -> None:
        """Drop of zero is inside the tolerance; freshness is the whole point."""
        _rebuild(tmp_path, probe, HOUR_ONE)
        outcome = _rebuild(tmp_path, probe, HOUR_TWO)

        assert outcome.promoted is True
        assert outcome.live_recall == pytest.approx(1.0)
        assert current(tmp_path / "CURRENT") == version_label(HOUR_TWO)

    def test_nothing_is_retired_below_the_depth(self, tmp_path: Path, probe: Probe) -> None:
        """Two versions, keep three: the list is empty and nothing is deleted."""
        _rebuild(tmp_path, probe, HOUR_ONE)
        outcome = _rebuild(tmp_path, probe, HOUR_TWO)

        assert outcome.retired == []
