"""The sidecar's ``content_similarity`` must equal the column the ranker was fitted on.

``serving/retrieval/index.py:content_similarity`` and
``models/ranking/dataset.py:build`` produce the same ranker column -- 12.3% of
the fitted importance, per ``docs/ranking.md`` -- from the same content table:
one per request in numpy, one batched in torch over a padded
``[R, MAX_HISTORY]`` history and its mask. Two implementations of one
definition, so they need a test that fails when they drift.

**The training side is transcribed rather than imported, and that is this
file's weakness.** ``build`` needs a tower, a ``SplitTensors`` and a
Spark-built split to reach the four lines that define the column, none of which
belong in a unit test. So ``_training_column`` copies those four lines, plus the
padded representation ``gold/user_history`` would have produced. A change to
``build`` that this file is not updated alongside will not be caught -- which is
why the asymmetry is also stated in words in that function's docstring, in
``content_similarity``'s, and on field 6 of ``serving/proto/retrieval.proto``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch

from data_pipeline.features.user_history import MAX_HISTORY
from serving.retrieval.index import content_similarity, real_history

CONTENT_DIM = 8
N_ITEMS = 12
N_ROWS = N_ITEMS + 1

# Deliberately not sorted and not a prefix: the column is positional against
# whatever order retrieval returned, and a sorted fixture would let an
# implementation that sorts internally pass.
CANDIDATES = [1, 4, 7, 12, 2]

HISTORIES = [
    pytest.param([3, 5, 8], id="three-articles"),
    pytest.param([1], id="one-article"),
    pytest.param([2, 2, 9], id="a-repeat"),
    pytest.param(list(range(1, N_ROWS)), id="the-whole-catalogue"),
]


@pytest.fixture
def content() -> torch.Tensor:
    """A content table shaped like the real one: row 0 zeroed, rows of UNEQUAL norm.

    Both properties are load-bearing.

    ``load_item_tables`` builds the table with ``torch.zeros`` and fills from
    index 1, so row 0 is exactly zeros here too -- see
    ``test_the_filter_cannot_be_seen_here_and_is_still_required``.

    And the norms vary because the training column dots an UN-normalised
    candidate vector against a normalised pool. Against a table of unit rows a
    both-sides-normalised implementation would agree with it exactly, and
    ``test_normalising_the_candidate_too_is_a_different_column`` would pass
    while asserting nothing.
    """
    rng = np.random.default_rng(20260117)
    table = rng.normal(size=(N_ROWS, CONTENT_DIM)).astype("float32")
    table *= rng.uniform(0.25, 4.0, size=(N_ROWS, 1)).astype("float32")
    table[0] = 0.0
    return torch.from_numpy(np.ascontiguousarray(table, dtype="float32"))


def _training_column(
    content: torch.Tensor, history: list[int], items: list[int]
) -> npt.NDArray[np.float32]:
    """``models/ranking/dataset.py:build`` for one request, transcribed.

    ``history`` is the real entries. This pads them to ``MAX_HISTORY`` with item
    0 and a zero mask, which is how the gold table stores them and therefore
    what ``build`` was reading when it produced every offline number.
    """
    table = content
    ids = torch.zeros((1, MAX_HISTORY), dtype=torch.long)
    mask = torch.zeros((1, MAX_HISTORY), dtype=torch.long)
    ids[0, : len(history)] = torch.tensor(history, dtype=torch.long)
    mask[0, : len(history)] = 1

    # The four lines that DEFINE the column, from build(). Copied verbatim
    # apart from indexing one request instead of gathering per candidate row.
    weight = mask.unsqueeze(-1).to(table.dtype)
    pooled = (table[ids] * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    similarity = (pooled[0] * table[torch.tensor(items, dtype=torch.long)]).sum(dim=-1)

    out: npt.NDArray[np.float32] = similarity.numpy()
    return out


class TestItAgreesWithTraining:
    @pytest.mark.parametrize("history", HISTORIES)
    def test_the_column_matches(self, content: torch.Tensor, history: list[int]) -> None:
        np.testing.assert_allclose(
            content_similarity(content, history, CANDIDATES),
            _training_column(content, history, CANDIDATES),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_an_empty_history_is_zero_on_both_sides(self, content: torch.Tensor) -> None:
        """Zero is the training answer for a cold user, not a serving fallback.

        ``build`` divides by ``clamp(min=1.0)`` and normalises a zero vector,
        which ``F.normalize`` returns unchanged rather than as NaN. Asserted
        against both an exact zero vector and the training side, because a
        near-zero column here would mean the serving floor and torch's
        ``eps`` disagree.
        """
        served = content_similarity(content, [], CANDIDATES)
        np.testing.assert_array_equal(served, np.zeros(len(CANDIDATES), dtype="float32"))
        np.testing.assert_array_equal(served, _training_column(content, [], CANDIDATES))

    def test_the_alignment_is_positional(self, content: torch.Tensor) -> None:
        """Reordering the candidates reorders the column and changes nothing else.

        The response pairs ``items``, ``scores`` and ``content_similarity`` by
        position, so a column that was right as a multiset and wrong in order
        would hand every candidate its neighbour's value.
        """
        history = [3, 5, 8]
        straight = content_similarity(content, history, CANDIDATES)
        backwards = content_similarity(content, history, list(reversed(CANDIDATES)))
        np.testing.assert_allclose(backwards, straight[::-1], rtol=1e-6)


class TestThePropertiesThatCouldDriftSilently:
    def test_normalising_the_candidate_too_is_a_different_column(
        self, content: torch.Tensor
    ) -> None:
        """The asymmetry is the whole trap, so prove the symmetric version differs.

        Normalising both sides is what a reader who knows the words "cosine
        similarity" will write, and it produces a column in the same range,
        with the same sign, of the same length. This test is the only thing
        standing between that and a silently re-scaled feature.
        """
        history = [3, 5, 8]
        asymmetric = content_similarity(content, history, CANDIDATES)

        table = content
        pooled = torch.nn.functional.normalize(
            table[torch.tensor(history)].mean(dim=0), p=2, dim=-1
        )
        unit_candidates = torch.nn.functional.normalize(
            table[torch.tensor(CANDIDATES)], p=2, dim=-1
        )
        symmetric = (unit_candidates * pooled).sum(dim=-1).numpy()

        assert not np.allclose(asymmetric, symmetric, rtol=1e-3, atol=1e-4)

    def test_the_filter_cannot_be_seen_here_and_is_still_required(
        self, content: torch.Tensor
    ) -> None:
        """This column is scale-invariant. The TOWER is not. Do not remove the filter.

        Row 0 of the content table is exactly zeros, so pooling a padding entry
        adds nothing to the numerator and only inflates the count -- which the
        L2 normalisation on the next line divides straight back out. So
        ``content_similarity`` genuinely cannot detect whether
        :func:`real_history` ran, and a test claiming it does would be
        asserting a coincidence of this fixture.

        The filter earns its place one caller over. ``encode_user`` pools
        ``item_id_emb``, whose row 0 is a LEARNED OOV vector and emphatically
        not zero, and a SASRec arm additionally attends over whatever the mask
        marks real. A 0 arriving inline in ``request.history`` moves the user
        embedding there, and moves the retrieved slate with it.
        """
        raw = [3, 5, 0, 0, 8]
        np.testing.assert_allclose(
            content_similarity(content, raw, CANDIDATES),
            content_similarity(content, real_history(raw, N_ROWS), CANDIDATES),
            rtol=1e-5,
            atol=1e-6,
        )


class TestRealHistory:
    def test_it_drops_the_reserved_row(self) -> None:
        assert real_history([0, 3, 0, 5], N_ROWS) == [3, 5]

    def test_it_drops_ids_the_table_does_not_hold(self) -> None:
        """An id past the table would IndexError in the pool, or wrap in torch."""
        assert real_history([3, N_ROWS, 99, -2, 5], N_ROWS) == [3, 5]

    def test_it_keeps_order_and_repeats(self) -> None:
        """Most recent first, and a re-read is a real second entry in the mean."""
        assert real_history([5, 3, 5], N_ROWS) == [5, 3, 5]

    def test_it_truncates_to_the_towers_window(self) -> None:
        history = [(index % N_ITEMS) + 1 for index in range(MAX_HISTORY + 17)]
        assert real_history(history, N_ROWS) == history[:MAX_HISTORY]

    def test_it_truncates_before_it_filters(self) -> None:
        """The order of the two operations is itself the behaviour.

        The window is the first ``MAX_HISTORY`` entries the gateway sent, which
        is the window the gold table materialised and the tower was fitted on.
        Filtering first would reach past it and pool entries training never
        saw -- so a request whose window is entirely padding has no history at
        all, rather than one article pulled in from position 51.
        """
        history = [0] * MAX_HISTORY + [7]
        assert real_history(history, N_ROWS) == []
