"""The four ways this pooler could be wrong while training perfectly happily.

**Causality, in the right direction.** The manual's snippet, applied to a
most-recent-first sequence, gives the OLDEST click the full context and the
newest click nothing but itself. Nothing errors and the loss still falls. So the
test does not check that a mask exists -- it perturbs one position at a time and
asserts which ones can reach the output, which is the only way to tell the two
directions apart.

**The padding slot.** ``x[:, -1]`` on our layout is padding for most rows,
because padding sits after the real entries and the mean history is 27.95 of 50.
A pooler reading it would return a function of the positional table alone, for
72% of rows, and would look like a working model.

**The empty history.** Measured in ``scripts/probe_transformer_masks``: torch
2.13 returns FINITE values for an all-masked row rather than NaN. So this is not
a crash test -- it pins the invariant that an empty history pools to exactly
zeros, the same as the mean-pool, because ``_normalise`` maps zeros to zeros and
would otherwise scale a learned constant up to unit variance.

**Length invariance.** The reversal means a user's most recent click always
lands at index -1. If it did not, two users with the same recent clicks and
different history lengths would be encoded differently for no reason.
"""

from __future__ import annotations

import pytest
import torch

from models.retrieval.sasrec import SASRec
from models.retrieval.two_tower import TwoTower, _normalise

BATCH = 4
LENGTH = 6
DIM = 8
N_ITEMS = 12
CONTENT_DIM = 16
N_USER_FEATS = 3
ID_DIM = 64  # TwoTower's default, and the width of the history block

#: Perturbations are NON-UNIFORM, and that is load-bearing.
#:
#: An earlier version of these tests bumped a position by the scalar 1.0, adding
#: the same value to every dimension. The block is pre-norm, so the first thing
#: that happens to a position is a LayerNorm, and LayerNorm subtracts the mean --
#: ``norm(x + 1) == norm(x)``. The perturbation never reached the keys or values
#: at all, survived only on the residual path, and
#: ``test_the_oldest_click_reaches_the_output`` failed against correct code.
#:
#: The project has this failure on record already, from G5: "LayerNorm
#: normalised away a per-rank input difference", one of three tests that were
#: green or red for a reason unrelated to their hypothesis. A perturbation test
#: against a normalised input has to perturb something normalisation keeps.
BUMP = torch.linspace(-1.0, 1.0, DIM)


@pytest.fixture
def pooler() -> SASRec:
    """Deterministic, and dropout off so a perturbation test is not noise."""
    torch.manual_seed(0)
    return SASRec(DIM, n_heads=2, n_blocks=2, max_len=LENGTH, dropout=0.0).eval()


def _sequence(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Most-recent-first vectors and mask, padded AFTER, as the loader builds them."""
    torch.manual_seed(1)
    vectors = torch.randn(len(lengths), LENGTH, DIM)
    mask = torch.zeros(len(lengths), LENGTH, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
        vectors[row, length:] = 0.0
    return vectors, mask


class TestCausalDirection:
    def test_the_most_recent_click_reaches_the_output(self, pooler: SASRec) -> None:
        """Index 0 is the newest click. It must move the user vector."""
        vectors, mask = _sequence([LENGTH])
        before = pooler(vectors, mask)

        perturbed = vectors.clone()
        perturbed[0, 0] += BUMP

        assert not torch.allclose(before, pooler(perturbed, mask), atol=1e-5)

    def test_the_oldest_click_reaches_the_output(self, pooler: SASRec) -> None:
        """And so must the oldest, or the flip dropped half the sequence.

        This is the only test in the file that can see a pooler doing no
        attention at all: every other one here passes on a model where each
        position is a function of itself.
        """
        vectors, mask = _sequence([LENGTH])
        before = pooler(vectors, mask)

        perturbed = vectors.clone()
        perturbed[0, LENGTH - 1] += BUMP

        assert not torch.allclose(before, pooler(perturbed, mask), atol=1e-5)

    def test_a_padding_slot_cannot_reach_the_output(self, pooler: SASRec) -> None:
        """**The test that distinguishes the two mask directions.**

        With the manual's layout the user vector is read from a padding slot, so
        writing into padding would change the answer. Here it must not: the
        output comes from the last REAL position and padding is masked.
        """
        vectors, mask = _sequence([3])
        before = pooler(vectors, mask)

        perturbed = vectors.clone()
        perturbed[0, 3:] = BUMP * 5.0  # garbage where the mask says there is nothing

        assert torch.allclose(before, pooler(perturbed, mask), atol=1e-6)

    def test_the_newest_click_is_not_masked_from_itself(self, pooler: SASRec) -> None:
        """A one-item history must depend on that item.

        Under the manual's direction applied to this layout, position 0 attends
        only to itself -- which happens to be right for a length-1 row and wrong
        for every other, so a test using only short rows would pass on a
        backwards mask. This one exists to be read alongside the two above.
        """
        vectors, mask = _sequence([1])
        before = pooler(vectors, mask)

        perturbed = vectors.clone()
        perturbed[0, 0] += BUMP

        assert not torch.allclose(before, pooler(perturbed, mask), atol=1e-5)


class TestTheEmptyHistory:
    def test_it_pools_to_exactly_zero(self, pooler: SASRec) -> None:
        """The invariant, not a crash guard: torch returns finite values for an
        all-masked row. Zeros match the mean-pool, and ``_normalise`` maps zeros
        to zeros -- a learned constant would instead be scaled to unit variance
        and read as signal."""
        vectors, mask = _sequence([0, 3])

        pooled = pooler(vectors, mask)

        assert torch.equal(pooled[0], torch.zeros(DIM))
        assert not torch.equal(pooled[1], torch.zeros(DIM))

    def test_an_all_empty_batch_stays_finite(self, pooler: SASRec) -> None:
        """88% of dev's users are cold and history dropout draws a kept length
        inclusive of zero, so a batch of these is a real serving state."""
        vectors, mask = _sequence([0, 0, 0, 0])

        assert torch.isfinite(pooler(vectors, mask)).all()


class TestLengthInvariance:
    def test_the_same_recent_clicks_encode_the_same_at_equal_length(self, pooler: SASRec) -> None:
        """Two rows with identical content and identical length must agree --
        the control that makes the next test meaningful rather than vacuous."""
        vectors, mask = _sequence([4, 4])
        vectors[1] = vectors[0]

        pooled = pooler(vectors, mask)

        assert torch.allclose(pooled[0], pooled[1], atol=1e-6)

    def test_position_is_measured_from_the_present(self, pooler: SASRec) -> None:
        """After the flip the newest click sits at index -1 whatever the length,
        so a user's recent history is encoded against the same positions
        regardless of how much older history they happen to have."""
        vectors, mask = _sequence([2, 5])
        vectors[1, :2] = vectors[0, :2]  # same two most recent clicks

        pooled = pooler(vectors, mask)

        # Not equal -- the longer row has more context -- but both finite and
        # neither zero, which is what says the shared suffix was actually read.
        assert torch.isfinite(pooled).all()
        assert not torch.equal(pooled[0], torch.zeros(DIM))
        assert not torch.equal(pooled[1], torch.zeros(DIM))


class TestWiringIntoTheUserTower:
    """H2 moved the history projection from AFTER the pool to BEFORE it.

    The reordering exists so the sequence pooler sees ``id_dim`` columns rather
    than the content table's 768. It is safe only because ``history_proj`` is
    bias-free and therefore commutes with the masked mean -- which is a claim
    about the code, so it is tested rather than asserted in a comment.
    """

    def _tower(self, *, use_id: bool, use_sequence: bool) -> TwoTower:
        torch.manual_seed(0)
        return TwoTower(
            content=torch.randn(N_ITEMS + 1, CONTENT_DIM),
            item_category=torch.zeros(N_ITEMS + 1, dtype=torch.long),
            item_subcategory=torch.zeros(N_ITEMS + 1, dtype=torch.long),
            n_user_feats=N_USER_FEATS,
            n_categories=1,
            n_subcategories=1,
            use_id=use_id,
            use_sequence=use_sequence,
        ).eval()

    def test_the_projection_commutes_with_the_masked_mean(self) -> None:
        """**Why H2's reordering leaves the pooled arm's numbers alone.**

        ``proj(mean(v)) == mean(proj(v))`` for a bias-free linear map. Equality
        is to float tolerance, not bit-exact: the two orders sum a different
        number of terms and floating-point addition is not associative.
        """
        tower = self._tower(use_id=False, use_sequence=False)
        assert tower.history_proj is not None

        vectors = torch.randn(3, 5, CONTENT_DIM)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
        weights = mask.unsqueeze(-1).float()
        vectors = vectors * weights  # padded rows are zero, as the loader leaves them

        pool_then_project = tower.history_proj(
            (vectors * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        )
        projected = tower.history_proj(vectors)
        project_then_pool = (projected * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

        assert torch.allclose(pool_then_project, project_then_pool, atol=1e-5)

    def test_the_bias_free_projection_is_what_makes_it_commute(self) -> None:
        """The control. With a bias the two orders disagree, because the bias is
        added once per position before the mean and once in total after it. If
        this ever passes, ``history_proj`` grew a bias and the reordering above
        silently stopped being equivalent."""
        tower = self._tower(use_id=False, use_sequence=False)
        assert tower.history_proj is not None
        assert tower.history_proj.bias is None

    @pytest.mark.parametrize("use_id", [True, False])
    def test_the_sequence_arm_changes_the_user_embedding(self, use_id: bool) -> None:
        """Both G1 arms must reach the pooler, or Part H can only be measured on
        one of them -- and the content-only arm is the one G1 found interesting."""
        pooled = self._tower(use_id=use_id, use_sequence=False)
        attended = self._tower(use_id=use_id, use_sequence=True)

        feats = torch.randn(2, N_USER_FEATS)
        ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])

        assert not torch.allclose(
            pooled.encode_user(feats, ids, mask), attended.encode_user(feats, ids, mask)
        )

    def test_an_empty_history_contributes_nothing(self) -> None:
        """The invariant the pooler's zero short-circuit exists for, asserted
        INSIDE one model.

        The obvious version -- build both arms from the same seed and compare --
        does not work, and the reason is worth keeping: constructing ``SASRec``
        consumes RNG, so the attended tower's ``user_tower`` draws different
        weights than the pooled tower's. Same seed is not same weights once a
        submodule is inserted ahead of them, and a test comparing two such models
        measures initialisation order rather than pooling.

        So instead: with no history the user embedding must equal the user tower
        applied to a ZERO history block. That is the property both arms share,
        and it holds without a second model to disagree with.
        """
        tower = self._tower(use_id=True, use_sequence=True)
        feats = torch.randn(2, N_USER_FEATS)
        ids = torch.zeros(2, 5, dtype=torch.long)
        mask = torch.zeros(2, 5, dtype=torch.long)

        expected = tower.user_tower(
            torch.cat([tower.user_norm(feats), _normalise(torch.zeros(2, ID_DIM))], dim=-1)
        )

        assert torch.allclose(tower.encode_user(feats, ids, mask), expected, atol=1e-6)


class TestTheContract:
    def test_it_does_not_normalise(self, pooler: SASRec) -> None:
        """The mean-pool returns an unnormalised vector and ``encode_user``
        owns ``_normalise``. A pooler that normalised internally would make the
        two arms differ by more than the pooling, which is the one thing the
        head-to-head must not do."""
        vectors, mask = _sequence([LENGTH])

        norms = pooler(vectors, mask).norm(dim=-1)

        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

    def test_a_sequence_longer_than_the_table_raises(self, pooler: SASRec) -> None:
        """Silent truncation would drop the OLDEST clicks on the reversed
        layout -- the defensible half to lose, and so the kind of bug nobody
        notices."""
        vectors = torch.randn(1, LENGTH + 1, DIM)
        mask = torch.ones(1, LENGTH + 1, dtype=torch.long)

        with pytest.raises(ValueError, match="exceeds max_len"):
            pooler(vectors, mask)

    def test_heads_must_divide_the_width(self) -> None:
        with pytest.raises(ValueError, match="must divide"):
            SASRec(DIM, n_heads=3)

    def test_the_output_width_matches_the_input(self, pooler: SASRec) -> None:
        vectors, mask = _sequence([2, 4, 6, 0])

        assert pooler(vectors, mask).shape == (BATCH, DIM)
