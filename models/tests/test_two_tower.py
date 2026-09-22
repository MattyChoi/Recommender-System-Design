"""Shapes, the unit sphere, the mask, and what ``use_id`` actually removes.

Two of these are worth more than the rest. The mean-pool must divide by the
TRUE history length rather than the padded width, or every short history is
pulled toward the origin in proportion to how short it is -- and on this corpus
most dev histories are short, so the bias would fall hardest on exactly the
requests the model exists to serve. And ``use_id=False`` must remove IDs from
both towers, or the ID table keeps training through the user side and the
content-only arm of the ablation quietly benefits from the thing it is meant to
be doing without.

``use_content`` is the mirror of that, and G1 asks for three arms rather than
two: ID-only, content-only, and both. The same contamination applies in reverse,
one level out -- category and subcategory are content, since a cold article has
both, so an ID-only arm that kept them would not be ID-only.

Everything runs on CPU by construction. The device ladder is
``common/torch_env.py``'s job, and ``tests/test_torch_env.py`` fails the build
if anything under ``models/`` names a device.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from common.torch_env import deterministic, select_device
from models.retrieval.two_tower import Tower, TwoTower, _normalise

N_ITEMS = 5
CONTENT_DIM = 4
N_USER_FEATS = 3
N_CATEGORIES = 3
N_SUBCATEGORIES = 5
OUT_DIM = 8


@pytest.fixture
def tables() -> dict[str, torch.Tensor]:
    """Item-indexed tables, row 0 reserved for OOV and padding (B2)."""
    content = torch.arange((N_ITEMS + 1) * CONTENT_DIM, dtype=torch.float32).reshape(
        N_ITEMS + 1, CONTENT_DIM
    )
    content[0].zero_()
    return {
        "content": content,
        "item_category": torch.tensor([0, 1, 1, 2, 2, 3]),
        "item_subcategory": torch.tensor([0, 1, 2, 3, 4, 5]),
    }


def _model(
    tables: dict[str, torch.Tensor], *, use_id: bool = True, use_content: bool = True
) -> TwoTower:
    return TwoTower(
        content=tables["content"],
        item_category=tables["item_category"],
        item_subcategory=tables["item_subcategory"],
        n_user_feats=N_USER_FEATS,
        n_categories=N_CATEGORIES,
        n_subcategories=N_SUBCATEGORIES,
        out_dim=OUT_DIM,
        use_id=use_id,
        use_content=use_content,
    ).to(select_device("cpu"))


def _in_features(tower: Tower) -> int:
    """Input width of a tower's first Linear.

    ``nn.Sequential.__getitem__`` is typed as returning ``Module``, and
    attribute access on a Module is ``Tensor | Module``, so ``in_features``
    compares fine with ``==`` and not at all with ``<``. Narrowing once here
    keeps that out of the assertions.
    """
    first = tower.net[0]
    assert isinstance(first, nn.Linear)
    return first.in_features


class TestTheUnitSphere:
    def test_the_tower_output_has_unit_norm(self) -> None:
        """Which is what makes the dot product cosine, and lets FAISS use
        METRIC_INNER_PRODUCT in Part J."""
        out = Tower(in_dim=6, out_dim=OUT_DIM)(torch.randn(4, 6))

        assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5)

    def test_both_encoders_land_on_the_sphere(self, tables: dict[str, torch.Tensor]) -> None:
        model = _model(tables)

        items = model.encode_item(torch.tensor([1, 2, 3]))
        users = model.encode_user(
            torch.randn(2, N_USER_FEATS),
            torch.tensor([[1, 2, 0], [3, 0, 0]]),
            torch.tensor([[1, 1, 0], [1, 0, 0]]),
        )

        assert torch.allclose(items.norm(dim=-1), torch.ones(3), atol=1e-5)
        assert torch.allclose(users.norm(dim=-1), torch.ones(2), atol=1e-5)
        assert items.shape == (3, OUT_DIM) and users.shape == (2, OUT_DIM)


class TestInputBlockScales:
    """The blocks must reach the tower comparable, and zero must survive.

    Measured on the first four runs before this existed: the 24 categorical
    dimensions carried 95.5% of the item tower's input energy against 4.2% for
    768 content dimensions, and the pooled history carried 0.13-0.28% of the
    user tower's. The item embeddings came out at an effective rank of 7 of
    128, the retriever served 774 of 65,238 articles, and Recall@100 on items
    with no training clicks was exactly zero. None of that errored.

    The second half matters as much as the first. The norms are parameterless
    so that a zero input stays zero, which is what keeps the reserved row 0 out
    of the candidate pool and an empty history contributing nothing. An affine
    LayerNorm would satisfy every scale assertion here and silently break both.
    """

    def test_every_item_block_reaches_the_tower_at_the_same_scale(
        self, tables: dict[str, torch.Tensor]
    ) -> None:
        model = _model(tables)
        assert model.content is not None
        assert model.category is not None
        assert model.subcategory is not None
        assert model.item_id_emb is not None
        widths = [
            model.content.embedding_dim,
            model.category.embedding_dim,
            model.subcategory.embedding_dim,
            model.item_id_emb.embedding_dim,
        ]

        features = model.item_features(torch.tensor([1, 2, 3, 4, 5]))

        start = 0
        for width in widths:
            block = features[:, start : start + width]
            # Not exactly 1: LayerNorm divides by sqrt(var + 1e-5), so a block
            # whose own variance is small loses a little scale to the epsilon.
            # The ID block inits at std 1/sqrt(64), variance 0.0156, and lands
            # at 0.99968. That is a property worth knowing rather than a
            # tolerance to hide -- see the test below.
            assert float(block.detach().pow(2).mean().sqrt()) == pytest.approx(1.0, abs=2e-3)
            start += width
        # If the model grows a block this test does not know about, the widths
        # stop covering the tensor and every assertion above becomes partial.
        assert start == features.shape[1]

    def test_a_very_quiet_block_is_not_amplified_to_full_scale(self) -> None:
        """LayerNorm's epsilon is a floor, and here it works in our favour.

        Normalising the ID block was the risk in this change: an item whose ID
        embedding is still near its initialisation would have that noise raised
        to the same volume as a trained one, which is the opposite of what a
        cold item needs. The epsilon bounds it. A row at std 1e-3 has variance
        1e-6 against eps 1e-5 and comes out at roughly a third of unit scale,
        so an untrained row stays quieter than a trained one without anything
        having to track which is which.
        """
        quiet = torch.randn(256, 64) * 1e-3
        loud = torch.randn(256, 64)

        assert float(_normalise(quiet).pow(2).mean().sqrt()) < 0.5
        assert float(_normalise(loud).pow(2).mean().sqrt()) == pytest.approx(1.0, abs=1e-3)

    def test_the_reserved_row_is_still_all_zero_after_normalising(
        self, tables: dict[str, torch.Tensor]
    ) -> None:
        """Row 0 is zero in every table, so its whole feature vector must be
        zero. With an affine norm it would be the learned bias instead -- a
        nonzero vector, competing in every softmax, for an index that is a
        bucket rather than an article."""
        model = _model(tables)

        features = model.item_features(torch.tensor([0]))

        assert torch.equal(features, torch.zeros_like(features))

    def test_a_zero_block_normalises_to_zero(self) -> None:
        """Both invariants above reduce to this one property of the helper."""
        assert torch.equal(_normalise(torch.zeros(2, 5)), torch.zeros(2, 5))

    def test_the_id_table_does_not_start_inside_numerical_noise(
        self, tables: dict[str, torch.Tensor]
    ) -> None:
        """At std=0.01 every row starts indistinguishable from every other, and
        an item appearing in a handful of batches never leaves that state."""
        model = _model(tables)
        assert model.item_id_emb is not None

        spread = float(model.item_id_emb.weight[1:].std())

        assert spread == pytest.approx(model.item_id_emb.embedding_dim**-0.5, rel=0.4)

    def test_a_checkpoint_from_before_the_block_norms_is_refused(
        self, tables: dict[str, torch.Tensor]
    ) -> None:
        """Those weights were fitted against inputs two orders of magnitude
        apart. Nothing about the tensor shapes changed, so without the buffer
        they would load cleanly and score a number that looks like a result."""
        model = _model(tables)
        state = model.state_dict()
        del state["input_scaling"]

        with pytest.raises(RuntimeError, match="input_scaling"):
            model.load_state_dict(state)


class TestThePool:
    def test_padding_width_does_not_change_the_user_embedding(
        self, tables: dict[str, torch.Tensor]
    ) -> None:
        """The test that matters. The same history padded to different widths
        must encode identically -- dividing by L instead of by the true length
        would shrink the longer-padded one toward the origin."""
        model = _model(tables).eval()
        feats = torch.zeros(1, N_USER_FEATS)

        narrow = model.encode_user(feats, torch.tensor([[1, 2]]), torch.tensor([[1, 1]]))
        wide = model.encode_user(feats, torch.tensor([[1, 2, 0, 0]]), torch.tensor([[1, 1, 0, 0]]))

        assert torch.allclose(narrow, wide, atol=1e-6)

    def test_a_masked_slot_is_ignored_even_when_it_holds_a_real_item(
        self, tables: dict[str, torch.Tensor]
    ) -> None:
        """The mask, not the value, decides. A loader that padded with a real
        index instead of 0 must not silently change the result."""
        model = _model(tables).eval()
        feats = torch.zeros(1, N_USER_FEATS)

        clean = model.encode_user(feats, torch.tensor([[1, 0]]), torch.tensor([[1, 0]]))
        dirty = model.encode_user(feats, torch.tensor([[1, 4]]), torch.tensor([[1, 0]]))

        assert torch.allclose(clean, dirty, atol=1e-6)

    def test_an_empty_history_does_not_divide_by_zero(
        self, tables: dict[str, torch.Tensor]
    ) -> None:
        """88% of dev's users are new; an all-zero mask is a common state, not
        an edge case."""
        model = _model(tables).eval()

        out = model.encode_user(
            torch.zeros(1, N_USER_FEATS), torch.tensor([[0, 0]]), torch.tensor([[0, 0]])
        )

        assert torch.isfinite(out).all()


class TestUseId:
    def test_it_removes_the_table_entirely(self, tables: dict[str, torch.Tensor]) -> None:
        """Not merely bypassed: the parameter-count difference between the two
        ablation arms should be real and visible in the run's log."""
        assert _model(tables, use_id=True).item_id_emb is not None
        assert _model(tables, use_id=False).item_id_emb is None

    def test_the_history_pools_content_when_ids_are_off(
        self, tables: dict[str, torch.Tensor]
    ) -> None:
        """The decision this flag exists to make honest. With IDs on the user
        side, a nominally content-only run still trains the ID table through
        the pool and still benefits from it."""
        with_ids = _model(tables, use_id=True)
        without = _model(tables, use_id=False)
        ids = torch.tensor([[1, 2]])

        assert with_ids.item_id_emb is not None
        assert with_ids.history_vectors(ids).shape[-1] == with_ids.item_id_emb.weight.shape[1]
        assert torch.equal(without.history_vectors(ids), tables["content"][ids])

    def test_both_arms_feed_the_user_tower_the_same_width(
        self, tables: dict[str, torch.Tensor]
    ) -> None:
        """Otherwise the ablation measures a bigger model, not a different one.

        Pooling raw content would give the content arm a ~10x wider user tower
        input than the ID arm, and any difference between the runs would be
        unattributable.
        """
        with_ids = _model(tables, use_id=True)
        without = _model(tables, use_id=False)

        assert _in_features(with_ids.user_tower) == _in_features(without.user_tower)

    def test_the_history_projection_is_bias_free(self, tables: dict[str, torch.Tensor]) -> None:
        """A bias-free map commutes with the mean, which is what makes pooling
        first and projecting once equivalent to projecting all L entries -- and
        what keeps an empty history pooling to zeros in BOTH arms rather than to
        a learned constant in one of them."""
        model = _model(tables, use_id=False)

        assert model.history_proj is not None
        assert model.history_proj.bias is None

    def test_projecting_after_the_pool_matches_projecting_before(
        self, tables: dict[str, torch.Tensor]
    ) -> None:
        """The equivalence the cheap path relies on, asserted rather than
        assumed -- it is only true while the projection stays linear."""
        model = _model(tables, use_id=False).eval()
        assert model.history_proj is not None

        vectors = model.history_vectors(torch.tensor([[1, 2, 0]]))
        mask = torch.tensor([[1, 1, 0]]).unsqueeze(-1).to(vectors.dtype)
        counts = mask.sum(dim=1).clamp(min=1.0)

        pooled_first = model.history_proj((vectors * mask).sum(dim=1) / counts)
        projected_first = (model.history_proj(vectors) * mask).sum(dim=1) / counts

        assert torch.allclose(pooled_first, projected_first, atol=1e-5)

    def test_both_arms_still_reach_every_cold_item(self, tables: dict[str, torch.Tensor]) -> None:
        """An item never seen in train has a random ID embedding but a real
        content vector, so the content-only arm must still encode it."""
        out = _model(tables, use_id=False).encode_item(torch.tensor([N_ITEMS]))

        assert out.shape == (1, OUT_DIM) and torch.isfinite(out).all()


class TestUseContent:
    """G1's third arm. ``use_id`` alone gives only two of the three the manual
    asks for -- content-only and both -- because content was unconditional."""

    def test_it_removes_every_content_derived_table(self, tables: dict[str, torch.Tensor]) -> None:
        """Category and subcategory go with the vectors, not with the ID.

        A brand-new article has a category, so an arm that kept it could still
        represent cold items and the ID-only number would understate exactly the
        gap the ablation is measuring.
        """
        model = _model(tables, use_content=False)

        assert model.content is None
        assert model.category is None
        assert model.subcategory is None
        assert model.item_id_emb is not None

    def test_the_item_tower_narrows_to_the_id_alone(self, tables: dict[str, torch.Tensor]) -> None:
        """Not merely zeroed: the arms must differ in parameter count, or the
        ablation compares one model against a handicapped copy of itself."""
        both = _model(tables)
        id_only = _model(tables, use_content=False)

        assert _in_features(id_only.item_tower) < _in_features(both.item_tower)
        assert sum(p.numel() for p in id_only.parameters()) < sum(
            p.numel() for p in both.parameters()
        )

    def test_the_id_only_arm_still_encodes(self, tables: dict[str, torch.Tensor]) -> None:
        model = _model(tables, use_content=False).eval()

        items = model.encode_item(torch.tensor([1, 2, 3]))
        users = model.encode_user(
            torch.zeros(1, N_USER_FEATS), torch.tensor([[1, 2]]), torch.tensor([[1, 1]])
        )

        assert items.shape == (3, OUT_DIM) and users.shape == (1, OUT_DIM)
        assert torch.allclose(items.norm(dim=-1), torch.ones(3), atol=1e-5)

    def test_the_history_still_pools_ids(self, tables: dict[str, torch.Tensor]) -> None:
        """With content gone the user side must fall back to the ID table, not
        to the content table that no longer exists."""
        model = _model(tables, use_content=False)

        assert model.item_id_emb is not None
        assert (
            model.history_vectors(torch.tensor([[1, 2]])).shape[-1]
            == (model.item_id_emb.weight.shape[1])
        )

    def test_all_three_arms_share_the_user_tower_width(
        self, tables: dict[str, torch.Tensor]
    ) -> None:
        """The same reason the two-arm version of this test exists: a wider user
        tower in one arm makes any difference between runs unattributable."""
        widths = {
            _in_features(_model(tables, use_id=use_id, use_content=use_content).user_tower)
            for use_id, use_content in [(True, True), (False, True), (True, False)]
        }

        assert len(widths) == 1

    def test_dropping_both_is_refused(self, tables: dict[str, torch.Tensor]) -> None:
        """The one combination with no item representation at all. Silently
        allowed, it would train a model whose item tower takes a zero-width
        input -- which torch accepts and which learns nothing."""
        with pytest.raises(ValueError, match="cannot both be False"):
            _model(tables, use_id=False, use_content=False)


class TestReservedIndexZero:
    def test_the_tables_are_sized_n_items_plus_one(self, tables: dict[str, torch.Tensor]) -> None:
        """Off by one here is an IndexError on the highest-numbered item, which
        is the last one anybody tests by hand (B2)."""
        model = _model(tables)

        assert model.item_id_emb is not None and model.content is not None
        assert model.item_id_emb.weight.shape[0] == N_ITEMS + 1
        assert model.content.weight.shape[0] == N_ITEMS + 1
        assert model.encode_item(torch.tensor([N_ITEMS])).shape == (1, OUT_DIM)

    def test_index_zero_is_padding_on_every_learned_table(
        self, tables: dict[str, torch.Tensor]
    ) -> None:
        model = _model(tables)

        assert model.item_id_emb is not None
        assert model.category is not None and model.subcategory is not None
        assert model.item_id_emb.padding_idx == 0
        assert model.category.padding_idx == 0
        assert model.subcategory.padding_idx == 0
        assert torch.equal(
            model.item_id_emb.weight[0], torch.zeros(model.item_id_emb.weight[0].shape)
        )

    def test_mismatched_table_lengths_are_refused(self, tables: dict[str, torch.Tensor]) -> None:
        """Silent misalignment would read a different item's category forever."""
        with pytest.raises(ValueError, match="row count"):
            TwoTower(
                content=tables["content"],
                item_category=tables["item_category"][:-1],
                item_subcategory=tables["item_subcategory"],
                n_user_feats=N_USER_FEATS,
                n_categories=N_CATEGORIES,
                n_subcategories=N_SUBCATEGORIES,
            )


class TestContentIsFrozen:
    def test_no_gradient_reaches_the_sentence_vectors(
        self, tables: dict[str, torch.Tensor]
    ) -> None:
        """They are an input, not a parameter. Recomputing them per epoch would
        dominate training time, and fine-tuning them would quietly make the
        cached Parquet wrong for the next run."""
        model = _model(tables)

        # torch ships no annotation for Tensor.backward; the call is correct.
        model.encode_item(torch.tensor([1, 2])).sum().backward()  # type: ignore[no-untyped-call]

        assert model.content is not None
        assert model.content.weight.requires_grad is False
        assert model.content.weight.grad is None

    def test_the_matrix_is_copied_not_aliased(self, tables: dict[str, torch.Tensor]) -> None:
        """``from_pretrained`` would wrap the caller's storage; the copy does not.

        That is what makes zeroing row 0 below safe -- with an alias it would
        reach back into the caller's tensor, and through the numpy array behind
        it if it came from ``from_numpy``.
        """
        source = tables["content"].clone()
        model = _model({**tables, "content": source})

        source[1] = 99.0

        assert model.content is not None
        assert not torch.equal(model.content.weight[1], source[1])

    def test_a_nonzero_reserved_row_is_zeroed(self, tables: dict[str, torch.Tensor]) -> None:
        """``padding_idx`` does NOT zero a weight supplied after construction --
        measured, by either construction route. Every other test here uses a
        fixture that already zeroes row 0, so they would stay green while the
        guarantee was fiction. This one passes a dirty row.
        """
        dirty = tables["content"].clone()
        dirty[0] = 7.0

        model = _model({**tables, "content": dirty})

        assert model.content is not None
        assert torch.equal(model.content.weight[0], torch.zeros(CONTENT_DIM))


class TestDeterminism:
    def test_the_same_seed_gives_the_same_model(self, tables: dict[str, torch.Tensor]) -> None:
        """An ablation whose arms differ by initialisation is not an ablation."""
        with deterministic(0):
            first = _model(tables).eval().encode_item(torch.tensor([1, 2, 3]))
        with deterministic(0):
            second = _model(tables).eval().encode_item(torch.tensor([1, 2, 3]))

        assert torch.allclose(first, second, atol=1e-6)


class TestPrecompute:
    def test_it_matches_encoding_one_at_a_time(self, tables: dict[str, torch.Tensor]) -> None:
        """This output becomes the Part J index, so a batching bug here would
        surface as inexplicably poor recall much later."""
        model = _model(tables).eval()

        everything = model.precompute_items(batch_size=2)
        singly = model.encode_item(torch.arange(N_ITEMS + 1))

        assert everything.shape == (N_ITEMS + 1, OUT_DIM)
        assert torch.allclose(everything, singly, atol=1e-6)

    def test_the_temperature_stays_positive(self, tables: dict[str, torch.Tensor]) -> None:
        """Learned in log space precisely so it cannot cross zero and flip the
        softmax inside out."""
        model = _model(tables)

        # sub_ rather than -=: the in-place operator on a Parameter returns a
        # plain Tensor, which would rebind the attribute and detach it from the
        # module's parameters.
        with torch.no_grad():
            model.log_temp.sub_(50.0)

        assert model.temperature > 0
