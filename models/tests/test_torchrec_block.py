"""The TorchRec categorical backend, and the four ways it could be silently wrong.

**Parity.** The claim the module makes is that it is a drop-in: swap the backend
and nothing above it changes. That is only true if the two produce the same
numbers from the same weights, so the first test is the load-bearing one.

**Ordering.** A ``KeyedJaggedTensor`` is a flat value buffer with feature as the
OUTER dimension, so the matrix has to be transposed before it is flattened.
Passing it row-major produces a tensor of exactly the right shape holding the
wrong lookups, and it trains. The control test shows the two orderings CAN
differ, because an assertion that the right one works proves nothing unless the
wrong one visibly fails.

**The reserved row.** ``FeatureBlock`` uses ``padding_idx=0``, which pins row 0
to zero and takes no gradient on it. TorchRec has no such concept. Row 0 is
zeroed at construction so the backends start equal; it is trainable afterwards,
and that difference is pinned here because the module docstring claims it.

**Initialisation** -- added last, and the only one of the four that was found by
a metric rather than by reading. Parity of the forward pass says nothing about
where the weights START, and TorchRec's default is roughly a seventieth of
``nn.Embedding``'s scale on a table of any size. A block that computes the same
function from the same weights and begins somewhere else is not a drop-in.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

pytest.importorskip(
    "torchrec",
    reason="`sharded` is an optional extra; run `uv sync --extra sharded` to include it",
)

from torchrec import EmbeddingBagCollection, EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from models.ranking.dcn import DCNv2
from models.ranking.torch_fit import Block, FeatureBlock
from models.ranking.torchrec_block import TorchRecFeatureBlock

CARDINALITIES = (4, 7)
N_DENSE = 3
EMB_DIM = 5
ROWS = 6

# Wide enough for a standard deviation to mean something, and wide enough that
# TorchRec's 1/sqrt(rows) default is unmistakably not unit scale.
WIDE = 2000


def matched_pair() -> tuple[FeatureBlock, TorchRecFeatureBlock]:
    """The two backends carrying identical weights."""
    dense = FeatureBlock(N_DENSE, CARDINALITIES, EMB_DIM)
    sharded = TorchRecFeatureBlock(N_DENSE, CARDINALITIES, EMB_DIM)
    with torch.no_grad():
        for index, table in enumerate(dense.embeddings):
            bag = sharded.collection.embedding_bags[f"table_{index}"]
            bag.weight.copy_(table.weight)
    return dense, sharded


def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """A batch whose two categorical columns hold DIFFERENT values per row.

    Equal columns would make the transpose invisible: the row-major and the
    column-major flattening of a matrix with identical columns are the same
    buffer, and the ordering control below would pass on a broken module.
    """
    torch.manual_seed(0)
    dense = torch.randn(ROWS, N_DENSE)
    sparse = torch.stack(
        [
            torch.arange(ROWS) % (CARDINALITIES[0] + 1),
            (torch.arange(ROWS) * 3 + 1) % (CARDINALITIES[1] + 1),
        ],
        dim=1,
    ).long()
    return dense, sparse


class TestParity:
    def test_the_two_backends_agree_given_the_same_weights(self) -> None:
        dense_block, sharded_block = matched_pair()
        dense, sparse = inputs()

        assert torch.allclose(dense_block(dense, sparse), sharded_block(dense, sparse), atol=1e-6)

    def test_the_widths_agree(self) -> None:
        dense_block, sharded_block = matched_pair()

        assert dense_block.width == sharded_block.width == N_DENSE + EMB_DIM * len(CARDINALITIES)

    def test_the_dense_columns_pass_through_first(self) -> None:
        """The concatenation order is part of the contract: every layer above
        reads a flat vector and would happily learn from a permuted one."""
        _, sharded_block = matched_pair()
        dense, sparse = inputs()

        assert torch.equal(sharded_block(dense, sparse)[:, :N_DENSE], dense)


def lookup(block: TorchRecFeatureBlock, values: torch.Tensor, n_bags: int) -> torch.Tensor:
    """Run the collection over a value buffer laid out however the caller chose."""
    out: torch.Tensor = block.collection(
        KeyedJaggedTensor.from_lengths_sync(
            keys=block.names,
            values=values,
            lengths=torch.ones(n_bags, dtype=torch.long),
        )
    ).values()
    return out


class TestOrdering:
    def test_row_major_and_column_major_give_different_lookups(self) -> None:
        """The control, on EQUAL cardinalities so both orderings are in range.

        This is the shape the mistake really has. With same-sized tables every
        index is valid under either flattening, nothing raises, the tensor comes
        back the right shape, and the model trains on the wrong embeddings.
        """
        sizes = (7, 7)
        block = TorchRecFeatureBlock(N_DENSE, sizes, EMB_DIM)
        sparse = torch.stack(
            [torch.arange(ROWS) % 8, (torch.arange(ROWS) * 3 + 1) % 8], dim=1
        ).long()
        bags = sparse.numel()

        correct = lookup(block, sparse.t().reshape(-1), bags)
        wrong = lookup(block, sparse.reshape(-1), bags)

        assert correct.shape == wrong.shape, "the wrong ordering is not caught by a shape"
        assert not torch.allclose(correct, wrong)

    def test_unequal_cardinalities_may_raise_instead_and_that_is_luck(self) -> None:
        """Written because it is what this test did on its first fixture.

        Feeding a wide column's values into a narrow table indexes past the end,
        so the mistake sometimes announces itself. That depends entirely on the
        cardinalities lining up badly and is not a guard -- the test above is
        the one that describes the general case.
        """
        _, block = matched_pair()  # cardinalities (4, 7): table_0 has 5 rows
        _, sparse = inputs()

        with pytest.raises(RuntimeError):
            lookup(block, sparse.reshape(-1), sparse.numel())


class TestTheReservedRow:
    def test_every_table_is_one_row_wider_than_its_level_count(self) -> None:
        _, sharded_block = matched_pair()

        for index, size in enumerate(CARDINALITIES):
            bag = sharded_block.collection.embedding_bags[f"table_{index}"]
            assert bag.weight.shape == (size + 1, EMB_DIM)

    def test_row_zero_starts_at_zero(self) -> None:
        sharded_block = TorchRecFeatureBlock(N_DENSE, CARDINALITIES, EMB_DIM)

        for bag in sharded_block.collection.embedding_bags.values():
            assert torch.equal(bag.weight[0], torch.zeros(EMB_DIM))

    def test_row_zero_is_trainable_unlike_padding_idx(self) -> None:
        """The stated behavioural difference, pinned as a gradient rather than
        as prose. Unknown becomes a LEARNED vector here and stays an absent one
        in the dense backend, so a checkpoint is not portable between them."""
        dense_block = FeatureBlock(N_DENSE, CARDINALITIES, EMB_DIM)
        _, sharded_block = matched_pair()
        dense, sparse = inputs()
        sparse[:, 0] = 0  # every row looks up the reserved level

        dense_block(dense, sparse).sum().backward()
        sharded_block(dense, sparse).sum().backward()

        table = cast("nn.Embedding", dense_block.embeddings[0])
        assert table.padding_idx == 0
        dense_grad = table.weight.grad
        assert dense_grad is not None
        assert torch.equal(dense_grad[0], torch.zeros(EMB_DIM))

        bag_grad = sharded_block.collection.embedding_bags["table_0"].weight.grad
        assert bag_grad is not None
        assert bag_grad[0].abs().sum() > 0


class TestInitialisation:
    """Written after an end-to-end arm came back 0.0114 NDCG worse and the
    difference turned out to be here, not in the backend."""

    def test_the_tables_start_at_unit_scale_like_nn_embedding(self) -> None:
        block = TorchRecFeatureBlock(N_DENSE, (WIDE,), EMB_DIM)
        weight = block.collection.embedding_bags["table_0"].weight.detach()

        # Row 0 is zeroed by hand, so it is excluded from the statistic it would
        # otherwise drag down.
        assert 0.9 < float(weight[1:].std()) < 1.1

    def test_the_torchrec_default_would_not_and_that_is_why_it_is_overridden(self) -> None:
        """The control. Without it, the assertion above reads as a restatement
        of the library's behaviour rather than a correction of it."""
        default = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="table_0",
                    embedding_dim=EMB_DIM,
                    num_embeddings=WIDE + 1,
                    feature_names=["cat_0"],
                )
            ],
            device=torch.device("cpu"),
        )

        assert float(default.embedding_bags["table_0"].weight.detach().std()) < 0.1


class TestItDropsIntoTheRanker:
    @pytest.mark.parametrize("block", [FeatureBlock, TorchRecFeatureBlock])
    def test_dcn_builds_and_scores_under_either_backend(self, block: Block) -> None:
        model = DCNv2(N_DENSE, CARDINALITIES, emb_dim=EMB_DIM, block=block)
        dense, sparse = inputs()

        assert model(dense, sparse).shape == (ROWS,)
