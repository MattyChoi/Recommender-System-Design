"""G4's module, and the four ways it could be wrong while looking right.

**The first is parity.** The whole claim this module supports is negative -- "at
65K items TorchRec is not worth its complexity, and here is the number." That
claim is only honest if the sharded table is a drop-in for the one it is being
compared against. If ``EmbeddingBagCollection`` and ``nn.Embedding`` disagree,
the comparison is between two different models and the conclusion is about
neither. The equivalence is also CONDITIONAL -- an embedding *bag* pools -- so
the second parity test shows the condition failing, because an equivalence
nobody has seen break reads as unconditional.

**The second is the reserved row.** Sizing a table at N instead of N+1 is an
``IndexError`` on the highest-numbered item, which is the last one anybody
reaches by hand and the first one a full catalogue sweep hits.

**The third is the compute device.** TorchRec's sharder offers seven sharding
types on an accelerator and four on CPU. Planning a GPU design target against
``"cpu"`` deletes row-wise from the search space, returns a perfectly valid plan,
and the report concludes the planner never chooses row-wise.

**The fourth is that the report might be vacuous.** G4c's deliverable is a plan
that CHANGES as the constraint changes. If the planner returns ``table_wise``
whatever topology it is handed, there is nothing to report -- so a test checks
the decision can move at all, before a sweep is built on the assumption that it
does. That is the standing rule about checking A and B can differ, applied
before the measurement rather than after it.
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

from torchrec import KeyedJaggedTensor

from models.layers.sharded_embeddings import (
    DENSE_RESERVATION,
    FEATURE_NAMES,
    ITEM_FEATURE,
    ITEM_TABLE,
    USER_FEATURE,
    USER_TABLE,
    build_ebc,
    placements,
    plan_for,
    synthetic_topology,
)
from models.layers.sizing import DESIGN_TARGET_ITEMS, table_bytes

N_ITEMS = 32
N_USERS = 16
DIM = 8

# The planner needs an accelerator to offer row-wise at all, and the topologies
# this file plans against are deliberately ones the running machine need not
# have -- that is what a design-target topology IS. The device-literal rule
# exists to stop a module quietly training somewhere other than it was tested;
# naming hypothetical hardware for a dry run is the case it is not aimed at.
ACCELERATOR = "cuda"  # allow-device-literal


def _table_weight(collection: nn.Module, table: str) -> torch.Tensor:
    """One table's weight, found by state-dict key rather than by attribute.

    Reaching for ``collection.embedding_bags[table].weight`` couples this test to
    an internal attribute name; the state-dict key is the public surface and is
    what a checkpoint would carry anyway.
    """
    state = collection.state_dict()
    key = next(name for name in state if name.endswith(f"{table}.weight"))
    return cast("torch.Tensor", state[key])


def _lookup(collection: nn.Module, key: str, ids: torch.Tensor) -> torch.Tensor:
    """One embedding per id, by handing the bag a bag of exactly one.

    **Every** feature the collection declares has to appear in the KJT, not just
    the one being read -- ``forward`` iterates all tables and indexes the input
    by each table's feature name. The features not under test are fed the
    reserved row, which is the cheapest way to satisfy a requirement that exists
    whether or not the caller wants that table's output.
    """
    batch = len(ids)
    padding = torch.zeros(batch, dtype=torch.long)
    jagged = KeyedJaggedTensor.from_lengths_sync(
        keys=list(FEATURE_NAMES),
        values=torch.cat([ids if name == key else padding for name in FEATURE_NAMES]),
        lengths=torch.ones(batch * len(FEATURE_NAMES), dtype=torch.long),
    )
    return cast("torch.Tensor", collection(jagged).to_dict()[key])


class TestParityWithNnEmbedding:
    def test_the_same_indices_return_the_same_vectors(self) -> None:
        """The load-bearing test. "We measured TorchRec and did not adopt it" is
        only a measurement if the two tables are the same table."""
        collection = build_ebc(N_ITEMS, N_USERS, dim=DIM, device=torch.device("cpu"))
        weight = _table_weight(collection, ITEM_TABLE)
        reference = nn.Embedding.from_pretrained(  # type: ignore[no-untyped-call]
            weight.clone(), freeze=True
        )

        ids = torch.arange(N_ITEMS + 1, dtype=torch.long)

        assert torch.allclose(_lookup(collection, ITEM_FEATURE, ids), reference(ids), atol=1e-6)

    def test_a_bag_of_two_sums_them_which_is_why_the_parity_is_conditional(self) -> None:
        """The control. The equivalence above holds because every bag has exactly
        one id; an EmbeddingBag pools, and SUM over one element is the identity.
        Without this, a reader cannot tell whether the parity is a property of
        the module or an accident of the fixture -- and the two-tower feeds one
        item id per row, so it is the fixture that makes it true."""
        collection = build_ebc(N_ITEMS, N_USERS, dim=DIM, device=torch.device("cpu"))
        weight = _table_weight(collection, ITEM_TABLE)

        paired = KeyedJaggedTensor.from_lengths_sync(
            keys=list(FEATURE_NAMES),
            values=torch.tensor([3, 7, 0], dtype=torch.long),
            lengths=torch.tensor([2, 1], dtype=torch.long),
        )
        pooled = collection(paired).to_dict()[ITEM_FEATURE]

        assert torch.allclose(pooled[0], weight[3] + weight[7], atol=1e-6)
        assert not torch.allclose(pooled[0], weight[3], atol=1e-6)

    def test_a_kjt_missing_any_feature_raises(self) -> None:
        """Measured, and it is the reason ``_lookup`` pads the other tables.

        A collection's forward walks EVERY table and indexes the KJT by that
        table's feature name, so asking for the item embedding alone is not
        expressible. ``nn.Embedding`` has no such coupling, and
        ``TwoTower.encode_item`` is a function of ``item_idx`` alone -- which is
        the property the whole Part J precomputation rests on. Pinned here
        because it is an argument in the writeup, not an incidental.
        """
        collection = build_ebc(N_ITEMS, N_USERS, dim=DIM, device=torch.device("cpu"))
        item_only = KeyedJaggedTensor.from_lengths_sync(
            keys=[ITEM_FEATURE],
            values=torch.tensor([3], dtype=torch.long),
            lengths=torch.tensor([1], dtype=torch.long),
        )

        with pytest.raises(KeyError, match=USER_FEATURE):
            collection(item_only)


class TestTheReservedRow:
    @pytest.mark.parametrize("table,size", [(ITEM_TABLE, N_ITEMS), (USER_TABLE, N_USERS)])
    def test_every_table_is_one_row_wider_than_its_id_space(self, table: str, size: int) -> None:
        """Indices are 1-based with 0 reserved for OOV. Sizing at N is an
        IndexError on the highest-numbered row, which is the last one anybody
        reaches by hand."""
        collection = build_ebc(N_ITEMS, N_USERS, dim=DIM, device=torch.device("cpu"))

        assert _table_weight(collection, table).shape == (size + 1, DIM)

    def test_the_highest_index_is_addressable(self) -> None:
        """The off-by-one stated as the failure it produces rather than as a
        shape assertion, because a shape assertion passes on a table that is
        one row too wide as readily as on a correct one."""
        collection = build_ebc(N_ITEMS, N_USERS, dim=DIM, device=torch.device("cpu"))

        got = _lookup(collection, ITEM_FEATURE, torch.tensor([N_ITEMS], dtype=torch.long))

        assert got.shape == (1, DIM)


class TestTheMetaDevice:
    def test_a_design_target_table_allocates_no_storage(self) -> None:
        """What makes planning a 2M-row table possible on a machine that could
        not hold two of them. If this ever became a real allocation, the plan
        report would silently turn into a benchmark of this GPU's memory."""
        collection = build_ebc(DESIGN_TARGET_ITEMS, N_USERS, dim=64)

        weight = _table_weight(collection, ITEM_TABLE)

        assert weight.is_meta
        assert weight.shape == (DESIGN_TARGET_ITEMS + 1, 64)


class TestTheComputeDeviceChangesTheSearchSpace:
    def test_row_wise_exists_on_an_accelerator_and_not_on_cpu(self) -> None:
        """Why ``compute_device`` is a required argument with no default.

        Planning a GPU design target against "cpu" returns a valid-looking plan
        drawn from four candidates instead of seven, and a report built on it
        would conclude the planner never chooses row-wise. Nothing errors.
        """
        from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder

        sharder = EmbeddingBagCollectionSharder()

        assert "row_wise" in sharder.sharding_types(ACCELERATOR)
        assert "row_wise" not in sharder.sharding_types("cpu")


class TestThePlan:
    def test_placements_names_every_table_once(self) -> None:
        collection = build_ebc(N_ITEMS, N_USERS, dim=DIM)
        plan = plan_for(collection, synthetic_topology(2, ACCELERATOR, hbm_gb=16.0), batch_size=64)

        placed = placements(plan)

        assert [row.table for row in placed] == sorted([ITEM_TABLE, USER_TABLE])
        assert all(row.ranks for row in placed)
        assert all(dim == DIM for row in placed for _, dim in row.shard_sizes)

    def test_a_table_that_fits_comfortably_is_placed_whole(self) -> None:
        """The baseline the sweep is measured against, and the reason the sweep
        has to exist: given room, the planner has no reason to split anything."""
        collection = build_ebc(DESIGN_TARGET_ITEMS, N_USERS, dim=64)
        plan = plan_for(collection, synthetic_topology(8, ACCELERATOR, hbm_gb=80.0))

        item = next(row for row in placements(plan) if row.table == ITEM_TABLE)

        assert item.sharding_type == "table_wise"
        assert item.n_shards == 1

    @pytest.mark.slow
    def test_a_table_too_large_for_one_device_is_split(self) -> None:
        """**The test that stops G4c being vacuous**, and the size is DERIVED.

        The report's whole content is which sharding type the planner picks. If
        the answer is ``table_wise`` whatever it is handed, a sweep prints the
        same row nine times and reads as a finding.

        An earlier version of this test guessed a tight ``hbm_gb`` and asserted
        two plans differ. It failed: at 1 GB per device the 2M-row design target
        is STILL placed whole, because TorchRec's fused optimiser is rowwise
        Adagrad -- one state float per ROW, about 8 MB, not one per element. The
        guess was off by the optimiser's memory model.

        So the table is sized from the topology rather than the topology guessed
        from the table: strictly more rows than one device's post-reservation
        HBM can hold. Then "not table_wise" is arithmetic, not a prediction.
        Which split it picks is still the planner's call and is not asserted --
        that would be reciting the cost model, which this file exists to avoid.
        """
        hbm_gb = 1.0
        usable = hbm_gb * 1024**3 * (1.0 - DENSE_RESERVATION)
        rows_that_fit = int(usable // (64 * 4))
        collection = build_ebc(rows_that_fit * 2, N_USERS, dim=64)

        plan = plan_for(collection, synthetic_topology(8, ACCELERATOR, hbm_gb=hbm_gb))
        item = next(row for row in placements(plan) if row.table == ITEM_TABLE)

        assert item.sharding_type != "table_wise"
        assert item.n_shards > 1


class TestTableBytes:
    def test_the_design_target_table_is_512_mb(self) -> None:
        """Pinned because the writeup quotes it as the reason the machinery is
        worth demonstrating at 2M and not at 65K."""
        assert table_bytes(DESIGN_TARGET_ITEMS, 64) == 512_000_000

    def test_this_corpus_table_is_under_17_mb(self) -> None:
        """The other half of the same sentence: 65,239 x 64 x fp32 is nothing,
        which is why G1's gate could measure the table as near-worthless without
        anyone noticing a memory problem first."""
        assert table_bytes(65_239, 64) < 17 * 1024**2

    def test_optimiser_state_is_excluded_and_that_is_the_point(self) -> None:
        """Adam carries two moments per parameter, so a fused table's real
        footprint is ~3x this. Quoting parameter bytes as "the memory" is the
        usual way an embedding table's cost gets understated threefold, and
        G4d's "memory saved" number is where that would bite."""
        assert table_bytes(1000, 64) == 1000 * 64 * 4
        assert table_bytes(1000, 64, bytes_per_element=2) == 1000 * 64 * 2
