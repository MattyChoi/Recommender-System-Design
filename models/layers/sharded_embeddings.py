"""TorchRec sharded embedding tables"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchrec import EmbeddingBagCollection, EmbeddingBagConfig
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner import EmbeddingShardingPlanner
from torchrec.distributed.planner.storage_reservations import (
    FixedPercentageStorageReservation,
)
from torchrec.distributed.planner.types import Topology
from torchrec.distributed.types import ShardingPlan

from common.schemas import OOV_IDX

ITEM_TABLE = "item"
USER_TABLE = "user"

#: Feature names, declared once because the KJT handed to ``forward`` must carry
#: EVERY one of them -- see :func:`build_ebc`.
ITEM_FEATURE = "item_id"
USER_FEATURE = "user_id"
FEATURE_NAMES = (ITEM_FEATURE, USER_FEATURE)

#: Held back for everything that is not an embedding table. The planner's default
#: (``HeuristicalStorageReservation``) estimates dense tensor size by measuring
#: it -- impossible under meta tensors, where it over-estimates and TorchRec
#: warns as much. An over-reservation is not a harmless safety margin here: it
#: manufactures HBM pressure, and HBM pressure is exactly what the plan responds
#: to, so the report would measure the reservation policy rather than the tables.
DENSE_RESERVATION = 0.15


def build_ebc(
    n_items: int,
    n_users: int,
    dim: int = 64,
    device: torch.device | None = None,
) -> EmbeddingBagCollection:
    """Two tables in one collection, allocated nowhere by default.

    A ``meta`` device allocates no storage: shapes and dtypes exist, bytes do
    not, and the sharder materialises each shard on its own rank later. That is
    what makes planning a 2M-row table possible on a machine that could not hold
    two of them, and it is why the planner run is a *dry run* rather than a
    benchmark of this GPU.

    Args:
        n_items: Catalogue size EXCLUDING the reserved row.
        n_users: User count EXCLUDING the reserved row.
        dim: Embedding width, matching the two-tower's ``id_dim``.
        device: Where to allocate. ``None`` means ``meta``.

    Returns:
        An uninitialised collection with an ``item`` and a ``user`` table.

    Note:
        Both tables are sized ``N + 1``. Indices are 1-based with
        :data:`~common.schemas.OOV_IDX` reserved at 0, and sizing at ``N``
        instead is an ``IndexError`` on the highest-numbered row -- the last one
        anybody reaches by hand.

    Note:
        **A collection's forward is all-or-nothing over its tables.** It iterates
        every table and indexes the input ``KeyedJaggedTensor`` by that table's
        feature name, so a KJT missing ``user_id`` raises ``KeyError`` even when
        only the item embedding is wanted. Measured, not read.

        That is a real difference from ``nn.Embedding`` and it lands on the one
        property the two-tower is built around:
        :meth:`~models.retrieval.two_tower.TwoTower.encode_item` is a function of
        ``item_idx`` ALONE, which is what allows the Part J index to be
        precomputed. Adopting a two-table collection would mean feeding a dummy
        user feature through every item-tower call. Two separate collections
        would avoid it -- at the cost of the planner losing the chance to balance
        both tables against each other, which is most of why TorchRec exists.
    """
    rows = 1 + OOV_IDX  # the reserved row, written as what it is rather than as 1
    return EmbeddingBagCollection(
        tables=[
            EmbeddingBagConfig(
                name=ITEM_TABLE,
                embedding_dim=dim,
                num_embeddings=n_items + rows,
                feature_names=[ITEM_FEATURE],
            ),
            EmbeddingBagConfig(
                name=USER_TABLE,
                embedding_dim=dim,
                num_embeddings=n_users + rows,
                feature_names=[USER_FEATURE],
            ),
        ],
        device=device if device is not None else torch.device("meta"),
    )


def synthetic_topology(
    world_size: int,
    compute_device: str,
    *,
    hbm_gb: float,
    local_world_size: int | None = None,
) -> Topology:
    """Hardware the running machine need not have.

    ``hbm_gb`` is the lever the whole report turns on. The planner reaches for
    row-wise or column-wise only when table-wise does not fit, so a topology with
    generous HBM returns ``table_wise`` for everything and demonstrates nothing.
    A 2M x 64 fp32 table is 512 MB against a 4090's 24 GB -- which is precisely
    why the design target alone is not enough to make the decision move.

    Constructed directly rather than through ``TopologyFactory``, which resolves
    *actual* hardware. TorchRec warns about direct construction; the warning is
    about launching a real sharded job, and a hypothetical topology is the one
    case where resolving this machine's hardware would be wrong.

    Args:
        world_size: Total ranks.
        compute_device: TorchRec's device string. Never a literal here -- see
            the module docstring.
        hbm_gb: Per-device high-bandwidth memory.
        local_world_size: Ranks per host. Defaults to ``world_size``, i.e. one
            host, which keeps ``inter_host_bw`` out of the cost model.

    Returns:
        A :class:`Topology` for the planner.
    """
    return Topology(
        world_size=world_size,
        compute_device=compute_device,
        hbm_cap=int(hbm_gb * 1024**3),
        local_world_size=local_world_size if local_world_size is not None else world_size,
    )


def plan_for(
    collection: EmbeddingBagCollection,
    topology: Topology,
    *,
    batch_size: int = 8192,
) -> ShardingPlan:
    """Ask the planner where the tables should live.

    Args:
        collection: From :func:`build_ebc`, typically on ``meta``.
        topology: From :func:`synthetic_topology`.
        batch_size: Feeds the cost model's all-to-all estimate, so it belongs to
            the *training configuration* being planned for rather than to the
            hardware. 8192 matches what G2 and G3 trained at.

    Returns:
        The chosen :class:`ShardingPlan`. Read it with :func:`placements`.
    """
    planner = EmbeddingShardingPlanner(
        topology=topology,
        batch_size=batch_size,
        storage_reservation=FixedPercentageStorageReservation(percentage=DENSE_RESERVATION),
    )
    return planner.plan(collection, sharders=[EmbeddingBagCollectionSharder()])


@dataclass(frozen=True)
class PlacedTable:
    """One table's placement, flattened out of the plan for rendering.

    Attributes:
        table: The ``EmbeddingBagConfig`` name.
        sharding_type: A :class:`~torchrec.distributed.types.ShardingType` value,
            e.g. ``"table_wise"`` or ``"row_wise"``. **This is the answer the
            report exists to report.**
        compute_kernel: What TorchRec will run the lookup with, e.g. ``"fused"``.
        ranks: Which ranks hold a piece.
        shard_sizes: ``(rows, dim)`` per shard, in plan order.
    """

    table: str
    sharding_type: str
    compute_kernel: str
    ranks: tuple[int, ...]
    shard_sizes: tuple[tuple[int, int], ...]

    @property
    def n_shards(self) -> int:
        return len(self.shard_sizes)

    @property
    def rows_placed(self) -> int:
        """Total rows across shards.

        Compare against the table's own row count: under ``data_parallel`` every
        rank holds a full replica, so this exceeding the table size is the plan
        telling you it chose replication over partitioning.
        """
        return sum(rows for rows, _ in self.shard_sizes)


def placements(plan: ShardingPlan) -> tuple[PlacedTable, ...]:
    """Flatten a plan into one row per table, sorted by name.

    ``ShardingPlan.plan`` is keyed by module path and then by parameter name.
    There is one module here, but reading it generically costs nothing and
    survives a second collection being added.

    Args:
        plan: From :func:`plan_for`.

    Returns:
        One :class:`PlacedTable` per table, in name order so two plans render
        comparably.
    """
    rows: list[PlacedTable] = []
    for module_plan in plan.plan.values():
        for name, sharding in module_plan.items():
            spec = getattr(sharding, "sharding_spec", None)
            shards = getattr(spec, "shards", None) or ()
            rows.append(
                PlacedTable(
                    table=str(name),
                    sharding_type=str(sharding.sharding_type),
                    compute_kernel=str(sharding.compute_kernel),
                    ranks=tuple(sharding.ranks or ()),
                    shard_sizes=tuple(
                        (int(s.shard_sizes[0]), int(s.shard_sizes[1])) for s in shards
                    ),
                )
            )
    return tuple(sorted(rows, key=lambda row: row.table))
