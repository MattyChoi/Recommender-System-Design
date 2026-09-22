"""The shared ranking fit, and the invariants distribution depends on."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from models.ranking.dataset import RankingRows, shard_by_request
from models.ranking.torch_fit import fit, prepare, split_columns, unwrap


def make_rows(
    n_requests: int = 12, per_request: int = 4, categories: np.ndarray | None = None
) -> RankingRows:
    """A small table with the two categorical columns in their usual places.

    ``categories`` sets ``category_idx`` per REQUEST, so a test can put a value
    in one stride and not another.
    """
    rows = n_requests * per_request
    rng = np.random.default_rng(0)
    request = np.repeat(np.arange(n_requests), per_request).astype(np.int64)
    per_row = (np.ones(n_requests, dtype=np.int64) if categories is None else categories)[request]

    features = np.stack(
        [
            rng.normal(size=rows),  # retrieval_score
            rng.integers(0, 50, size=rows),  # two_tower_rank
            rng.integers(0, 500, size=rows),  # prior_clicks
            per_row,  # category_idx
            np.ones(rows),  # subcategory_idx
        ],
        axis=1,
    ).astype(np.float32)

    labels = np.zeros(rows, dtype=np.int64)
    labels[::per_request] = 1
    return RankingRows(
        names=(
            "retrieval_score",
            "two_tower_rank",
            "prior_clicks",
            "category_idx",
            "subcategory_idx",
        ),
        features=features,
        labels=labels,
        groups=np.full(n_requests, per_request, dtype=np.int64),
        request=request,
        user_ids=np.arange(n_requests, dtype=np.int64),
        observed=np.ones(rows, dtype=bool),
        found=np.ones(n_requests, dtype=bool),
    )


class Tiny(nn.Module):
    """One linear layer over the dense columns, ignoring the categoricals."""

    def __init__(self, n_dense: int, cardinalities: tuple[int, ...]) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.embeddings = nn.ModuleList([nn.Embedding(size + 1, 2) for size in cardinalities])
        self.out = nn.Linear(n_dense, 1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.out(dense).squeeze(-1)
        return result


class Wrapper(nn.Module):
    """Stands in for DDP: forwards to ``.module`` and answers to ``unwrap``."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.module(*args)
        return result


def test_shards_partition_the_requests() -> None:
    rows = make_rows(12)
    world = 3
    seen = [shard_by_request(rows, rank, world).user_ids for rank in range(world)]

    assert sorted(np.concatenate(seen).tolist()) == list(range(12))
    for rank in range(world):
        for other in range(rank + 1, world):
            assert not set(seen[rank].tolist()) & set(seen[other].tolist())


def test_shards_keep_groups_whole() -> None:
    rows = make_rows(12, per_request=4)
    shard = shard_by_request(rows, 1, 3)
    assert shard.groups.sum() == len(shard.labels)
    assert set(shard.groups.tolist()) == {4}


def test_a_rank_outside_the_world_is_rejected() -> None:
    rows = make_rows(6)
    with pytest.raises(ValueError, match="world"):
        shard_by_request(rows, 3, 3)


def test_a_shard_can_disagree_with_the_whole_about_cardinality() -> None:
    """The precondition. Without it the next test proves nothing.

    Category 5 appears only on requests 0, 3, 6, 9 -- stride 3, rank 0 -- so a
    rank that sized its embedding table from its own shard would build a table
    of 6 rows where rank 1 built one of 2.
    """
    categories = np.ones(12, dtype=np.int64)
    categories[::3] = 5
    rows = make_rows(12, categories=categories)

    whole = prepare(rows).cardinalities
    per_shard = [prepare(shard_by_request(rows, rank, 3)).cardinalities for rank in range(3)]

    assert whole != per_shard[1], "the fixture does not exercise the disagreement"
    assert per_shard[0] != per_shard[1]


def test_the_shared_preparation_is_what_every_rank_builds_from() -> None:
    categories = np.ones(12, dtype=np.int64)
    categories[::3] = 5
    rows = make_rows(12, categories=categories)
    shared = prepare(rows)

    built: list[tuple[int, ...]] = []

    def factory(n_dense: int, cardinalities: tuple[int, ...]) -> nn.Module:
        built.append(cardinalities)
        return Tiny(n_dense, cardinalities)

    for rank in range(3):
        fit(
            factory,
            shard_by_request(rows, rank, 3),
            rows,
            torch.device("cpu"),
            epochs=1,
            preparation=shared,
        )

    assert built == [shared.cardinalities] * 3


def test_the_passed_scaler_is_returned_not_a_refitted_one() -> None:
    rows = make_rows(12)
    shared = prepare(rows)
    _, scaler = fit(
        Tiny,
        shard_by_request(rows, 0, 3),
        rows,
        torch.device("cpu"),
        epochs=1,
        preparation=shared,
    )
    assert np.array_equal(scaler.mean, shared.scaler.mean)
    assert np.array_equal(scaler.scale, shared.scaler.scale)


def test_a_wrapped_model_comes_back_unwrapped() -> None:
    rows = make_rows(12)
    wrapped: list[nn.Module] = []

    def wrap(model: nn.Module) -> nn.Module:
        holder = Wrapper(model)
        wrapped.append(holder)
        return holder

    model, _ = fit(Tiny, rows, rows, torch.device("cpu"), epochs=1, wrap=wrap)

    assert wrapped, "wrap was never applied"
    assert isinstance(model, Tiny)
    assert model is unwrap(wrapped[0])


def test_every_epoch_reports_including_the_one_that_stops() -> None:
    """Ranks stop together only if the last epoch reports like the others.

    ``ray.train.report`` is a barrier, so an epoch that breaks out of the loop
    before reporting leaves the other ranks waiting at it.
    """
    rows = make_rows(12)
    records: list[dict[str, float]] = []
    fit(
        Tiny,
        rows,
        rows,
        torch.device("cpu"),
        epochs=3,
        patience=1,
        on_epoch=records.append,
    )
    assert [record["epoch"] for record in records] == [0.0, 1.0, 2.0][: len(records)]
    assert len(records) >= 2
    assert {"epoch", "train_loss", "held_loss"} == set(records[0])


def test_split_columns_finds_the_categoricals_by_name() -> None:
    rows = make_rows(6)
    dense, sparse = split_columns(rows)
    assert [rows.names[index] for index in sparse] == ["category_idx", "subcategory_idx"]
    assert sorted(dense + sparse) == list(range(len(rows.names)))
