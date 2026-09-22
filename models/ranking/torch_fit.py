"""Input preparation and the training loop every neural ranker here shares.

Two models use this, and they must use the *same* one. If each carried its own
scaling and its own early-stopping rule, a difference between them would be a
difference between four things, and the comparison would not be a comparison.

**Input scaling is not a detail.** The features arrive as a cosine near 0, a
rank sentinel at 10,000 and click counts in the thousands. One ``Linear``
initialised at a single scale cannot hear the quiet ones -- this project has
lost days to exactly that failure, twice -- so the transforms are explicit:
reciprocal for ranks, ``log1p`` for counts, then standardisation fitted on the
FITTING rows alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from models.ranking.dataset import CATEGORICAL, RankingRows

# Rank columns are unbounded with a large sentinel for "absent". A reciprocal
# sends rank 0 to 1, rank 99 to 0.01 and the sentinel to ~0 -- bounded, monotone
# the right way, and it parks "absent" next to "worst" rather than fifty
# standard deviations from everything.
RANK_COLUMNS = ("two_tower_rank", "trending_rank")
COUNT_COLUMNS = ("prior_clicks", "train_clicks", "history_length")


@dataclass(frozen=True)
class Standardiser:
    """Per-column mean and scale, fitted once on the fitting rows.

    Fitted on the fitting split and applied to both. Statistics taken over the
    held-out rows as well would let the model see their distribution -- a small
    leak that flatters every neural baseline and is almost never checked.
    """

    mean: npt.NDArray[np.float32]
    scale: npt.NDArray[np.float32]

    @classmethod
    def fit(cls, matrix: npt.NDArray[np.float32]) -> Standardiser:
        scale = matrix.std(axis=0)
        # A constant column has zero spread; dividing by it gives NaN and the
        # whole forward pass goes NaN without raising.
        return cls(
            mean=matrix.mean(axis=0),
            scale=np.where(scale > 1e-6, scale, 1.0).astype(np.float32),
        )

    def apply(self, matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return ((matrix - self.mean) / self.scale).astype(np.float32)


def transform(rows: RankingRows) -> npt.NDArray[np.float32]:
    """Bound the unbounded columns before anything is standardised."""
    out = rows.features.astype(np.float32).copy()
    for name in RANK_COLUMNS:
        if name in rows.names:
            index = rows.names.index(name)
            out[:, index] = 1.0 / (1.0 + out[:, index])
    for name in COUNT_COLUMNS:
        if name in rows.names:
            index = rows.names.index(name)
            out[:, index] = np.log1p(np.maximum(out[:, index], 0.0))
    return out


def split_columns(rows: RankingRows) -> tuple[list[int], list[int]]:
    """Indices of the dense and the categorical columns, in matrix order."""
    sparse = [rows.names.index(name) for name in CATEGORICAL if name in rows.names]
    dense = [index for index in range(len(rows.names)) if index not in set(sparse)]
    return dense, sparse


class FeatureEmbedding(nn.Module):
    """What a categorical backend has to be, stated as a base class.

    Declared rather than left implicit because the backend is now swappable, and
    ``width`` is read by every model above it to size its first layer. Typed as
    a plain ``nn.Module`` instead, that read returns ``Tensor | Module`` -- the
    signature of ``nn.Module.__getattr__`` -- and the layer sizes downstream
    stop being checked at all.
    """

    width: int

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeatureBlock(FeatureEmbedding):
    """Dense columns and embedded categoricals, concatenated. One input layout.

    Shared so two architectures differ in what they do with the features and in
    nothing else.
    """

    def __init__(self, n_dense: int, cardinalities: Sequence[int], emb_dim: int = 16) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size + 1, emb_dim, padding_idx=0) for size in cardinalities]
        )
        self.width = n_dense + emb_dim * len(cardinalities)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        parts = [dense]
        for index, table in enumerate(self.embeddings):
            parts.append(table(sparse[:, index]))
        return torch.cat(parts, dim=-1)


Factory = Callable[[int, tuple[int, ...]], nn.Module]

#: A categorical-embedding backend: ``(n_dense, cardinalities, emb_dim)``. Both
#: rankers take one, so the TorchRec backend is a constructor argument rather
#: than a branch inside either model.
Block = Callable[[int, Sequence[int], int], "FeatureEmbedding"]


def unwrap(model: nn.Module) -> nn.Module:
    """The inner model, whether or not DDP is wrapping it."""
    return getattr(model, "module", model)


@dataclass(frozen=True)
class Preparation:
    """The quantities every rank must agree on, fitted over the whole table.

    In one process each of these is read off the fitting rows, which is correct
    there and wrong the moment the rows are sharded.

    *The standardiser* fitted on a shard scales that rank's inputs differently
    from its peers', so the gradients DDP averages are gradients of a different
    objective per rank -- and the statistics rank 0 saves next to the weights
    then describe one shard rather than the data.

    *The cardinalities* read off a shard can be SMALLER than rank 0's, because a
    rare category may be absent from that slice. Two ranks then build embedding
    tables of different shapes and DDP's opening broadcast fails, which is the
    good case; a one-worker rerun that happens to see every category makes it
    disappear again, which is the bad one.

    *The positive rate* sets the loss weight, so a per-shard estimate hands each
    rank a slightly different loss surface for nothing.

    So the driver fits this once, over the unsharded rows, and gives every
    worker the same object.
    """

    scaler: Standardiser
    cardinalities: tuple[int, ...]
    positive_rate: float


def prepare(rows: RankingRows) -> Preparation:
    """Fit the rank-invariant quantities on ``rows``."""
    dense_columns, sparse_columns = split_columns(rows)
    prepared = transform(rows)
    sparse = prepared[:, sparse_columns].astype(np.int64)
    return Preparation(
        scaler=Standardiser.fit(prepared[:, dense_columns]),
        cardinalities=tuple(int(sparse[:, index].max()) + 1 for index in range(sparse.shape[1])),
        positive_rate=float(rows.labels.mean()),
    )


def fit(
    factory: Factory,
    rows: RankingRows,
    validation: RankingRows,
    device: torch.device,
    *,
    epochs: int = 20,
    batch_size: int = 8192,
    learning_rate: float = 1e-3,
    patience: int = 3,
    seed: int = 0,
    preparation: Preparation | None = None,
    wrap: Callable[[nn.Module], nn.Module] | None = None,
    on_epoch: Callable[[dict[str, float]], None] | None = None,
) -> tuple[nn.Module, Standardiser]:
    """Fit pointwise with BCE, early-stopping on held-out loss.

    Positives are roughly one row in a hundred, so the loss is weighted by the
    observed ratio. Unweighted, predicting zero everywhere is a 99%-accurate
    model and the gradient barely leaves it.

    Args:
        preparation: Scaling, cardinalities and the positive rate. ``None`` fits
            them on ``rows``, which is right in one process and wrong on a
            shard -- see :class:`Preparation`.
        wrap: Applied to the model before training. This is where DDP goes; the
            loop is otherwise identical under one process and under many, so
            the tested path is the trained path.
        on_epoch: Called with each epoch's record, BEFORE the early-stop test,
            so the final epoch reports like every other one. Under Ray every
            rank must call it, because ``ray.train.report`` is a barrier and
            reporting on rank 0 alone hangs the others at the next boundary.

    Note:
        **Validation is not sharded.** Every rank scores every held-out row, so
        after the gradients synchronise the ranks hold identical weights,
        compute the identical loss and reach the same early-stop decision. Ranks
        disagreeing about which epoch to stop on is a hang, not a wrong number.
    """
    torch.manual_seed(seed)
    dense_columns, sparse_columns = split_columns(rows)
    preparation = prepare(rows) if preparation is None else preparation
    scaler = preparation.scaler

    def tensors(source: RankingRows) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prepared = transform(source)
        return (
            torch.from_numpy(scaler.apply(prepared[:, dense_columns])),
            torch.from_numpy(prepared[:, sparse_columns].astype(np.int64)),
            torch.from_numpy(source.labels.astype(np.float32)),
        )

    train_dense, train_sparse, train_labels = tensors(rows)
    held_dense, held_sparse, held_labels = tensors(validation)

    built = factory(len(dense_columns), preparation.cardinalities)
    # `wrap` is expected to place the model itself -- Ray's `prepare_model`
    # moves it to the worker's device and then wraps it in DDP -- so the
    # unwrapped path does the placing instead.
    model = built.to(device) if wrap is None else wrap(built)

    positives = preparation.positive_rate
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((1.0 - positives) / max(positives, 1e-6), device=device)
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    generator = torch.Generator().manual_seed(seed)
    best, stale = float("inf"), 0
    best_state = {
        name: value.detach().clone() for name, value in unwrap(model).state_dict().items()
    }

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_labels), generator=generator)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            # BatchNorm cannot take statistics over a single row.
            if len(batch) < 2:
                continue
            loss = loss_fn(
                model(train_dense[batch].to(device), train_sparse[batch].to(device)),
                train_labels[batch].to(device),
            )
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            # Detached before the float: without it the scalar conversion drags
            # a live graph node into a running total that is never backwarded,
            # which torch warns about and which holds the graph alive.
            total, seen = total + float(loss.detach()) * len(batch), seen + len(batch)

        model.eval()
        with torch.no_grad():
            held = float(
                loss_fn(
                    model(held_dense.to(device), held_sparse.to(device)), held_labels.to(device)
                )
            )
        if on_epoch is not None:
            on_epoch(
                {
                    "epoch": float(epoch),
                    "train_loss": total / max(seen, 1),
                    "held_loss": held,
                }
            )
        if held < best:
            best, stale = held, 0
            best_state = {
                name: value.detach().clone() for name, value in unwrap(model).state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                break

    # The unwrapped module, so a DDP-trained ranker and a single-process one are
    # the same object by the time anything scores with them.
    trained = unwrap(model)
    trained.load_state_dict(best_state)
    trained.eval()
    return trained, scaler


def predict(
    model: nn.Module, scaler: Standardiser, rows: RankingRows, device: torch.device
) -> npt.NDArray[np.float64]:
    """Scores for every candidate row."""
    dense_columns, sparse_columns = split_columns(rows)
    prepared = transform(rows)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(scaler.apply(prepared[:, dense_columns])).to(device),
            torch.from_numpy(prepared[:, sparse_columns].astype(np.int64)).to(device),
        )
    scores: npt.NDArray[np.float64] = logits.cpu().numpy().astype(np.float64)
    return scores
