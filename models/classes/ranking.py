from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


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
