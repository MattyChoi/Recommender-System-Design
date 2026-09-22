"""How much a set of embeddings varies, measured one way everywhere.

A retriever whose embeddings occupy a narrow cone returns nearly the same list
to everyone, and recall alone will not say so: a two-tower serving 774 of
65,238 articles scored 0.3861, within noise of a popularity baseline. These
three statistics are what separates the two, and they belong beside the metrics
rather than in a diagnostic script, because they are reported per epoch.

``||mean||`` is 1.0 when every row is identical and 0.0 when they cancel. The
most legible of the three.

``mean pairwise cosine`` is the same fact from the other side and is NOT
independent evidence: for unit rows ``||mean||^2 ~= mean pairwise cosine``.
Both are reported because different readers trust different ones, not because
their agreeing confirms anything.

``effective rank`` is the participation ratio of the singular spectrum of the
CENTRED rows, so the shared direction is already removed and this describes the
variation alone. It is the one that catches a tower using three of its
dimensions while still looking spread out. For scale: ALS at rank 32 on this
corpus reaches 25.9, and the two-tower reached 8.8 of 128.

One implementation, because the numbers get compared across models -- a
two-tower embedding against an ALS factor, a trained model against an untrained
one -- and two that drift make those comparisons quietly meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as f


@dataclass(frozen=True)
class Spread:
    """The three statistics for one set of embeddings."""

    centroid: float
    cosine: float
    effective_rank: float
    width: int
    rows: int


def spread_of(embeddings: torch.Tensor, sample: int = 2000, seed: int = 0) -> Spread:
    """Measure a set of embeddings.

    Rows are L2-normalised on the way in rather than assumed: raw content
    vectors and ALS factors are not unit-norm, and reporting dot products as
    cosines for one input and true cosines for another would compare two
    different quantities within one table.

    Args:
        embeddings: ``[N, D]``. Anything with rows to compare.
        sample: Rows drawn for the pairwise and spectral figures, which are
            quadratic and cubic in the row count. The centroid uses every row,
            being linear and cheap.
        seed: Fixed, so two runs are comparable.
    """
    rows, width = embeddings.shape
    normalised = f.normalize(embeddings.detach().float(), p=2, dim=-1)

    generator = torch.Generator().manual_seed(seed)
    index = torch.randperm(rows, generator=generator)[: min(sample, rows)]
    taken = normalised[index]

    similarity = taken @ taken.T
    off_diagonal = ~torch.eye(len(taken), dtype=torch.bool, device=taken.device)

    # Participation ratio: (sum L)^2 / sum L^2 over the covariance eigenvalues.
    # Equals width when variance is spread evenly and 1 when it lies on a line.
    spectrum = torch.linalg.svdvals(taken - taken.mean(dim=0, keepdim=True)) ** 2

    return Spread(
        centroid=float(normalised.mean(dim=0).norm()),
        cosine=float(similarity[off_diagonal].mean()) if len(taken) > 1 else float("nan"),
        effective_rank=float(spectrum.sum() ** 2 / (spectrum**2).sum()),
        width=width,
        rows=rows,
    )


def render_spread(name: str, spread: Spread) -> None:
    print(f"{name}  ({spread.rows:,} rows)")
    print(f"  ||mean embedding||    : {spread.centroid:.4f}   (1.0 = every row identical)")
    print(f"  mean pairwise cosine  : {spread.cosine:+.4f}")
    print(f"  effective rank        : {spread.effective_rank:.1f} of {spread.width}")
