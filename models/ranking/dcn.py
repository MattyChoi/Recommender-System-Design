"""DCN v2: explicit feature crosses beside a deep tower, as the neural contender.

The boring model was built first on purpose, and this has to beat it on a
measured number before it earns a second serving stack. What it brings that a
tree does not is *explicit* bounded-degree interaction: a cross layer computes
``x0 * (W xl + b) + xl``, so after three of them every output term is a product
of at most four input features, learned rather than enumerated. A tree can
represent interactions too, but only as axis-aligned boxes, and it needs a
split per region.

**Two things differ between this and the tree, not one.** Architecture, and the
objective: this is fitted pointwise with binary cross-entropy, which is what a
CTR model normally does, while the tree is fitted listwise on within-request
comparisons. So a difference in NDCG cannot be attributed to the architecture
alone. Isolating that would need a listwise loss here, and the confound is
stated rather than quietly carried.

Input preparation and the training loop are shared with the other neural
ranker -- see ``torch_fit`` -- so the two differ in architecture and in nothing
else.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from models.ranking.torch_fit import Block, FeatureBlock


class CrossLayer(nn.Module):
    """``x_{l+1} = x0 * V(U(xl)) + xl`` -- the low-rank variant.

    Factoring the weight into ``dim -> rank -> dim`` cuts parameters about
    fourfold at ``rank = dim / 4``. On a handful of tabular features the
    full-rank version has nothing to spend the difference on.
    """

    def __init__(self, dim: int, rank: int | None = None) -> None:
        super().__init__()
        width = rank or max(dim // 4, 4)
        self.down = nn.Linear(dim, width, bias=False)
        self.up = nn.Linear(width, dim, bias=True)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        crossed: torch.Tensor = x0 * self.up(self.down(xl)) + xl
        return crossed


class DCNv2(nn.Module):
    """Cross tower and deep tower in PARALLEL, concatenated before the head.

    Parallel rather than stacked: both read the same input and their outputs are
    joined. The stacked variant puts the deep tower on top of the cross tower
    and is the other option in the paper; knowing which one is built matters,
    because they are not interchangeable and the difference is invisible in a
    metric.

    Args:
        n_dense: Numeric feature count.
        cardinalities: One per categorical column.
        emb_dim: Width of each categorical embedding.
        n_cross: Cross layers. Depth d gives interactions of order up to d + 1.
        hidden: Deep tower widths.
        block: Categorical-embedding backend. Swapping in the TorchRec one must
            change nothing above this line, which is the whole test of whether
            the abstraction is real.
    """

    def __init__(
        self,
        n_dense: int,
        cardinalities: Sequence[int],
        emb_dim: int = 16,
        n_cross: int = 3,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        block: Block = FeatureBlock,
    ) -> None:
        super().__init__()
        self.features = block(n_dense, cardinalities, emb_dim)
        width = self.features.width

        self.cross = nn.ModuleList([CrossLayer(width) for _ in range(n_cross)])
        layers: list[nn.Module] = []
        deep = width
        for size in hidden:
            layers += [nn.Linear(deep, size), nn.BatchNorm1d(size), nn.ReLU(), nn.Dropout(dropout)]
            deep = size
        self.deep = nn.Sequential(*layers)
        self.head = nn.Linear(width + deep, 1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        x0 = self.features(dense, sparse)

        crossed = x0
        for layer in self.cross:
            crossed = layer(x0, crossed)

        joint = torch.cat([crossed, self.deep(x0)], dim=-1)
        logit: torch.Tensor = self.head(joint).squeeze(-1)
        return logit
