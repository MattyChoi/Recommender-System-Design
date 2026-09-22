"""The ranker's categorical block, backed by TorchRec instead of ``nn.Embedding``.

**This is scaffolding, and saying so is the point.** TorchRec exists to shard
embedding tables that do not fit on one device. This ranker's categoricals are
18 categories and 121 subcategories; at 16 dimensions and fp32 that is

    (18 + 1 + 121 + 1) x 16 x 4 bytes = 9.0 KB

for the pair. Nothing about 9 KB needs a planner, and a report claiming
otherwise would be measuring the framework rather than the problem. What the
module buys is that the wiring exists and is tested, so a categorical with
millions of levels -- an item id, a publisher, a user id -- is a config change
rather than a rewrite.

**It also reverses a conclusion from the two-tower, for a reason worth stating.**
``EmbeddingBagCollection.forward`` is all-or-nothing across its tables: it
indexes the input by every table's feature name, so a batch missing one raises.
That disqualified it for the two-tower, whose ``encode_item`` must be a function
of the item id ALONE -- that is what lets the Part J index be precomputed. **A
ranker has no such call.** It scores (user, item) pairs and therefore always has
every categorical in hand at once, which is exactly the access pattern a
collection assumes.

**No end-to-end comparison between the backends is reported, and that is
deliberate.** They cannot be seed-matched from the command line: even with the
initialiser forced to match, the two draw a different number of values from the
global RNG, so every layer built after the block -- the cross layers, the deep
tower, the head -- gets different weights. An arm run under each backend differs
by its whole initialisation, and a bootstrap over users cannot see that, because
it resamples users. Parity is claimed at the unit level, where it is checkable,
and nowhere else.

**One behaviour genuinely differs**, measured rather than assumed. ``FeatureBlock``
builds ``nn.Embedding(padding_idx=0)``, which pins row 0 to zero AND takes no
gradient on it, so the unknown-category vector is a permanent zero. An
``EmbeddingBagCollection`` has no ``padding_idx``. Row 0 is zeroed here at
construction so the two backends start identical, but it is an ordinary trainable
row afterwards and will drift: unknown becomes a *learned* vector rather than an
absent one. That is arguably the better behaviour and it is certainly a different
one, so a checkpoint is not portable between the backends after training.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torchrec import EmbeddingBagCollection, EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from models.ranking.torch_fit import FeatureEmbedding


def normal_init(weight: torch.Tensor) -> torch.Tensor:
    """``N(0, 1)``, because that is what ``nn.Embedding`` does.

    TorchRec's default is uniform on ``+/- 1/sqrt(num_embeddings)``, which for a
    2,000-row table is a standard deviation of **0.013 against 1.0** -- the
    embeddings start at roughly one seventy-seventh the scale. That is not a
    tuning difference. With a standardised dense block beside them, categorical
    features arrive effectively switched off and the model spends its budget
    learning to turn them back on.

    Both defaults are defensible on their own terms -- a small init is the
    sensible choice for a table with millions of rows feeding a sum-pooled bag.
    It is matched here because this backend's entire claim is that swapping it
    changes nothing, and starting somewhere else is a change.
    """
    return torch.nn.init.normal_(weight)


def feature_name(index: int) -> str:
    """The KJT key for categorical column ``index``.

    Positional, because the columns arrive as a matrix whose meaning is its
    order. The names are internal to this module and never persisted.
    """
    return f"cat_{index}"


class TorchRecFeatureBlock(FeatureEmbedding):
    """Dense columns and TorchRec-embedded categoricals, concatenated.

    Interchangeable with :class:`~models.ranking.torch_fit.FeatureBlock`: same
    constructor, same ``width``, same ``forward`` signature and, at
    construction, the same numbers given the same weights.

    Args:
        n_dense: Numeric feature count.
        cardinalities: One per categorical column, EXCLUDING the reserved row.
        emb_dim: Width of each categorical embedding.
    """

    def __init__(self, n_dense: int, cardinalities: Sequence[int], emb_dim: int = 16) -> None:
        super().__init__()
        self.names = [feature_name(index) for index in range(len(cardinalities))]
        self.collection = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name=f"table_{index}",
                    embedding_dim=emb_dim,
                    # +1 for the reserved unknown row at 0, matching the dense
                    # backend. Sizing at the cardinality instead is an
                    # IndexError on the highest-numbered level.
                    num_embeddings=size + 1,
                    feature_names=[feature_name(index)],
                    init_fn=normal_init,
                )
                for index, size in enumerate(cardinalities)
            ],
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            for bag in self.collection.embedding_bags.values():
                bag.weight[0].zero_()
        self.width = n_dense + emb_dim * len(cardinalities)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        rows, columns = sparse.shape
        # Column-major: every row's value for feature 0, then for feature 1.
        # A KJT is a flat value buffer plus per-(feature, sample) lengths, and
        # feature is the OUTER dimension -- passing the matrix row-major gives a
        # tensor of the right shape holding the wrong lookups, which trains.
        jagged = KeyedJaggedTensor.from_lengths_sync(
            keys=self.names,
            values=sparse.t().reshape(-1),
            # One value per bag, so each "bag" is a plain lookup and the
            # collection's pooling never combines two ids.
            lengths=torch.ones(columns * rows, dtype=torch.long, device=sparse.device),
        )
        # `to_dict` and index by name rather than taking `values()` whole. The
        # probe showed the concatenated columns come back in declared order, but
        # that is an observation about one build; indexing by the names this
        # module declared is true by construction.
        embedded = self.collection(jagged).to_dict()
        return torch.cat([dense, *(embedded[name] for name in self.names)], dim=-1)
