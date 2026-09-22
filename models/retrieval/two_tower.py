"""The two-tower retrieval model.

**No hourly features in the item tower.** ``item_ctr_smoothed`` and
``item_age_hours`` are deliberately absent: an item embedding that moves every
hour is an ANN index that must be rebuilt every hour. Those features belong to
the ranker, which sees the user and the item together anyway. The USER tower is
different -- it is encoded at request time, so dynamic user features are free
there.

Index 0 is ``OOV_IDX`` and every table is sized ``n_items + 1`` (B2). It is also
``padding_idx`` on the learned tables, so a padded history slot holds a vector
that is zero and stays zero.

**Every input block is normalised before it is concatenated.** Both towers glue
together blocks whose natural scales differ by two orders of magnitude and hand
the result to one ``Linear`` initialised at a single scale, so without this the
loud blocks decide the output and the quiet ones are not inputs at all. See
:meth:`TwoTower.item_features` for what that measured.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as f

from models.retrieval.sasrec import SASRec


def _normalise(block: torch.Tensor) -> torch.Tensor:
    """Zero-mean unit-variance across a block's own last dimension, no parameters."""
    normalised: torch.Tensor = f.layer_norm(block, block.shape[-1:])
    return normalised


class Tower(nn.Module):
    """An MLP onto the unit sphere.

    Args:
        in_dim: Width of the concatenated input features.
        hidden: Hidden layer widths.
        out_dim: Embedding width. Both towers must agree.
        dropout: Applied after each hidden activation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...] = (512, 256),
        out_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = in_dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.LayerNorm(size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode and project onto the unit sphere."""
        encoded: torch.Tensor = self.net(features)
        return f.normalize(encoded, p=2, dim=-1)


class TwoTower(nn.Module):
    """User and item encoders into one space, meeting only at the dot product.

    Args:
        content: ``[n_items + 1, content_dim]`` frozen sentence-encoder vectors,
            row 0 reserved for OOV/padding. Copied into an embedding whose
            weight has ``requires_grad=False``: it is an input, not a parameter.
            Recomputing it per epoch would dominate training time, and letting
            gradients into it would quietly make the cached Parquet wrong for
            the next run. Not allocated at all when ``use_content`` is False,
            though the argument is still required -- it carries ``content_dim``,
            which the history projection needs.
        item_category: ``[n_items + 1]`` category index per item.
        item_subcategory: ``[n_items + 1]`` subcategory index per item.
        n_user_feats: Width of the static user feature vector.
        n_categories: Distinct categories, excluding the reserved 0.
        n_subcategories: Distinct subcategories, excluding the reserved 0.
        id_dim: Width of the learned item-ID embedding. Unused when
            ``use_id`` is False, and then not allocated at all, so the
            parameter count difference between the ablation arms is real.
        cat_dim: Width of the learned category embedding.
        subcat_dim: Width of the learned subcategory embedding.
        out_dim: Shared embedding width.
        temperature: Initial softmax temperature. Learned in log space so it
            cannot go negative.
        use_id: False removes item IDs from both towers.
        use_content: False removes the sentence vectors AND the category and
            subcategory embeddings, leaving the item tower with nothing but the
            learned ID.
        use_sequence: True replaces the mean-pool over the click history with
            causal attention. The ITEM side is untouched, so a head-to-head
            differs only in how the sequence is reduced.
        sequence_heads: Attention heads, when ``use_sequence``.
        sequence_blocks: Stacked encoder layers, when ``use_sequence``.
        sequence_dropout: Inside the pooler, when ``use_sequence``.
    """

    item_category: torch.Tensor
    item_subcategory: torch.Tensor
    input_scaling: torch.Tensor

    def __init__(
        self,
        content: torch.Tensor,
        item_category: torch.Tensor,
        item_subcategory: torch.Tensor,
        n_user_feats: int,
        n_categories: int,
        n_subcategories: int,
        id_dim: int = 64,
        cat_dim: int = 8,
        subcat_dim: int = 16,
        out_dim: int = 128,
        temperature: float = 0.05,
        use_id: bool = True,
        use_content: bool = True,
        use_sequence: bool = False,
        sequence_heads: int = 2,
        sequence_blocks: int = 2,
        sequence_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        n_rows, content_dim = content.shape
        if item_category.shape[0] != n_rows or item_subcategory.shape[0] != n_rows:
            raise ValueError(
                "content, item_category and item_subcategory must agree on row "
                f"count; got {n_rows}, {item_category.shape[0]}, "
                f"{item_subcategory.shape[0]}. Off by one here is an IndexError "
                "on the highest-numbered item, which is the last one anybody "
                "tests by hand."
            )

        if not (use_id or use_content):
            raise ValueError(
                "use_id and use_content cannot both be False: the item tower "
                "would have no input and the user tower nothing to pool."
            )

        self.use_id = use_id
        self.use_content = use_content
        self.n_items = n_rows - 1  # index 0 is reserved for OOV index

        # The indices are registered whatever the arm, so precompute_items has a
        # device to read and the ID-only arm can still be told what it dropped.
        self.register_buffer("item_category", item_category.long())
        self.register_buffer("item_subcategory", item_subcategory.long())

        # Nothing reads this. It exists so that a checkpoint written BEFORE the
        # blocks were normalised fails `load_state_dict(strict=True)` instead of
        # loading cleanly. Bump it whenever the input layout changes in a way old
        # weights cannot survive.
        self.register_buffer("input_scaling", torch.tensor(1))

        # self.content = nn.Embedding.from_pretrained(  # type: ignore[no-untyped-call]
        #     content, freeze=True, padding_idx=0
        # )
        self.content: nn.Embedding | None = None
        self.category: nn.Embedding | None = None
        self.subcategory: nn.Embedding | None = None
        if use_content:
            self.content = nn.Embedding(n_rows, content_dim, padding_idx=0)
            with torch.no_grad():
                self.content.weight.copy_(content)
                self.content.weight[0].zero_()
            self.content.weight.requires_grad_(False)

            # Create learnable embeddings for category and subcategory
            self.category = nn.Embedding(n_categories + 1, cat_dim, padding_idx=0)
            self.subcategory = nn.Embedding(n_subcategories + 1, subcat_dim, padding_idx=0)

        # If flagged, create a learnable item_embedding instead of pooling the history
        self.item_id_emb: nn.Embedding | None = None
        self.history_proj: nn.Linear | None = None
        if use_id:
            self.item_id_emb = nn.Embedding(n_rows, id_dim, padding_idx=0)
            nn.init.normal_(self.item_id_emb.weight, std=id_dim**-0.5)
            with torch.no_grad():
                self.item_id_emb.weight[0].zero_()
        else:
            self.history_proj = nn.Linear(content_dim, id_dim, bias=False)

        item_in = (content_dim + cat_dim + subcat_dim if use_content else 0) + (
            id_dim if use_id else 0
        )
        history_in = id_dim

        self.sequence: SASRec | None = None
        if use_sequence:
            self.sequence = SASRec(
                history_in,
                n_heads=sequence_heads,
                n_blocks=sequence_blocks,
                dropout=sequence_dropout,
            )

        self.user_norm = nn.LayerNorm(n_user_feats)
        self.user_tower = Tower(n_user_feats + history_in, out_dim=out_dim)
        self.item_tower = Tower(item_in, out_dim=out_dim)
        self.log_temp = nn.Parameter(torch.tensor(float(temperature)).log())

    @property
    def temperature(self) -> torch.Tensor:
        """The learned temperature, always positive."""
        return self.log_temp.exp()

    def item_features(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Concatenated item inputs, each block normalised to a common scale."""
        parts: list[torch.Tensor] = []
        content, category, subcategory = self.content, self.category, self.subcategory
        if content is not None and category is not None and subcategory is not None:
            parts += [
                _normalise(content(item_ids)),
                _normalise(category(self.item_category[item_ids])),
                _normalise(subcategory(self.item_subcategory[item_ids])),
            ]
        if self.item_id_emb is not None:
            parts.append(_normalise(self.item_id_emb(item_ids)))
        return torch.cat(parts, dim=-1)

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Embed items by index alone.

        Args:
            item_ids: ``[...]`` of item indices.

        Returns:
            ``[..., out_dim]`` unit-norm embeddings.
        """
        embedded: torch.Tensor = self.item_tower(self.item_features(item_ids))
        return embedded

    def history_vectors(self, history_ids: torch.Tensor) -> torch.Tensor:
        """What the user tower pools: ID embeddings, or content when IDs are off."""
        table = self.item_id_emb if self.item_id_emb is not None else self.content
        if table is None:  # unreachable: the constructor refuses both arms off
            raise RuntimeError("no item table to pool")
        vectors: torch.Tensor = table(history_ids)
        return vectors

    def encode_user(
        self,
        user_feats: torch.Tensor,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed a request from its static features and its click history.

        Args:
            user_feats: ``[B, n_user_feats]``.
            history_ids: ``[B, L]`` item indices, 0 where padded.
            history_mask: ``[B, L]`` 1 for a real entry.

        Returns:
            ``[B, out_dim]`` unit-norm embeddings.
        """
        vectors = self.history_vectors(history_ids)
        if self.history_proj is not None:
            projected: torch.Tensor = self.history_proj(vectors)
            vectors = projected

        if self.sequence is not None:
            # SASRec pooling
            pooled = self.sequence(vectors, history_mask)
        else:
            # mean pool
            mask = history_mask.unsqueeze(-1).to(vectors.dtype)
            pooled = (vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        embedded: torch.Tensor = self.user_tower(
            torch.cat([self.user_norm(user_feats), _normalise(pooled)], dim=-1)
        )
        return embedded

    def forward(
        self,
        user_feats: torch.Tensor,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Both towers for one batch, and the temperature.

        The temperature is RETURNED rather than read off the module because
        ``log_temp`` is consumed by the loss, not by either tower: a forward
        returning only embeddings leaves it with no gradient, and DDP waits for
        every tracked parameter before it reduces anything.

        Args:
            user_feats: ``[B, n_user_feats]``.
            history_ids: ``[B, L]``.
            history_mask: ``[B, L]``.
            item_ids: ``[N]`` every candidate column -- positives and negatives
                together, so the item tower runs once per step rather than twice.

        Returns:
            ``(user_emb [B, out_dim], item_emb [N, out_dim], temperature)``.
        """
        return (
            self.encode_user(user_feats, history_ids, history_mask),
            self.encode_item(item_ids),
            self.temperature,
        )

    @torch.no_grad()
    def precompute_items(self, batch_size: int = 4096) -> torch.Tensor:
        """Every item's embedding

        Row 0 is included and is whatever the towers make of the reserved index.
        It is a bucket, not an entity -- drop it before indexing.

        Returns:
            ``[n_items + 1, out_dim]``.
        """
        was_training = self.training
        self.eval()
        try:
            device = self.item_category.device
            ids = torch.arange(self.n_items + 1, device=device)
            chunks = [
                self.encode_item(ids[i : i + batch_size]) for i in range(0, len(ids), batch_size)
            ]
            return torch.cat(chunks, dim=0)
        finally:
            self.train(was_training)
