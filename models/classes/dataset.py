"""The tensor records ``models.retrieval.dataset`` loads the gold tables into."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch


@dataclass(frozen=True)
class ItemTables:
    """Item-indexed inputs, sized ``n_items + 1`` with row 0 reserved.

    Attributes:
        content: ``[n_items + 1, dim]``, row 0 all-zero. ``TwoTower`` copies it
            into a frozen embedding and zeroes row 0 on the way in.
        category: ``[n_items + 1]`` category index per item.
        subcategory: ``[n_items + 1]`` subcategory index per item.
        n_categories: Distinct categories, excluding the reserved 0.
        n_subcategories: Distinct subcategories, excluding the reserved 0.
    """

    content: torch.Tensor
    category: torch.Tensor
    subcategory: torch.Tensor
    n_categories: int
    n_subcategories: int


@dataclass(frozen=True)
class SplitTensors:
    """One split, ready for a DataLoader.

    Attributes:
        user_feats: ``[R, F]`` static + context features.
        history_ids: ``[R, L]`` item indices, 0 where padded.
        history_mask: ``[R, L]`` 1 for a real entry.
        item_ids: ``[R]`` the clicked item.
        impression_ids: ``[R]`` kept so slates can be regrouped at evaluation.
        user_ids: ``[R]`` who made the request. Carried so a paired comparison
            can pair on the USER: impressions from one user are correlated, and
            pairing on them instead inflates the effective sample size by
            roughly the impressions per user.
        neg_ids: ``[R, K]`` items shown in the same slate and not clicked, 0
            where padded. Hard negatives.
        neg_mask: ``[R, K]`` 1 for a real negative. All-zero for a request whose
            every item was clicked, and for one whose slate held fewer than K
            others.
    """

    user_feats: torch.Tensor
    history_ids: torch.Tensor
    history_mask: torch.Tensor
    item_ids: torch.Tensor
    impression_ids: torch.Tensor
    user_ids: torch.Tensor
    neg_ids: torch.Tensor
    neg_mask: torch.Tensor

    def head(self, rows: int) -> SplitTensors:
        """The first ``rows`` requests, every field cut together.

        A PREFIX, not a sample. These rows arrive in time order, so a prefix is
        a shorter window ending earlier -- coherent, if not the window anything
        was measured on. A random subsample would instead redraw the popularity
        distribution the logQ counters estimate, which is the one axis a smoke
        test must not quietly change.

        Cutting every field by iterating ``fields()`` rather than naming eight
        of them: a ninth column added later is included automatically, where a
        hand-written list would silently leave it full-length and misaligned
        against the rest.
        """
        if rows >= len(self.item_ids):
            return self
        return replace(
            self, **{field.name: getattr(self, field.name)[:rows] for field in fields(self)}
        )
