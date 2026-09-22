"""The training-loop records: the logQ counters and the per-row hit table."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.retrieval.distributed import all_gather_detached
from models.retrieval.sampling import StreamingLogQ, uniform_log_q


@dataclass
class Counters:
    """One frequency estimator per negative source, and the ablation switch.

    The correction assumes every softmax column was drawn from the distribution
    ``q`` describes, and the pool mixes two observed distributions -- positives
    drawn by popularity, slate negatives drawn by exposure. Uniform draws need
    no estimator; their probability is exact.

    Attributes:
        positives: Frequency of the clicked items.
        slate: Frequency of the impression negatives.
        corrected: False makes every ``log_q`` zero, which is G2's gate --
            "train with and without the correction". A per-row constant is
            invisible to softmax, so zeros are exactly "no correction". Living
            here rather than as a flag threaded through ``fit``, ``run_epoch``
            and ``train_step`` keeps the thing being ablated in one place.
    """

    positives: StreamingLogQ
    slate: StreamingLogQ
    corrected: bool = True

    def to(self, device: torch.device) -> Counters:
        return Counters(self.positives.to(device), self.slate.to(device), self.corrected)

    def log_q_for(
        self,
        item_ids: torch.Tensor,
        negatives: torch.Tensor,
        is_slate: torch.Tensor,
        n_items: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The correction for the positive and negative columns.

        Call BEFORE :meth:`observe` for a given batch, or it conditions its own
        correction on its own labels.
        """
        if not self.corrected:
            return (
                torch.zeros(item_ids.shape, device=device),
                torch.zeros(negatives.shape, device=device),
            )
        return (
            self.positives.log_q(item_ids),
            torch.where(
                is_slate,
                self.slate.log_q(negatives),
                uniform_log_q((negatives.numel(),), n_items, device),
            ),
        )

    def observe(
        self, item_ids: torch.Tensor, negatives: torch.Tensor, is_slate: torch.Tensor
    ) -> None:
        """Count what appeared as a column, across every rank.

        The GATHERED ids, because the distribution being modelled is the whole
        pool -- and every rank sees the identical gathered set, so the counters
        stay in step without a collective of their own.

        Gather the fixed-width tensors and mask afterwards: the masked subset has
        a different length per rank, and a ragged gather either hangs or
        misaligns.
        """
        if not self.corrected:
            return
        self.positives.update(all_gather_detached(item_ids))
        drawn = all_gather_detached(negatives)
        self.slate.update(drawn[all_gather_detached(is_slate.long()).bool()])

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {"positives": self.positives.state_dict(), "slate": self.slate.state_dict()}

    def load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.positives.load_state_dict(state["positives"])
        self.slate.load_state_dict(state["slate"])


@dataclass(frozen=True)
class Hits:
    """One row per scored request, on CPU.

    Attributes:
        hit: ``[R]`` bool, the clicked item was in the top k.
        item_ids: ``[R]`` the clicked item, so a result can be grouped by
            anything item-indexed -- popularity, category, age.
        user_ids: ``[R]`` who asked, so a paired comparison can pair on the user.
    """

    hit: torch.Tensor
    item_ids: torch.Tensor
    user_ids: torch.Tensor
