from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from evaluation.offline.stats import PairedResult

RESULTS = Path("evaluation/results/sources")


@dataclass(frozen=True)
class Blend:
    """One allocation of the slot budget, and what it recalls.

    Attributes:
        label: How the slots were divided, spelled out -- the allocation IS the
            condition, so it travels with the number.
        recall: Recall@k of the union under that allocation.
        pool: Mean distinct candidates put forward per request.
    """

    label: str
    recall: float
    pool: float


@dataclass(frozen=True)
class Contribution:
    """Contribution of one source to the blend, and what it loses if dropped.

    Attributes:
        source: Which source.
        share: Fraction of the union's (request, item) pairs this source offers.
        unique: Fraction of the union's pairs **only** this source offers.
        alone: Recall@k with nothing else in the blend.
        found_only_here: Clicked items this source found and no other did, as a
            share of all requests. The number that decides whether a source
            earns its slot -- a source can contribute 30% of the candidates and
            none of the answers.
        loss: Recall lost by dropping it from the quota blend, paired per user.
    """

    source: str
    share: float
    unique: float
    alone: float
    found_only_here: float
    loss: PairedResult | None


@dataclass(frozen=True)
class Retrieved:
    """One source's candidates for every validation request.

    Attributes:
        source: Which source produced this.
        top: ``[R, k]`` item indices, best first, **0 where the source ran out
            of candidates**. Never padded with filler.
        item_ids: ``[R]`` the item actually clicked.
        user_ids: ``[R]`` who made the request, so a comparison can pair on the
            user rather than on the row.
        k: The cutoff.
    """

    source: str
    top: torch.Tensor
    item_ids: torch.Tensor
    user_ids: torch.Tensor
    k: int

    @property
    def hit(self) -> torch.Tensor:
        """``[R]`` whether the clicked item is anywhere in the top k."""
        found: torch.Tensor = (self.top == self.item_ids.unsqueeze(1)).any(dim=1)
        return found

    @property
    def pool(self) -> torch.Tensor:
        """``[R]`` real candidates returned, which is <= k for a short source."""
        sizes: torch.Tensor = (self.top > 0).sum(dim=1)
        return sizes

    @property
    def recall(self) -> float:
        return float(self.hit.float().mean()) if len(self.hit) else float("nan")

    @property
    def reach(self) -> float:
        """Share of requests the source could answer at all."""
        return float((self.pool > 0).float().mean()) if len(self.top) else float("nan")

    def save(self, directory: Path = RESULTS) -> Path:
        """Keep the candidates, not just the hit -- the blend needs the lists."""
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{self.source}.npz"
        np.savez_compressed(
            destination,
            source=np.asarray(self.source),
            top=self.top.numpy(),
            item_ids=self.item_ids.numpy(),
            user_ids=self.user_ids.numpy(),
            k=np.asarray(self.k),
        )
        return destination


def load_retrieved(source: str, directory: Path = RESULTS) -> Retrieved:
    """Read one source's candidates back."""
    with np.load(directory / f"{source}.npz", allow_pickle=False) as data:
        return Retrieved(
            source=str(data["source"]),
            top=torch.from_numpy(data["top"]),
            item_ids=torch.from_numpy(data["item_ids"]),
            user_ids=torch.from_numpy(data["user_ids"]),
            k=int(data["k"]),
        )
