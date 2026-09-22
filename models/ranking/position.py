"""Position debiasing: the module, and why its ablation is deferred rather than run.

A user clicks the top of a list more often than the bottom, whatever is in it.
A ranker fitted on that log learns to predict position as much as quality, and
then reproduces the incumbent ordering. The standard correction trains a
shallow tower whose only input is the displayed position, adds its output to the
logit during training, and drops it at serving -- so the main tower has to
explain the click without the position, because at inference there is no
position to lean on.

**The ablation cannot be run on this corpus, and running it would look like it
worked.** Microsoft documents that MIND's impression lists are presented in
shuffled order, which is why the exploded index is named ``slot`` throughout
this project rather than ``position``: it is the order rows came out of a list,
carrying no display-order signal. Fit a position tower on it and it fits noise.
Worse, the usual verification -- plot average predicted CTR by position and
check it is flat -- comes out flat **by construction**, because the input was
noise to begin with. A correction that cannot fail its own test is not evidence.

So the module is kept and left unwired. Once the serving layer records the real
slot it chose, the same code trains on a log this project generated itself, and
the ablation becomes a measurement of a bias we created and then removed --
which is a stronger claim than correcting someone else's.
"""

from __future__ import annotations

import torch
from torch import nn

# Index 0 means "no position": an inference-time request, or a training row
# whose position was not recorded. Kept as a real embedding row rather than a
# sentinel so the two cases are expressible and are the SAME case.
UNKNOWN_POSITION = 0


class PositionDebiasedRanker(nn.Module):
    """Wraps a ranker with an additive, training-only position term.

    Args:
        main: The ranker whose logit is being corrected.
        n_positions: Largest slot the serving layer can report.

    Note:
        The position tower is initialised to **zero**, not randomly. At step
        zero the wrapper must be exactly the model it wraps, or the correction
        starts by injecting noise into the logit and the main tower spends its
        first epochs undoing it.
    """

    def __init__(self, main: nn.Module, n_positions: int = 50) -> None:
        super().__init__()
        self.main = main
        self.position_tower = nn.Embedding(n_positions + 1, 1, padding_idx=UNKNOWN_POSITION)
        nn.init.zeros_(self.position_tower.weight)

    def forward(self, features: torch.Tensor, position: torch.Tensor | None = None) -> torch.Tensor:
        """Score, with the position term added only while training.

        Raises:
            ValueError: If training without positions. Silently skipping the
                term would train an ordinary ranker while every name in the
                stack claimed it was debiased -- and nothing downstream could
                tell the difference.
        """
        logit: torch.Tensor = self.main(features)
        if not self.training:
            return logit
        if position is None:
            raise ValueError(
                "training a position-debiased ranker without positions; pass "
                f"{UNKNOWN_POSITION} for rows whose slot was not recorded"
            )
        corrected: torch.Tensor = logit + self.position_tower(position).squeeze(-1)
        return corrected
