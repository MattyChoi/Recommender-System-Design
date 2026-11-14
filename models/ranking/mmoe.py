"""Multi-gate Mixture-of-Experts, built with one task on purpose.

A click-only ranker learns to produce clickbait, so real systems optimise
several objectives at once and let a shared set of experts serve all of them,
with per-task gates deciding how much of each expert each task gets.

**This corpus has one label.** MIND records click or no-click: no dwell, no
cart, no purchase, no explicit rating. So the structure is here and only one
head is wired, deliberately, and it is worth being exact about what that costs:

* the **mixture-of-experts** half still works. The gate is a softmax over
  experts computed from the input, so routing is input-dependent and different
  requests genuinely use different experts.
* the **multi-gate** half is inert. With one task there is nothing to share a
  representation *with*, and the cross-task regularisation that is the whole
  argument for MMoE cannot happen.

So a single-task MMoE is a gated mixture, not a multi-task model, and it should
not be reported as one.

**The rule for adding a second head.** A task must come from an OBSERVED signal.
The tempting constructed label here -- "clicked, and in a category this user has
clicked before" -- is computable from features the model already receives, so a
head predicting it adds no supervision. It is a feature wearing a task's
clothes, and multi-task learning works precisely because the second signal is
information the first did not contain. Fabricating dwell time is worse still.
When a real second signal exists, widen ``tasks`` and the rest of this works.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from models.ranking.torch_fit import Block, FeatureBlock

# The task this corpus can actually supervise. Named rather than counted, so a
# second entry has to be justified by whoever adds it.
TASKS = ("click",)


@dataclass(frozen=True)
class GateUsage:
    """How the gates spread their weight across experts.

    **Two entropies, and one of them alone tells you nothing.** The entropy OF
    THE MEAN and the MEAN OF THE entropies answer different questions, and a
    mixture that routes every request sharply to a different expert looks
    identical, on the first, to a mixture that routes every request uniformly
    to all of them.

    ======================  ====================  ==========================
    entropy of the mean     mean per-row entropy  what the mixture is doing
    ======================  ====================  ==========================
    high                    **low**               real per-request routing
    high                    high                  no routing -- an ensemble
    low                     low                   collapsed onto one expert
    ======================  ====================  ==========================

    Attributes:
        weights: Mean gate weight per expert, over the rows measured.
        entropy: Shannon entropy of those means, in nats. Detects GLOBAL
            collapse -- one expert winning for everybody.
        per_row: Mean of each row's own entropy. Detects whether any individual
            request commits to a subset of experts.
        ceiling: ``log(n_experts)`` -- the entropy of perfectly even routing.
    """

    weights: list[float]
    entropy: float
    per_row: float
    ceiling: float

    @property
    def used(self) -> float:
        """Share of the available routing diversity in use across requests."""
        return self.entropy / self.ceiling if self.ceiling else float("nan")

    @property
    def specialised(self) -> float:
        """How far a typical request is from using every expert equally.

        0.0 means the average request spreads itself evenly over all experts,
        so the gate is not routing -- it is averaging, and the architecture has
        bought nothing over a single wider expert. 1.0 means each request picks
        exactly one.
        """
        return 1.0 - (self.per_row / self.ceiling) if self.ceiling else float("nan")


def gate_usage(model: MMoE, dense: torch.Tensor, sparse: torch.Tensor, task: int = 0) -> GateUsage:
    """Measure whether the mixture routes, and whether it routes per request."""
    with torch.no_grad():
        rows = model.gate_weights(dense, sparse)[:, task]

    def shannon(distribution: torch.Tensor) -> torch.Tensor:
        return -(distribution * torch.log(distribution.clamp(min=1e-12))).sum(dim=-1)

    averaged = rows.mean(dim=0)
    averaged = averaged / averaged.sum()

    return GateUsage(
        weights=[float(value) for value in averaged],
        entropy=float(shannon(averaged)),
        # Per row FIRST, then averaged. Averaging the rows first and taking one
        # entropy is the other statistic entirely, and it is the one that
        # cannot see a gate that never commits to anything.
        per_row=float(shannon(rows).mean()),
        ceiling=math.log(rows.shape[-1]),
    )


def combine(scores: torch.Tensor, weights: Sequence[float]) -> torch.Tensor:
    """Fold several task scores into one ranking score.

    ``score = p1^w1 * p2^w2 * ...`` in log space, which is a weighted sum of
    log-probabilities and is numerically better behaved than the product.

    Each head is an INDEPENDENT binary prediction, so the per-task probability
    is ``sigmoid(logit)`` and the log of it is ``logsigmoid``. A softmax across
    the task dimension would be wrong in a way that still runs: it would make
    the tasks compete for one unit of probability, so a request scoring high on
    everything would be normalised back down to the same total as one scoring
    high on nothing.

    **The weights are a product decision and cannot be chosen offline.** They
    encode how much a click is worth against a completion against a complaint,
    and no held-out set contains that exchange rate -- the data only says what
    happened under the policy that produced it. They are set by an A/B test, or
    learned against a north-star metric with a bandit over the weight simplex.
    Anything fitted offline is fitting the incumbent policy's preferences.

    Raises:
        ValueError: If the weight count does not match the score columns.
    """
    if scores.ndim == 1:
        scores = scores.unsqueeze(-1)
    if scores.shape[-1] != len(weights):
        raise ValueError(f"{len(weights)} weights for {scores.shape[-1]} task scores")

    coefficients = torch.tensor(weights, dtype=scores.dtype, device=scores.device)
    combined: torch.Tensor = (nn.functional.logsigmoid(scores) * coefficients).sum(dim=-1)
    return combined


class MMoE(nn.Module):
    """Shared experts, one softmax gate per task, one tower per task.

    Args:
        n_dense: Numeric feature count.
        cardinalities: One per categorical column.
        tasks: Task names. Length decides how many gates and towers exist.
        n_experts: Shared expert networks.
        expert_dim: Width of each expert's output.
        emb_dim: Width of each categorical embedding.
        block: Categorical-embedding backend. Swapping in the TorchRec one must
            change nothing above this line.

    Raises:
        ValueError: If ``tasks`` is empty, or if there are fewer experts than
            tasks -- a configuration in which the experts cannot specialise and
            the gates have nothing to choose between.
    """

    def __init__(
        self,
        n_dense: int,
        cardinalities: Sequence[int],
        tasks: Sequence[str] = TASKS,
        n_experts: int = 8,
        expert_dim: int = 128,
        emb_dim: int = 16,
        dropout: float = 0.2,
        block: Block = FeatureBlock,
    ) -> None:
        super().__init__()
        if not tasks:
            raise ValueError("MMoE needs at least one task")
        if n_experts < len(tasks):
            raise ValueError(f"{n_experts} experts cannot serve {len(tasks)} tasks")

        self.tasks = tuple(tasks)
        self.features = block(n_dense, cardinalities, emb_dim)
        width = self.features.width

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, expert_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim, expert_dim),
                    nn.ReLU(),
                )
                for _ in range(n_experts)
            ]
        )
        self.gates = nn.ModuleList([nn.Linear(width, n_experts) for _ in self.tasks])
        self.towers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(expert_dim, 64), nn.ReLU(), nn.Linear(64, 1))
                for _ in self.tasks
            ]
        )

    def gate_weights(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        """``[B, n_tasks, n_experts]`` routing weights, for the collapse check."""
        shared = self.features(dense, sparse)
        return torch.stack([torch.softmax(gate(shared), dim=-1) for gate in self.gates], dim=1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        """``[B]`` for a single task, ``[B, n_tasks]`` for several.

        The single-task case is squeezed so this drops into a loop written for
        a one-output model. A shape that changes with configuration is a trap,
        so it is stated here rather than discovered at the call site.
        """
        shared = self.features(dense, sparse)
        stacked = torch.stack([expert(shared) for expert in self.experts], dim=1)

        outputs: list[torch.Tensor] = []
        for gate, tower in zip(self.gates, self.towers, strict=True):
            weights = torch.softmax(gate(shared), dim=-1).unsqueeze(-1)
            outputs.append(tower((stacked * weights).sum(dim=1)).squeeze(-1))

        if len(outputs) == 1:
            return outputs[0]
        return torch.stack(outputs, dim=-1)
