"""Funnel NDCG, and the denominators it has to be quoted with.

**One metric path, deliberately.** A booster and a torch model reach this file
by the same route -- :func:`evaluate_scores` takes scores a caller already has,
so neither learner carries an evaluation of its own. Two evaluate functions is
how two models end up compared on two subtly different denominators, and the
difference is invisible in the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from evaluation.offline.metrics import Aggregate, gauc, ndcg_at_k
from evaluation.offline.stats import Interval, bootstrap_ci
from models.ranking.dataset import RankingRows


@dataclass(frozen=True)
class RankingResult:
    """What a ranker scored, and on which denominator.

    Attributes:
        ndcg: Funnel NDCG@k, the mean over USERS -- over every request, zero
            where retrieval missed. Matches ``interval``, which is what makes
            the pair quotable together.
        ndcg_by_request: The same quantity averaged over REQUESTS instead. A
            user with forty requests outweighs one with a single request here
            and does not there, so the two differ by the weighting rather than
            by noise and no interval brackets both.
        ndcg_retrieved: NDCG@k over the requests retrieval solved. Always the
            larger number, and it is not the system's performance.
        ceiling: Retrieval's recall over the candidate list, which funnel NDCG
            cannot exceed.
        interval: ``ndcg`` bootstrapped over users.
        requests: How many requests the denominator covers.
        users: How many users. The unit the interval resamples.
        gauc: Per-request AUC, weighted by candidate-list size. **The metric
            that tracks online lift**, because ranking only ever happens WITHIN
            a request. Its ``skipped`` count is the requests with no positive --
            there is no correct order to have found there -- so GAUC's
            denominator is the RETRIEVED subset and it cannot see a retrieval
            miss. Funnel NDCG can. The two answer different questions and both
            are on the table for that reason.
        auc: Global AUC over every candidate row, pooled across requests. On
            here to be undercut: it rewards separating a strong request's
            negatives from a weak request's positives, which no user ever
            experiences. Expect it to flatter the model relative to GAUC.
    """

    gauc: Aggregate
    auc: float
    ndcg: float
    ndcg_by_request: float
    ndcg_retrieved: float
    ceiling: float
    interval: Interval
    requests: int
    users: int

    @property
    def headroom_used(self) -> float:
        """Share of what retrieval made reachable that the ranker converted."""
        return self.ndcg / self.ceiling if self.ceiling else float("nan")


def global_auc(scores: npt.NDArray[np.float64], labels: npt.NDArray[np.int64]) -> float:
    """AUC over every row at once, ignoring which request each came from.

    Computed with sklearn rather than this project's :func:`impression_auc`,
    which is an explicit double loop over positives and negatives -- correct,
    and O(P x N), which on half a million pooled rows is billions of
    comparisons. Within one request it is a few hundred.
    """
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def per_request_ndcg(
    scores: npt.NDArray[np.float64], rows: RankingRows, k: int = 10
) -> npt.NDArray[np.float64]:
    """NDCG@k for each request, zero where the candidate list holds no positive."""
    out = np.zeros(len(rows.groups), dtype=float)
    start = 0
    for request, size in enumerate(rows.groups):
        end = start + int(size)
        if rows.found[request]:
            out[request] = ndcg_at_k(rows.labels[start:end].tolist(), scores[start:end].tolist(), k)
        start = end
    return out


def evaluate(model: Any, rows: RankingRows, k: int = 10) -> RankingResult:
    """Score a fitted model on the funnel, with the misses counted."""
    return evaluate_scores(np.asarray(model.predict(rows.features), dtype=float), rows, k)


def evaluate_scores(
    scores: npt.NDArray[np.float64], rows: RankingRows, k: int = 10
) -> RankingResult:
    """The same, from scores a caller already has."""
    per_request = per_request_ndcg(scores, rows, k)
    users = [str(value) for value in rows.user_ids]
    interval = bootstrap_ci(per_request.tolist(), users)

    return RankingResult(
        gauc=gauc(scores, rows.labels, rows.request),
        auc=global_auc(scores, rows.labels),
        # From the interval, not recomputed: the headline and its bounds must be
        # the SAME estimand, and a mean taken twice over two different units is
        # how a point estimate ends up outside its own confidence interval.
        ndcg=interval.mean,
        ndcg_by_request=float(per_request.mean()),
        ndcg_retrieved=float(per_request[rows.found].mean()) if rows.found.any() else float("nan"),
        ceiling=rows.recall_ceiling,
        interval=interval,
        requests=len(rows.groups),
        users=interval.n_users,
    )


def render(result: RankingResult, ranked: list[tuple[str, float]], k: int) -> str:
    """The table, with both denominators on it.

    Args:
        ranked: Feature importances, or empty. Only the tree produces them, and
            an empty list prints nothing rather than a heading over no rows.
    """
    lines = [
        f"  NDCG@{k}, funnel, per USER  ({result.users:,} users)    {result.ndcg:.4f}"
        f"  [{result.interval.lo:.4f}, {result.interval.hi:.4f}]",
        f"  NDCG@{k}, funnel, per REQUEST ({result.requests:,})      {result.ndcg_by_request:.4f}"
        "   <- different weighting, NOT bracketed above",
        f"  NDCG@{k}, retrieved requests only             {result.ndcg_retrieved:.4f}",
        f"  ceiling (retrieval recall over the list)     {result.ceiling:.4f}",
        f"  headroom used                                {result.headroom_used:.1%}",
        "",
        f"  GAUC, per request, size-weighted             {result.gauc.mean:.4f}"
        f"   <- the headline for ORDERING",
        f"    scored on {result.gauc.scored:,} requests, "
        f"{result.gauc.skipped:,} skipped for having no positive",
        f"  AUC, pooled over all rows                    {result.auc:.4f}"
        "   <- weaker: mixes requests no user compares",
    ]
    if ranked:
        lines += ["", "  feature importance, by gain:"]
        lines += [f"    {name:<22}{share:>8.1%}" for name, share in ranked]
    return "\n".join(lines)
