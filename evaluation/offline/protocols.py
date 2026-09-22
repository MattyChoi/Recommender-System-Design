"""The two evaluation protocols, kept apart on purpose.

``metrics.py`` holds pure functions with no opinions. This module is where the
opinions live, because the same primitive means two different things depending
on the pool it is given:

* ``recall_at_k`` over the items shown together in one impression is a RANKING
  metric -- did the model order that slate well?
* ``recall_at_k`` over the whole catalogue is a RETRIEVAL metric -- did the
  candidate generator find the clicked item among two hundred thousand?

We make two decisions here:

1. **Retrieval is scored against the full catalogue, never sampled negatives.**
   Scoring one positive against 100 random negatives is biased and does not
   rank-correlate with full ranking (Krichene & Rendle, 2020).
2. **Ranking is scored within the impression**, which is MIND's actual task and
   what its leaderboard reports. Scoring ranking against the catalogue would
   produce numbers that cannot be compared to any published result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from evaluation.offline.metrics import (
    Aggregate,
    aggregate,
    catalog_coverage,
    gauc_from_slates,
    group_slates,
    ndcg_at_k,
    novelty,
    recall_at_k,
    reciprocal_rank,
)


class RankingResult(NamedTuple):
    """Within-impression quality. Comparable to published MIND numbers.

    Attributes:
        gauc: Impression-weighted per-slate AUC. The headline.
        mrr: Mean reciprocal rank of the first click. Reported because MIND's
            leaderboard reports it, so it is the number that makes this
            comparable to published results.
        ndcg: Mean NDCG@k over slates where it is defined.
        recall: Mean recall@k over slates where it is defined.
        k: The cutoff NDCG and recall were taken at. MRR has no cutoff.
    """

    gauc: Aggregate
    mrr: Aggregate
    ndcg: Aggregate
    recall: Aggregate
    k: int


class RetrievalResult(NamedTuple):
    """Catalogue-wide candidate generation quality.

    Attributes:
        recall: Share of clicked items the retriever surfaced in its top k.
        coverage: Distinct items recommended, over the addressable catalogue.
        novelty: Mean self-information, in bits, against train popularity.
        k: The cutoff recall was taken at.
        catalogue_size: The pool scored against, recorded so a later reader can
            tell whether this was a full-catalogue run.
        reach: Share of requests the retriever answered with anything at all.
            1.0 for a retriever that scores the whole catalogue; below it for a
            source with limited reach, and that distinction is a finding rather
            than a defect -- co-visitation on news is the worked example.
        mean_pool: Mean candidates returned per request. Reported beside
            ``reach`` so a short pool is visible in the result rather than
            inferred from a recall that looks low for the wrong reason.
    """

    recall: Aggregate
    coverage: float
    novelty: float
    k: int
    catalogue_size: int
    reach: float = 1.0
    mean_pool: float = float("nan")


def evaluate_ranking(
    scores: Sequence[float],
    labels: Sequence[int],
    impression_ids: Sequence[str],
    k: int = 10,
) -> RankingResult:
    """Score a model on the ordering of each slate.

    Args:
        scores: Model score per row.
        labels: 1 for clicked, 0 otherwise.
        impression_ids: Slate id per row. Rows sharing one were shown together.
        k: Cutoff for NDCG and recall.

    Returns:
        A :class:`RankingResult`.

    Note:
        Slates with no click, or with nothing but clicks, are undefined rather
        than zero and are excluded. The counts in each ``Aggregate`` say how
        many -- which matters, because that number moves with every filter and
        two runs with different denominators are not comparable.
    """
    by_slate = group_slates(scores, labels, impression_ids)

    return RankingResult(
        gauc=gauc_from_slates(by_slate),
        mrr=aggregate([reciprocal_rank(lab, sc) for lab, sc in by_slate.values()]),
        ndcg=aggregate([ndcg_at_k(lab, sc, k) for lab, sc in by_slate.values()]),
        recall=aggregate([recall_at_k(lab, sc, k) for lab, sc in by_slate.values()]),
        k=k,
    )


def evaluate_retrieval(
    retrieved: Mapping[str, Sequence[str]],
    relevant: Mapping[str, Sequence[str]],
    catalogue_size: int,
    train_popularity: Mapping[str, float],
    k: int = 100,
    allow_short_pools: bool = False,
) -> RetrievalResult:
    """Score a candidate generator against the whole catalogue.

    Args:
        retrieved: Request id to the ranked item ids the retriever returned,
            best first. These must have been selected from the FULL catalogue.
        relevant: Request id to the item ids actually clicked. Requests with no
            click contribute nothing and are counted as skipped.
        catalogue_size: Items the retriever could have returned.
        train_popularity: p(item) on TRAIN ONLY, for novelty.
        k: Cutoff for recall.
        allow_short_pools: Permit a retriever that returns fewer than ``k``.

    Returns:
        A :class:`RetrievalResult`.

    Raises:
        ValueError: If ``catalogue_size`` is not a plausible catalogue, or if a
            request's pool is shorter than ``k`` and ``allow_short_pools`` is
            False. The check exists because sampled-negative evaluation produces
            a perfectly reasonable-looking number that is not comparable to
            anything -- and the output gives no hint that it happened. A guard
            that fires is cheaper than a table that has to be retracted.
    """
    if catalogue_size < k:
        raise ValueError(
            f"catalogue_size={catalogue_size} is smaller than k={k}; this is "
            "not a full-catalogue evaluation"
        )

    # guard against candidate pools that are smaller than the catalogue size
    short = {
        request: len(candidates) for request, candidates in retrieved.items() if len(candidates) < k
    }
    if short and not allow_short_pools:
        example = next(iter(short.items()))
        raise ValueError(
            f"{len(short)} request(s) returned fewer than k={k} candidates "
            f"(e.g. {example[0]!r} returned {example[1]}). Retrieval must be "
            "scored over the full catalogue, not a sampled pool. If this is a "
            "source with genuinely limited reach rather than a sampled pool, "
            "pass allow_short_pools=True and read `reach` in the result."
        )

    recalls: list[float] = []
    served: list[str] = []
    pools: list[int] = []
    answered = 0
    for request, candidates in retrieved.items():
        top = list(candidates)[:k]
        served.extend(top)
        pools.append(len(top))
        answered += bool(top)

        clicked = set(relevant.get(request, ()))
        if not clicked:
            recalls.append(float("nan"))
            continue
        recalls.append(len(clicked & set(top)) / len(clicked))

    requests = len(retrieved)
    return RetrievalResult(
        recall=aggregate(recalls),
        coverage=catalog_coverage(served, catalogue_size),
        novelty=novelty(served, train_popularity),
        k=k,
        catalogue_size=catalogue_size,
        reach=answered / requests if requests else float("nan"),
        mean_pool=sum(pools) / requests if requests else float("nan"),
    )
