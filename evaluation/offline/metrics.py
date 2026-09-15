"""Ranking and beyond-accuracy metrics, over per-impression lists."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt


class Aggregate(NamedTuple):
    """A mean over impressions, with the denominator it was taken over.

    Returning a bare float would hide the thing that makes two runs
    incomparable.

    Attributes:
        mean: Mean over the impressions where the metric was defined.
        scored: How many impressions contributed.
        skipped: How many were undefined and excluded.
    """

    mean: float
    scored: int
    skipped: int


def impression_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based AUC within one impression.

    The probability that a randomly chosen clicked item outranks a randomly
    chosen non-clicked one, with ties counting a half.

    Args:
        labels: 1 for clicked, 0 otherwise.
        scores: Model scores, aligned with ``labels``.

    Returns:
        The AUC, or NaN when the impression is all-clicks or no-clicks.

    Note:
        NaN, never 0.0. An impression with no click does not represent a failed
        ranking -- there is no correct order to have found. Scoring it zero
        drags the mean down in proportion to how many such impressions a filter
        happens to leave in.
    """
    positives = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    negatives = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not positives or not negatives:
        return float("nan")

    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return float(wins / (len(positives) * len(negatives)))


def gauc(preds: npt.ArrayLike, labels: npt.ArrayLike, group_ids: npt.ArrayLike) -> Aggregate:
    """Impression-weighted per-group AUC -- the metric that tracks online lift.

    Weighting is by impression SIZE, so a slate of 40 counts for more than a
    slate of 4 -- larger slates are harder and more informative. Weighting by
    positives instead would let a single heavily-clicked impression dominate.

    Args:
        preds: Model scores.
        labels: 1 for clicked, 0 otherwise.
        group_ids: Impression id per row; rows sharing one are one slate.

    Returns:
        The weighted mean, and how many impressions were scored and skipped.
    """
    return gauc_from_slates(group_slates(preds, labels, group_ids))


def group_slates(
    preds: npt.ArrayLike, labels: npt.ArrayLike, group_ids: npt.ArrayLike
) -> dict[Any, tuple[list[int], list[float]]]:
    """Bucket rows by impression in ONE pass.

    Args:
        preds: Model scores.
        labels: 1 for clicked, 0 otherwise.
        group_ids: Impression id per row.

    Returns:
        Impression id to its ``(labels, scores)``, in first-seen order.
    """
    slates: dict[Any, tuple[list[int], list[float]]] = {}
    for pred, label, group in zip(
        np.asarray(preds).tolist(),
        np.asarray(labels).tolist(),
        np.asarray(group_ids).tolist(),
        strict=True,
    ):
        slate_labels, slate_scores = slates.setdefault(group, ([], []))
        slate_labels.append(int(label))
        slate_scores.append(float(pred))
    return slates


def gauc_from_slates(
    slates: Mapping[Any, tuple[list[int], list[float]]],
) -> Aggregate:
    """GAUC over rows already bucketed by :func:`group_slates`.

    Separate from :func:`gauc` so a caller computing several slate metrics pays
    for the grouping once rather than once per metric.
    """
    total, total_weight, scored, skipped = 0.0, 0.0, 0, 0
    for slate_labels, slate_scores in slates.values():
        auc = impression_auc(slate_labels, slate_scores)
        if math.isnan(auc):
            skipped += 1
            continue
        weight = float(len(slate_labels))
        total += auc * weight
        total_weight += weight
        scored += 1

    mean = total / total_weight if total_weight else float("nan")
    return Aggregate(mean=mean, scored=scored, skipped=skipped)


def aggregate(values: Sequence[float]) -> Aggregate:
    """Mean over per-impression values, carrying the NaN count.

    Calculate mean by hand to count number of skipped values
    """
    arr = np.asarray(values, dtype=float)
    defined = ~np.isnan(arr)
    scored = int(defined.sum())
    mean = float(arr[defined].mean()) if scored else float("nan")
    return Aggregate(mean=mean, scored=scored, skipped=int(len(arr) - scored))


def recall_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> float:
    """Share of an impression's clicked items that land in the top k.

    Returns NaN when nothing was clicked -- same reasoning as impression_auc.
    """
    total_positives = int(sum(labels))
    if total_positives == 0:
        return float("nan")

    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    top = order[:k]
    hits = sum(int(labels[i]) for i in top)
    return float(hits / total_positives)


def ndcg_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> float:
    """Normalized Discounted Cumulative Gain over one impression, binary relevance.

    The ideal DCG is computed from the labels this impression actually has, so
    an impression with two clicks is scored against the best achievable ordering
    of two clicks -- not against an unreachable ideal.
    """
    total_positives = int(sum(labels))
    if total_positives == 0:
        return float("nan")

    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    gains = [int(labels[i]) for i in order[:k]]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.dot(gains, discounts))

    ideal_gains = [1] * min(total_positives, k)
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal_gains) + 2))
    idcg = float(np.dot(ideal_gains, ideal_discounts))
    return dcg / idcg


def reciprocal_rank(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Reciprocal of the rank of the FIRST clicked item in one impression.

    1.0 if the top-ranked item was clicked, 0.5 if the second was, and so on.
    Averaging this across impressions gives MRR.

    Unlike NDCG it ignores every click after the first, which is the right
    emphasis when the product question is "did the user find something
    immediately" rather than "was the whole slate well ordered". On MIND most
    slates with a click have exactly one, so the two usually agree here -- MRR
    earns its place as the metric that is comparable to published MIND results,
    not as an independent signal.

    Args:
        labels: 1 for clicked, 0 otherwise.
        scores: Model scores, aligned with ``labels``.

    Returns:
        The reciprocal rank, or NaN when nothing was clicked -- undefined
        rather than zero, the same reasoning as :func:`impression_auc`. Scoring
        a clickless slate 0.0 would say the model ranked badly, when there was
        no correct answer to rank.
    """
    if sum(labels) == 0:
        return float("nan")

    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    for position, index in enumerate(order, start=1):
        if labels[int(index)] == 1:
            return 1.0 / position
    return float("nan")


def catalog_coverage(recommended_items: Sequence[str], catalogue_size: int) -> float:
    """Distinct items recommended, over the catalogue that COULD be recommended.

    Args:
        recommended_items: Every item id served, across all impressions.
        catalogue_size: Size of the addressable catalogue. Required rather than
            inferred: measuring against only the items that appeared in the
            evaluation window flatters the number badly on this corpus, where
            the item map spans the whole catalogue (ADR 0005) and most of it is
            never shown. Pass ``len(item_map)`` for the honest denominator.

    Returns:
        A fraction in [0, 1].
    """
    if catalogue_size <= 0:
        raise ValueError("catalogue_size must be positive")
    return len(set(recommended_items)) / catalogue_size


def novelty(recommended_items: Sequence[str], train_popularity: Mapping[str, float]) -> float:
    """Mean self-information of the recommended items, in bits.

    Args:
        recommended_items: Item ids served.
        train_popularity: p(item) estimated on TRAIN ONLY. Required rather than
            derived inside: a popularity vector computed over the full corpu
            carries information from the evaluation window, and a novelty score
            built on it is quietly leaked. Making the caller supply it is what
            makes the provenance reviewable.

    Returns:
        Mean ``-log2(p)``. Items absent from the popularity map are treated as
        maximally novel using the smallest probability present, rather than
        dropped -- a cold item IS the novel case, and dropping it would remove
        exactly the items this metric exists to reward.
    """
    if not recommended_items:
        return float("nan")
    if not train_popularity:
        raise ValueError("train_popularity is empty; novelty would be undefined")

    floor = min(train_popularity.values())
    return float(
        np.mean([-math.log2(train_popularity.get(item, floor)) for item in recommended_items])
    )


def intra_list_diversity(
    recommended_items: Sequence[str], vectors: Mapping[str, npt.ArrayLike]
) -> float:
    """One minus the mean pairwise cosine similarity of a single slate.

    Args:
        recommended_items: The items in one slate, in any order.
        vectors: Item id to embedding. Use CONTENT vectors, not ID embeddings:
            five write-ups of one breaking story are near-duplicates that ID
            embeddings cannot recognise, because all five are new. That is the
            case this metric exists to catch on a news corpus.

    Returns:
        A value in [0, 2] for arbitrary vectors, [0, 1] for non-negative ones.
        NaN for a slate of fewer than two items with known vectors, where
        pairwise similarity is undefined.
    """
    known = [
        np.asarray(vectors[item], dtype=float) for item in recommended_items if item in vectors
    ]
    if len(known) < 2:
        return float("nan")

    matrix = np.vstack(known)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalised = matrix / np.where(norms == 0, 1.0, norms)
    similarity = normalised @ normalised.T

    upper = np.triu_indices(len(known), k=1)
    return float(1.0 - similarity[upper].mean())


def gini(item_impression_counts: npt.ArrayLike) -> float:
    """Concentration of exposure across the catalogue.

    Rising Gini over time is the signature of a feedback loop closing: the
    model shows popular items, they accrue engagement, they look better.

    Returns:
        0.0 for perfectly even exposure, approaching 1.0 as exposure
        concentrates on a single item.
    """
    x = np.sort(np.asarray(item_impression_counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
