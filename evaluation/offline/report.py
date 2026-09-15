"""The one output format every model in this project is scored by.

Sliced rather than aggregated, because a single headline number hides exactly
the thing worth discussing. On this corpus 46.2% of the dev catalogue never
appears in train, so a model can be excellent overall and useless on the items
that make the domain hard -- and the overall number will not say so.

Every model goes through this, including the ones you are proud of. A baseline
scored by one path and a neural model scored by another is not a comparison.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from evaluation.offline.metrics import catalog_coverage, group_slates, ndcg_at_k
from evaluation.offline.protocols import evaluate_ranking
from evaluation.offline.stats import (
    bootstrap_ci,
    bootstrap_ci_by_impression,
    per_user_means,
)

_ROUND = 4


def _git_sha() -> str | None:
    """The commit a result was produced at, when there is one.

    A report card without provenance cannot be compared to the next one: two
    files with different numbers and no way to say what changed between them
    are worse than one file.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def default_cohorts(
    is_cold_user: npt.ArrayLike,
    is_cold_item: npt.ArrayLike,
    impression_ids: npt.ArrayLike,
    labels: npt.ArrayLike,
) -> dict[str, np.ndarray]:
    """Five slices every model is reported over, each keeping slates WHOLE.

    Args:
        is_cold_user: Per row, whether the user is absent from train.
        is_cold_item: Per row, whether that row's item is absent from train.
        impression_ids: Slate id per row.
        labels: 1 for clicked, 0 otherwise.

    Returns:
        Cohort name to a boolean row mask. Every mask selects whole slates.
    """
    cold_user = np.asarray(is_cold_user, dtype=bool)
    cold_item = np.asarray(is_cold_item, dtype=bool)
    slates = np.asarray(impression_ids).tolist()
    clicked = np.asarray(labels, dtype=bool).tolist()

    # A slate is a cold-item case when something the user actually CLICKED was
    # absent from train. A slate with no click belongs to neither item cohort:
    # there is no target item to classify, and inventing one would put it in
    # whichever cohort the mask happened to favour.
    cold_target: dict[Any, bool] = {}
    for slate, is_cold, was_clicked in zip(slates, cold_item.tolist(), clicked, strict=True):
        if was_clicked:
            cold_target[slate] = cold_target.get(slate, False) or is_cold

    slate_is_cold = np.array([cold_target.get(s, False) for s in slates], dtype=bool)
    slate_has_click = np.array([s in cold_target for s in slates], dtype=bool)

    return {
        "overall": np.ones_like(cold_user, dtype=bool),
        "warm_user": ~cold_user,
        "cold_user": cold_user,
        "warm_item": slate_has_click & ~slate_is_cold,
        "cold_item": slate_has_click & slate_is_cold,
    }


def _served_items(
    scores: np.ndarray, item_ids: np.ndarray, impression_ids: np.ndarray, k: int
) -> list[Any]:
    """Every item this model ranks into some slate's top k.

    Ties are broken by the order rows arrive in, which is the same resolution
    the metrics themselves use -- a model that ties everything therefore
    "serves" the first k of each slate rather than nothing, and its coverage
    reads as narrow rather than as zero. That is the honest reading: it does
    surface items, it just is not choosing them.
    """
    buckets: dict[Any, list[tuple[float, Any]]] = {}
    for score, item, slate in zip(
        scores.tolist(), item_ids.tolist(), impression_ids.tolist(), strict=True
    ):
        buckets.setdefault(slate, []).append((score, item))

    served: list[Any] = []
    for pairs in buckets.values():
        pairs.sort(key=lambda pair: -pair[0])
        served.extend(item for _, item in pairs[:k])
    return served


def _slice_report(
    scores: np.ndarray,
    labels: np.ndarray,
    impression_ids: np.ndarray,
    user_ids: np.ndarray,
    k: int,
    item_ids: np.ndarray | None = None,
    catalogue_size: int | None = None,
) -> dict[str, Any]:
    """Every number reported for one cohort, including how much of the cohort
    the model could discriminate within at all."""
    ranking = evaluate_ranking(scores.tolist(), labels.tolist(), impression_ids.tolist(), k=k)

    by_slate = group_slates(scores, labels, impression_ids)

    slate_owner: dict[Any, str] = {}
    for slate, user in zip(impression_ids.tolist(), user_ids.tolist(), strict=True):
        slate_owner.setdefault(slate, str(user))

    slate_ndcg = [ndcg_at_k(lab, sc, k) for lab, sc in by_slate.values()]
    slate_users = [slate_owner[slate] for slate in by_slate]

    # A slate whose scores are all equal contributes exactly 0.5 to GAUC no
    # matter what those scores are: every comparison is a tie, and a tie counts
    # half. Counting those slates is what separates "the model ranked badly"
    # from "the model had nothing to rank with" -- two states that otherwise
    # produce near-identical cards. Co-visitation sat on random's GAUC to four
    # decimals because 91% of its slates were flat, and nothing on the card
    # said so.
    #
    # Defined on the SCORES alone, so it means the same thing for every model.
    # A per-model rule such as "score > 0" would not: zero is a real score for
    # popularity and an absence for co-visitation.
    scorable = [(lab, sc) for lab, sc in by_slate.values() if 0 < sum(lab) < len(lab)]
    flat = sum(1 for _, sc in scorable if max(sc) == min(sc))
    flat_share = flat / len(scorable) if scorable else float("nan")

    report: dict[str, Any] = {
        "rows": len(scores),
        "impressions": len(by_slate),
        "users": len(set(slate_users)),
        "gauc": round(ranking.gauc.mean, _ROUND),
        "mrr": round(ranking.mrr.mean, _ROUND),
        f"ndcg@{k}": round(ranking.ndcg.mean, _ROUND),
        f"recall@{k}": round(ranking.recall.mean, _ROUND),
        "scored_impressions": ranking.gauc.scored,
        "skipped_impressions": ranking.gauc.skipped,
        "flat_impressions": flat,
    }

    if scorable:
        ceiling = 1.0 - 0.5 * flat_share
        report["gauc_ceiling"] = round(ceiling, _ROUND)
        # None, not 0.0: with every slate flat there is no budget to take a
        # share OF, and 0.0 would read as "used none of a budget it had".
        report["headroom_used"] = (
            round((report["gauc"] - 0.5) / (ceiling - 0.5), _ROUND) if ceiling > 0.5 else None
        )

    # A cohort can be entirely clickless -- cold items especially -- in which
    # case there is no interval to quote and saying so beats a NaN triple.
    if per_user_means(slate_ndcg, slate_users):
        interval = bootstrap_ci(slate_ndcg, slate_users)

        # The ndcg above is a mean over IMPRESSIONS; this interval is a
        # percentile bootstrap of the mean over USERS.
        report[f"ndcg@{k}_by_user"] = round(interval.mean, _ROUND)
        report["ci95"] = [round(interval.lo, _ROUND), round(interval.hi, _ROUND)]
        report["ci_users"] = interval.n_users

        # The same interval computed over the WRONG unit
        wrong = bootstrap_ci_by_impression(slate_ndcg)
        report["ci95_by_impression_DO_NOT_QUOTE"] = [
            round(wrong.lo, _ROUND),
            round(wrong.hi, _ROUND),
        ]
        report["ci_impressions"] = wrong.n_users
    else:
        report["ci95"] = None
        report["ci_users"] = 0

    # Coverage answers a question no accuracy metric asks: a model can win on
    # NDCG by surfacing the same few popular items to everyone, and the whole
    # catalogue beyond them is dead inventory. Measured against the FULL item
    # map rather than the items that happened to appear in the evaluation
    # window -- per catalog_coverage's own contract and ADR 0005, the window
    # denominator flatters the number badly on this corpus.
    if item_ids is not None and catalogue_size:
        served = _served_items(scores, item_ids, impression_ids, k)
        report[f"coverage@{k}"] = round(catalog_coverage(served, catalogue_size), _ROUND)
        report["distinct_items_served"] = len(set(served))
        report["catalogue_size"] = catalogue_size

    return report


def report_card(
    model_name: str,
    scores: npt.ArrayLike,
    labels: npt.ArrayLike,
    impression_ids: npt.ArrayLike,
    user_ids: npt.ArrayLike,
    cohorts: Mapping[str, npt.ArrayLike],
    split: str = "dev",
    k: int = 10,
    item_ids: npt.ArrayLike | None = None,
    catalogue_size: int | None = None,
) -> dict[str, Any]:
    """Score one model and return the comparable record of it.

    Args:
        model_name: Identifies the run in ``evaluation/results/``.
        scores: Model score per row.
        labels: 1 for clicked, 0 otherwise.
        impression_ids: Slate id per row.
        user_ids: User id per row.
        cohorts: Cohort name to a row mask, from :func:`default_cohorts`.
        split: Which evaluation split produced these rows.
        k: Cutoff for NDCG and recall.
        item_ids: Item id per row. Omit and coverage is left OFF the card
            rather than reported as zero -- an absent measurement and a
            measured zero are different claims.
        catalogue_size: Items that COULD be served, i.e. ``len(item_map)``.
            Required alongside ``item_ids``; without it there is no denominator.

    Returns:
        A JSON-serialisable report card.
    """
    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    impressions_arr = np.asarray(impression_ids, dtype=object)
    users_arr = np.asarray(user_ids, dtype=object)

    card: dict[str, Any] = {
        "model": model_name,
        "split": split,
        "generated": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "k": k,
        "cohorts": {},
    }

    items_arr = None if item_ids is None else np.asarray(item_ids, dtype=object)

    for name, mask in cohorts.items():
        selector = np.asarray(mask, dtype=bool)
        if not selector.any():
            card["cohorts"][name] = {"rows": 0}
            continue
        card["cohorts"][name] = _slice_report(
            scores_arr[selector],
            labels_arr[selector],
            impressions_arr[selector],
            users_arr[selector],
            k=k,
            item_ids=None if items_arr is None else items_arr[selector],
            catalogue_size=catalogue_size,
        )

    return card
