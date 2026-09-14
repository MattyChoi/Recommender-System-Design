"""Confidence intervals and paired comparison. Anyone can print a mean.

The single decision this module exists to enforce: **the resampling unit is the
user, not the impression.** One user generates many impressions and their scores
are correlated, so resampling impressions treats dependent observations as
independent. The interval that comes out is far too narrow and the wins it
certifies are not real.

:func:`bootstrap_ci_by_impression` exists solely to make the error visible side
by side, and says so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

_DEFAULT_RESAMPLES = 1_000
_DEFAULT_ALPHA = 0.05


class Interval(NamedTuple):
    """A point estimate with its interval and the sample it rests on.

    ``n_users`` is not decoration. An interval quoted without its sample size
    cannot be compared against another run, and the size moves whenever a filter
    or a cohort definition changes.

    Attributes:
        mean: The point estimate.
        lo: Lower percentile bound.
        hi: Upper percentile bound.
        n_users: Independent units resampled.
        n_resamples: Bootstrap iterations.
    """

    mean: float
    lo: float
    hi: float
    n_users: int
    n_resamples: int


class PairedResult(NamedTuple):
    """A paired comparison of two systems over the same users.

    Attributes:
        difference: Mean per-user difference, ``b - a``.
        lo: Lower bound on that difference.
        hi: Upper bound on that difference.
        n_users: Users scored under BOTH systems.
        significant: Whether the interval excludes zero.
    """

    difference: float
    lo: float
    hi: float
    n_users: int
    significant: bool


def per_user_means(scores: Sequence[float], user_ids: Sequence[str]) -> dict[str, float]:
    """Reduce per-impression scores to one score per user.

    NaN scores are skipped, not zeroed -- they mark slates with no click, where
    the metric is undefined rather than bad. A user whose every impression was
    clickless therefore has NO score and is absent from the result, rather than
    contributing a zero that would drag the mean and tighten the interval.

    Args:
        scores: Per-impression metric values, possibly NaN.
        user_ids: The user each impression belongs to.

    Returns:
        User id to mean score, over that user's defined impressions only.
    """
    totals: dict[str, list[float]] = {}
    for score, user in zip(scores, user_ids, strict=True):
        value = float(score)
        if not np.isnan(value):
            totals.setdefault(user, []).append(value)

    return {user: float(np.mean(values)) for user, values in totals.items()}


def _resample_means(values: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_resamples, len(values)))
    return values[draws].mean(axis=1)


def bootstrap_ci(
    scores: Sequence[float],
    user_ids: Sequence[str],
    n_resamples: int = _DEFAULT_RESAMPLES,
    alpha: float = _DEFAULT_ALPHA,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap over USERS.

    Args:
        scores: Per-impression metric values. Reduced internally; do not
            pre-average, and do not pass a per-user array -- the user ids are
            what make the unit correct.
        user_ids: The user each impression belongs to.
        n_resamples: Bootstrap iterations.
        alpha: 0.05 gives a 95% interval.
        seed: Fixed so a rerun reproduces the interval exactly.

    Returns:
        An :class:`Interval`.

    Raises:
        ValueError: If no user has a defined score.
    """
    by_user = per_user_means(scores, user_ids)
    if not by_user:
        raise ValueError("no user has a defined score; the metric is undefined here")

    values = np.fromiter(by_user.values(), dtype=float)
    means = _resample_means(values, n_resamples, seed)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return Interval(
        mean=float(values.mean()),
        lo=float(lo),
        hi=float(hi),
        n_users=len(values),
        n_resamples=n_resamples,
    )


def bootstrap_ci_by_impression(
    scores: Sequence[float],
    n_resamples: int = _DEFAULT_RESAMPLES,
    alpha: float = _DEFAULT_ALPHA,
    seed: int = 0,
) -> Interval:
    """**Deliberately wrong.** Resamples impressions as if they were independent.

    This is not a utility. It exists so the README can print the two intervals
    side by side, because the gap is far more convincing than the assertion that
    a gap exists. Impressions from one user are correlated; treating them as
    independent inflates the effective sample size by roughly the impressions
    per user, and the interval narrows by roughly its square root.

    Never use this for a reported number. ``n_users`` in the returned Interval
    is set to the impression count, which is exactly the lie being illustrated.
    """
    values = np.asarray(scores, dtype=float)
    values = values[~np.isnan(values)]
    if not len(values):
        raise ValueError("no defined scores")

    means = _resample_means(values, n_resamples, seed)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return Interval(
        mean=float(values.mean()),
        lo=float(lo),
        hi=float(hi),
        n_users=len(values),
        n_resamples=n_resamples,
    )


def paired_bootstrap(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    n_resamples: int = _DEFAULT_RESAMPLES,
    alpha: float = _DEFAULT_ALPHA,
    seed: int = 0,
) -> PairedResult:
    """Bootstrap the per-user DIFFERENCE between two systems.

    Paired, because variance across users dwarfs the effect size: some users are
    simply easier to serve than others, and that spread is far larger than the
    2% lift being looked for. Comparing two independent samples of users buries
    the effect under between-user variance; comparing each user against
    themselves cancels it.

    Bootstrapped rather than a t-test, because per-user NDCG and AUC differences
    are bounded, skewed and often spiky -- normality is not a safe assumption,
    and it buys nothing here.

    Args:
        baseline: User id to score under system A.
        candidate: User id to score under system B.
        n_resamples: Bootstrap iterations.
        alpha: 0.05 gives a 95% interval.
        seed: Fixed for reproducibility.

    Returns:
        A :class:`PairedResult` for ``candidate - baseline``.

    Raises:
        ValueError: If no user appears in both. Scoring different user sets and
            comparing the means is the unpaired test this function exists to
            replace, so it is refused rather than silently approximated.
    """
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise ValueError(
            "no user is scored under both systems; a paired test needs the same users on both sides"
        )

    differences = np.array([candidate[user] - baseline[user] for user in shared], dtype=float)
    means = _resample_means(differences, n_resamples, seed)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return PairedResult(
        difference=float(differences.mean()),
        lo=float(lo),
        hi=float(hi),
        n_users=len(shared),
        significant=bool(lo > 0.0 or hi < 0.0),
    )
