"""Interval width, the resampling-unit gate, and pairing.

The gate is asserted, not described: the impression-level interval is measurably
narrower than the user-level one on identical data, and that narrowness is the
bug.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.offline.stats import (
    bootstrap_ci,
    bootstrap_ci_by_impression,
    paired_bootstrap,
    per_user_means,
)


def _correlated_corpus(
    n_users: int = 60, per_user: int = 20, seed: int = 7
) -> tuple[list[float], list[str]]:
    """Users with genuinely different ability, many impressions each.

    The structure that makes the resampling unit matter: within a user the
    scores barely move, across users they move a lot. Independent noise would
    hide the whole effect.
    """
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    users: list[str] = []
    for u in range(n_users):
        level = rng.uniform(0.2, 0.8)
        for _ in range(per_user):
            scores.append(float(np.clip(level + rng.normal(0, 0.01), 0, 1)))
            users.append(f"U{u}")
    return scores, users


class TestPerUserMeans:
    def test_nan_impressions_are_skipped_not_zeroed(self) -> None:
        got = per_user_means([1.0, float("nan"), 0.0], ["U1", "U1", "U1"])

        assert got == {"U1": 0.5}

    def test_a_user_with_only_nan_impressions_disappears(self) -> None:
        """Not a zero. That user has no measurement, and inventing one would
        both drag the mean and tighten the interval."""
        got = per_user_means([float("nan"), float("nan"), 0.4], ["U1", "U1", "U2"])

        assert got == {"U2": 0.4}


class TestBootstrapCi:
    def test_the_interval_brackets_the_mean(self) -> None:
        scores, users = _correlated_corpus()
        result = bootstrap_ci(scores, users)

        assert result.lo < result.mean < result.hi
        assert result.n_users == 60

    def test_more_users_narrows_the_interval(self) -> None:
        few = bootstrap_ci(*_correlated_corpus(n_users=20))
        many = bootstrap_ci(*_correlated_corpus(n_users=400))

        assert (many.hi - many.lo) < (few.hi - few.lo)

    def test_it_is_deterministic_under_a_fixed_seed(self) -> None:
        scores, users = _correlated_corpus()

        assert bootstrap_ci(scores, users) == bootstrap_ci(scores, users)

    def test_it_refuses_data_with_no_defined_score(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            bootstrap_ci([float("nan")], ["U1"])


class TestTheResamplingUnitGate:
    def test_resampling_impressions_gives_a_falsely_narrow_interval(self) -> None:
        """THE gate of E3, as an assertion rather than a warning.

        Identical data. Resampling 1,200 correlated impressions instead of 60
        users inflates the effective sample size by roughly the impressions per
        user, so the interval narrows by roughly its square root -- here, several
        times over. Every width in that interval is unearned.
        """
        scores, users = _correlated_corpus(n_users=60, per_user=20)

        honest = bootstrap_ci(scores, users)
        wrong = bootstrap_ci_by_impression(scores)

        assert (wrong.hi - wrong.lo) < (honest.hi - honest.lo) / 2

    def test_both_agree_on_the_point_estimate(self) -> None:
        """Only the interval is wrong, which is what makes it hard to notice.

        With equal impressions per user the means coincide exactly; the error
        lives entirely in the uncertainty, where nothing looks out of place.
        """
        scores, users = _correlated_corpus()

        honest = bootstrap_ci(scores, users)
        wrong = bootstrap_ci_by_impression(scores)

        assert honest.mean == pytest.approx(wrong.mean, abs=1e-9)


class TestPairedBootstrap:
    def test_a_small_consistent_lift_is_detected(self) -> None:
        """Between-user spread of 0.6, per-user lift of 0.02.

        Pairing cancels the spread, so the lift is visible. An unpaired
        comparison of the two means would be swamped by it.
        """
        rng = np.random.default_rng(3)
        baseline = {f"U{i}": float(rng.uniform(0.2, 0.8)) for i in range(200)}
        candidate = {user: score + 0.02 for user, score in baseline.items()}

        result = paired_bootstrap(baseline, candidate)

        assert result.difference == pytest.approx(0.02, abs=1e-6)
        assert result.significant
        assert result.lo > 0

    def test_no_real_difference_is_not_called_significant(self) -> None:
        rng = np.random.default_rng(11)
        baseline = {f"U{i}": float(rng.uniform(0.2, 0.8)) for i in range(200)}
        candidate = {user: score + float(rng.normal(0, 0.05)) for user, score in baseline.items()}

        result = paired_bootstrap(baseline, candidate)

        assert not result.significant
        assert result.lo < 0 < result.hi

    def test_a_regression_is_significant_in_the_other_direction(self) -> None:
        baseline = {f"U{i}": 0.5 for i in range(100)}
        candidate = {f"U{i}": 0.4 for i in range(100)}

        result = paired_bootstrap(baseline, candidate)

        assert result.significant
        assert result.hi < 0

    def test_only_users_in_both_systems_are_compared(self) -> None:
        result = paired_bootstrap({"U1": 0.5, "U2": 0.5, "U3": 0.5}, {"U1": 0.6, "U2": 0.6})

        assert result.n_users == 2

    def test_disjoint_user_sets_are_refused(self) -> None:
        """Comparing means over different users IS the unpaired test."""
        with pytest.raises(ValueError, match="same"):
            paired_bootstrap({"U1": 0.5}, {"U2": 0.6})


def test_the_wrong_function_reports_impressions_as_its_unit() -> None:
    """n_users on the impression-level interval is the lie, made legible."""
    scores, users = _correlated_corpus(n_users=10, per_user=5)

    assert bootstrap_ci(scores, users).n_users == 10
    assert bootstrap_ci_by_impression(scores).n_users == 50
    assert not math.isnan(bootstrap_ci_by_impression(scores).mean)
