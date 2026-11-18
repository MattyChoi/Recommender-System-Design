"""Sample ratio mismatch -- the check that runs before any result is read.

If SRM fires, the experiment is **invalid regardless of the p-value on the
primary metric**. Arms that did not receive the traffic they were allocated did
not run the experiment that was designed, so a lift computed over them is a
measurement of something nobody specified. Assignment, logging or filtering is
broken, and that is what gets investigated -- not the metric.

``serving/monitoring/alerts.yml``'s ``RecsysExperimentImbalance`` is the crude
live version: a 1.2 max/min ratio over 30 minutes. A ratio cannot tell a 20%
imbalance over 200 assignments from the same imbalance over 200,000, and the
first is noise while the second is a serious defect. This is the real test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

#: Relative convergence target, and the iteration cap that bounds both series.
#: 1e-14 is near double precision; the series converge in well under 100 terms
#: for every degrees-of-freedom this test will ever see.
_EPSILON = 1e-14
_MAX_ITERATIONS = 500

#: Guards the continued fraction against a zero denominator.
_TINY = 1e-300

#: The conventional floor for the chi-square approximation. Below five expected
#: observations in a cell the asymptotic distribution is not a good fit and the
#: p-value is not trustworthy -- reported rather than enforced, because the
#: right response is to wait for traffic, not to refuse the reading.
MIN_EXPECTED = 5.0

#: 0.001, not 0.05. SRM is checked on EVERY experiment, so a 5% false-positive
#: rate means one healthy experiment in twenty gets declared invalid -- and a
#: guard that cries wolf is turned off, which is strictly worse than not having
#: one. The cost of the tighter threshold is sensitivity to small imbalances,
#: and a small genuine imbalance will be caught by the next readthrough anyway
#: because SRM grows with sample size.
DEFAULT_ALPHA = 0.001


class SRM(NamedTuple):
    """The verdict, with everything needed to argue with it.

    Attributes:
        statistic: Pearson's chi-square.
        p_value: Probability of a statistic at least this extreme under the
            null that assignment matched the allocation.
        degrees_of_freedom: Arms minus one.
        detected: ``p_value < alpha``. True means STOP.
        observed: Counts as given.
        expected: Counts the allocation implies, same order.
        min_expected: The smallest expected count. Below :data:`MIN_EXPECTED`
            the approximation is unreliable, which ``reliable`` reports.
    """

    statistic: float
    p_value: float
    degrees_of_freedom: int
    detected: bool
    observed: tuple[int, ...]
    expected: tuple[float, ...]
    min_expected: float

    @property
    def reliable(self) -> bool:
        """Whether the chi-square approximation holds for these counts."""
        return self.min_expected >= MIN_EXPECTED

    def summary(self) -> str:
        """One line, for a log or a report."""
        verdict = "SRM DETECTED" if self.detected else "no SRM"
        caveat = "" if self.reliable else f" (UNRELIABLE: min expected {self.min_expected:.1f})"
        return (
            f"{verdict}: chi2={self.statistic:.3f} df={self.degrees_of_freedom} "
            f"p={self.p_value:.2e}{caveat}"
        )


def _lower_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x), by series. For x < a + 1."""
    term = 1.0 / a
    total = term
    for n in range(1, _MAX_ITERATIONS + 1):
        term *= x / (a + n)
        total += term
        if abs(term) < abs(total) * _EPSILON:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _upper_fraction(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x), by continued fraction.

    Modified Lentz. Used for x >= a + 1, where the series above converges
    slowly and loses precision to cancellation.
    """
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITERATIONS + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPSILON:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi_square_sf(statistic: float, degrees_of_freedom: int) -> float:
    """P(X > statistic) for a chi-square with the given degrees of freedom.

    The survival function, not the CDF: the p-value is the upper tail, and
    computing it as ``1 - cdf`` throws away every digit that matters once the
    CDF approaches 1 -- which is exactly the regime a p-value lives in.
    """
    if degrees_of_freedom < 1:
        raise ValueError(f"degrees_of_freedom must be >= 1, got {degrees_of_freedom}")
    if statistic <= 0.0:
        return 1.0

    a, x = degrees_of_freedom / 2.0, statistic / 2.0
    if x < a + 1.0:
        return 1.0 - _lower_series(a, x)
    return _upper_fraction(a, x)


def srm_check(
    observed: Sequence[int],
    expected_ratios: Sequence[float],
    alpha: float = DEFAULT_ALPHA,
) -> SRM:
    """Test observed arm counts against the allocation they were meant to get.

    Args:
        observed: Assignments per arm.
        expected_ratios: The allocation, same order, summing to 1.
        alpha: Significance. See :data:`DEFAULT_ALPHA`.

    Returns:
        The verdict. ``detected`` true means stop and investigate.

    Raises:
        ValueError: On fewer than two arms, a length mismatch, a negative
            count, no assignments at all, or ratios that do not sum to 1.

    Note:
        **The holdback is an arm.** ``experiments.Experiment`` allocates less
        than 100% to create one, and ``Experiment.Names()`` includes
        ``Unassigned`` for exactly this reason. Passing only the named arms
        makes the ratios miss their sum and this refuses -- deliberately, since
        the alternative is silently normalising and testing against a
        denominator the experiment does not have.
    """
    if len(observed) != len(expected_ratios):
        raise ValueError(f"{len(observed)} arms against {len(expected_ratios)} ratios")
    if len(observed) < 2:
        raise ValueError("SRM needs at least two arms")
    if any(count < 0 for count in observed):
        raise ValueError(f"negative counts: {list(observed)}")
    if any(ratio <= 0.0 for ratio in expected_ratios):
        raise ValueError(f"every arm needs a positive share: {list(expected_ratios)}")

    total = sum(observed)
    if total == 0:
        raise ValueError("no assignments to test")

    allocated = math.fsum(expected_ratios)
    if abs(allocated - 1.0) > 1e-6:
        raise ValueError(
            f"ratios sum to {allocated:.6f}, not 1. An under-allocated experiment has a "
            "holdback arm -- include it (Experiment.Names() does) rather than renormalising"
        )

    expected = tuple(ratio * total for ratio in expected_ratios)
    statistic = math.fsum(
        (count - want) ** 2 / want for count, want in zip(observed, expected, strict=True)
    )
    degrees_of_freedom = len(observed) - 1
    p_value = chi_square_sf(statistic, degrees_of_freedom)

    return SRM(
        statistic=statistic,
        p_value=p_value,
        degrees_of_freedom=degrees_of_freedom,
        detected=p_value < alpha,
        observed=tuple(observed),
        expected=expected,
        min_expected=min(expected),
    )
