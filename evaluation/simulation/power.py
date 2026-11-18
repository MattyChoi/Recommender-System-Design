"""Power, sample size, and the minimum detectable effect.

Three questions, one formula, asked in different directions:

* **How long do I run?** Given a lift worth detecting, how many units per arm.
* **Was this experiment capable of finding anything?** Given the units it got,
  the smallest lift it could have detected.
* **What were the odds?** Given both, the probability of a significant result
  if the effect is real.

The third is the one that makes a non-significant result interpretable. An
experiment powered at 35% that came back flat has said almost nothing, and
reporting "no significant difference" without the power behind it is how a
real effect gets shipped as a null.

⚠️ **"A 1% lift" is ambiguous and this module refuses to guess.** On a 4% CTR,
a 1% RELATIVE lift is 0.04 points and needs roughly a hundred times the traffic
of a 1% ABSOLUTE lift, which would be a 25% improvement. Every function here
takes ``relative_lift`` explicitly and the absolute change is reported back, so
a plan can be checked against the thing it was meant to say.

**No scipy.** The normal quantile and CDF come from :mod:`statistics` in the
standard library, which is exact to double precision -- unlike
:mod:`evaluation.experiments.srm`, where the chi-square had no stdlib
equivalent and the incomplete gamma had to be written out.

Scope: a two-arm test of a BINARY metric (clicked or not), the normal
approximation, equal arm sizes. Enough for a CTR decision and honest about
being so. Unequal allocation, more than two arms, and sequential testing with
its alpha spending are all out of scope and would each change the arithmetic.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import NamedTuple

#: Convention, not a law. 80% power means a real effect of the stated size is
#: missed one time in five, which is the risk most teams have agreed to accept.
DEFAULT_POWER = 0.80

#: Two-sided by default: a change that makes the metric WORSE is a result, and
#: a one-sided test cannot see it. The trap is that one-sided at 0.05 is a
#: two-sided test at 0.10 wearing a smaller number.
DEFAULT_ALPHA = 0.05

_NORMAL = NormalDist()


class SampleSize(NamedTuple):
    """What an experiment needs, and what it is being asked to detect.

    Attributes:
        per_arm: Units required in each arm.
        total: Both arms.
        baseline_rate: The control rate assumed.
        relative_lift: The effect as a fraction of baseline.
        absolute_lift: The same effect in percentage points -- reported
            because this is the number people actually disagree about.
        treatment_rate: ``baseline_rate * (1 + relative_lift)``.
        power: Probability of detecting that effect if it is real.
        alpha: Significance level.
        variance_reduction: The CUPED reduction assumed, if any.
    """

    per_arm: int
    total: int
    baseline_rate: float
    relative_lift: float
    absolute_lift: float
    treatment_rate: float
    power: float
    alpha: float
    variance_reduction: float

    def summary(self) -> str:
        saved = ""
        if self.variance_reduction > 0.0:
            saved = f", after a {self.variance_reduction:.0%} CUPED variance reduction"
        return (
            f"detecting a {self.relative_lift:+.1%} relative lift "
            f"({self.baseline_rate:.3%} -> {self.treatment_rate:.3%}, "
            f"{self.absolute_lift:+.3%} absolute) at {self.power:.0%} power "
            f"needs {self.per_arm:,} units per arm{saved}"
        )


def _validate(baseline_rate: float, alpha: float, power: float) -> None:
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")


def achieved_power(
    baseline_rate: float,
    relative_lift: float,
    per_arm: int,
    alpha: float = DEFAULT_ALPHA,
    variance_reduction: float = 0.0,
) -> float:
    """Probability of a significant result, if the effect is exactly this size.

    Args:
        baseline_rate: Control conversion rate.
        relative_lift: Effect as a fraction of baseline. May be negative.
        per_arm: Units in each arm.
        alpha: Two-sided significance level.
        variance_reduction: Fraction of variance removed by CUPED, in [0, 1).
            Enters as an effective sample size of ``n / (1 - reduction)``,
            which is the whole reason P3 exists.

    Returns:
        Power in [0, 1].
    """
    _validate(baseline_rate, alpha, 0.5)
    if per_arm < 1:
        raise ValueError(f"per_arm must be at least 1, got {per_arm}")
    if not 0.0 <= variance_reduction < 1.0:
        raise ValueError(f"variance_reduction must be in [0, 1), got {variance_reduction}")

    treatment = baseline_rate * (1.0 + relative_lift)
    if not 0.0 < treatment < 1.0:
        raise ValueError(f"a {relative_lift:+.1%} lift puts the treatment rate at {treatment}")

    delta = abs(treatment - baseline_rate)
    if delta == 0.0:
        return alpha  # no effect: significance happens at exactly the false-positive rate

    effective = per_arm / (1.0 - variance_reduction)
    pooled = (baseline_rate + treatment) / 2.0

    # Under the null both arms share the pooled rate; under the alternative
    # each has its own. Using the pooled variance for the critical value and
    # the unpooled one for the effect is the standard two-proportion form.
    null_sd = math.sqrt(2.0 * pooled * (1.0 - pooled))
    alt_sd = math.sqrt(baseline_rate * (1.0 - baseline_rate) + treatment * (1.0 - treatment))
    critical = _NORMAL.inv_cdf(1.0 - alpha / 2.0)

    z = (delta * math.sqrt(effective) - critical * null_sd) / alt_sd
    return float(_NORMAL.cdf(z))


def sample_size_per_arm(
    baseline_rate: float,
    relative_lift: float,
    power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
    variance_reduction: float = 0.0,
) -> SampleSize:
    """Units per arm needed to detect a relative lift at the given power.

    Args:
        baseline_rate: Control conversion rate.
        relative_lift: Effect as a fraction of baseline, e.g. ``0.01`` for a
            1% relative lift. **Not** percentage points -- see the module
            docstring.
        power: Target power.
        alpha: Two-sided significance level.
        variance_reduction: CUPED reduction, which divides the requirement.

    Returns:
        The requirement, with the assumptions it rests on.

    Raises:
        ValueError: On a rate outside (0, 1), a zero lift, or a lift that puts
            the treatment rate outside (0, 1).
    """
    _validate(baseline_rate, alpha, power)
    if relative_lift == 0.0:
        raise ValueError("a zero lift needs infinite traffic; state the lift worth detecting")
    if not 0.0 <= variance_reduction < 1.0:
        raise ValueError(f"variance_reduction must be in [0, 1), got {variance_reduction}")

    treatment = baseline_rate * (1.0 + relative_lift)
    if not 0.0 < treatment < 1.0:
        raise ValueError(f"a {relative_lift:+.1%} lift puts the treatment rate at {treatment}")

    delta = abs(treatment - baseline_rate)
    pooled = (baseline_rate + treatment) / 2.0
    null_sd = math.sqrt(2.0 * pooled * (1.0 - pooled))
    alt_sd = math.sqrt(baseline_rate * (1.0 - baseline_rate) + treatment * (1.0 - treatment))

    critical = _NORMAL.inv_cdf(1.0 - alpha / 2.0)
    beta = _NORMAL.inv_cdf(power)
    required = ((critical * null_sd + beta * alt_sd) / delta) ** 2
    required *= 1.0 - variance_reduction

    per_arm = math.ceil(required)
    return SampleSize(
        per_arm=per_arm,
        total=2 * per_arm,
        baseline_rate=baseline_rate,
        relative_lift=relative_lift,
        absolute_lift=treatment - baseline_rate,
        treatment_rate=treatment,
        power=power,
        alpha=alpha,
        variance_reduction=variance_reduction,
    )


def minimum_detectable_effect(
    baseline_rate: float,
    per_arm: int,
    power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
    variance_reduction: float = 0.0,
) -> float:
    """The smallest relative lift this many units could detect.

    The question to ask of an experiment that came back flat: if the MDE is
    +12% and the hoped-for effect was +2%, the result is not evidence of no
    effect, it is evidence that the experiment could not have seen one.

    Found by bisection on :func:`achieved_power` rather than by inverting the
    formula, so this and :func:`sample_size_per_arm` are guaranteed to be
    consistent with each other instead of merely nearly so.

    Returns:
        The relative lift, positive.
    """
    _validate(baseline_rate, alpha, power)
    if per_arm < 1:
        raise ValueError(f"per_arm must be at least 1, got {per_arm}")

    # The largest lift the rate admits: beyond it the treatment rate reaches 1
    # and the question stops being well posed. Held strictly below, because
    # achieved_power refuses a rate of exactly 1.0 -- and a bisection that
    # proposes an impossible lift would raise rather than converge.
    ceiling = ((1.0 - baseline_rate) / baseline_rate) * (1.0 - 1e-9)
    low, high = 1e-9, min(1.0, ceiling)

    # Grow until the bracket covers the target power.
    while (
        high < ceiling
        and achieved_power(baseline_rate, high, per_arm, alpha, variance_reduction) < power
    ):
        high = min(high * 2.0, ceiling)

    for _ in range(200):
        middle = (low + high) / 2.0
        if achieved_power(baseline_rate, middle, per_arm, alpha, variance_reduction) < power:
            low = middle
        else:
            high = middle
    return high
