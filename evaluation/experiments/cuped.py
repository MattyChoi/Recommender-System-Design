"""CUPED -- strip the variance a pre-experiment covariate already explains.

The treatment effect is unchanged in expectation; only the noise around it
shrinks, so the same experiment reaches significance on less traffic. The
reduction is exactly the squared correlation between the metric and the
covariate, which is why choosing the covariate matters far more than anything
in this module: a covariate correlated 0.7 with the metric removes half the
variance, one correlated 0.2 removes four percent and is not worth the code.

**The covariate must be measured strictly BEFORE assignment.** A covariate the
treatment can touch does not de-noise the estimate, it biases it -- the same
point-in-time discipline as Part D, wearing a different costume. Nothing here
can check that for you; :func:`covariate_balance` is the closest available,
and it detects only the loud cases.

Two corrections to the usual formulation, both of which change the answer:

1. **θ is estimated once, over BOTH arms, and applied to both.** Estimating it
   per arm, or centring the covariate on each arm's own mean, silently cancels
   the adjustment out of the difference between arms and leaves the raw
   estimate wearing a CUPED label -- see :func:`cuped`.
2. **The variance terms share a ddof.** Writing ``np.cov(y, x)[0, 1] /
   np.var(x)`` mixes ddof=1 in the numerator with ddof=0 in the denominator, so
   θ comes out scaled by (n-1)/n. Harmless at a million rows, wrong at two
   hundred, and wrong for no reason at either.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class CUPED(NamedTuple):
    """An adjusted metric and the evidence for trusting it.

    Attributes:
        adjusted: The de-noised metric, aligned with the input.
        theta: The coefficient applied, whether estimated or supplied.
        correlation: Pearson correlation of metric and covariate. This is what
            decides the reduction; everything else is bookkeeping.
        variance_reduction: Fraction of variance removed, in [0, 1). Equal to
            ``correlation ** 2`` up to estimation error when theta was fitted
            on this same sample.
        degenerate: The covariate was constant, so there was nothing to strip.
            ``adjusted`` is the input unchanged and ``theta`` is 0.
    """

    adjusted: npt.NDArray[np.float64]
    theta: float
    correlation: float
    variance_reduction: float
    degenerate: bool

    def summary(self) -> str:
        if self.degenerate:
            return "CUPED inert: the covariate is constant"
        return (
            f"CUPED: theta={self.theta:.4f} r={self.correlation:.3f} "
            f"variance -{self.variance_reduction:.1%}"
        )


def theta_for(y: npt.ArrayLike, x: npt.ArrayLike) -> float:
    """The OLS slope of the metric on the covariate, cov(y, x) / var(x).

    Both terms take ddof=1. The ratio is what matters, so the ddof cancels --
    which is exactly why mixing them does not.

    Returns:
        0.0 when the covariate has no variance, so the caller gets an inert
        adjustment rather than a NaN that poisons every downstream number.
    """
    metric = np.asarray(y, dtype=np.float64)
    covariate = np.asarray(x, dtype=np.float64)

    variance = float(np.var(covariate, ddof=1))
    if variance == 0.0:
        return 0.0
    covariance = float(np.cov(metric, covariate, ddof=1)[0, 1])
    return covariance / variance


def cuped(y: npt.ArrayLike, x: npt.ArrayLike, theta: float | None = None) -> CUPED:
    """Adjust a metric by a pre-experiment covariate.

    Args:
        y: The metric, one value per unit (user, session) -- **both arms
            together**.
        x: The covariate, same order and length, measured before assignment.
        theta: Override the fitted coefficient. Supply one estimated on a
            previous period to remove even the small bias from fitting θ on
            the same data the effect is read from.

    Returns:
        The adjusted metric and its diagnostics.

    Raises:
        ValueError: On a length mismatch or fewer than two units.

    Note:
        **Pass both arms in one call.** The adjustment is
        ``y - theta * (x - mean(x))``, and the ``mean(x)`` must be the SAME
        constant for every unit. Call this once per arm and each arm is centred
        on its own mean, so the per-arm correction becomes
        ``theta * (x_arm_mean - x_arm_mean) = 0`` in the difference -- the
        estimate reverts to the unadjusted one while every label still says
        CUPED. It does not error, it does not look wrong, and the variance
        reduction it reports is real within each arm. That is the trap.
    """
    metric = np.asarray(y, dtype=np.float64)
    covariate = np.asarray(x, dtype=np.float64)

    if metric.shape != covariate.shape:
        raise ValueError(f"metric is {metric.shape}, covariate is {covariate.shape}")
    if metric.ndim != 1:
        raise ValueError(f"expected one value per unit, got shape {metric.shape}")
    if metric.size < 2:
        raise ValueError("CUPED needs at least two units")

    coefficient = theta_for(metric, covariate) if theta is None else float(theta)
    if coefficient == 0.0 and float(np.var(covariate, ddof=1)) == 0.0:
        return CUPED(
            adjusted=metric.copy(),
            theta=0.0,
            correlation=0.0,
            variance_reduction=0.0,
            degenerate=True,
        )

    adjusted = metric - coefficient * (covariate - covariate.mean())

    before = float(np.var(metric, ddof=1))
    after = float(np.var(adjusted, ddof=1))
    reduction = 0.0 if before == 0.0 else max(0.0, 1.0 - after / before)

    correlation = 0.0
    if before > 0.0 and float(np.var(covariate, ddof=1)) > 0.0:
        correlation = float(np.corrcoef(metric, covariate)[0, 1])

    return CUPED(
        adjusted=adjusted,
        theta=coefficient,
        correlation=correlation,
        variance_reduction=reduction,
        degenerate=False,
    )


def covariate_balance(x: npt.ArrayLike, arms: Sequence[str]) -> dict[str, float]:
    """Standardised mean difference of the covariate, per arm against the rest.

    The continuous analogue of :mod:`evaluation.experiments.srm`. SRM asks
    whether the arms got the right NUMBER of units; this asks whether they got
    comparable ones. Under working randomisation both are near zero, and a
    covariate measured before assignment cannot be imbalanced by the treatment
    -- so a large value here means either the randomisation is broken or the
    covariate is not actually pre-assignment, and both invalidate the CUPED
    adjustment built on it.

    Args:
        x: The covariate, one value per unit.
        arms: Arm label per unit, same order and length.

    Returns:
        Arm name to ``(mean_in_arm - mean_elsewhere) / pooled_sd``. Cohen's
        conventions put 0.1 at negligible and 0.2 at small; anything past 0.1
        on a pre-period covariate is worth explaining before it is used.

    Raises:
        ValueError: On a length mismatch, or fewer than two arms.
    """
    covariate = np.asarray(x, dtype=np.float64)
    labels = np.asarray(arms, dtype=object)

    if covariate.shape != labels.shape:
        raise ValueError(f"covariate is {covariate.shape}, arms is {labels.shape}")
    names = sorted(set(labels.tolist()))
    if len(names) < 2:
        raise ValueError(f"need at least two arms, got {names}")

    pooled_sd = float(np.std(covariate, ddof=1))
    out: dict[str, float] = {}
    for name in names:
        inside = covariate[labels == name]
        outside = covariate[labels != name]
        if inside.size == 0 or outside.size == 0 or pooled_sd == 0.0:
            out[str(name)] = 0.0
            continue
        out[str(name)] = float(inside.mean() - outside.mean()) / pooled_sd
    return out
