"""Turn two simulated arms into a launch recommendation.

The order of the checks is the point, and it is not the order people
instinctively use:

1. **SRM first.** If assignment is broken the experiment did not happen, and
   every number below it is a description of the breakage. This is checked
   before the primary metric is even computed.
2. **Guardrails next.** A win on the primary metric that costs latency or
   catalogue coverage is a trade, and the trade has to be named before the win
   is celebrated -- §1's coverage floor exists because optimising engagement
   alone collapses onto the head of the catalogue.
3. **The primary metric last**, with its interval rather than its p-value.
4. **And if it is flat, the MDE.** A non-significant result from an
   underpowered experiment is not evidence of no effect; it is evidence the
   experiment could not have found one. Reporting "no significant difference"
   without the power behind it is how a real effect gets shipped as a null --
   or a real regression gets shipped as a nothing.

The recommendation is a function of those four, not a judgement call, so the
same inputs always produce the same call and the rule itself can be argued
with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np

from evaluation.experiments.cuped import cuped
from evaluation.experiments.srm import SRM, srm_check
from evaluation.offline.stats import two_sample_bootstrap
from evaluation.simulation.ab import Arm, Corpus
from evaluation.simulation.power import achieved_power, minimum_detectable_effect

Direction = Literal["higher_is_better", "lower_is_better"]

#: The relative lift the experiment was designed to detect. Quoted in the
#: report so a flat result can be read against what it was powered for.
DEFAULT_TARGET_LIFT = 0.02


@dataclass(frozen=True)
class Guardrail:
    """A metric that must not degrade, and by how much it may.

    Attributes:
        name: Label for the report.
        direction: Which way is good.
        tolerance: Relative degradation tolerated, as a positive fraction.
            0.05 on a latency guardrail means "5% slower is acceptable".
    """

    name: str
    direction: Direction
    tolerance: float


class MetricResult(NamedTuple):
    """One metric compared across arms.

    Attributes:
        name: Label.
        control: Control value.
        treatment: Treatment value.
        absolute: ``treatment - control``.
        relative: The same as a fraction of control.
        lo: Lower bound on the absolute difference.
        hi: Upper bound.
        significant: Interval excludes zero.
        direction: Which way is good.
        breached: Degraded past tolerance with confidence. Guardrails only.
    """

    name: str
    control: float
    treatment: float
    absolute: float
    relative: float
    lo: float
    hi: float
    significant: bool
    direction: Direction
    breached: bool

    def summary(self) -> str:
        mark = " BREACHED" if self.breached else ""
        return (
            f"{self.name}: {self.control:.4g} -> {self.treatment:.4g} "
            f"({self.relative:+.2%}, 95% CI [{self.lo:+.4g}, {self.hi:+.4g}]){mark}"
        )


class Experiment(NamedTuple):
    """Everything the launch decision rests on."""

    srm: SRM
    primary: MetricResult
    primary_cuped: MetricResult
    variance_reduction: float
    guardrails: tuple[MetricResult, ...]
    sessions_per_arm: int
    target_lift: float
    minimum_detectable: float
    power_for_target: float
    recommendation: Literal["LAUNCH", "DO NOT LAUNCH", "INCONCLUSIVE"]
    reasons: tuple[str, ...]


def _compare(
    name: str,
    control: np.ndarray,
    treatment: np.ndarray,
    direction: Direction,
    tolerance: float | None,
    seed: int,
) -> MetricResult:
    """Bootstrap one metric and decide whether a guardrail was breached."""
    result = two_sample_bootstrap(control, treatment, seed=seed)
    control_mean = float(np.mean(control))
    relative = result.difference / control_mean if control_mean else float("nan")

    breached = False
    if tolerance is not None and control_mean:
        # The bound NEAREST zero on the bad side: a breach needs the interval
        # to clear the tolerance, not merely the point estimate. A noisy
        # guardrail should not block a launch on its own noise.
        if direction == "lower_is_better":
            breached = (result.lo / control_mean) > tolerance
        else:
            breached = (result.hi / control_mean) < -tolerance

    return MetricResult(
        name=name,
        control=control_mean,
        treatment=float(np.mean(treatment)),
        absolute=result.difference,
        relative=relative,
        lo=result.lo,
        hi=result.hi,
        significant=result.significant,
        direction=direction,
        breached=breached,
    )


def analyse(
    control: Arm,
    treatment: Arm,
    corpus: Corpus,
    guardrails: tuple[Guardrail, ...] = (),
    target_lift: float = DEFAULT_TARGET_LIFT,
    seed: int = 0,
) -> Experiment:
    """Compare two arms and produce a recommendation.

    Args:
        control: The control arm's sessions.
        treatment: The treatment arm's sessions.
        corpus: Used for the long-tail head size.
        guardrails: Which secondary metrics may block a launch, and by how
            much they may move. An empty tuple means nothing can block, which
            is a choice worth making explicitly.
        target_lift: The relative lift the experiment was designed to detect.
        seed: Bootstrap seed.

    Returns:
        The full result, including the recommendation and its reasons.
    """
    # --- 1. SRM, before anything else ---------------------------------------
    #
    # Equal allocation by construction in the simulator, so this is a
    # formality here and a real check on live counts. It runs anyway, because
    # a pipeline that skips it when it is convenient will skip it when it is
    # not.
    srm = srm_check([control.sessions, treatment.sessions], [0.5, 0.5])

    # --- 2. the primary metric, raw and de-noised ---------------------------
    control_clicks = control.clicked.astype(float)
    treatment_clicks = treatment.clicked.astype(float)
    primary = _compare(
        "session CTR", control_clicks, treatment_clicks, "higher_is_better", None, seed
    )

    # CUPED over BOTH arms in one call -- per-arm would cancel the adjustment
    # out of the difference. See evaluation.experiments.cuped.
    pooled_metric = np.concatenate([control_clicks, treatment_clicks])
    pooled_covariate = np.concatenate([control.pre_ctr, treatment.pre_ctr])
    adjustment = cuped(pooled_metric, pooled_covariate)
    split = control.sessions
    primary_cuped = _compare(
        "session CTR (CUPED)",
        adjustment.adjusted[:split],
        adjustment.adjusted[split:],
        "higher_is_better",
        None,
        seed,
    )

    # --- 3. guardrails ------------------------------------------------------
    head = corpus.head_mask()
    # Every guardrail is a PER-SESSION quantity, so the bootstrap resamples
    # the same independent units as the primary metric. A p99 or a distinct-item
    # count is one number for the whole arm, not one per session -- bootstrapping
    # a constant array returns a zero-width interval that looks like certainty
    # and is nothing of the kind. Those belong in the report as observations,
    # not here as blocking checks.
    observed = {
        "latency (ms)": (control.latency_ms, treatment.latency_ms),
        # By popularity RANK via the mask, never by item index -- see
        # Corpus.head_mask for what the index comparison silently computes.
        "long-tail share": (
            (~head[control.items]).mean(axis=1),
            (~head[treatment.items]).mean(axis=1),
        ),
    }
    results: list[MetricResult] = []
    for rail in guardrails:
        if rail.name not in observed:
            raise ValueError(f"no observation for guardrail {rail.name!r}")
        arm_control, arm_treatment = observed[rail.name]
        results.append(
            _compare(rail.name, arm_control, arm_treatment, rail.direction, rail.tolerance, seed)
        )

    # --- 4. power, read against what was asked of the experiment ------------
    baseline = float(control_clicks.mean())
    sessions = min(control.sessions, treatment.sessions)
    reduction = adjustment.variance_reduction
    mde = minimum_detectable_effect(baseline, sessions, variance_reduction=reduction)
    power = achieved_power(baseline, target_lift, sessions, variance_reduction=reduction)

    recommendation, reasons = _recommend(srm, primary_cuped, tuple(results), target_lift, mde)

    return Experiment(
        srm=srm,
        primary=primary,
        primary_cuped=primary_cuped,
        variance_reduction=reduction,
        guardrails=tuple(results),
        sessions_per_arm=sessions,
        target_lift=target_lift,
        minimum_detectable=mde,
        power_for_target=power,
        recommendation=recommendation,
        reasons=reasons,
    )


def _recommend(
    srm: SRM,
    primary: MetricResult,
    guardrails: tuple[MetricResult, ...],
    target_lift: float,
    mde: float,
) -> tuple[Literal["LAUNCH", "DO NOT LAUNCH", "INCONCLUSIVE"], tuple[str, ...]]:
    """The rule, applied in order. Deterministic, so it can be argued with."""
    if srm.detected:
        return "DO NOT LAUNCH", (
            f"Sample ratio mismatch (p={srm.p_value:.2e}). The experiment is invalid "
            "regardless of any metric below it; assignment, logging or filtering is wrong.",
        )

    breached = [rail.name for rail in guardrails if rail.breached]
    if breached:
        return "DO NOT LAUNCH", (
            f"Guardrail breached: {', '.join(breached)}. A primary-metric win that "
            "costs one of these is a trade, and this one is past the tolerance agreed "
            "in advance.",
        )

    if primary.significant and primary.absolute > 0:
        return "LAUNCH", (
            f"{primary.name} is up {primary.relative:+.2%}, 95% CI "
            f"[{primary.lo:+.4g}, {primary.hi:+.4g}] excluding zero, with no guardrail breached.",
        )

    if primary.significant and primary.absolute < 0:
        return "DO NOT LAUNCH", (
            f"{primary.name} is DOWN {primary.relative:+.2%} and the interval excludes zero.",
        )

    if mde > target_lift:
        return "INCONCLUSIVE", (
            f"{primary.name} is flat, but this experiment could only have detected a "
            f"{mde:+.2%} lift and was asked about {target_lift:+.2%}. That is not evidence "
            "of no effect; it is an underpowered experiment. Run longer or raise the bar.",
        )

    return "DO NOT LAUNCH", (
        f"{primary.name} is flat with adequate power: the experiment could have detected "
        f"{mde:+.2%} and found nothing. Shipping costs complexity for no measured gain.",
    )
