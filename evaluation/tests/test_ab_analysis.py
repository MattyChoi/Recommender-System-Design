"""The launch rule: that each branch fires for its own reason."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.experiments.srm import srm_check
from evaluation.offline.stats import two_sample_bootstrap
from evaluation.simulation.ab import Corpus, Policy, simulate
from evaluation.simulation.analysis import Guardrail, analyse

LATENCY = Guardrail(name="latency (ms)", direction="lower_is_better", tolerance=0.05)
TAIL = Guardrail(name="long-tail share", direction="higher_is_better", tolerance=0.02)
RAILS = (LATENCY, TAIL)


def zipf_weights(size: int = 2_000, exponent: float = 1.1) -> np.ndarray:
    raw = 1.0 / (np.arange(size, dtype=np.float64) + 1.0) ** exponent
    return raw / raw.sum()


def corpus(session_ctr: float = 0.25) -> Corpus:
    return Corpus(
        slate_size=10,
        session_ctr=session_ctr,
        item_weights=zipf_weights(),
        head_size=50,
        latency_p50_ms=8.0,
        latency_sigma=0.35,
    )


CONTROL = Policy(name="control")


class TestTwoSampleBootstrap:
    def test_it_finds_a_real_difference(self) -> None:
        rng = np.random.default_rng(1)
        control = rng.normal(0.0, 1.0, 5_000)
        treatment = rng.normal(0.4, 1.0, 5_000)
        result = two_sample_bootstrap(control, treatment)
        assert result.significant
        assert result.difference == pytest.approx(0.4, abs=0.06)
        assert result.lo < 0.4 < result.hi

    def test_it_does_not_invent_one(self) -> None:
        rng = np.random.default_rng(2)
        control = rng.normal(0.0, 1.0, 5_000)
        treatment = rng.normal(0.0, 1.0, 5_000)
        assert not two_sample_bootstrap(control, treatment).significant

    def test_unequal_arms_are_handled_rather_than_truncated(self) -> None:
        rng = np.random.default_rng(3)
        result = two_sample_bootstrap(rng.normal(0, 1, 9_000), rng.normal(0.5, 1, 1_000))
        assert result.n_users == 10_000
        assert result.significant

    def test_an_empty_arm_is_refused(self) -> None:
        with pytest.raises(ValueError, match="both arms need units"):
            two_sample_bootstrap([], [1.0, 2.0])


class TestTheRecommendation:
    def test_a_clean_win_launches(self) -> None:
        treatment = Policy("treatment", relevance_lift=0.25, epsilon=0.05)
        control, treated = simulate(CONTROL, treatment, corpus(), 40_000, seed=4)
        result = analyse(control, treated, corpus(), RAILS)

        assert result.recommendation == "LAUNCH"
        assert result.primary.significant
        assert not any(rail.breached for rail in result.guardrails)
        assert "excluding zero" in result.reasons[0]

    def test_a_latency_regression_blocks_a_win(self) -> None:
        """The trade has to be named before the win is celebrated."""
        treatment = Policy("treatment", relevance_lift=0.25, latency_ms_shift=6.0)
        control, treated = simulate(CONTROL, treatment, corpus(), 40_000, seed=5)
        result = analyse(control, treated, corpus(), RAILS)

        assert result.primary.significant and result.primary.absolute > 0
        assert result.recommendation == "DO NOT LAUNCH"
        assert "Guardrail breached" in result.reasons[0]
        assert "latency" in result.reasons[0]

    def test_a_regression_does_not_launch(self) -> None:
        treatment = Policy("treatment", relevance_lift=-0.25)
        control, treated = simulate(CONTROL, treatment, corpus(), 40_000, seed=6)
        result = analyse(control, treated, corpus(), RAILS)

        assert result.recommendation == "DO NOT LAUNCH"
        assert "DOWN" in result.reasons[0]

    def test_a_flat_result_with_power_does_not_launch(self) -> None:
        """Flat and well powered is a decision, not a gap."""
        control, treated = simulate(CONTROL, Policy("treatment"), corpus(), 200_000, seed=7)
        result = analyse(control, treated, corpus(), RAILS, target_lift=0.10)

        assert result.recommendation == "DO NOT LAUNCH"
        assert result.minimum_detectable < 0.10
        assert "adequate power" in result.reasons[0]

    def test_a_flat_result_without_power_is_inconclusive(self) -> None:
        """The branch that stops a null being read as a negative."""
        control, treated = simulate(CONTROL, Policy("treatment"), corpus(), 400, seed=8)
        result = analyse(control, treated, corpus(), RAILS, target_lift=0.02)

        assert result.recommendation == "INCONCLUSIVE"
        assert result.minimum_detectable > 0.02
        assert "underpowered" in result.reasons[0]

    def test_srm_outranks_everything(self) -> None:
        """Even a large, clean win is unreadable if assignment broke."""
        treatment = Policy("treatment", relevance_lift=0.40)
        control, treated = simulate(CONTROL, treatment, corpus(), 40_000, seed=9)
        broken = control._replace(
            clicked=control.clicked[:20_000],
            latency_ms=control.latency_ms[:20_000],
            pre_ctr=control.pre_ctr[:20_000],
            items=control.items[:20_000],
            logging_prob=control.logging_prob[:20_000],
            counterfactual_prob=control.counterfactual_prob[:20_000],
        )
        result = analyse(broken, treated, corpus(), RAILS)

        assert result.srm.detected
        assert result.recommendation == "DO NOT LAUNCH"
        assert "Sample ratio mismatch" in result.reasons[0]


class TestCuped:
    def test_the_adjusted_interval_is_tighter(self) -> None:
        """P3's payoff, inside the decision it was built for."""
        treatment = Policy("treatment", relevance_lift=0.10)
        control, treated = simulate(CONTROL, treatment, corpus(), 60_000, seed=10)
        result = analyse(control, treated, corpus(), RAILS)

        raw_width = result.primary.hi - result.primary.lo
        adjusted_width = result.primary_cuped.hi - result.primary_cuped.lo
        assert adjusted_width < raw_width
        assert result.variance_reduction > 0.0

    def test_the_decision_uses_the_adjusted_metric(self) -> None:
        treatment = Policy("treatment", relevance_lift=0.25)
        control, treated = simulate(CONTROL, treatment, corpus(), 40_000, seed=11)
        result = analyse(control, treated, corpus(), RAILS)
        assert "CUPED" in result.reasons[0]


class TestGuardrailSemantics:
    def test_an_unknown_guardrail_is_refused(self) -> None:
        control, treated = simulate(CONTROL, Policy("t"), corpus(), 2_000, seed=12)
        unknown = Guardrail(name="revenue", direction="higher_is_better", tolerance=0.01)
        with pytest.raises(ValueError, match="no observation for guardrail"):
            analyse(control, treated, corpus(), (unknown,))

    def test_no_guardrails_means_nothing_can_block(self) -> None:
        """An explicit choice, so it reads as one in the report."""
        treatment = Policy("treatment", relevance_lift=0.25, latency_ms_shift=20.0)
        control, treated = simulate(CONTROL, treatment, corpus(), 40_000, seed=13)
        assert analyse(control, treated, corpus(), ()).recommendation == "LAUNCH"

    def test_a_breach_needs_the_interval_not_just_the_estimate(self) -> None:
        """A noisy guardrail must not block a launch on its own noise.

        A latency shift well inside the tolerance moves the point estimate but
        not the bound nearest zero, so it does not breach.
        """
        treatment = Policy("treatment", relevance_lift=0.25, latency_ms_shift=0.1)
        control, treated = simulate(CONTROL, treatment, corpus(), 40_000, seed=14)
        result = analyse(control, treated, corpus(), RAILS)
        latency = next(rail for rail in result.guardrails if rail.name == "latency (ms)")
        assert latency.relative > 0.0
        assert not latency.breached


class TestSrmIsNotSkipped:
    def test_equal_arms_pass_cleanly(self) -> None:
        assert not srm_check([40_000, 40_000], [0.5, 0.5]).detected
