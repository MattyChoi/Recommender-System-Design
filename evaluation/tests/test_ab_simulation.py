"""The simulator: that it reproduces what it was calibrated to, and reacts."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.simulation.ab import (
    DEFAULT_USER_SIGMA,
    Arm,
    Corpus,
    Policy,
    calibrate_relevance,
    examination,
    simulate,
    simulate_arm,
)

CATALOGUE = 500


def zipf_weights(size: int = CATALOGUE, exponent: float = 1.1) -> np.ndarray:
    """A popularity distribution shaped like a real one."""
    raw = 1.0 / (np.arange(size, dtype=np.float64) + 1.0) ** exponent
    return raw / raw.sum()


def corpus(session_ctr: float = 0.25, slate_size: int = 10, catalogue: int = CATALOGUE) -> Corpus:
    return Corpus(
        slate_size=slate_size,
        session_ctr=session_ctr,
        item_weights=zipf_weights(catalogue),
        head_size=50,
        latency_p50_ms=8.0,
        latency_sigma=0.35,
    )


CONTROL = Policy(name="control")
TREATMENT = Policy(name="treatment", relevance_lift=0.15, epsilon=0.05, latency_ms_shift=2.0)


class TestCalibration:
    @pytest.mark.parametrize("target", [0.02, 0.10, 0.25, 0.50])
    def test_the_simulated_rate_is_the_calibrated_rate(self, target: float) -> None:
        """Calibration is the claim that makes the baseline meaningful."""
        control, _ = simulate(CONTROL, CONTROL, corpus(target), 60_000, seed=1)
        assert control.ctr == pytest.approx(target, abs=0.01)

    def test_relevance_solves_the_click_equation_over_the_population(self) -> None:
        """The solve integrates over users rather than evaluating at the mean.

        P(click | u) is concave in u, so the two differ -- and calibrating at
        the mean user overstates the realised rate. The second assertion is
        the bug this exists to avoid, pinned so a revert would fail here.
        """
        relevance = calibrate_relevance(0.25, 10)
        weights = examination(10)

        rng = np.random.default_rng(99)
        users = rng.lognormal(0.0, DEFAULT_USER_SIGMA, 200_000)
        users /= users.mean()
        per_slot = np.clip(weights[None, :] * relevance * users[:, None], 0.0, 1.0)
        marginal = float((1.0 - np.prod(1.0 - per_slot, axis=1)).mean())
        assert marginal == pytest.approx(0.25, abs=0.003)

        at_the_mean_user = 1.0 - np.prod(1.0 - np.clip(weights * relevance, 0.0, 1.0))
        assert at_the_mean_user > 0.25

    def test_an_unreachable_rate_is_refused_rather_than_approximated(self) -> None:
        with pytest.raises(ValueError, match="unreachable"):
            calibrate_relevance(0.999, slate_size=2)

    def test_gamma_moves_slots_not_the_session_rate(self) -> None:
        """Why conclusions about the primary metric survive the assumption."""
        for gamma in (0.4, 0.8, 1.4):
            control, _ = simulate(CONTROL, CONTROL, corpus(0.25), 40_000, seed=2, gamma=gamma)
            assert control.ctr == pytest.approx(0.25, abs=0.01), gamma


class TestTheTreatmentShowsUp:
    def test_a_relevance_lift_raises_the_click_rate(self) -> None:
        control, treatment = simulate(CONTROL, TREATMENT, corpus(), 60_000, seed=3)
        assert treatment.ctr > control.ctr
        # Not a claim about the exact mapping from relevance lift to CTR lift,
        # which the examination curve mediates -- only that it is substantial.
        assert (treatment.ctr - control.ctr) / control.ctr > 0.05

    def test_no_lift_means_no_difference_beyond_noise(self) -> None:
        control, treatment = simulate(CONTROL, Policy("treatment"), corpus(), 60_000, seed=4)
        assert abs(treatment.ctr - control.ctr) < 0.01

    def test_exploration_widens_exposure(self) -> None:
        """The guardrail that the exposure work in Part O made legible.

        The catalogue has to be large relative to the draws or the measure
        saturates: 200,000 impressions over 500 items reach every one of them
        in BOTH arms, and coverage stops discriminating -- which is the same
        denominator trap docs/exposure.md documents.
        """
        wide = corpus(catalogue=20_000)
        head = wide.head_mask()
        control, treatment = simulate(CONTROL, TREATMENT, wide, 20_000, seed=5)
        assert treatment.distinct_items() > control.distinct_items()
        assert treatment.long_tail_share(head) > control.long_tail_share(head)

    def test_the_head_is_by_popularity_not_by_item_index(self) -> None:
        """The bug a sorted-by-construction fixture cannot catch.

        zipf_weights happens to be ordered descending, so an item's index IS
        its popularity rank and comparing ``items >= head_size`` works by
        coincidence. Shuffle the catalogue -- as a real item_map is, since it
        orders items by nothing -- and the index comparison collapses to
        "is this identifier small", which is a long-tail share of ~1.0 and a
        guardrail that cannot move.
        """
        rng = np.random.default_rng(42)
        shuffled = zipf_weights(5_000)
        rng.shuffle(shuffled)
        scrambled = Corpus(
            slate_size=10,
            session_ctr=0.25,
            item_weights=shuffled,
            head_size=50,
            latency_p50_ms=8.0,
            latency_sigma=0.35,
        )
        head = scrambled.head_mask()
        assert int(head.sum()) == 50
        # The 50 flagged items are the 50 heaviest, wherever they landed.
        assert np.allclose(np.sort(shuffled[head])[::-1], np.sort(shuffled)[::-1][:50])

        control, _ = simulate(CONTROL, CONTROL, scrambled, 5_000, seed=43)
        by_rank = control.long_tail_share(head)
        by_index = float(
            np.count_nonzero(control.items >= scrambled.head_size) / control.items.size
        )
        assert by_rank < 0.5  # the head carries most of the exposure, as it should
        assert by_index > 0.9  # what the broken comparison reported

    def test_a_latency_cost_is_visible(self) -> None:
        control, treatment = simulate(CONTROL, TREATMENT, corpus(), 20_000, seed=6)
        assert np.median(treatment.latency_ms) > np.median(control.latency_ms)
        assert np.median(treatment.latency_ms) == pytest.approx(10.0, abs=0.5)


class TestPropensities:
    def test_an_exploring_policy_logs_varying_propensities(self) -> None:
        """What P4 needs, and what the real log lacked until the flags were wired."""
        _, treatment = simulate(CONTROL, TREATMENT, corpus(), 5_000, seed=7)
        assert treatment.logging_prob.min() < treatment.logging_prob.max()
        assert np.all(treatment.logging_prob > 0.0)

    def test_the_counterfactual_probability_is_the_other_policy(self) -> None:
        control, _ = simulate(CONTROL, TREATMENT, corpus(), 5_000, seed=8)
        # Control explores not at all, treatment at epsilon=0.05, so the two
        # policies assign different probabilities to the same shown item.
        assert not np.allclose(control.logging_prob, control.counterfactual_prob)

    def test_a_non_exploring_pair_has_identical_propensities(self) -> None:
        """Two deterministic policies leave the log unable to tell them apart."""
        control, _ = simulate(CONTROL, Policy("treatment"), corpus(), 2_000, seed=9)
        assert np.allclose(control.logging_prob, control.counterfactual_prob)


class TestTheCovariate:
    def test_it_predicts_the_outcome(self) -> None:
        """Otherwise CUPED has nothing to strip."""
        control, _ = simulate(CONTROL, CONTROL, corpus(), 40_000, seed=10)
        correlation = np.corrcoef(control.pre_ctr, control.clicked.astype(float))[0, 1]
        assert correlation > 0.05

    def test_it_does_not_depend_on_the_arm(self) -> None:
        """Pre-assignment by construction: the treatment cannot touch it."""
        control, treatment = simulate(CONTROL, TREATMENT, corpus(), 40_000, seed=11)
        pooled_sd = float(np.std(np.concatenate([control.pre_ctr, treatment.pre_ctr]), ddof=1))
        gap = abs(control.pre_ctr.mean() - treatment.pre_ctr.mean()) / pooled_sd
        assert gap < 0.05


class TestReproducibility:
    def test_one_seed_gives_one_experiment(self) -> None:
        first = simulate(CONTROL, TREATMENT, corpus(), 5_000, seed=12)
        second = simulate(CONTROL, TREATMENT, corpus(), 5_000, seed=12)
        for a, b in zip(first, second, strict=True):
            assert np.array_equal(a.clicked, b.clicked)
            assert np.array_equal(a.items, b.items)

    def test_different_seeds_give_different_experiments(self) -> None:
        first, _ = simulate(CONTROL, TREATMENT, corpus(), 5_000, seed=13)
        second, _ = simulate(CONTROL, TREATMENT, corpus(), 5_000, seed=14)
        assert not np.array_equal(first.clicked, second.clicked)


class TestShapesAndRefusals:
    def test_the_record_is_shaped_per_session_and_slot(self) -> None:
        control, _ = simulate(CONTROL, TREATMENT, corpus(slate_size=7), 1_000, seed=15)
        assert isinstance(control, Arm)
        assert control.sessions == 1_000
        assert control.items.shape == (1_000, 7)
        assert control.logging_prob.shape == (1_000, 7)
        assert control.clicked.shape == (1_000,)

    def test_weights_that_do_not_sum_to_one_are_refused(self) -> None:
        with pytest.raises(ValueError, match="sum to"):
            Corpus(
                slate_size=5,
                session_ctr=0.2,
                item_weights=np.full(100, 0.5),
                head_size=10,
                latency_p50_ms=8.0,
                latency_sigma=0.35,
            )

    def test_a_catalogue_smaller_than_the_slate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="slate_size long"):
            Corpus(
                slate_size=50,
                session_ctr=0.2,
                item_weights=zipf_weights(10),
                head_size=5,
                latency_p50_ms=8.0,
                latency_sigma=0.35,
            )

    def test_zero_sessions_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sessions must be"):
            simulate_arm(CONTROL, TREATMENT, corpus(), 0, np.random.default_rng(0))
