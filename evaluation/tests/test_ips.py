"""Off-policy estimation: that it recovers a known value, and that it refuses."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.counterfactual.ips import (
    DEFAULT_CLIP,
    effective_sample_size,
    snips,
)


class TestEffectiveSampleSize:
    def test_equal_weights_give_back_the_row_count(self) -> None:
        assert effective_sample_size(np.ones(500)) == pytest.approx(500.0)
        assert effective_sample_size(np.full(500, 7.0)) == pytest.approx(500.0)

    def test_one_dominant_weight_collapses_it(self) -> None:
        weights = np.concatenate([[1_000.0], np.full(999, 1e-6)])
        assert effective_sample_size(weights) < 1.01

    def test_it_sits_between_one_and_n(self) -> None:
        rng = np.random.default_rng(1)
        weights = rng.exponential(1.0, 2_000)
        ess = effective_sample_size(weights)
        assert 1.0 <= ess <= 2_000.0
        # An exponential spread loses roughly half the sample.
        assert 800 < ess < 1_200

    def test_empty_and_all_zero_are_zero_rather_than_nan(self) -> None:
        assert effective_sample_size([]) == 0.0
        assert effective_sample_size([0.0, 0.0]) == 0.0


class TestItRecoversAKnownValue:
    def test_a_uniform_shift_is_estimated_correctly(self) -> None:
        """Two actions, a known logging policy, a known target policy.

        The target's true value is computable by hand, which is the only way
        to test an estimator without reimplementing it in the test.
        """
        rng = np.random.default_rng(2)
        n = 200_000

        # Logging policy: action 1 with probability 0.5.
        # Target policy:  action 1 with probability 0.8.
        # Rewards: action 1 pays 1.0 with p=0.30, action 0 with p=0.10.
        action = rng.random(n) < 0.5
        reward = np.where(
            action,
            (rng.random(n) < 0.30).astype(float),
            (rng.random(n) < 0.10).astype(float),
        )
        logging = np.where(action, 0.5, 0.5)
        target = np.where(action, 0.8, 0.2)

        truth = 0.8 * 0.30 + 0.2 * 0.10  # 0.26
        result = snips(reward, logging, target)

        assert result.snips == pytest.approx(truth, abs=0.005)
        assert result.reliable
        # 0.5 throughout is a working randomisation, not a deterministic log.
        assert not result.deterministic_logging

    def test_an_unchanged_policy_returns_the_logged_average(self) -> None:
        rng = np.random.default_rng(3)
        reward = rng.random(10_000)
        probs = np.full(10_000, 0.25)
        result = snips(reward, probs, probs)
        assert result.snips == pytest.approx(reward.mean(), rel=1e-9)
        assert result.effective_sample_size == pytest.approx(10_000.0)


class TestTheGuards:
    def test_a_zero_logging_probability_is_refused(self) -> None:
        """The defect in the usual formulation.

        target/0 is inf, and clipping inf yields the MAXIMUM weight -- so the
        row carrying the least information gets the most influence.
        """
        with pytest.raises(ValueError, match="logging probability <= 0"):
            snips([1.0, 1.0, 1.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.5])

    def test_the_naive_formulation_would_have_maximised_that_weight(self) -> None:
        """Demonstrates what is being prevented, rather than asserting it."""
        with np.errstate(divide="ignore"):
            naive = np.clip(np.asarray([0.5]) / np.asarray([0.0]), 0, DEFAULT_CLIP)
        assert naive[0] == DEFAULT_CLIP

    def test_unsupported_rows_can_be_dropped_deliberately(self) -> None:
        result = snips(
            [1.0, 99.0, 1.0],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.5],
            drop_unsupported=True,
        )
        assert result.dropped_unsupported == 1
        assert result.rows == 2
        assert result.snips == pytest.approx(1.0)  # the 99.0 row is gone
        assert "dropped" in result.summary()

    def test_a_deterministic_log_is_flagged_as_the_naive_average(self) -> None:
        """The state this repo's own log was in until the reranker was fixed."""
        rng = np.random.default_rng(4)
        reward = rng.random(5_000)
        result = snips(reward, np.ones(5_000), np.ones(5_000))
        assert result.deterministic_logging
        assert not result.reliable
        assert result.snips == pytest.approx(reward.mean(), rel=1e-9)
        assert "DETERMINISTIC LOG" in result.summary()

    def test_equal_propensities_below_one_are_not_deterministic(self) -> None:
        """The distinction the flag turns on.

        A 50/50 randomisation logs 0.5 for every action it takes. The
        propensities are all equal and the log supports IPS perfectly well,
        because the alternative action WAS reachable.
        """
        rng = np.random.default_rng(5)
        n = 20_000
        action = rng.random(n) < 0.5
        reward = np.where(action, 1.0, 0.0)
        result = snips(reward, np.full(n, 0.5), np.where(action, 0.9, 0.1))
        assert not result.deterministic_logging
        assert result.reliable

    def test_clipping_is_counted_and_its_mass_reported(self) -> None:
        # One row wants a weight of 100; the cap is 10.
        result = snips(
            [1.0, 1.0, 1.0, 1.0],
            [0.01, 0.5, 0.5, 0.5],
            [1.0, 0.5, 0.5, 0.5],
        )
        assert result.clipped == 1
        # 10 of a total weight of 13.
        assert result.clipped_mass == pytest.approx(10.0 / 13.0, rel=1e-9)
        assert not result.reliable
        assert "cap" in result.summary()

    def test_a_collapsed_ess_makes_it_unreliable(self) -> None:
        logging = np.concatenate([[0.001], np.full(999, 0.5)])
        target = np.concatenate([[0.5], np.full(999, 0.01)])
        result = snips(np.ones(1_000), logging, target)
        assert result.ess_fraction < 0.1
        assert not result.reliable

    def test_a_target_that_would_never_act_gives_nan_rather_than_zero(self) -> None:
        result = snips([1.0, 1.0], [0.5, 0.5], [0.0, 0.0])
        assert np.isnan(result.snips)
        assert result.effective_sample_size == 0.0


class TestSnipsAgainstIps:
    def test_self_normalisation_lowers_the_variance(self) -> None:
        """The bias/variance trade, measured rather than asserted.

        SNIPS divides by the realised weight sum, so a replication that drew
        unusually heavy weights is not also credited with unusually large
        rewards. It is slightly biased and markedly steadier.
        """
        truth = 0.3
        snips_estimates: list[float] = []
        ips_estimates: list[float] = []

        for seed in range(60):
            rng = np.random.default_rng(500 + seed)
            n = 2_000
            action = rng.random(n) < 0.2
            reward = np.where(action, (rng.random(n) < truth).astype(float), 0.0)
            logging = np.where(action, 0.2, 0.8)
            target = np.where(action, 0.9, 0.1)
            result = snips(reward, logging, target)
            snips_estimates.append(result.snips)
            ips_estimates.append(result.ips)

        assert np.std(snips_estimates, ddof=1) < np.std(ips_estimates, ddof=1)

    def test_both_are_reported_so_a_gap_is_visible(self) -> None:
        """A large SNIPS/IPS gap means the weights are badly behaved.

        With every weight at 1.0 the two coincide exactly, which is the
        reference case; the summary carries both either way so the comparison
        never needs a second call.
        """
        result = snips([1.0, 0.0, 1.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert result.snips == pytest.approx(2.0 / 3.0)
        assert result.ips == pytest.approx(2.0 / 3.0)
        assert "SNIPS=" in result.summary()
        assert "IPS=" in result.summary()


class TestMalformedInput:
    @pytest.mark.parametrize(
        ("rewards", "logging", "target", "match"),
        [
            ([1.0, 1.0], [0.5], [0.5, 0.5], "must match"),
            ([], [], [], "no rows"),
            ([1.0], [0.5], [-0.1], "negative probability"),
        ],
    )
    def test_it_is_refused(
        self, rewards: list[float], logging: list[float], target: list[float], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            snips(rewards, logging, target)

    def test_a_non_positive_clip_is_refused(self) -> None:
        with pytest.raises(ValueError, match="clip must be positive"):
            snips([1.0], [0.5], [0.5], clip=0.0)

    def test_two_dimensional_input_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one value per logged row"):
            snips([[1.0, 1.0]], [[0.5, 0.5]], [[0.5, 0.5]])
