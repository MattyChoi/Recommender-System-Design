"""CUPED: that it de-noises, that it does not move the effect, and the trap."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from evaluation.experiments.cuped import covariate_balance, cuped, theta_for


def correlated(
    n: int, rho: float, seed: int, shift: float = 0.0
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A covariate and a metric correlated at approximately ``rho``.

    ``shift`` is added to the metric only, so it stands in for a treatment
    effect that the covariate -- being pre-assignment -- knows nothing about.
    """
    rng = np.random.default_rng(seed)
    covariate = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 1.0, n)
    metric = rho * covariate + np.sqrt(1.0 - rho**2) * noise + shift
    return metric, covariate


class TestTheta:
    def test_it_is_the_ols_slope(self) -> None:
        metric, covariate = correlated(5_000, 0.6, seed=1)
        slope = np.polyfit(covariate, metric, 1)[0]
        assert theta_for(metric, covariate) == pytest.approx(slope, rel=1e-9)

    def test_the_ddof_does_not_leak_into_the_ratio(self) -> None:
        """The manual's formulation mixes ddof=1 and ddof=0 and is off by (n-1)/n.

        Small n, so the two would differ by 10% rather than by nothing.
        """
        metric, covariate = correlated(10, 0.8, seed=2)
        mixed = float(np.cov(metric, covariate)[0, 1] / np.var(covariate))
        correct = theta_for(metric, covariate)
        assert correct != pytest.approx(mixed, rel=1e-6)
        assert correct == pytest.approx(mixed * 9 / 10, rel=1e-9)

    def test_a_constant_covariate_gives_zero_rather_than_nan(self) -> None:
        assert theta_for([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) == 0.0


class TestVarianceReduction:
    @pytest.mark.parametrize("rho", [0.2, 0.5, 0.7, 0.9])
    def test_the_reduction_is_the_squared_correlation(self, rho: float) -> None:
        metric, covariate = correlated(50_000, rho, seed=3)
        result = cuped(metric, covariate)
        assert result.variance_reduction == pytest.approx(rho**2, abs=0.02)
        assert result.correlation == pytest.approx(rho, abs=0.02)

    def test_an_uncorrelated_covariate_removes_almost_nothing(self) -> None:
        metric, covariate = correlated(20_000, 0.0, seed=4)
        assert cuped(metric, covariate).variance_reduction < 0.01

    def test_a_constant_covariate_is_inert_rather_than_an_error(self) -> None:
        result = cuped([1.0, 2.0, 3.0, 4.0], [7.0, 7.0, 7.0, 7.0])
        assert result.degenerate
        assert result.theta == 0.0
        assert np.allclose(result.adjusted, [1.0, 2.0, 3.0, 4.0])
        assert "inert" in result.summary()


class TestTheEffectSurvives:
    def test_it_corrects_a_chance_covariate_imbalance(self) -> None:
        """On ANY ONE sample CUPED does not reproduce the raw estimate.

        It corrects it, and the gap between the two IS the correction. These
        two arms differ slightly in pre-period covariate by chance, which drags
        the raw difference away from the true effect; the adjustment removes
        exactly that much. A test asserting the two estimates agree would be
        asserting that CUPED does nothing.
        """
        effect = 0.30
        control_y, control_x = correlated(20_000, 0.7, seed=5)
        treat_y, treat_x = correlated(20_000, 0.7, seed=6, shift=effect)

        metric = np.concatenate([control_y, treat_y])
        covariate = np.concatenate([control_x, treat_x])
        is_treated = np.concatenate([np.zeros(20_000, bool), np.ones(20_000, bool)])

        raw = metric[is_treated].mean() - metric[~is_treated].mean()
        adjusted = cuped(metric, covariate).adjusted
        lifted = adjusted[is_treated].mean() - adjusted[~is_treated].mean()

        assert abs(lifted - effect) < abs(raw - effect)

    def test_it_is_unbiased_across_replications_with_less_spread(self) -> None:
        """The claim that actually justifies CUPED, over many experiments.

        Unbiasedness is a property of the SAMPLING DISTRIBUTION, so it cannot
        be read off one sample -- it needs replications. With rho = 0.7 the
        variance reduction is rho^2 = 0.49, so the spread should fall by about
        sqrt(1 - 0.49) = 0.71.
        """
        effect = 0.30
        units = 2_000
        raw_effects: list[float] = []
        adjusted_effects: list[float] = []

        for replication in range(40):
            control_y, control_x = correlated(units, 0.7, seed=100 + 2 * replication)
            treat_y, treat_x = correlated(units, 0.7, seed=101 + 2 * replication, shift=effect)
            metric = np.concatenate([control_y, treat_y])
            covariate = np.concatenate([control_x, treat_x])
            treated = np.concatenate([np.zeros(units, bool), np.ones(units, bool)])

            raw_effects.append(float(metric[treated].mean() - metric[~treated].mean()))
            adjusted = cuped(metric, covariate).adjusted
            adjusted_effects.append(float(adjusted[treated].mean() - adjusted[~treated].mean()))

        raw = np.asarray(raw_effects)
        adjusted_array = np.asarray(adjusted_effects)

        assert raw.mean() == pytest.approx(effect, abs=0.02)
        assert adjusted_array.mean() == pytest.approx(effect, abs=0.02)
        assert adjusted_array.std(ddof=1) < 0.8 * raw.std(ddof=1)

    def test_the_standard_error_of_the_effect_actually_shrinks(self) -> None:
        """The claim that justifies the whole technique, measured."""
        control_y, control_x = correlated(10_000, 0.8, seed=7)
        treat_y, treat_x = correlated(10_000, 0.8, seed=8, shift=0.2)

        metric = np.concatenate([control_y, treat_y])
        covariate = np.concatenate([control_x, treat_x])
        adjusted = cuped(metric, covariate).adjusted

        def standard_error(values: npt.NDArray[np.float64]) -> float:
            half = values.size // 2
            a, b = values[:half], values[half:]
            return float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))

        assert standard_error(adjusted) < 0.7 * standard_error(metric)

    def test_per_arm_calls_silently_cancel_the_adjustment(self) -> None:
        """The trap the module docstring names, demonstrated.

        Adjusting each arm separately centres the covariate on that arm's own
        mean, so the correction vanishes from the DIFFERENCE between arms --
        and nothing errors, nothing looks wrong, and the per-arm variance
        reduction it reports is real.
        """
        control_y, control_x = correlated(20_000, 0.7, seed=9)
        treat_y, treat_x = correlated(20_000, 0.7, seed=10, shift=0.30)

        raw = treat_y.mean() - control_y.mean()
        per_arm = (
            cuped(treat_y, treat_x).adjusted.mean() - cuped(control_y, control_x).adjusted.mean()
        )

        # Identical to the unadjusted estimate: the adjustment did nothing.
        assert per_arm == pytest.approx(raw, abs=1e-9)

    def test_a_supplied_theta_is_used_verbatim(self) -> None:
        metric, covariate = correlated(1_000, 0.6, seed=11)
        result = cuped(metric, covariate, theta=0.25)
        assert result.theta == 0.25
        assert np.allclose(result.adjusted, metric - 0.25 * (covariate - covariate.mean()))


class TestCovariateBalance:
    def test_randomised_arms_are_balanced(self) -> None:
        rng = np.random.default_rng(12)
        covariate = rng.normal(0.0, 1.0, 40_000)
        arms = rng.choice(["control", "treatment"], size=40_000)
        balance = covariate_balance(covariate, list(arms))
        assert all(abs(value) < 0.05 for value in balance.values())

    def test_a_covariate_that_depends_on_the_arm_is_caught(self) -> None:
        """What a post-treatment covariate looks like from here."""
        rng = np.random.default_rng(13)
        arms = ["control"] * 10_000 + ["treatment"] * 10_000
        covariate = np.concatenate([rng.normal(0.0, 1.0, 10_000), rng.normal(0.5, 1.0, 10_000)])
        balance = covariate_balance(covariate, arms)
        assert balance["treatment"] > 0.2
        assert balance["control"] < -0.2

    def test_one_arm_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two arms"):
            covariate_balance([1.0, 2.0, 3.0], ["only"] * 3)

    def test_a_length_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="covariate is"):
            covariate_balance([1.0, 2.0, 3.0], ["a", "b"])


class TestMalformedInput:
    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="metric is"):
            cuped([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_too_few_units(self) -> None:
        with pytest.raises(ValueError, match="at least two units"):
            cuped([1.0], [1.0])

    def test_two_dimensional_input(self) -> None:
        with pytest.raises(ValueError, match="one value per unit"):
            cuped([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]])
