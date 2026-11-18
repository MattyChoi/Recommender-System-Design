"""Power and sample size: against published values, and round-tripping."""

from __future__ import annotations

import pytest

from evaluation.simulation.power import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    achieved_power,
    minimum_detectable_effect,
    sample_size_per_arm,
)


class TestSampleSize:
    def test_it_matches_the_textbook_case(self) -> None:
        """5% baseline, 10% relative lift, 80% power, two-sided 5%.

        The standard worked example; published tables put this near 31,000
        per arm. Checked loosely because sources differ on whether to use the
        pooled or unpooled variance for the critical value.
        """
        result = sample_size_per_arm(0.05, 0.10)
        assert 30_000 <= result.per_arm <= 32_000
        assert result.total == 2 * result.per_arm
        assert result.absolute_lift == pytest.approx(0.005)
        assert result.treatment_rate == pytest.approx(0.055)

    def test_a_smaller_effect_costs_quadratically(self) -> None:
        """Halving the lift roughly quadruples the traffic."""
        big = sample_size_per_arm(0.04, 0.20).per_arm
        small = sample_size_per_arm(0.04, 0.10).per_arm
        assert small / big == pytest.approx(4.0, rel=0.05)

    def test_relative_and_absolute_are_not_confused(self) -> None:
        """The ambiguity the module refuses to guess at.

        On a 4% baseline, a 1% RELATIVE lift is 0.04 points; a 1% ABSOLUTE
        lift is a 25% relative improvement. The traffic differs by orders of
        magnitude, which is why the API takes the relative one by name.
        """
        relative_one_percent = sample_size_per_arm(0.04, 0.01)
        absolute_one_point = sample_size_per_arm(0.04, 0.25)

        assert relative_one_percent.absolute_lift == pytest.approx(0.0004)
        assert absolute_one_point.absolute_lift == pytest.approx(0.01)
        assert relative_one_percent.per_arm > 100 * absolute_one_point.per_arm

    def test_more_power_costs_more_traffic(self) -> None:
        assert (
            sample_size_per_arm(0.04, 0.05, power=0.95).per_arm
            > sample_size_per_arm(0.04, 0.05, power=0.80).per_arm
        )

    def test_a_tighter_alpha_costs_more_traffic(self) -> None:
        assert (
            sample_size_per_arm(0.04, 0.05, alpha=0.01).per_arm
            > sample_size_per_arm(0.04, 0.05, alpha=0.05).per_arm
        )

    def test_a_negative_lift_needs_the_same_order_of_traffic(self) -> None:
        up = sample_size_per_arm(0.04, 0.10).per_arm
        down = sample_size_per_arm(0.04, -0.10).per_arm
        assert down == pytest.approx(up, rel=0.1)

    def test_cuped_cuts_the_requirement_proportionally(self) -> None:
        """P3's payoff, stated as traffic."""
        plain = sample_size_per_arm(0.04, 0.05).per_arm
        reduced = sample_size_per_arm(0.04, 0.05, variance_reduction=0.49).per_arm
        assert reduced / plain == pytest.approx(0.51, rel=0.01)

    def test_the_summary_names_both_framings(self) -> None:
        summary = sample_size_per_arm(0.04, 0.05).summary()
        assert "+5.0%" in summary
        assert "absolute" in summary
        assert "per arm" in summary


class TestAchievedPower:
    def test_it_hits_the_target_at_the_computed_sample_size(self) -> None:
        """The consistency that makes the two directions trustworthy."""
        for baseline in (0.01, 0.04, 0.20, 0.50):
            for lift in (0.02, 0.05, 0.20):
                required = sample_size_per_arm(baseline, lift).per_arm
                power = achieved_power(baseline, lift, required)
                assert power == pytest.approx(DEFAULT_POWER, abs=0.005), (baseline, lift)

    def test_power_rises_with_traffic(self) -> None:
        previous = 0.0
        for per_arm in (1_000, 5_000, 20_000, 100_000, 500_000):
            power = achieved_power(0.04, 0.05, per_arm)
            assert power >= previous
            previous = power
        assert previous > 0.99

    def test_an_underpowered_experiment_reports_itself(self) -> None:
        """The number that makes a flat result interpretable."""
        assert achieved_power(0.04, 0.02, 1_000) < 0.10

    def test_no_effect_gives_back_the_false_positive_rate(self) -> None:
        assert achieved_power(0.04, 0.0, 10_000) == pytest.approx(DEFAULT_ALPHA)

    def test_cuped_raises_power_at_fixed_traffic(self) -> None:
        plain = achieved_power(0.04, 0.05, 20_000)
        reduced = achieved_power(0.04, 0.05, 20_000, variance_reduction=0.49)
        assert reduced > plain


class TestMinimumDetectableEffect:
    def test_it_round_trips_with_the_sample_size(self) -> None:
        """MDE(sample_size(lift)) == lift, which is why it bisects."""
        for baseline in (0.02, 0.04, 0.30):
            for lift in (0.03, 0.10, 0.50):
                required = sample_size_per_arm(baseline, lift).per_arm
                recovered = minimum_detectable_effect(baseline, required)
                assert recovered == pytest.approx(lift, rel=0.01), (baseline, lift)

    def test_more_traffic_detects_smaller_effects(self) -> None:
        previous = 1.0
        for per_arm in (1_000, 10_000, 100_000, 1_000_000):
            mde = minimum_detectable_effect(0.04, per_arm)
            assert mde < previous
            previous = mde

    def test_a_small_experiment_can_only_see_a_large_effect(self) -> None:
        assert minimum_detectable_effect(0.04, 500) > 0.30

    def test_cuped_shrinks_the_detectable_effect(self) -> None:
        plain = minimum_detectable_effect(0.04, 20_000)
        reduced = minimum_detectable_effect(0.04, 20_000, variance_reduction=0.49)
        assert reduced < plain


class TestMalformedInput:
    @pytest.mark.parametrize(
        ("baseline", "lift", "match"),
        [
            (0.0, 0.1, "baseline_rate"),
            (1.0, 0.1, "baseline_rate"),
            (-0.1, 0.1, "baseline_rate"),
            (0.04, 0.0, "zero lift"),
            (0.5, 1.5, "treatment rate"),
        ],
    )
    def test_sample_size_refuses(self, baseline: float, lift: float, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            sample_size_per_arm(baseline, lift)

    def test_a_variance_reduction_of_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="variance_reduction"):
            sample_size_per_arm(0.04, 0.05, variance_reduction=1.0)

    def test_zero_units_is_refused(self) -> None:
        with pytest.raises(ValueError, match="per_arm"):
            achieved_power(0.04, 0.05, 0)
