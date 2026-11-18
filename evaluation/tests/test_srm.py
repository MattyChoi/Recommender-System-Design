"""The SRM guard, and the chi-square survival function it is built on.

The numerics get their own class, checked against published critical values
rather than against another implementation of the same thing -- a hand-rolled
special function tested only against itself is a tautology.
"""

from __future__ import annotations

import math

import pytest

# The two incomplete-gamma branches, imported privately on purpose: the point
# of the test below is that they agree at the switch, and that cannot be asked
# through the public function, which picks one and never shows its working.
from evaluation.experiments.srm import (
    DEFAULT_ALPHA,
    MIN_EXPECTED,
    SRM,
    _lower_series,
    _upper_fraction,
    chi_square_sf,
    srm_check,
)


class TestChiSquareSurvival:
    # Upper-tail critical values from standard chi-square tables: the
    # statistic at which the survival function equals the stated alpha.
    @pytest.mark.parametrize(
        ("df", "critical", "alpha"),
        [
            (1, 3.841, 0.05),
            (1, 6.635, 0.01),
            (1, 10.828, 0.001),
            (2, 5.991, 0.05),
            (2, 9.210, 0.01),
            (2, 13.816, 0.001),
            (3, 7.815, 0.05),
            (3, 11.345, 0.01),
            (4, 9.488, 0.05),
            (5, 11.070, 0.05),
            (10, 18.307, 0.05),
            (10, 29.588, 0.001),
            (20, 31.410, 0.05),
        ],
    )
    def test_matches_published_critical_values(
        self, df: int, critical: float, alpha: float
    ) -> None:
        # Tables are quoted to three decimals, so the tolerance is the table's,
        # not the implementation's.
        assert chi_square_sf(critical, df) == pytest.approx(alpha, abs=5e-4)

    def test_the_two_branches_agree_where_they_meet(self) -> None:
        """x < a+1 takes the series, x >= a+1 the continued fraction.

        Both are evaluated at the SAME point, which is the only way to compare
        them: sampling either side of the switch instead measures the slope of
        the function, not the agreement of the algorithms. Both converge at
        x = a + 1; they merely converge at different rates, which is what the
        switch is for.

        A disagreement here would be invisible in the table above, whose
        critical values never land near a boundary.
        """
        for df in (1, 2, 3, 7, 12):
            a = df / 2.0
            x = a + 1.0
            series = 1.0 - _lower_series(a, x)
            fraction = _upper_fraction(a, x)
            assert series == pytest.approx(fraction, rel=1e-9), f"df={df}"

    def test_it_is_continuous_across_the_switch(self) -> None:
        """No step at the boundary, to the precision the slope allows.

        The survival function has a slope there, so two nearby points cannot
        be equal -- they may only differ by about the density times the step.
        """
        for df in (1, 2, 3, 7, 12):
            boundary = float(df + 2)  # x = a + 1, in statistic units
            step = 1e-9
            gap = abs(chi_square_sf(boundary - step, df) - chi_square_sf(boundary + step, df))
            assert gap < 1e-7, f"df={df}: a step of {gap:.2e} is a discontinuity, not a slope"

    def test_it_is_monotone_and_bounded(self) -> None:
        previous = 1.0
        for statistic in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0]:
            value = chi_square_sf(statistic, 3)
            assert 0.0 <= value <= 1.0
            assert value <= previous
            previous = value

    def test_zero_is_certain_and_the_far_tail_underflows_to_zero(self) -> None:
        assert chi_square_sf(0.0, 1) == 1.0
        assert chi_square_sf(-1.0, 1) == 1.0
        assert chi_square_sf(1000.0, 2) == pytest.approx(0.0, abs=1e-200)

    def test_zero_degrees_of_freedom_is_refused(self) -> None:
        with pytest.raises(ValueError, match="degrees_of_freedom"):
            chi_square_sf(1.0, 0)


class TestSRM:
    def test_a_balanced_split_does_not_fire(self) -> None:
        result = srm_check([50_123, 49_877], [0.5, 0.5])
        assert not result.detected
        assert result.degrees_of_freedom == 1
        assert result.reliable

    def test_a_broken_split_fires(self) -> None:
        # 52/48 over 100k is a catastrophic imbalance, not a coin flip.
        result = srm_check([52_000, 48_000], [0.5, 0.5])
        assert result.detected
        assert result.p_value < DEFAULT_ALPHA

    def test_the_same_imbalance_is_noise_when_small(self) -> None:
        """What the ratio alert in alerts.yml cannot do.

        A 52/48 split is a defect at 100,000 assignments and nothing at all at
        200. `RecsysExperimentImbalance` fires on the ratio in both cases; this
        test is the reason the chi-square belongs in the analysis path.
        """
        big = srm_check([52_000, 48_000], [0.5, 0.5])
        small = srm_check([104, 96], [0.5, 0.5])
        assert big.detected
        assert not small.detected

    def test_an_uneven_allocation_is_tested_against_its_own_ratios(self) -> None:
        result = srm_check([8_980, 1_020], [0.9, 0.1])
        assert not result.detected
        assert result.expected == pytest.approx((9_000.0, 1_000.0))

    def test_the_holdback_counts_as_an_arm(self) -> None:
        """Experiment.Names() includes Unassigned; so must the ratios."""
        result = srm_check([2_500, 2_500, 5_000], [0.25, 0.25, 0.5])
        assert not result.detected
        assert result.degrees_of_freedom == 2

    def test_omitting_the_holdback_is_refused_rather_than_renormalised(self) -> None:
        # The failure this prevents: testing two arms against each other while
        # half the traffic sat in a holdback nobody passed in.
        with pytest.raises(ValueError, match="holdback"):
            srm_check([2_500, 2_500], [0.25, 0.25])

    def test_small_arms_are_flagged_unreliable_rather_than_refused(self) -> None:
        result = srm_check([3, 2], [0.5, 0.5])
        assert not result.reliable
        assert result.min_expected < MIN_EXPECTED
        assert "UNRELIABLE" in result.summary()

    def test_a_clean_result_says_so_in_one_line(self) -> None:
        summary = srm_check([50_000, 50_000], [0.5, 0.5]).summary()
        assert summary.startswith("no SRM")
        assert "UNRELIABLE" not in summary

    @pytest.mark.parametrize(
        ("observed", "ratios", "match"),
        [
            ([100], [1.0], "at least two"),
            ([100, 100, 100], [0.5, 0.5], "against"),
            ([-1, 100], [0.5, 0.5], "negative"),
            ([0, 0], [0.5, 0.5], "no assignments"),
            ([100, 100], [0.5, 0.6], "holdback"),
            ([100, 100], [1.0, 0.0], "positive share"),
        ],
    )
    def test_malformed_input_is_refused(
        self, observed: list[int], ratios: list[float], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            srm_check(observed, ratios)

    def test_the_statistic_is_the_textbook_one(self) -> None:
        """Pearson by hand, so the implementation cannot drift silently."""
        observed, ratios = [60, 40], [0.5, 0.5]
        result = srm_check(observed, ratios)
        expected = 100 * 0.5
        by_hand = sum((count - expected) ** 2 / expected for count in observed)
        assert result.statistic == pytest.approx(by_hand)

    def test_the_result_carries_what_it_was_computed_from(self) -> None:
        result = srm_check([30, 70], [0.4, 0.6])
        assert isinstance(result, SRM)
        assert result.observed == (30, 70)
        assert math.isclose(sum(result.expected), 100.0)
