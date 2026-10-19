from __future__ import annotations

import numpy as np
import pytest

from common.features import (
    LOG1P_USER_FEATURES,
    USER_FEATURES,
    USER_TOWER_COLUMNS,
    build_user_features,
    spark_day_of_week,
)

# Spelled out literally rather than derived from LOG1P_USER_FEATURES. Deriving
# it would make the arithmetic test below reduce to "the loop ran": the
# implementation branches on that same constant, so both sides would move
# together and the membership -- which is the thing Checkpoint 1 turned on --
# would stop being asserted at all.
#
# Three transformed, not two. `user_tenure_hours` is a DURATION and it is
# transformed, because what log1p is for here is unbounded range, not counting.
# Take the membership from LOG1P_USER_FEATURES, never from the word "count".
TRANSFORMED = ("user_impressions_24h", "user_clicks_24h", "user_tenure_hours")
UNTRANSFORMED = ("user_ctr_smoothed",)


def test_tower_width_is_nine_not_four() -> None:
    assert len(USER_TOWER_COLUMNS) == len(USER_FEATURES) + 5


def test_the_flag_changes_the_vector() -> None:
    static = {name: np.zeros(1, dtype="float32") for name in USER_FEATURES}
    hour, dow = np.zeros(1, "float32"), np.ones(1, "float32")
    absent = build_user_features(static, np.zeros(1, "float32"), hour, dow)
    present = build_user_features(static, np.ones(1, "float32"), hour, dow)
    # A zeroed KNOWN user and an UNKNOWN user must not encode identically.
    assert not np.array_equal(absent, present)


@pytest.mark.parametrize(
    ("iso", "spark"),
    [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 1)],
)
def test_day_of_week_matches_sparks_origin(iso: int, spark: int) -> None:
    class Fake:
        def isoweekday(self) -> int:
            return iso

    assert spark_day_of_week(Fake()) == spark  # type: ignore[arg-type]


def test_cyclic_columns_are_on_the_unit_circle() -> None:
    static = {name: np.zeros(3, dtype="float32") for name in USER_FEATURES}
    hours = np.asarray([0.0, 6.0, 23.0], dtype="float32")
    dow = np.asarray([1.0, 4.0, 7.0], dtype="float32")
    out = build_user_features(static, np.ones(3, "float32"), hours, dow)
    hour_sin, hour_cos = out[:, -4], out[:, -3]
    np.testing.assert_allclose(hour_sin**2 + hour_cos**2, 1.0, atol=1e-6)


def test_the_literal_membership_still_matches_the_constants() -> None:
    """The drift guard for the test below, and the only place the two meet.

    If someone adds a feature to ``USER_FEATURES`` or moves one into
    ``LOG1P_USER_FEATURES``, this fails and makes them update ``TRANSFORMED``
    deliberately. Without it the arithmetic test would quietly keep passing
    while no longer covering the new column.
    """
    assert set(TRANSFORMED) | set(UNTRANSFORMED) == set(USER_FEATURES)
    assert set(TRANSFORMED) == set(LOG1P_USER_FEATURES)


def test_log1p_applies_to_the_counts_and_the_duration_but_not_the_rate() -> None:
    """Every slot, against a literal expectation -- not just the rate.

    The rate column must pass through untransformed: log1p on a smoothed CTR
    is a different feature that still lies in [0, 1) and looks entirely fine,
    so nothing downstream would flag it.
    """
    static = {name: np.full(1, 3.0, dtype="float32") for name in USER_FEATURES}
    out = build_user_features(
        static,
        np.ones(1, "float32"),
        np.zeros(1, "float32"),
        np.ones(1, "float32"),
    )
    for slot, name in enumerate(USER_FEATURES):
        expected = np.log1p(3.0) if name in TRANSFORMED else 3.0
        # `name` in the message: a bare index tells you a slot is wrong, not
        # which feature it was meant to hold.
        assert out[0, slot] == pytest.approx(expected), name
