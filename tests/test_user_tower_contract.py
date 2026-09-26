"""The serving path must produce the vector the tower was fitted on.

``test_served_covers_trained.py`` already checks that every trained feature is
*available in the store*. That is a weaker claim than it sounds, and it passed
while the feature gateway was five columns short: the store held
``hour_of_day`` and ``day_of_week``, the gateway simply never assembled them.

The failure that slipped through was total -- every retrieval call returned
InvalidArgument and the orchestrator served the popularity fallback on every
request -- and the only thing that caught it was the sidecar's runtime width
check, on the first real request the system ever handled. These tests move that
check to CI.

Annotated in full despite ``tests/*`` being exempt from ruff's ANN rules:
mypy runs strict over the whole tree and does not share that exemption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from models.retrieval.dataloader.dataset import (
    LOG1P_FEATURES,
    USER_FEATURES,
    USER_FLAGS,
    USER_TOWER_COLUMNS,
    build_user_features,
)
from serving.features.columns import USER_COLUMNS

#: The checkpoint every serving target defaults to. Absent in CI, where the
#: width assertion is skipped rather than faked.
CHECKPOINT = Path("data/checkpoints/both-logq-n4u0-b8192e10lr0.001-ab7e1d500b9bf792.pt")


def _row(
    raw: float = 1.0,
    has_user_features: float = 1.0,
    hour_of_day: float = 13.0,
    day_of_week: float = 3.0,
) -> npt.NDArray[np.float32]:
    """One request's worth of inputs, as serving supplies them."""
    return build_user_features(
        static={name: np.array([raw], dtype="float32") for name in USER_FEATURES},
        has_user_features=np.array([has_user_features], dtype="float32"),
        hour_of_day=np.array([hour_of_day], dtype="float32"),
        day_of_week=np.array([day_of_week], dtype="float32"),
    )


def test_the_builder_emits_one_column_per_declared_name() -> None:
    assert _row().shape == (1, len(USER_TOWER_COLUMNS))


def test_the_gateway_columns_are_the_prefix_of_the_tower_columns() -> None:
    """The gateway owns the RAW block; the builder appends the derived five.

    A prefix, not a subset: the builder stacks the static columns first and in
    order, so a gateway that reordered them would still pass a set comparison
    and would still produce a believable embedding for a user who does not
    exist.
    """
    assert tuple(USER_COLUMNS) == USER_TOWER_COLUMNS[: len(USER_COLUMNS)]


def test_the_derived_columns_are_the_flag_and_two_cyclic_pairs() -> None:
    derived = USER_TOWER_COLUMNS[len(USER_COLUMNS) :]

    assert derived == (*USER_FLAGS, "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos")


def test_log1p_is_applied_to_counts_and_not_to_the_rate() -> None:
    """``user_ctr_smoothed`` is a rate in [0, 1]; log1p on it is not a no-op.

    Pinned with a value whose transformed and untransformed forms differ, so a
    LOG1P_FEATURES that quietly gained or lost a member fails here.
    """
    raw = 7.0
    row = _row(raw=raw, has_user_features=0.0, hour_of_day=0.0, day_of_week=1.0)[0]

    for position, name in enumerate(USER_FEATURES):
        expected = np.log1p(raw) if name in LOG1P_FEATURES else raw
        assert row[position] == pytest.approx(expected), name


def test_the_day_of_week_origin_is_sparks_and_not_pythons() -> None:
    """Spark's dayofweek is 1..7 (Sunday = 1); datetime.weekday() is 0..6.

    Passing the wrong one rotates the encoding by a constant and is invisible to
    every width, name, dtype and range check. The origin is pinned by asserting
    that the FIRST day of the week lands at angle zero: sin 0, cos 1.
    """
    row = _row(raw=0.0, has_user_features=0.0, hour_of_day=0.0, day_of_week=1.0)[0]

    sin_at = USER_TOWER_COLUMNS.index("day_of_week_sin")
    cos_at = USER_TOWER_COLUMNS.index("day_of_week_cos")

    assert row[sin_at] == pytest.approx(0.0, abs=1e-6)
    assert row[cos_at] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="no checkpoint on this machine")
def test_the_tower_was_fitted_with_exactly_these_columns() -> None:
    """The assertion that matters, against a real checkpoint.

    ``user_norm`` is a LayerNorm over the static block, so its weight width IS
    the feature count the tower was fitted with -- the same quantity
    ``serving/retrieval/service.py`` reads back at load time to validate an
    incoming request. Skipped rather than faked where no checkpoint exists,
    because a mocked width would assert this file against itself.
    """
    import torch

    state = torch.load(CHECKPOINT, map_location="cpu")
    state = state.get("model", state)

    assert int(state["user_norm.weight"].shape[0]) == len(USER_TOWER_COLUMNS)
