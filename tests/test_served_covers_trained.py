"""Everything the model trains on must be something serving can obtain.

The failure this prevents is silent and one-directional: a feature added to
``attach_point_in_time_features`` and forgotten in the feature store trains a
model whose input the serving layer cannot produce. Nothing offline notices,
because offline never reads the store.

Three sources have to agree:

* ``asof.py``'s constants -- what the as-of joins attach.
* The derived block -- the cold-start fallback and the missingness flags.
* ``ranker_v1`` -- what serving is allowed to request.
"""

from __future__ import annotations

import pytest

from data_pipeline.features.asof import (
    _CATEGORY_FEATURES,
    _ITEM_FEATURES,
    _USER_CATEGORY_FEATURES,
    _USER_FEATURES,
    ZERO_FILLED,
)
from data_pipeline.features.recsys_store.feature_repo.definition import (
    derived_features,
    ranker_v1,
)

JOINED = (
    set(_ITEM_FEATURES)
    | set(_USER_FEATURES)
    | set(_CATEGORY_FEATURES)
    | set(_USER_CATEGORY_FEATURES)
)

# Computed from the label's own timestamp rather than joined from a series, so
# they appear in no asof.py constant.
CONTEXT = {"hour_of_day", "day_of_week"}


def _served() -> set[str]:
    """Every feature name ranker_v1 can return."""
    names: set[str] = set()
    for projection in ranker_v1.feature_view_projections:
        names.update(field.name for field in projection.features)
    return names


def test_every_joined_feature_is_served() -> None:
    missing = JOINED - _served()

    assert not missing, (
        f"trained on but not in ranker_v1: {sorted(missing)}. A model using these cannot be served."
    )


def test_every_derived_feature_is_served() -> None:
    missing = {field.name for field in derived_features.features} - _served()

    assert not missing, f"derived but not served: {sorted(missing)}"


def test_context_features_are_served() -> None:
    assert _served() >= CONTEXT


def test_the_raw_and_effective_ctr_are_both_available_and_distinct() -> None:
    """The bug this whole split exists to prevent.

    item_ctr_smoothed used to be overwritten in place with its own coalesce, so
    the name meant the raw rate online and the fallback-applied rate offline.
    Both must now exist, under different names.
    """
    served = _served()

    assert "item_ctr_smoothed" in served
    assert "item_ctr_effective" in served


@pytest.mark.parametrize("column", sorted(ZERO_FILLED))
def test_zero_filled_columns_are_served_raw(column: str) -> None:
    """The fill is a serving-side contract, so the unfilled value must be there.

    An on-demand view cannot emit a feature under an input's name, so the fill
    is not computed in the store -- it is written down in ZERO_FILLED and
    applied by whoever assembles the request. That only works if serving can
    actually read the column.
    """
    assert column in _served()


def test_zero_filled_are_counts_and_durations_not_rates() -> None:
    """Zero-filling a RATE asserts a measured click-through of zero.

    Zero-filling a COUNT says nothing was observed, which is true. The list must
    never acquire a member whose name reads like a rate.
    """
    assert not [name for name in ZERO_FILLED if "ctr" in name or "affinity" in name]
