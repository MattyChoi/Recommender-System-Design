"""Feature names and orderings shared across training and serving for both retrieval and ranking."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import numpy.typing as npt

#: Spark's ``dayofweek`` is 1..7 with **Sunday = 1**, which is why
#: :func:`build_user_features` subtracts one before the cyclic encoding.
#: Python's ``datetime.weekday()`` is 0..6 with Monday = 0 -- a caller that
#: passes that directly produces a correctly-shaped, correctly-named vector
#: rotated by a constant, which no width or range check can see.
DAY_OF_WEEK_ORIGIN = 1

USER_VIEW = "user_stats"

# The RAW user block, in the order GetUserResponse.values uses.
USER_FEATURES = (
    "user_impressions_24h",
    "user_clicks_24h",
    "user_ctr_smoothed",
    "user_tenure_hours",
)
# Unbounded counts and durations. log1p before the tower's LayerNorm, or the
# first Linear sees a column ranging over four orders of magnitude next to one
# bounded in [0, 1].
LOG1P_USER_FEATURES = frozenset({"user_impressions_24h", "user_clicks_24h", "user_tenure_hours"})
USER_FLAGS = ("has_user_features",)
CONTENT_VARIANTS = ("vec_title_abstract", "vec_title")

USER_TOWER_COLUMNS: tuple[str, ...] = (
    *USER_FEATURES,
    *USER_FLAGS,
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
)

ITEM_VIEW = "item_stats"

# The per-item block GetItems serves, row-major, one row per candidate.
ITEM_COLUMNS: tuple[str, ...] = (
    "item_impressions_24h",
    "item_clicks_24h",
    "item_ctr_smoothed",
    "item_age_hours",
    "item_ctr_effective",
    "has_item_features",
)

# Counts are log1p'd
LOG1P_ITEM_COLUMNS = frozenset({"item_impressions_24h", "item_clicks_24h"})


def effective_item_ctr(
    item_ctr_smoothed: float | None, cat_expanding_ctr: float | None
) -> float | None:
    """``item_ctr_effective``: an item's own smoothed CTR, else its category prior."""
    return item_ctr_smoothed if item_ctr_smoothed is not None else cat_expanding_ctr


def item_has_own_history(item_ctr_smoothed: float | None) -> bool:
    """``has_item_features``: whether the item had a rate before the fallback."""
    return item_ctr_smoothed is not None


def _cyclic(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Sin/cos encoding of a cyclic integer feature."""
    radians = 2.0 * np.pi * values / period
    return np.sin(radians), np.cos(radians)


def spark_day_of_week(when: datetime) -> int:
    """Python's Monday=1 .. Sunday=7 -> Spark's Sunday=1 .. Saturday=7.

    Getting this wrong rotates the cyclic encoding by a constant and passes
    every shape, name, dtype and range check there is.
    """
    return (when.isoweekday() % 7) + 1


def build_user_features(
    static: dict[str, npt.NDArray[np.float32]],
    has_user_features: npt.NDArray[np.float32],
    hour_of_day: npt.NDArray[np.float32],
    day_of_week: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """The tower's ``[R, 9]`` static input, from raw columns.

    Args:
        static: The four :data:`USER_FEATURES`, already null-filled. Each
            ``[R]``. ``log1p`` is applied here, to :data:`LOG1P_USER_FEATURES`
            only, so callers pass RAW counts.
        has_user_features: ``[R]`` 1.0 where the feature store had a row. At
            serving this is ``GetUserResponse.found``; offline it is the
            missingness flag the as-of join wrote.
        hour_of_day: ``[R]`` 0..23.
        day_of_week: ``[R]`` in **Spark's 1..7, Sunday = 1** convention. See
            :data:`DAY_OF_WEEK_ORIGIN`.

    Returns:
        ``[R, 9]`` float32 in :data:`USER_TOWER_COLUMNS` order.

    Raises:
        KeyError: If ``static`` is missing one of :data:`USER_FEATURES`. Raised
            rather than zero-filled: a silently absent column shifts every
            later column left by one.
    """
    numeric: list[npt.NDArray[np.float32]] = []
    for name in USER_FEATURES:
        column = static[name]
        numeric.append(np.log1p(column) if name in LOG1P_USER_FEATURES else column)
    numeric.append(has_user_features)
    numeric.extend(_cyclic(hour_of_day, 24))
    numeric.extend(_cyclic(day_of_week - DAY_OF_WEEK_ORIGIN, 7))

    stacked: npt.NDArray[np.float32] = np.stack(numeric, axis=1).astype("float32")
    return stacked


def feature_refs(view: str, columns: tuple[str, ...]) -> list[str]:
    """Feast's "view:column" references, in this module's declared order."""
    return [f"{view}:{column}" for column in columns]
