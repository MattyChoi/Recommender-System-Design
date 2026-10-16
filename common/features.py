"""Feature names and orderings shared across training and serving for both retrieval and ranking."""

from __future__ import annotations

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


# The per-item block GetItems serves, row-major, one row per candidate.
ITEM_COLUMNS: tuple[str, ...] = (
    "item_impressions_24h",
    "item_clicks_24h",
    "item_ctr_smoothed",
    "item_age_hours",
    "item_ctr_effective",
    "has_item_features",
    "user_cat_affinity",
)

# Counts are log1p'd
LOG1P_ITEM_COLUMNS = frozenset({"item_impressions_24h", "item_clicks_24h"})
