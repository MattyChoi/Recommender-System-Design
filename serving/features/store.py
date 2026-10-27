"""Everything that talks to Feast or reads an artifact. No gRPC in this file."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from feast import FeatureStore

from common.features import (
    ITEM_COLUMNS,
    ITEM_VIEW,
    USER_FEATURES,
    USER_VIEW,
    effective_item_ctr,
    feature_refs,
    item_has_own_history,
)
from serving.features.cache import ItemFeatureCache

FEATURE_REPO = Path("data_pipeline/features/recsys_store/feature_repo")
ITEM_MAP = Path("serving/artifacts/item_map.json")
CATEGORY_MAP = Path("serving/artifacts/category_map.json")

DEFAULT_MAX_HISTORY = 50

# ITEM_COLUMNS, split by where each value actually comes from.
#
# `cat_expanding_ctr` is FETCHED BUT NOT SERVED: it is the category prior that
# backs item_ctr_effective for an item with no history of its own.
_FETCH_FROM_ITEM_STATS: tuple[str, ...] = (
    "item_impressions_24h",
    "item_clicks_24h",
    "item_ctr_smoothed",
    "item_age_hours",
    "cat_expanding_ctr",
)

# Derived here from the columns above rather than read from Feast's
# `derived_features` on-demand view. That view also sources
# `user_category_stats`, which is keyed on (user_id, category) -- join keys an
# item-keyed call does not carry and must not need.
_DERIVED: tuple[str, ...] = ("item_ctr_effective", "has_item_features")

# The check that catches a column with nowhere to come from.
_UNASSIGNED = set(ITEM_COLUMNS) - set(_FETCH_FROM_ITEM_STATS) - set(_DERIVED)
if _UNASSIGNED:
    raise RuntimeError(
        f"ITEM_COLUMNS has columns with no source: {sorted(_UNASSIGNED)}. Add "
        "them to _FETCH_FROM_ITEM_STATS or _DERIVED -- or, if they are "
        "user-keyed, serve them from GetUser instead."
    )


@dataclass(frozen=True)
class UserRow:
    """User block parallel to GetUserResponse."""

    values: list[float]
    found: bool
    history: list[int]
    unmapped: int
    cat_affinity: list[float]


@dataclass(frozen=True)
class ItemRows:
    """The flattened per-item block, plus how much of it came from cache.

    Parallel to GetItemsResponse
    """

    values: list[float]
    found: list[bool]
    cached: int


class Store:
    """Feast's online store."""

    def __init__(
        self,
        repo: Path = FEATURE_REPO,
        item_map: Path = ITEM_MAP,
        category_map: Path = CATEGORY_MAP,
        cache: ItemFeatureCache | None = None,
    ) -> None:
        self._store = FeatureStore(repo_path=str(repo))

        item_id_idx_map = json.loads(item_map.read_text())
        self.item_map_version: str = item_id_idx_map["version"]
        self._to_index: dict[str, int] = {
            str(id): int(idx) for id, idx in item_id_idx_map["items"].items()
        }
        self._to_id: dict[int, str] = {
            item_idx: item_id for item_id, item_idx in self._to_index.items()
        }

        vocabulary = json.loads(category_map.read_text())
        self.category_map_version: str = vocabulary["version"]
        self._categories: list[str] = vocabulary["categories"]

        self._cache = cache

    def _translate(self, external: list[str], limit: int) -> tuple[list[int], int]:
        """item_id -> item_idx, most recent first.

        An id the map does not know is DROPPED and COUNTED, never mapped to 0:
        index 0 is the reserved OOV row, and sending it as history would feed
        the tower the padding vector as if it were a click.
        """
        out: list[int] = []
        unmapped = 0
        for item_id in external[:limit]:
            index = self._to_index.get(item_id)
            if index is None:
                unmapped += 1
                continue
            out.append(index)
        return out, unmapped

    def _affinity(self, user_id: str) -> list[float]:
        """The user's affinity for EVERY category, indexed by category index."""
        names = self._categories[1:]
        if not names:
            return []

        got = self._store.get_online_features(
            features=["user_category_stats:user_cat_affinity"],
            entity_rows=[{"user_id": user_id, "category": name} for name in names],
        ).to_dict()

        # Position 0 is the reserved bucket and is ALWAYS 0.0.
        affinity = [0.0]
        affinity.extend(
            float(value) if value is not None else 0.0 for value in got["user_cat_affinity"]
        )
        return affinity

    def _fetch_items(self, indices: list[int]) -> dict[int, tuple[tuple[float, ...], bool]]:
        """ONE batched read for every miss, then the two local derivations.

        One call, not one per item. Returns ``index -> (values, found)``, the
        shape `ItemFeatureCache.put` takes. An index the item map does not know
        is left out: Feast was never asked, so there is no answer to cache.
        """
        known = [index for index in indices if index in self._to_id]
        if not known:
            return {}

        entity_rows = [{"item_id": self._to_id[index]} for index in known]
        got = self._store.get_online_features(
            features=feature_refs(ITEM_VIEW, _FETCH_FROM_ITEM_STATS), entity_rows=entity_rows
        ).to_dict()

        rows: dict[int, tuple[tuple[float, ...], bool]] = {}
        for position, index in enumerate(known):
            computed: dict[str, float | bool | None] = {
                name: got[name][position] for name in _FETCH_FROM_ITEM_STATS
            }
            # Feast answers a missing entity with all-None, not an error.
            found = any(value is not None for value in computed.values())

            smoothed = got["item_ctr_smoothed"][position]
            prior = got["cat_expanding_ctr"][position]
            computed["item_ctr_effective"] = effective_item_ctr(smoothed, prior)
            computed["has_item_features"] = item_has_own_history(smoothed)

            values = tuple(
                float(value) if value is not None else 0.0
                for value in (computed[name] for name in ITEM_COLUMNS)
            )
            rows[index] = (values, found)

        return rows

    def user(self, user_id: str, max_history: int = DEFAULT_MAX_HISTORY) -> UserRow:
        """The raw block, the found flag, the translated history, the OOV count."""
        refs = feature_refs(USER_VIEW, USER_FEATURES)
        refs.append("user_realtime:last_50_items")

        got = self._store.get_online_features(
            features=refs, entity_rows=[{"user_id": user_id}]
        ).to_dict()

        raw = [got[name][0] for name in USER_FEATURES]
        found = all(value is not None for value in raw)

        # Zeros on a miss
        values = [float(value) if value is not None else 0.0 for value in raw]

        history, unmapped = self._translate(got["last_50_items"][0] or [], max_history)

        return UserRow(
            values=values,
            found=found,
            history=history,
            unmapped=unmapped,
            cat_affinity=self._affinity(user_id),
        )

    def items(self, indices: list[int]) -> ItemRows:
        """ITEM_COLUMNS per index, FLATTENED row-major, plus the cache count.

        Row-major because the response is one packed `repeated float`, not 400
        messages. The reshape arithmetic lives in the Go client, which checks
        `len(values) == len(items) * len(names)` and refuses a mismatch.
        """
        if self._cache is None:
            hits, wanted = {}, list(dict.fromkeys(indices))
        else:
            hits, wanted = self._cache.take(indices)

        rows = {index: (entry.values, entry.found) for index, entry in hits.items()}
        if wanted:
            fetched = self._fetch_items(wanted)
            if self._cache is not None:
                self._cache.put(fetched)
            rows.update(fetched)

        missing = ((0.0,) * len(ITEM_COLUMNS), False)
        values: list[float] = []
        found: list[bool] = []
        cached = 0
        for index in indices:
            row, row_found = rows.get(index, missing)
            values.extend(row)
            found.append(row_found)
            # Per position, not per distinct index: a repeated hit is served twice.
            cached += index in hits

        return ItemRows(values=values, found=found, cached=cached)

    def warm(self) -> float:
        """One throwaway read per view, so the first request skips Feast's lazy setup."""
        started = time.perf_counter()
        for view, columns, key in (
            (USER_VIEW, USER_FEATURES, "user_id"),
            (ITEM_VIEW, _FETCH_FROM_ITEM_STATS, "item_id"),
        ):
            try:
                self._store.get_online_features(
                    features=feature_refs(view, columns),
                    entity_rows=[{key: "__warmup__"}],
                ).to_dict()
            except Exception as exc:
                print(f"  warmup of {view} failed ({exc}); the first request will pay for it")
        return time.perf_counter() - started