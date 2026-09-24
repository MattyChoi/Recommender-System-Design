"""The feature gateway: Feast's online store, behind gRPC.

Go cannot read Feast. Its online store is written through Feast's SDK in a
Feast-internal protobuf layout, and reimplementing that encoding in the request
path would couple serving to a third-party wire format whose changes look like
corrupt features rather than like a version mismatch. So Feast is used the way
it is meant to be used -- from Python -- and the orchestrator asks this.

Two jobs, deliberately separated by WHEN they happen:

  - `GetUser` runs before the retrieval fan-out. One call, not two, so the
    retriever and the ranker cannot see different snapshots of the same user
    mid-request.
  - `GetItems` runs after filtering, because the candidates do not exist until
    then.

Everything this returns carries its column NAMES alongside its values. See
columns.py: order is the one part of the contract that no downstream shape
check can catch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.pb import features_pb2, features_pb2_grpc
from serving.features.columns import (
    HISTORY_COLUMN,
    HISTORY_VIEW,
    ITEM_COLUMNS,
    ITEM_VIEW,
    MISSING,
    USER_COLUMNS,
    USER_VIEW,
    feature_refs,
)

#: Matches data_pipeline/features/parity.py, which reads the same store.
FEATURE_REPO = Path("data_pipeline/features/recsys_store/feature_repo")

#: The tower pools at most this many, so more is bytes nobody reads. Mirrors
#: data_pipeline.features.user_history.MAX_HISTORY; imported there rather than
#: here to keep this module importable without the Spark-side package.
DEFAULT_MAX_HISTORY = 50


class ItemIndex:
    """External item id to internal index.

    The same `item_map` the Spark job writes, dumped by
    scripts/dump_item_map.py. It lives HERE rather than in the orchestrator
    because this is the only place that needs it in the serving path: Feast
    stores history as external id strings and the tower indexes an embedding
    table by position, so the translation has an obvious owner.
    """

    def __init__(self, mapping: dict[str, int], version: str) -> None:
        self.mapping = mapping
        self.version = version
        # Both directions, built once. GetUser goes external -> internal
        # (history from Feast); GetItems goes internal -> external (candidates
        # from the orchestrator, which speaks indices).
        self.reverse: dict[int, str] = {}
        for external, index in mapping.items():
            if index in self.reverse:
                # Two ids on one index means the reverse has no answer, and
                # whichever won would have this gateway fetch a DIFFERENT
                # article's features for that candidate -- a correctly-shaped
                # row describing the wrong item.
                raise RuntimeError(
                    f"index {index} maps to both {self.reverse[index]!r} and "
                    f"{external!r}; the item map is not invertible"
                )
            self.reverse[index] = external

    def external(self, index: int) -> str | None:
        """The catalogue id for an internal index, or None if unknown."""
        return self.reverse.get(index)

    @classmethod
    def load(cls, path: Path) -> ItemIndex:
        import json

        payload = json.loads(path.read_text())
        items = payload.get("items") or {}
        if not items:
            # An empty map parses fine and turns every user into a cold user.
            # Refused at load, where it is one line in a startup log rather
            # than a silent quality regression nobody can attribute.
            raise RuntimeError(f"item map {path} is empty")
        return cls({str(k): int(v) for k, v in items.items()}, str(payload.get("version", "")))

    def translate(self, external: list[str]) -> tuple[list[int], int]:
        """Internal indices, and how many ids were not in the map.

        Unknown ids are dropped, not zero-filled: index 0 is the reserved OOV
        row, and pooling it into a user's history moves their vector toward a
        row that is not an article. The COUNT is returned because a rising one
        means the map and the store are drifting apart, which otherwise shows
        up as users mysteriously looking colder than they are.
        """
        indices: list[int] = []
        unmapped = 0
        for item in external:
            found = self.mapping.get(item)
            if found is None or found <= 0:
                unmapped += 1
            else:
                indices.append(found)
        return indices, unmapped


@dataclass
class Loaded:
    store: Any
    items: ItemIndex


def load(repo: Path = FEATURE_REPO, item_map: Path | None = None) -> Loaded:
    from feast import FeatureStore

    path = item_map or Path("serving/artifacts/item_map.json")
    return Loaded(store=FeatureStore(repo_path=str(repo)), items=ItemIndex.load(path))


def _column(row: dict[str, Any], name: str, position: int) -> tuple[float, bool]:
    """One value out of Feast's response, and whether it was actually there.

    Feast returns None for a missing entity rather than omitting the key, so
    "absent" and "present but null" arrive identically. Both are treated as a
    miss, which is right: a null is not a number the model can use either.
    """
    values = row.get(name)
    if values is None or position >= len(values) or values[position] is None:
        return MISSING, False
    return float(values[position]), True


class FeaturesServicer(features_pb2_grpc.FeaturesServicer):
    def __init__(self, loaded: Loaded) -> None:
        self.loaded = loaded

    # --- users ---------------------------------------------------------------

    def GetUser(  # noqa: N802 - the name is the proto's
        self, request: features_pb2.GetUserRequest, context: Any
    ) -> features_pb2.GetUserResponse:
        import grpc

        if not request.user_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "user_id is required")

        limit = request.max_history or DEFAULT_MAX_HISTORY
        entity = [{"user_id": request.user_id}]

        row = self.loaded.store.get_online_features(
            features=[
                *feature_refs(USER_VIEW, USER_COLUMNS),
                f"{HISTORY_VIEW}:{HISTORY_COLUMN}",
            ],
            entity_rows=entity,
        ).to_dict()

        values: list[float] = []
        found = False
        for name in USER_COLUMNS:
            value, present = _column(row, name, 0)
            values.append(value)
            # ANY real column counts as a hit. A user with a row but a null
            # tenure is not the same event as a user the store has never seen,
            # and collapsing them would make the miss rate unreadable.
            found = found or present

        raw_history = row.get(HISTORY_COLUMN) or [None]
        external = list(raw_history[0] or [])[:limit]
        history, unmapped = self.loaded.items.translate([str(item) for item in external])

        return features_pb2.GetUserResponse(
            values=values,
            names=list(USER_COLUMNS),
            history=history,
            unmapped_history=unmapped,
            found=found,
        )

    # --- items ---------------------------------------------------------------

    def GetItems(  # noqa: N802 - the name is the proto's
        self, request: features_pb2.GetItemsRequest, context: Any
    ) -> features_pb2.GetItemsResponse:
        import grpc

        if not request.items:
            # Not an error: an empty candidate set is a legal thing for the
            # orchestrator to have after filtering, and it will not call the
            # ranker either.
            return features_pb2.GetItemsResponse(names=list(ITEM_COLUMNS))

        # Validated in one pass before translating, so the error names EVERY
        # unknown index rather than the first. One unknown index is a stale
        # caller; five hundred is two different item maps, and the difference
        # is the first thing whoever reads this wants to know.
        unknown = [index for index in request.items if self.loaded.items.external(index) is None]
        if unknown:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"{len(unknown)} item indices are absent from item map "
                f"{self.loaded.items.version!r} (first: {unknown[:5]}); the caller "
                "and this gateway are using different maps",
            )
        # The `or ""` is unreachable past that guard; it is there so the types
        # line up without asserting.
        external = [self.loaded.items.external(index) or "" for index in request.items]

        rows = self.loaded.store.get_online_features(
            features=feature_refs(ITEM_VIEW, ITEM_COLUMNS),
            entity_rows=[{"item_id": item} for item in external],
        ).to_dict()

        values: list[float] = []
        found: list[bool] = []
        for position in range(len(external)):
            present_any = False
            for name in ITEM_COLUMNS:
                value, present = _column(rows, name, position)
                values.append(value)
                present_any = present_any or present
            found.append(present_any)

        # Flattened row-major, in REQUEST order. The orchestrator reshapes it,
        # and a message per candidate would allocate ~500 objects per request
        # to carry four floats each.
        return features_pb2.GetItemsResponse(values=values, names=list(ITEM_COLUMNS), found=found)

    # --- health --------------------------------------------------------------

    def Health(  # noqa: N802 - the name is the proto's
        self, request: features_pb2.FeaturesHealthRequest, context: Any
    ) -> features_pb2.FeaturesHealthResponse:
        detail = ""
        ready = True
        try:
            self.loaded.store.get_online_features(
                features=feature_refs(USER_VIEW, USER_COLUMNS),
                entity_rows=[{"user_id": "__health__"}],
            ).to_dict()
        except Exception as exc:
            # A probe read rather than a flag: "the SDK constructed" says
            # nothing about whether Redis is reachable, and this endpoint
            # exists to answer that.
            ready = False
            detail = f"online store unreadable: {exc}"

        # oldest_feature_age_seconds is deliberately left at 0, not guessed.
        # Feast's event timestamps come back only when requested per view, and
        # a freshness number that is actually "the SDK answered" would be worse
        # than an absent one -- the alert §16.1 wants (lag > 10 minutes) would
        # then be firing on a field that means nothing. Owed, and named here
        # rather than quietly zero.
        return features_pb2.FeaturesHealthResponse(
            ready=ready,
            detail=detail,
            item_map_version=self.loaded.items.version,
        )
