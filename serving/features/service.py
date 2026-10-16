from __future__ import annotations

from typing import override

import grpc

from common.features import ITEM_COLUMNS, USER_FEATURES
from common.pb import features_pb2, features_pb2_grpc
from serving.features.store import DEFAULT_MAX_HISTORY, Store


class FeaturesServicer(features_pb2_grpc.FeaturesServicer):
    def __init__(self, store: Store) -> None:
        self._store = store

    @override
    def GetUser(
        self, request: features_pb2.GetUserRequest, context: grpc.ServicerContext
    ) -> features_pb2.GetUserResponse:
        limit = request.max_history if request.max_history else DEFAULT_MAX_HISTORY
        row = self._store.user(request.user_id, limit)

        return features_pb2.GetUserResponse(
            values=row.values,
            names=list(USER_FEATURES),
            history=row.history,
            found=row.found,
            unmapped_history=row.unmapped,
            cat_affinity=row.cat_affinity,
        )

    @override
    def GetItems(
        self, request: features_pb2.GetItemsRequest, context: grpc.ServicerContext
    ) -> features_pb2.GetItemsResponse:
        if not request.items:
            # An empty request is valid, and would degrade from the ranking to the popularity slate
            return features_pb2.GetItemsResponse(names=list(ITEM_COLUMNS))

        rows = self._store.items(list(request.items))

        return features_pb2.GetItemsResponse(
            values=rows.values,
            names=list(ITEM_COLUMNS),
            found=rows.found,
            cached=rows.cached,
        )

    @override
    def Health(
        self, request: features_pb2.FeaturesHealthRequest, context: grpc.ServicerContext
    ) -> features_pb2.FeaturesHealthResponse:
        return features_pb2.FeaturesHealthResponse(
            ready=True,
            detail="",
            # Feast's to_dict() hands back values without their event
            # timestamps, so the staleness number needs either a second read
            # path or a direct Redis peek at the `_ts:` field.
            oldest_feature_age_seconds=0,
            item_map_version=self._store.item_map_version,
        )
