from __future__ import annotations

from datetime import UTC, datetime
from typing import override

import faiss
import grpc

from common.features import USER_FEATURES
from common.pb import retrieval_pb2, retrieval_pb2_grpc
from indexing.build_index import to_item_ids
from serving.retrieval.cache import UserEmbeddingCache
from serving.retrieval.index import (
    LoadedIndex,
    build_user_tower_columns,
    content_similarity,
    encode_user,
    real_history,
)


class RetrievalServicer(retrieval_pb2_grpc.RetrievalServicer):
    def __init__(self, loaded: LoadedIndex, cache: UserEmbeddingCache | None = None) -> None:
        self._loaded = loaded
        self._tower_width = loaded.tower.user_norm.normalized_shape[0]
        self._cache = cache

    @override
    def Retrieve(
        self, request: retrieval_pb2.RetrieveRequest, context: grpc.ServicerContext
    ) -> retrieval_pb2.RetrieveResponse:
        if not request.user_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "user_id is required")
        if request.k <= 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"k must be positive, got {request.k}")
        if request.ef_search < 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"ef_search must be non-negative, got {request.ef_search}",
            )

        if len(request.user_feats) != len(USER_FEATURES):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"user_feats has width {len(request.user_feats)}, "
                f"tower expects {len(USER_FEATURES)}",
            )

        feats = build_user_tower_columns(
            list(request.user_feats),
            request.has_user_features,
            datetime.now(UTC),
        )

        history = real_history(request.history, len(self._loaded.content))

        cached, user_vector = False, None
        if self._cache is not None:
            user_vector = self._cache.get(request.user_id)
            if user_vector is not None:
                user_vector = user_vector
                cached = True
        if not cached:
            user_vector = encode_user(
                self._loaded.tower, feats, history, device=self._loaded.device
            )
            if self._cache is not None:
                self._cache.put(request.user_id, user_vector)

        # Per-call params, not `index.hnsw.efSearch`: the index is shared across
        # the server's worker threads and FAISS releases the GIL during search.
        ef = request.ef_search or self._loaded.ef_search
        params = None
        if self._loaded.kind == "hnsw":
            params = faiss.SearchParametersHNSW(efSearch=ef)  # type: ignore[attr-defined]

        scores, positions = self._loaded.index.search(user_vector, request.k, params=params)

        # FAISS holds items 1..n at positions 0..n-1 and returns -1 where it
        # found fewer than k neighbours. `to_item_ids` fixes this shift
        ids = to_item_ids(positions[0])
        pairs = [(int(i), float(s)) for i, s in zip(ids, scores[0], strict=True) if i > 0]
        items = [item for item, _ in pairs]

        return retrieval_pb2.RetrieveResponse(
            items=items,
            scores=[score for _, score in pairs],
            content_similarity=[
                float(value) for value in content_similarity(self._loaded.content, history, items)
            ],
            index_version=self._loaded.version,
            embedding_cached=cached,
        )

    @override
    def Health(
        self, request: retrieval_pb2.RetrievalHealthRequest, context: grpc.ServicerContext
    ) -> retrieval_pb2.RetrievalHealthResponse:
        return retrieval_pb2.RetrievalHealthResponse(
            ready=True,
            detail="",
            index_version=self._loaded.version,
            index_kind=self._loaded.kind,
            index_ef_search=self._loaded.ef_search,
            item_count=self._loaded.index.ntotal,
        )
