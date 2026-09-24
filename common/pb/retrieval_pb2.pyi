from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RetrieveRequest(_message.Message):
    __slots__ = ("user_id", "k", "history", "user_feats", "ef_search")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    USER_FEATS_FIELD_NUMBER: _ClassVar[int]
    EF_SEARCH_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    k: int
    history: _containers.RepeatedScalarFieldContainer[int]
    user_feats: _containers.RepeatedScalarFieldContainer[float]
    ef_search: int
    def __init__(self, user_id: _Optional[str] = ..., k: _Optional[int] = ..., history: _Optional[_Iterable[int]] = ..., user_feats: _Optional[_Iterable[float]] = ..., ef_search: _Optional[int] = ...) -> None: ...

class RetrieveResponse(_message.Message):
    __slots__ = ("items", "scores", "content_similarity", "embedding_cached", "index_kind", "index_version")
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    CONTENT_SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_CACHED_FIELD_NUMBER: _ClassVar[int]
    INDEX_KIND_FIELD_NUMBER: _ClassVar[int]
    INDEX_VERSION_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedScalarFieldContainer[int]
    scores: _containers.RepeatedScalarFieldContainer[float]
    content_similarity: _containers.RepeatedScalarFieldContainer[float]
    embedding_cached: bool
    index_kind: str
    index_version: str
    def __init__(self, items: _Optional[_Iterable[int]] = ..., scores: _Optional[_Iterable[float]] = ..., content_similarity: _Optional[_Iterable[float]] = ..., embedding_cached: _Optional[bool] = ..., index_kind: _Optional[str] = ..., index_version: _Optional[str] = ...) -> None: ...

class RetrievalHealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RetrievalHealthResponse(_message.Message):
    __slots__ = ("ready", "detail", "index_kind", "index_version", "index_ef_search", "item_count")
    READY_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    INDEX_KIND_FIELD_NUMBER: _ClassVar[int]
    INDEX_VERSION_FIELD_NUMBER: _ClassVar[int]
    INDEX_EF_SEARCH_FIELD_NUMBER: _ClassVar[int]
    ITEM_COUNT_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    detail: str
    index_kind: str
    index_version: str
    index_ef_search: int
    item_count: int
    def __init__(self, ready: _Optional[bool] = ..., detail: _Optional[str] = ..., index_kind: _Optional[str] = ..., index_version: _Optional[str] = ..., index_ef_search: _Optional[int] = ..., item_count: _Optional[int] = ...) -> None: ...
