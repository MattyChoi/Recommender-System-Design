from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RecommendRequest(_message.Message):
    __slots__ = ("user_id", "surface", "num_results", "exclude_item_ids", "context")
    class ContextEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    SURFACE_FIELD_NUMBER: _ClassVar[int]
    NUM_RESULTS_FIELD_NUMBER: _ClassVar[int]
    EXCLUDE_ITEM_IDS_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    surface: str
    num_results: int
    exclude_item_ids: _containers.RepeatedScalarFieldContainer[str]
    context: _containers.ScalarMap[str, str]
    def __init__(self, user_id: _Optional[str] = ..., surface: _Optional[str] = ..., num_results: _Optional[int] = ..., exclude_item_ids: _Optional[_Iterable[str]] = ..., context: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ScoredItem(_message.Message):
    __slots__ = ("item_id", "score", "debug_scores", "sources", "propensity", "position")
    class DebugScoresEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    DEBUG_SCORES_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    PROPENSITY_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    item_id: str
    score: float
    debug_scores: _containers.ScalarMap[str, float]
    sources: _containers.RepeatedScalarFieldContainer[str]
    propensity: float
    position: int
    def __init__(self, item_id: _Optional[str] = ..., score: _Optional[float] = ..., debug_scores: _Optional[_Mapping[str, float]] = ..., sources: _Optional[_Iterable[str]] = ..., propensity: _Optional[float] = ..., position: _Optional[int] = ...) -> None: ...

class RecommendResponse(_message.Message):
    __slots__ = ("items", "model_version", "index_version", "experiment_variant", "latency_ms", "stage_latency_ms", "used_fallback", "degraded_sources", "stage_counts")
    class StageLatencyMsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    class StageCountsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    INDEX_VERSION_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_VARIANT_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    STAGE_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    USED_FALLBACK_FIELD_NUMBER: _ClassVar[int]
    DEGRADED_SOURCES_FIELD_NUMBER: _ClassVar[int]
    STAGE_COUNTS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[ScoredItem]
    model_version: str
    index_version: str
    experiment_variant: str
    latency_ms: int
    stage_latency_ms: _containers.ScalarMap[str, int]
    used_fallback: bool
    degraded_sources: _containers.RepeatedScalarFieldContainer[str]
    stage_counts: _containers.ScalarMap[str, int]
    def __init__(self, items: _Optional[_Iterable[_Union[ScoredItem, _Mapping]]] = ..., model_version: _Optional[str] = ..., index_version: _Optional[str] = ..., experiment_variant: _Optional[str] = ..., latency_ms: _Optional[int] = ..., stage_latency_ms: _Optional[_Mapping[str, int]] = ..., used_fallback: _Optional[bool] = ..., degraded_sources: _Optional[_Iterable[str]] = ..., stage_counts: _Optional[_Mapping[str, int]] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("ready", "detail", "model_version", "index_version", "index_kind", "index_ef_search")
    READY_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    INDEX_VERSION_FIELD_NUMBER: _ClassVar[int]
    INDEX_KIND_FIELD_NUMBER: _ClassVar[int]
    INDEX_EF_SEARCH_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    detail: str
    model_version: str
    index_version: str
    index_kind: str
    index_ef_search: int
    def __init__(self, ready: _Optional[bool] = ..., detail: _Optional[str] = ..., model_version: _Optional[str] = ..., index_version: _Optional[str] = ..., index_kind: _Optional[str] = ..., index_ef_search: _Optional[int] = ...) -> None: ...
