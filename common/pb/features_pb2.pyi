from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GetUserRequest(_message.Message):
    __slots__ = ("user_id", "max_history")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_HISTORY_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    max_history: int
    def __init__(self, user_id: _Optional[str] = ..., max_history: _Optional[int] = ...) -> None: ...

class GetUserResponse(_message.Message):
    __slots__ = ("values", "names", "history", "unmapped_history", "found")
    VALUES_FIELD_NUMBER: _ClassVar[int]
    NAMES_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    UNMAPPED_HISTORY_FIELD_NUMBER: _ClassVar[int]
    FOUND_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    names: _containers.RepeatedScalarFieldContainer[str]
    history: _containers.RepeatedScalarFieldContainer[int]
    unmapped_history: int
    found: bool
    def __init__(self, values: _Optional[_Iterable[float]] = ..., names: _Optional[_Iterable[str]] = ..., history: _Optional[_Iterable[int]] = ..., unmapped_history: _Optional[int] = ..., found: _Optional[bool] = ...) -> None: ...

class GetItemsRequest(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, items: _Optional[_Iterable[int]] = ...) -> None: ...

class GetItemsResponse(_message.Message):
    __slots__ = ("values", "names", "found", "cached")
    VALUES_FIELD_NUMBER: _ClassVar[int]
    NAMES_FIELD_NUMBER: _ClassVar[int]
    FOUND_FIELD_NUMBER: _ClassVar[int]
    CACHED_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    names: _containers.RepeatedScalarFieldContainer[str]
    found: _containers.RepeatedScalarFieldContainer[bool]
    cached: int
    def __init__(self, values: _Optional[_Iterable[float]] = ..., names: _Optional[_Iterable[str]] = ..., found: _Optional[_Iterable[bool]] = ..., cached: _Optional[int] = ...) -> None: ...

class FeaturesHealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class FeaturesHealthResponse(_message.Message):
    __slots__ = ("ready", "detail", "oldest_feature_age_seconds", "item_map_version")
    READY_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    OLDEST_FEATURE_AGE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ITEM_MAP_VERSION_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    detail: str
    oldest_feature_age_seconds: int
    item_map_version: str
    def __init__(self, ready: _Optional[bool] = ..., detail: _Optional[str] = ..., oldest_feature_age_seconds: _Optional[int] = ..., item_map_version: _Optional[str] = ...) -> None: ...
