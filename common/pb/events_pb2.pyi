from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class InteractionEvent(_message.Message):
    __slots__ = ("event_id", "user_id", "item_id", "event_type", "event_ts_ms", "ingest_ts_ms", "session_id", "position", "request_id", "dwell_ms", "propensity", "device", "surface")
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    EVENT_TS_MS_FIELD_NUMBER: _ClassVar[int]
    INGEST_TS_MS_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    DWELL_MS_FIELD_NUMBER: _ClassVar[int]
    PROPENSITY_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    SURFACE_FIELD_NUMBER: _ClassVar[int]
    event_id: str
    user_id: str
    item_id: str
    event_type: str
    event_ts_ms: int
    ingest_ts_ms: int
    session_id: str
    position: int
    request_id: str
    dwell_ms: int
    propensity: float
    device: str
    surface: str
    def __init__(self, event_id: _Optional[str] = ..., user_id: _Optional[str] = ..., item_id: _Optional[str] = ..., event_type: _Optional[str] = ..., event_ts_ms: _Optional[int] = ..., ingest_ts_ms: _Optional[int] = ..., session_id: _Optional[str] = ..., position: _Optional[int] = ..., request_id: _Optional[str] = ..., dwell_ms: _Optional[int] = ..., propensity: _Optional[float] = ..., device: _Optional[str] = ..., surface: _Optional[str] = ...) -> None: ...
