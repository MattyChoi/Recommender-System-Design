"""``Store.oldest_feature_age_seconds``: the number the gateway's Health reports.

It was hard-coded to 0 -- "unknown" -- so the orchestrator's gauge could never
move. These pin what it now means, against a stub registry so neither Feast's
registry nor Redis has to exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from serving.features.store import MATERIALISED_VIEWS, Store

NOW = datetime(2026, 11, 9, 12, tzinfo=UTC).timestamp()


class StubRegistry:
    def __init__(
        self, ends: Mapping[str, datetime | None], missing: frozenset[str] = frozenset()
    ) -> None:
        self.ends = ends
        self.missing = missing

    def get_feature_view(self, name: str) -> Any:
        if name in self.missing:
            raise LookupError(name)
        return SimpleNamespace(most_recent_end_time=self.ends.get(name))


def store_with(registry: StubRegistry) -> Store:
    store = Store.__new__(Store)  # skip __init__: no repo, no artifacts
    store._store = registry  # type: ignore[assignment]
    return store


def hours_ago(hours: int) -> datetime:
    return datetime.fromtimestamp(NOW - hours * 3600, tz=UTC)


def test_the_stalest_view_wins_not_the_mean() -> None:
    ends = dict(zip(MATERIALISED_VIEWS, (hours_ago(1), hours_ago(1), hours_ago(48)), strict=True))

    assert store_with(StubRegistry(ends)).oldest_feature_age_seconds(NOW) == 48 * 3600


def test_never_materialised_is_unknown_not_fresh() -> None:
    assert store_with(StubRegistry({})).oldest_feature_age_seconds(NOW) == 0


def test_an_unreadable_registry_does_not_fail_the_health_check() -> None:
    ends = {MATERIALISED_VIEWS[0]: hours_ago(2)}
    registry = StubRegistry(ends, missing=frozenset(MATERIALISED_VIEWS[1:]))

    assert store_with(registry).oldest_feature_age_seconds(NOW) == 2 * 3600


def test_the_push_view_is_not_counted() -> None:
    """user_realtime has no materialisation intervals; counting it would pin the
    reading to 0 or to whatever its batch_source last held."""
    assert "user_realtime" not in MATERIALISED_VIEWS
