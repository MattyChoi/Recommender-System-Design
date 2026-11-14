"""Publishing serving artifacts, against a dict standing in for S3.

No MinIO: the publish logic is ordering and retention, both pure, and a test
that needs a container is a test CI skips. The Protocol in publish.py exists so
this substitution is a supported use rather than monkeypatching.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from common.config import Settings, load_settings
from data_pipeline.serving_artifacts.publish import (
    POINTER,
    current,
    promote,
    publish,
    retire,
    root,
    versions,
)


class FakeStore:
    """A dict with just enough of fsspec's surface.

    `exists` answers for prefixes as well as keys, because S3 has no
    directories and publish.py leans on that for the version listing.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def pipe_file(self, path: str, value: bytes) -> None:
        self.objects[path] = value

    def cat_file(self, path: str) -> bytes:
        return self.objects[path]

    def exists(self, path: str) -> bool:
        return path in self.objects or any(
            key.startswith(path.rstrip("/") + "/") for key in self.objects
        )

    def ls(self, path: str, detail: bool = False) -> list[str]:
        prefix = path.rstrip("/") + "/"
        children = {
            key[len(prefix) :].split("/", 1)[0] for key in self.objects if key.startswith(prefix)
        }
        return [prefix + child for child in sorted(children)]

    def rm(self, path: str, recursive: bool = False) -> None:
        doomed = [key for key in self.objects if key == path or key.startswith(path + "/")]
        for key in doomed:
            del self.objects[key]


@pytest.fixture
def settings() -> Settings:
    """Real settings with the storage block pinned, so conf/config.yml cannot
    move the keys this test asserts on."""
    loaded = load_settings()
    loaded.storage.bucket = "test-bucket"
    loaded.storage.artifacts_prefix = "artifacts"
    return loaded


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    """A plausible artifact set, including one nested file."""
    source = tmp_path / "artifacts"
    (source / "nested").mkdir(parents=True)
    (source / "item_map.json").write_text('{"version": "abc", "items": {}}')
    (source / "popular.json").write_text('{"version": "abc", "items": [1]}')
    (source / "nested" / "content.bin").write_bytes(b"\x00\x01")
    return source


def at(hour: int) -> datetime:
    return datetime(2026, 1, 1, hour, tzinfo=UTC)


def test_every_file_lands_under_the_version_with_its_layout_intact(
    settings: Settings, artifacts: Path
) -> None:
    store = FakeStore()
    version = publish(store, settings, artifacts, at(5))

    base = f"{root(settings)}/{version}"
    assert set(store.objects) == {
        f"{base}/item_map.json",
        f"{base}/popular.json",
        f"{base}/nested/content.bin",
    }
    assert store.objects[f"{base}/nested/content.bin"] == b"\x00\x01"


def test_extra_files_land_at_the_version_root_by_basename(
    settings: Settings, artifacts: Path, tmp_path: Path
) -> None:
    """model.columns.json is written beside the ONNX graph, under Triton's model
    repository rather than in serving/artifacts. A pod gets only what is
    published, so it has to come along -- flattened, because the orchestrator
    takes it as a path flag and should not have to recreate that directory."""
    columns = tmp_path / "triton" / "ranker" / "1" / "model.columns.json"
    columns.parent.mkdir(parents=True)
    columns.write_text('{"features": ["a"]}')

    store = FakeStore()
    version = publish(store, settings, artifacts, at(5), extra=[columns])

    assert f"{root(settings)}/{version}/model.columns.json" in store.objects


def test_a_missing_extra_is_refused_before_anything_uploads(
    settings: Settings, artifacts: Path, tmp_path: Path
) -> None:
    """Checked up front rather than as it goes.

    A version missing its columns file would upload cleanly and then kill every
    pod that pulled it, since the orchestrator refuses to start without it. And
    a part-uploaded version left behind on each failure is how the retention
    count starts counting things that are not versions.
    """
    store = FakeStore()

    with pytest.raises(FileNotFoundError, match=re.escape("model.columns.json")):
        publish(store, settings, artifacts, at(5), extra=[tmp_path / "model.columns.json"])

    assert not store.objects


def test_publishing_does_not_touch_the_pointer(settings: Settings, artifacts: Path) -> None:
    """**The ordering guarantee.** Files first, pointer last -- so a pod that
    starts mid-publish loads the previous version whole instead of a mixture."""
    store = FakeStore()
    publish(store, settings, artifacts, at(5))

    assert current(store, settings) is None
    assert f"{root(settings)}/{POINTER}" not in store.objects


def test_promote_then_current_round_trips(settings: Settings, artifacts: Path) -> None:
    store = FakeStore()
    version = publish(store, settings, artifacts, at(5))
    promote(store, settings, version)

    assert current(store, settings) == version


def test_publishing_nothing_is_refused(settings: Settings, tmp_path: Path) -> None:
    """An empty publish followed by a pointer flip would take serving down on
    the next roll, so it fails here instead."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError):
        publish(FakeStore(), settings, empty)


def test_versions_are_listed_chronologically(settings: Settings, artifacts: Path) -> None:
    store = FakeStore()
    published = [publish(store, settings, artifacts, at(hour)) for hour in (9, 5, 7)]

    assert versions(store, settings) == sorted(published)


def test_retire_never_deletes_the_live_version(settings: Settings, artifacts: Path) -> None:
    """The case retention arithmetic gets wrong.

    Five versions, keep three, and the LIVE one is the oldest -- so the plain
    count would delete the only thing there is to roll back to.
    """
    store = FakeStore()
    published = [publish(store, settings, artifacts, at(hour)) for hour in range(1, 6)]
    promote(store, settings, published[0])

    retired = retire(store, settings, keep=3)

    assert published[0] not in retired
    assert store.exists(f"{root(settings)}/{published[0]}")
    assert published[1] in retired
    assert not store.exists(f"{root(settings)}/{published[1]}")


def test_retire_keeps_the_newest_and_reports_what_went(settings: Settings, artifacts: Path) -> None:
    store = FakeStore()
    published = [publish(store, settings, artifacts, at(hour)) for hour in range(1, 6)]
    promote(store, settings, published[-1])

    retired = retire(store, settings, keep=3)

    assert retired == published[:2]
    for version in published[2:]:
        assert store.exists(f"{root(settings)}/{version}")
