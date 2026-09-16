"""Which gold tables follow storage.backend, and which deliberately do not.

The split is a decision, not an accident: the three feature-store sources have
to be reachable from wherever serving runs, while training_examples is the
largest table, is read by the trainer on the machine that wrote it, and has no
consumer outside this repo. A test pins the decision so a later "make gold
consistent" change has to argue with something.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from common.config import Settings, load_settings
from common.utils import FEATURE_TABLES, GOLD_TABLES, _is_built, gold_location
from tests.test_config import REPO_ROOT


@pytest.fixture
def local_settings() -> Settings:
    """Explicitly local: the committed default is s3, and this fixture is about
    what the local backend does."""
    settings = load_settings(REPO_ROOT).model_copy(deep=True)
    settings.storage.backend = "local"
    return settings


@pytest.fixture
def s3_settings(local_settings: Settings) -> Settings:
    local_settings.storage.backend = "s3"
    return local_settings


def test_remote_tables_are_a_subset_of_the_gold_tables() -> None:
    assert set(FEATURE_TABLES) < set(GOLD_TABLES)


def test_training_examples_is_not_remote() -> None:
    """The decision this module exists to record."""
    assert "training_examples" not in FEATURE_TABLES


@pytest.mark.parametrize("table", GOLD_TABLES)
def test_the_local_backend_moves_nothing(local_settings: Settings, table: str) -> None:
    assert gold_location(local_settings, table) == str(local_settings.paths.gold / table)


@pytest.mark.parametrize("table", FEATURE_TABLES)
def test_the_series_follow_the_backend(s3_settings: Settings, table: str) -> None:
    assert gold_location(s3_settings, table) == f"s3a://recsys/gold/{table}"
    assert gold_location(s3_settings, table, scheme="s3") == f"s3://recsys/gold/{table}"


def test_training_examples_stays_local_even_on_s3(s3_settings: Settings) -> None:
    """A per-split suffix must not defeat the table-name match."""
    got = gold_location(s3_settings, "training_examples/train")
    assert got == str(s3_settings.paths.gold / "training_examples" / "train")
    assert not got.startswith("s3")


def test_user_history_stays_local_even_on_s3(s3_settings: Settings) -> None:
    """Part G's click sequence is a training-side table, not a store source.

    Same reasoning as ``training_examples``: the trainer reads it on the machine
    that wrote it, and nothing outside this repo consumes it. Pinned separately
    because "it is in gold, so it should follow the backend" is the exact
    tidying-up this module exists to make someone argue with.
    """
    assert "user_history" not in FEATURE_TABLES

    got = gold_location(s3_settings, "user_history/dev")
    assert got == str(s3_settings.paths.gold / "user_history" / "dev")
    assert not got.startswith("s3")


def test_is_built_reads_the_marker_for_a_local_table(
    local_settings: Settings, tmp_path: Path
) -> None:
    local_settings.paths.gold = tmp_path
    assert not _is_built(local_settings, "gold", "item_hourly_features")

    marker = tmp_path / "item_hourly_features" / "_SUCCESS"
    marker.parent.mkdir(parents=True)
    marker.touch()
    assert _is_built(local_settings, "gold", "item_hourly_features")


class _FakeS3:
    """Records what was probed and answers however the test asks it to."""

    def __init__(self, answer: object) -> None:
        self.answer = answer
        self.probed: list[str] = []

    def exists(self, path: str) -> bool:
        self.probed.append(path)
        if isinstance(self.answer, Exception):
            raise self.answer
        return bool(self.answer)


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> Callable[[object], _FakeS3]:
    """Replace s3fs.S3FileSystem with a recorder. No network, no credentials."""

    def install(answer: object) -> _FakeS3:
        fs = _FakeS3(answer)
        monkeypatch.setattr("s3fs.S3FileSystem", lambda **kwargs: fs)
        return fs

    return install


def test_a_remote_series_is_built_when_the_marker_is_there(
    s3_settings: Settings, fake_s3: Callable[[object], _FakeS3]
) -> None:
    fs = fake_s3(True)

    assert _is_built(s3_settings, "gold", "item_hourly_features")
    # Scheme stripped, marker appended. gold_location returns s3a://, which is
    # Spark's registration and means nothing to s3fs -- probing it unstripped
    # would raise rather than answer, and the bug would look like an outage.
    assert fs.probed == ["recsys/gold/item_hourly_features/_SUCCESS"]


def test_a_remote_series_is_unbuilt_without_the_marker(
    s3_settings: Settings, fake_s3: Callable[[object], _FakeS3]
) -> None:
    """Part-files with no marker are the remains of a run that died."""
    fake_s3(False)

    assert not _is_built(s3_settings, "gold", "item_hourly_features")


def test_an_unreachable_store_is_an_error_not_a_rebuild(
    s3_settings: Settings, fake_s3: Callable[[object], _FakeS3]
) -> None:
    """Answering 'not built' would rebuild, then die inside S3A a minute later."""
    fake_s3(OSError("connection refused"))

    with pytest.raises(RuntimeError, match="is MinIO running"):
        _is_built(s3_settings, "gold", "item_hourly_features")


def test_local_tables_never_reach_for_a_client(
    s3_settings: Settings, tmp_path: Path, fake_s3: Callable[[object], _FakeS3]
) -> None:
    """training_examples stays on disk even under the s3 backend."""
    s3_settings.paths.gold = tmp_path
    fs = fake_s3(True)

    assert not _is_built(s3_settings, "gold", "training_examples/train")
    assert fs.probed == []


def test_is_built_still_skips_training_examples_on_s3(
    s3_settings: Settings, tmp_path: Path
) -> None:
    """The per-split skip keeps working, because that table never moved."""
    s3_settings.paths.gold = tmp_path
    marker = tmp_path / "training_examples" / "train" / "_SUCCESS"
    marker.parent.mkdir(parents=True)
    marker.touch()

    assert _is_built(s3_settings, "gold", "training_examples/train")
