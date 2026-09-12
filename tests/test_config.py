"""The config invariants nothing else checks.

``conf/config.yml`` and the pydantic models each carry a full set of values, and
nothing forces them to agree. They drift the same way every time: a field is
added to a model with a default, the YAML is never updated, and the two disagree
silently until someone deletes the YAML block and the behaviour changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from common.config import Settings, load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_YML = REPO_ROOT / "conf" / "config.yml"

# `paths` is required and has no defaults, so there is nothing to compare it
# against. Its presence in the file is asserted separately.
_REQUIRED_BLOCKS = {"paths"}


def _raw() -> dict[str, Any]:
    return dict(yaml.safe_load(CONFIG_YML.read_text()))


def test_every_yaml_block_is_a_settings_field() -> None:
    """A block nobody reads is worse than a missing one: it looks load-bearing."""
    unknown = set(_raw()) - set(Settings.model_fields)
    assert not unknown, f"conf/config.yml has blocks Settings ignores: {sorted(unknown)}"


def test_every_optional_settings_field_appears_in_the_yaml() -> None:
    """The drift direction that actually happens -- a new model, no YAML block."""
    missing = set(Settings.model_fields) - set(_raw()) - _REQUIRED_BLOCKS
    assert not missing, f"Settings blocks absent from conf/config.yml: {sorted(missing)}"


@pytest.mark.parametrize("block", sorted(set(_raw()) - _REQUIRED_BLOCKS))
def test_code_defaults_equal_the_yaml_values(block: str) -> None:
    """Parametrised per block so a failure names the one that drifted."""
    defaults = Settings.model_fields[block].default
    for key, want in _raw()[block].items():
        assert getattr(defaults, key) == want, (
            f"{block}.{key}: YAML says {want!r}, the code default is {getattr(defaults, key)!r}"
        )


def test_required_blocks_are_present() -> None:
    assert set(_raw()) >= _REQUIRED_BLOCKS


def test_load_settings_accepts_a_root_from_another_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason the root argument exists.

    The Feast CLI chdirs into the feature repo before importing its definitions,
    so a load relative to the working directory fails there. Without an explicit
    root, ``definition.py`` would need its own copy of the bucket and endpoint.
    """
    monkeypatch.chdir(tmp_path)

    with pytest.raises(Exception):  # noqa: B017 -- pydantic's own ValidationError
        load_settings()

    settings = load_settings(REPO_ROOT)
    assert settings.storage.bucket


def test_an_explicit_root_does_not_leak_into_the_next_load() -> None:
    """The ContextVar is reset, so one caller's root is not another's default."""
    load_settings(REPO_ROOT)
    assert load_settings().storage.bucket == load_settings(REPO_ROOT).storage.bucket


def test_an_explicit_root_anchors_the_data_paths_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this caught: a resolved YAML file with unresolved paths inside.

    The Feast CLI chdirs into the feature repo, so a relative ``data/gold``
    resolved there points at a directory under the feature repo that has never
    existed -- surfacing as pyarrow's FileNotFoundError rather than as anything
    that names the configuration.
    """
    monkeypatch.chdir(tmp_path)
    settings = load_settings(REPO_ROOT)

    for _, value in settings.paths:
        assert value.is_absolute(), value
    assert settings.paths.gold == REPO_ROOT / "data" / "gold"


def test_no_root_leaves_the_paths_relative() -> None:
    """Unchanged for everything launched from the repo root, which is the rest."""
    assert not load_settings().paths.gold.is_absolute()


def test_gold_uri_keeps_both_slashes_in_the_scheme() -> None:
    """The bug this config exists to avoid: Path() collapses ``s3://`` to ``s3:/``."""
    settings = load_settings(REPO_ROOT).model_copy(deep=True)
    settings.storage.backend = "s3"

    assert settings.gold_uri("item_hourly_features") == ("s3a://recsys/gold/item_hourly_features")
    assert settings.gold_uri("item_hourly_features", scheme="s3") == (
        "s3://recsys/gold/item_hourly_features"
    )
    assert settings.gold_uri("training_examples/train").startswith("s3a://recsys/gold/")
