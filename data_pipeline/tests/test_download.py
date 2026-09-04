"""Tests for the MIND downloader.

``hf_hub_download`` is stubbed out, so no network call and no gated-repo
authentication is needed -- these run in CI exactly as they do locally.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError

from data_pipeline.ingest import download_mind as dl

MEMBERS = ["behaviors.tsv", "news.tsv", "entity_embedding.vec", "relation_embedding.vec"]


def _write_zip(path: Path, members: list[str]) -> Path:
    """Build a zip at ``path`` holding ``members``, each with token TSV content."""
    with zipfile.ZipFile(path, "w") as zf:
        for name in members:
            zf.writestr(path.stem + "/" + name, "col_a\tcol_b\n")
    return path


@pytest.fixture
def stub_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Replace hf_hub_download with a local fake; record how it was called."""
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    calls: list[dict[str, object]] = []

    def fake(**kwargs: object) -> str:
        calls.append(kwargs)
        filename = str(kwargs["filename"])
        return str(_write_zip(cache / filename, MEMBERS))

    monkeypatch.setattr(dl, "hf_hub_download", fake)
    return calls


def test_extracts_every_member(stub_hub: list[dict[str, object]], tmp_path: Path) -> None:
    """All four MIND files land in ``raw_root/<split>`` after a successful fetch."""
    raw = tmp_path / "raw"
    dl.fetch_split("small", "train", raw)
    for name in MEMBERS:
        assert (raw / "train" / name).is_file(), name


def test_requests_the_right_artifact(stub_hub: list[dict[str, object]], tmp_path: Path) -> None:
    """fetch_split asks the hub for the artifact its arguments name."""
    dl.fetch_split("large", "dev", tmp_path / "raw")
    assert stub_hub[0]["repo_id"] == dl.REPO_ID
    assert stub_hub[0]["filename"] == "MINDlarge_dev.zip"
    assert stub_hub[0]["repo_type"] == "dataset"


def test_writes_a_manifest(stub_hub: list[dict[str, object]], tmp_path: Path) -> None:
    """A successful fetch records what was downloaded, beside the data."""
    raw = tmp_path / "raw"
    dl.fetch_split("small", "dev", raw)
    manifest = json.loads((raw / "dev" / "_manifest.json").read_text())
    assert manifest["filename"] == "MINDsmall_dev.zip"
    assert manifest["size"] == "small"
    assert len(manifest["sha256"]) == 64


def test_second_call_skips(
    stub_hub: list[dict[str, object]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Re-running an already-downloaded split does no work."""
    raw = tmp_path / "raw"
    dl.fetch_split("small", "train", raw)
    capsys.readouterr()
    dl.fetch_split("small", "train", raw)
    assert "skipping" in capsys.readouterr().out
    assert len(stub_hub) == 1


def test_force_refetches(stub_hub: list[dict[str, object]], tmp_path: Path) -> None:
    """``force=True`` overrides the skip *and* reaches the hub as force_download."""
    raw = tmp_path / "raw"
    dl.fetch_split("small", "train", raw)
    dl.fetch_split("small", "train", raw, force=True)
    assert len(stub_hub) == 2
    assert stub_hub[1]["force_download"] is True


def test_missing_member_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An archive missing an expected file is refused before extraction."""
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    monkeypatch.setattr(
        dl,
        "hf_hub_download",
        lambda **kw: str(_write_zip(cache / "x.zip", ["behaviors.tsv"])),
    )
    with pytest.raises(RuntimeError, match="missing"):
        dl.fetch_split("small", "train", tmp_path / "raw")


def test_gated_repo_gives_actionable_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gated-repo refusal becomes instructions rather than a traceback."""

    def raise_gated(**kw: object) -> str:
        # HfHubHTTPError reads response.request, so the response needs one attached
        request = httpx.Request("GET", f"https://huggingface.co/datasets/{dl.REPO_ID}")
        raise GatedRepoError("403", response=httpx.Response(403, request=request))

    monkeypatch.setattr(dl, "hf_hub_download", raise_gated)
    with pytest.raises(RuntimeError, match="hf auth login"):
        dl.fetch_split("small", "train", tmp_path / "raw")
