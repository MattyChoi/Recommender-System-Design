"""``make artifacts-push`` -- publish serving/artifacts to the object store.

**Versioned, then pointer-switched.** Every file lands under a version prefix
and only then does ``CURRENT`` change, so a pod that starts mid-publish reads
the previous version whole rather than a half-written mixture of two. Nothing
ever overwrites a published version.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from common.config import Settings, load_settings
from indexing.lifecycle import KEEP_VERSIONS, expired, version_label

# The object naming the version to load. Upper-case and extensionless, matching
# data/index/CURRENT, which the retrieval sidecar already reads.
POINTER = "CURRENT"


class ObjectStore(Protocol):
    """The five operations this module needs of a filesystem.

    A Protocol rather than ``s3fs.S3FileSystem``: it keeps the publish logic
    testable without a running MinIO, and s3fs satisfies it as-is.
    """

    def pipe_file(self, path: str, value: bytes) -> None: ...
    # str | bytes, not bytes: that is how fsspec annotates it, and s3fs must fit.
    def cat_file(self, path: str) -> str | bytes: ...
    def exists(self, path: str) -> bool: ...
    def ls(self, path: str, detail: bool = False) -> list[str]: ...
    def rm(self, path: str, recursive: bool = False) -> None: ...


def store(settings: Settings) -> ObjectStore:
    """An s3fs filesystem built from the storage config.

    Imported lazily for the reason ``common.utils`` gives: s3fs drags in
    aiobotocore and an event loop, and this module is cheap to import otherwise.

    NOTE: this is the third copy of this wiring (``common.utils`` and
    ``dump_popular`` in this package have the others). A shared
    ``common.utils.s3_filesystem`` is owed.
    """
    import s3fs

    filesystem: ObjectStore = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": settings.storage.endpoint_url},
        skip_instance_cache=True,
    )
    return filesystem


def root(settings: Settings) -> str:
    """``bucket/prefix``, with no scheme -- s3fs paths carry none."""
    return f"{settings.storage.bucket}/{settings.storage.artifacts_prefix}"


def publish(
    filesystem: ObjectStore,
    settings: Settings,
    source: Path,
    moment: datetime | None = None,
    extra: Sequence[Path] = (),
) -> str:
    """Upload every file under ``source``, plus ``extra``, to a fresh version prefix.

    Files are read whole and PUT individually rather than handed to a recursive
    upload: the set is a handful of JSON documents and two vector files, and one
    object per file is what makes the pointer flip meaningful.

    Args:
        filesystem: The object store.
        settings: Supplies bucket and prefix.
        source: Local artifacts directory, usually ``serving/artifacts``.
        moment: Timestamp the version is labelled from. Defaults to now, UTC.
        extra: Files from OUTSIDE ``source`` to publish at the version root.
            ``model.columns.json`` is the one that matters: it is written beside
            the ONNX graph under ``serving/triton/ranker/1/``, because that is
            Triton's model repository, but the orchestrator refuses to start
            without it. On one filesystem both paths are reachable; a pod gets
            only what is published, so it has to come along.

    Returns:
        The version label written, e.g. ``v=2026-11-12T07:00Z``.

    Raises:
        FileNotFoundError: If ``source`` holds no files, or an ``extra`` is
            missing. Publishing an incomplete version and then flipping the
            pointer at it would take serving down on the next roll, so both are
            refused before anything is uploaded.
    """
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"{source} holds no artifacts; run `make artifacts` first")

    missing = [str(path) for path in extra if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot publish, missing: {', '.join(missing)}")

    version = version_label(moment or datetime.now(UTC))
    base = f"{root(settings)}/{version}"

    for path in files:
        # Relative, so a nested artifact keeps its layout under the version.
        key = path.relative_to(source).as_posix()
        filesystem.pipe_file(f"{base}/{key}", path.read_bytes())

    for path in extra:
        filesystem.pipe_file(f"{base}/{path.name}", path.read_bytes())

    return version


def promote(filesystem: ObjectStore, settings: Settings, version: str) -> None:
    """Point CURRENT at ``version``.

    Called only after :func:`publish` returns, which is the whole ordering
    guarantee: readers see a complete version or the previous one.
    """
    filesystem.pipe_file(f"{root(settings)}/{POINTER}", version.encode())


def current(filesystem: ObjectStore, settings: Settings) -> str | None:
    """Which version serving should load, or None if nothing is published."""
    pointer = f"{root(settings)}/{POINTER}"
    if not filesystem.exists(pointer):
        return None
    content = filesystem.cat_file(pointer)
    text = content.decode() if isinstance(content, bytes) else content
    return text.strip()


def versions(filesystem: ObjectStore, settings: Settings) -> list[str]:
    """Published version labels, chronological.

    Chronological because :func:`version_label` is zero-padded ISO, so
    lexicographic order is time order -- which is the reason for that format
    rather than a preference about it.
    """
    base = root(settings)
    if not filesystem.exists(base):
        return []
    names = (entry.rstrip("/").rsplit("/", 1)[-1] for entry in filesystem.ls(base))
    return sorted(name for name in names if name.startswith("v="))


def retire(filesystem: ObjectStore, settings: Settings, keep: int = KEEP_VERSIONS) -> list[str]:
    """Delete versions past the retention depth, newest kept, and report them.

    The version CURRENT names is never deleted whatever the arithmetic says --
    the same exemption :func:`indexing.lifecycle.retire` makes for the live
    index, and for the same reason: the rollback target must outlive the count.
    """
    live = current(filesystem, settings)
    published = versions(filesystem, settings)
    stale = [version for version in expired(published, keep) if version != live]
    for version in stale:
        filesystem.rm(f"{root(settings)}/{version}", recursive=True)
    return stale


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish serving artifacts to the object store.")
    parser.add_argument("--source", type=Path, default=Path("serving/artifacts"))
    parser.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=[],
        help="a file outside --source to publish at the version root, repeatable",
    )
    parser.add_argument("--keep", type=int, default=KEEP_VERSIONS)
    parser.add_argument("--show", action="store_true", help="print CURRENT and exit")
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="upload the version but leave CURRENT alone, so it can be pulled and checked first",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = load_settings()
    if settings.storage.backend != "s3":
        raise SystemExit(
            f"storage.backend is {settings.storage.backend!r}; publishing needs the object store"
        )

    filesystem = store(settings)
    if args.show:
        print(current(filesystem, settings) or "nothing published")
        return 0

    version = publish(filesystem, settings, args.source, extra=args.extra)
    print(f"published {version} to {root(settings)}")

    if args.no_promote:
        print(f"  CURRENT still {current(filesystem, settings)}; promote once checked")
        return 0

    promote(filesystem, settings, version)
    print(f"  CURRENT -> {version}")

    for stale in retire(filesystem, settings, args.keep):
        print(f"  retired {stale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
