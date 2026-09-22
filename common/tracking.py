"""Run provenance, and the MLflow wrapper that records it.

"A run you cannot reproduce is a run you cannot cite." Three things have to be
recorded for that to be true, and one of them is usually recorded wrongly.

**The git SHA is worthless without a dirty flag.** A SHA taken from a modified
working tree names a commit that does not contain the code that ran. It is the
most common way a run stops being reproducible while still looking traceable, so
``git_dirty`` is logged beside it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from common.config import Settings
from common.utils import gold_location

# An untracked file is read up to here when digesting the working tree. Enough
# for any source file; bounded so an accidentally-unignored artifact cannot make
# provenance slow.
UNTRACKED_BYTE_CAP = 1_000_000


def _git(arguments: Sequence[str], root: Path | None = None) -> str | None:
    """One git command's stdout, or ``None`` where git cannot answer."""
    try:
        return subprocess.run(
            ["git", *arguments],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def working_tree_hash(sha: str, root: Path | None = None) -> str:
    """A digest of the code that actually ran, uncommitted changes included."""
    digest = hashlib.sha256(sha.encode())
    digest.update((_git(["diff", "HEAD"], root) or "").encode())

    listing = _git(["ls-files", "--others", "--exclude-standard"], root) or ""
    base = root if root is not None else Path.cwd()
    for name in sorted(line for line in listing.splitlines() if line):
        digest.update(name.encode())
        try:
            blob = (base / name).read_bytes()[:UNTRACKED_BYTE_CAP]
        except OSError:
            # A dangling symlink or a permission error. Recorded as its own
            # state rather than skipped, so the file's presence still counts.
            blob = b"<unreadable>"
        digest.update(hashlib.sha256(blob).digest())
    return digest.hexdigest()[:12]


def git_revision(root: Path | None = None) -> dict[str, str]:
    """The commit that is checked out, whether the tree matches it, and what ran."""
    sha = _git(["rev-parse", "HEAD"], root)
    status = _git(["status", "--porcelain"], root)
    if sha is None or status is None:
        # A tarball or a container with no git. Say so rather than logging a
        # plausible-looking blank.
        return {"git_sha": "unknown", "git_dirty": "unknown", "code_hash": "unknown"}

    dirty = bool(status)
    return {
        "git_sha": sha,
        "git_dirty": str(dirty).lower(),
        "code_hash": working_tree_hash(sha, root) if dirty else sha[:12],
    }


def dataset_version(settings: Settings, tables: Sequence[str]) -> str:
    """A digest over the commit markers of the gold tables a run reads.

    A table with no marker contributes ``missing`` rather than being skipped:
    training against an unbuilt table is a different run from training against a
    built one, and the version must say so.
    """
    parts = []
    for table in sorted(tables):
        marker = Path(gold_location(settings, table)) / "_SUCCESS"
        stamp = str(int(marker.stat().st_mtime)) if marker.is_file() else "missing"
        parts.append(f"{table}={stamp}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def config_hash(settings: Settings, arguments: Mapping[str, Any]) -> str:
    """A digest over the resolved settings AND the command line.

    Both halves are needed: two runs of the same YAML with different ``--lr``
    are different configurations, and so are two runs of the same command line
    against different ``RECSYS_`` overrides.
    """
    payload = {
        "settings": json.loads(settings.model_dump_json()),
        "arguments": {key: str(value) for key, value in sorted(arguments.items())},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def provenance(
    settings: Settings,
    arguments: Mapping[str, Any],
    tables: Sequence[str],
    root: Path | None = None,
) -> dict[str, str]:
    """Everything needed to say which code, config and data produced a number."""
    return {
        **git_revision(root),
        "config_hash": config_hash(settings, arguments),
        "dataset_version": dataset_version(settings, tables),
    }


def run_label(arm: str, marks: Mapping[str, str]) -> str:
    """The name a run is recorded and checkpointed under."""
    return f"{arm}-{marks['config_hash']}-{marks['code_hash']}"


def require_reachable(settings: Settings, timeout: float = 2.0) -> None:
    """Fail now if the tracking server is down, rather than after the data load.

    :func:`track` is entered late, so without this a missing server costs a full
    Spark read before surfacing as twelve frames of urllib3 ending in
    ``Connection refused``.

    Deliberately NOT a degrade-to-no-tracking path. A run whose metrics silently
    vanish is exactly the "run you cannot cite" failure the rest of this module
    exists to prevent: you would believe you had a logged run and not have one.
    Turning tracking off is a decision, spelled ``RECSYS_MLFLOW__ENABLED=false``.

    Raises:
        RuntimeError: If tracking is enabled and the server does not answer.
    """
    if not settings.mlflow.enabled:
        return

    import requests

    health = f"{settings.mlflow.tracking_uri.rstrip('/')}/health"
    try:
        requests.get(health, timeout=timeout).raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"MLflow is not answering at {settings.mlflow.tracking_uri} -- is it up? "
            "(`make up`). To train without tracking: RECSYS_MLFLOW__ENABLED=false"
        ) from exc


def mlflow_name(name: str) -> str:
    """A metric or parameter name, in the charset MLflow accepts.

    MLflow allows alphanumerics, ``_-. :/`` and nothing else, so ``recall@100``
    is rejected outright. The ``@`` notation is this project's own vocabulary --
    every evaluation card and every row of ``docs/baselines.md`` uses it -- so the
    translation happens HERE, at the one backend that objects, rather than by
    renaming the metric everywhere to suit it.
    """
    return name.replace("@", "_at_")


class Recorder:
    """What :func:`track` yields. The base class IS the disabled arm."""

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Discard."""

    def log_artifact(self, path: str) -> None:
        """Discard."""


class _MlflowRecorder(Recorder):
    """The enabled arm, wrapping the module rather than exposing it.

    Wrapping is what makes "one code path" true: the caller sees the same two
    methods either way, and the name translation cannot be forgotten at a call
    site because there is only one.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._client.log_metrics(
            {mlflow_name(name): value for name, value in metrics.items()}, step=step
        )

    def log_artifact(self, path: str) -> None:
        self._client.log_artifact(path)


@contextmanager
def track(
    settings: Settings,
    run_name: str,
    params: Mapping[str, Any],
    experiment: str | None = None,
) -> Iterator[Recorder]:
    """An MLflow run, or a no-op when tracking is disabled.

    mlflow is imported lazily: it pulls in a large dependency tree, and every
    test runs with tracking off.

    Args:
        experiment: Which experiment to record under. ``None`` takes the one in
            the settings. Callers name **the model they trained**, so the
            experiment list reads as the set of models this project has and a
            run can be found without knowing which stage produced it. Every run
            landing in one experiment is the alternative, and it makes the
            experiment name carry no information at all.
    """
    if not settings.mlflow.enabled:
        yield Recorder()
        return

    import mlflow

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(experiment or settings.mlflow.experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({mlflow_name(key): value for key, value in params.items()})
        yield _MlflowRecorder(mlflow)
