"""Versioning, the promotion gate, and retention for a rebuilt index.

The scheduler file that calls this is thin wiring; everything that can be wrong
lives here, where it can be tested without an Airflow installation.

**The gate compares CLICK-RECALL, not agreement with exact search.** Those come
apart badly: an index measured here lost 1.3% of exact search's candidates for
0.05% of the clicks, and a coarser one disagreed with exact search on 9% of
candidates while finding MORE clicks. A gate on agreement would block a better
index and pass a worse one whose errors happened to fall on items nobody clicks.

**A bad index is worse than a stale one.** The previous version is already
serving and already known to work, so the failure mode to design against is
promoting a regression, not missing a refresh. Promotion is therefore
fail-closed: anything the gate cannot evaluate aborts it.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# How far a rebuild may fall below the live index before promotion aborts.
# Absolute rather than relative: recall here is ~0.38, and a 1% relative
# tolerance would be 0.0038, which is inside the run-to-run spread of the
# training that produced the embeddings.
RECALL_TOLERANCE = 0.005

# Rollback depth. Three is two more than anyone plans to use and one more than
# the number of times "the previous one was also broken" has happened.
KEEP_VERSIONS = 3

# Dropped into a version directory the gate refused. Retention reads it so a run
# of refusals cannot crowd the last good rollback target out of the kept window.
REFUSED_MARKER = "REFUSED"


@dataclass(frozen=True)
class Promotion:
    """The gate's answer, and why.

    Attributes:
        allowed: Whether to swap the pointer.
        reason: Human-readable, and logged either way -- a gate that only
            explains itself on failure gives no evidence when it passes.
    """

    allowed: bool
    reason: str


def version_label(moment: datetime) -> str:
    """The artifact prefix for a build, to the hour.

    Hour granularity, not day: on a news corpus an article can be published,
    peak and die between two nightly builds, so an item missing from the index
    is missing for its entire useful life.
    """
    return f"v={moment.strftime('%Y-%m-%dT%H:00Z')}"


def gate(
    candidate_recall: float,
    live_recall: float | None,
    tolerance: float = RECALL_TOLERANCE,
) -> Promotion:
    """Whether a freshly built index may replace the one serving.

    Args:
        candidate_recall: The rebuild's click-recall on the probe set.
        live_recall: The serving index's click-recall on the SAME probe set.
            ``None`` means there is nothing live yet, which is the only case
            where an unmeasured comparison is allowed to promote.
        tolerance: How far below live the candidate may fall.

    Returns:
        A :class:`Promotion`.
    """
    if candidate_recall != candidate_recall:  # NaN
        return Promotion(False, "candidate recall is undefined; refusing to promote")
    if live_recall is None:
        return Promotion(True, f"first index, recall {candidate_recall:.4f}")

    drop = live_recall - candidate_recall
    if drop > tolerance:
        return Promotion(
            False,
            f"recall {candidate_recall:.4f} is {drop:.4f} below live "
            f"{live_recall:.4f}, past the {tolerance:.4f} tolerance",
        )
    return Promotion(
        True, f"recall {candidate_recall:.4f} against live {live_recall:.4f} (drop {drop:+.4f})"
    )


def expired(versions: list[str], keep: int = KEEP_VERSIONS) -> list[str]:
    """Versions to delete, oldest first, once ``keep`` newest are retained.

    Sorted lexicographically, which is chronological because
    :func:`version_label` is zero-padded ISO. That is the reason for the format
    rather than a preference about it.
    """
    return sorted(versions)[: max(0, len(versions) - keep)]


def mark_refused(version_dir: Path, reason: str) -> None:
    """Record that the gate refused this build, and why, beside the index."""
    (version_dir / REFUSED_MARKER).write_text(reason + "\n")


def retire(artifacts: Path, live: str | None, keep: int = KEEP_VERSIONS) -> list[str]:
    """Delete versions past the retention depth, and return what was deleted.

    Promoted and refused builds are retained SEPARATELY, ``keep`` of each. Counted
    together, a run of refusals would fill the window: live good v1, refused v2
    and v3, promoted v4, keep three -- and v1, the one known-good rollback
    target, is the one deleted. Refused builds are kept at all because they are
    what an incident is diagnosed from.

    The live version is never deleted, whatever the arithmetic says. Deleting is
    safe for a sidecar already serving an older version: FAISS reads the index
    fully into memory (no mmap), so the process does not need the file.
    """
    dirs = [path for path in artifacts.glob("v=*") if path.is_dir()]
    refused = [path.name for path in dirs if (path / REFUSED_MARKER).is_file()]
    promoted = [path.name for path in dirs if path.name not in refused]

    stale = sorted(v for v in expired(promoted, keep) + expired(refused, keep) if v != live)
    for version in stale:
        shutil.rmtree(artifacts / version)
    return stale


def promote(pointer: Path, version: str) -> Path:
    """Point serving at ``version``, atomically.

    Written to a temporary file in the same directory and then renamed. A rename
    within one filesystem is atomic, so a reader either sees the old version or
    the new one and never a half-written pointer -- which a plain
    truncate-and-write does allow, and which would take serving down rather than
    merely serving something stale. Same directory for the same reason: a rename
    across filesystems is a copy, and a copy is not atomic.
    """
    pointer.parent.mkdir(parents=True, exist_ok=True)
    handle, staged = tempfile.mkstemp(dir=pointer.parent, suffix=".tmp")
    with os.fdopen(handle, "w") as out:
        out.write(version)
    Path(staged).replace(pointer)
    return pointer


def current(pointer: Path) -> str | None:
    """Which version is serving, or None if nothing has been promoted."""
    return pointer.read_text().strip() if pointer.is_file() else None
