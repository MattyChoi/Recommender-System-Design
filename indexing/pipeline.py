"""The work an index rebuild actually does, with no scheduler in sight.

Every step a DAG task calls lives here as a plain function: encode the
catalogue, build an index, score it on a probe set. The scheduler file wires
these together and does nothing else, so the part that can be wrong is the part
CI runs.

**The probe set is the validation window, and it is fixed.** A gate that scores
the candidate index on fresher data than the live one measured would compare a
model against a different question and call the difference a regression.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from common.config import Settings, load_settings
from common.features import CONTENT_VARIANTS
from common.spark import get_spark
from common.torch_env import select_device
from common.utils import load_tower
from data_pipeline.features.user_history import MAX_HISTORY
from indexing.build_index import build_flat, build_hnsw, build_ivfpq, save, search
from indexing.lifecycle import (
    KEEP_VERSIONS,
    RECALL_TOLERANCE,
    current,
    expired,
    gate,
    promote,
    version_label,
)
from models.retrieval.dataloader.dataset import load_item_tables, load_train_and_validation
from models.retrieval.sources import encode_users

INDEX_FILENAME = "index.faiss"
INDEX_KINDS = ("flat", "hnsw", "ivfpq")

DEFAULT_KIND = "flat"
POINTER_FILENAME = "CURRENT"
DEFAULT_CHECKPOINT = Path("data/checkpoints/both-logq-n4u0-b8192e10lr0.001-ab7e1d500b9bf792.pt")


@dataclass(frozen=True)
class Probe:
    """Everything needed to build an index and to judge it.

    Attributes:
        vectors: ``[n_items, dim]`` unit-norm item embeddings, row 0 dropped.
        queries: ``[R, dim]`` request vectors from the validation window.
        clicked: ``[R]`` the item each request actually clicked.
    """

    vectors: npt.NDArray[np.float32]
    queries: npt.NDArray[np.float32]
    clicked: npt.NDArray[np.int64]


@dataclass(frozen=True)
class Outcome:
    """What one rebuild did, and why.

    Attributes:
        label: The version this build wrote, whether or not it was promoted.
        candidate_recall: Click-recall of the new index on the probe set.
        live_recall: The same for whatever was serving, or None if nothing was.
        promoted: Whether the pointer now names ``label``.
        reason: The gate's sentence, logged either way -- a gate that explains
            itself only on failure gives no evidence when it passes.
        retired: Versions past the retention depth. Reported, NOT deleted; see
            :func:`rebuild_and_promote`.
    """

    label: str
    candidate_recall: float
    live_recall: float | None
    promoted: bool
    reason: str
    retired: list[str]


def embed_catalogue(
    checkpoint: Path,
    settings: Settings | None = None,
    variant: str = CONTENT_VARIANTS[0],
    holdout_hours: int = 12,
    max_negs: int = 4,
) -> Probe:
    """Run the towers once: every item, and every probe request."""
    resolved = settings or load_settings()
    device = select_device()

    spark = get_spark(resolved, app="index-rebuild")
    try:
        items = load_item_tables(resolved, variant)
        train_split, val_split = load_train_and_validation(
            spark, resolved, MAX_HISTORY, max_negs, holdout_hours
        )
    finally:
        spark.stop()

    tower = load_tower(checkpoint, items, train_split.user_feats.shape[1], device)
    from indexing.build_index import item_vectors

    return Probe(
        vectors=item_vectors(tower),
        queries=encode_users(tower, val_split, device).numpy().astype("float32"),
        clicked=val_split.item_ids.numpy(),
    )


def build(vectors: npt.NDArray[np.float32], kind: str = "flat") -> Any:
    """One of the three index kinds, by name.

    Raises:
        ValueError: On an unknown kind. A typo would otherwise fall through to
            a default and promote an index nobody asked for.
    """
    if kind == "flat":
        return build_flat(vectors)
    if kind == "hnsw":
        return build_hnsw(vectors)
    if kind == "ivfpq":
        return build_ivfpq(vectors)
    raise ValueError(f"unknown index kind {kind!r}; expected flat, hnsw or ivfpq")


def probe_recall(index: Any, probe: Probe, k: int = 100) -> float:
    """Share of probe requests whose clicked item the index returns.

    Click-recall, not overlap with exact search. The two come apart: an index
    can disagree with exact search about a tenth of its candidates and find more
    of the answers, so overlap is the wrong thing to promote on.
    """
    got = search(index, probe.queries, k)
    return float((got == probe.clicked[:, None]).any(axis=1).mean())


def write_version(index: Any, root: Path, label: str) -> Path:
    """Write the index under its own version directory. Never in place."""
    return save(index, root / label / INDEX_FILENAME)


def load_version(root: Path, label: str) -> Any:
    from indexing.build_index import load

    return load(root / label / INDEX_FILENAME)


def rebuild_and_promote(
    checkpoint: Path,
    artifacts: Path,
    pointer: Path,
    kind: str = DEFAULT_KIND,
    moment: datetime | None = None,
    settings: Settings | None = None,
    variant: str = CONTENT_VARIANTS[0],
    tolerance: float = RECALL_TOLERANCE,
    keep: int = KEEP_VERSIONS,
    probe: Probe | None = None,
) -> Outcome:
    """Encode, build, score against live, and promote if the gate allows.

    Args:
        checkpoint: The two-tower run whose item vectors become the index. Must
            be the checkpoint serving will encode queries with.
        artifacts: Root holding one directory per version.
        pointer: The file serving reads to learn which version is live.
        kind: ``flat``, ``hnsw`` or ``ivfpq``.
        moment: Instant the version label is derived from. Defaults to now.
        settings: Root configuration; loaded if omitted.
        variant: Which cached content vectors to encode from.
        tolerance: How far below live the candidate may fall.
        keep: Retention depth.
        probe: Pre-encoded probe set, for tests. Supplying it skips
            :func:`embed_catalogue` and with it Spark, torch and the checkpoint
            -- which is what makes the gate's effect on the pointer assertable
            without an Airflow installation or a GPU. Production leaves it None.

    Returns:
        An :class:`Outcome`. ``promoted`` False is a normal result, not an error.
    """
    label = version_label(moment or datetime.now(UTC))

    if probe is None:
        probe = embed_catalogue(checkpoint, settings, variant)
    written = write_version(build(probe.vectors, kind), artifacts, label)
    print(f"{label}: wrote {written}")

    candidate = probe_recall(load_version(artifacts, label), probe)
    live_label = current(pointer)
    live = probe_recall(load_version(artifacts, live_label), probe) if live_label else None

    decision = gate(candidate, live, tolerance)
    if not decision.allowed:
        return Outcome(label, candidate, live, False, decision.reason, [])

    promote(pointer, label)
    retired = expired([path.name for path in artifacts.glob("v=*")], keep)
    return Outcome(label, candidate, live, True, decision.reason, retired)


def main(argv: Sequence[str] | None = None) -> int:
    """``make index-promote`` -- one rebuild, from the command line.

    Exits 1 when the gate refuses. A refusal is a correct outcome of the gate
    and a failed outcome for the caller, who asked for a promotion and did not
    get one; the pointer is unchanged and the reason is on stdout.
    """
    parser = argparse.ArgumentParser(description="Build, gate and promote a retrieval index.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("data/index"))
    parser.add_argument(
        "--pointer",
        type=Path,
        default=None,
        help=f"Defaults to <artifacts>/{POINTER_FILENAME}, which is what the sidecar reads.",
    )
    parser.add_argument("--kind", choices=INDEX_KINDS, default=DEFAULT_KIND)
    parser.add_argument("--variant", choices=CONTENT_VARIANTS, default=CONTENT_VARIANTS[0])
    parser.add_argument(
        "--at",
        default=None,
        help="ISO instant for the version label. Defaults to now; pass one to rebuild "
        "a label deterministically.",
    )
    args = parser.parse_args(argv)

    outcome = rebuild_and_promote(
        checkpoint=args.checkpoint,
        artifacts=args.artifacts,
        pointer=args.pointer or args.artifacts / POINTER_FILENAME,
        kind=args.kind,
        moment=datetime.fromisoformat(args.at) if args.at else None,
        variant=args.variant,
    )

    print(f"{outcome.label}: {outcome.reason}")
    for stale in outcome.retired:
        print(f"  retire {stale} (not deleted)")
    if not outcome.promoted:
        print("  pointer unchanged")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
