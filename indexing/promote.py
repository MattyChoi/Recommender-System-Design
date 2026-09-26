"""Build, gate and promote a retrieval index, with no scheduler in sight.

``serving/retrieval`` refuses to start when nothing has been promoted, and its
error said "run the hourly_index DAG or promote one by hand" -- but there was no
by-hand route, so the first index on any machine required a heredoc. This is
that route, and it is the same one: every step here is the function
``orchestration/dags/hourly_index.py`` calls, in the same order, so the manual
and scheduled paths cannot drift.

**The catalogue is encoded ONCE here, twice in the DAG.** That is not an
improvement on the DAG's part -- it re-encodes in ``validate`` because the only
thing it could carry from ``rebuild`` is XCom, and XCom is a column in a
metadata database that a 65k x 128 float table has no business in. A single
process has no such constraint, so it keeps the :class:`~indexing.pipeline.Probe`
in memory and the candidate and the live index are scored on the same object
rather than on two encodings that are merely supposed to match.

**Promotion is fail-closed**, exactly as in the DAG: the live index already
works, so the risk to design against is replacing it with a regression rather
than missing a refresh. Anything the gate cannot evaluate leaves the pointer
alone.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from indexing.lifecycle import (
    KEEP_VERSIONS,
    RECALL_TOLERANCE,
    Promotion,
    current,
    expired,
    gate,
    promote,
    version_label,
)
from indexing.pipeline import (
    build,
    embed_catalogue,
    load_version,
    probe_recall,
    write_version,
)
from models.retrieval.dataloader.dataset import CONTENT_VARIANTS

#: Where the sidecar looks by default. Matches ``serving/retrieval/service.py``'s
#: DEFAULT_ARTIFACTS and the hourly DAG's params; all three must agree or the
#: service serves an index nobody promoted.
DEFAULT_ROOT = Path("/srv/recsys/index")

#: HNSW, not the DAG's ``flat``. ADR 0002 ships efSearch 512, the sidecar
#: feature-detects ``SearchParametersHNSW`` and reports ``index_kind`` from the
#: OBJECT -- none of which means anything for a flat index. A flat index is a
#: legitimate choice on a 65k catalogue; it is just not the one the rest of the
#: serving path is written for, so it is opted into rather than defaulted to.
DEFAULT_KIND = "hnsw"


@dataclass(frozen=True)
class Outcome:
    """What a promotion run did, and to what.

    Attributes:
        label: The version built, whether or not it was promoted.
        path: Where the index was written. It stays on disk even when the gate
            refuses, because a rejected candidate is evidence.
        candidate_recall: The rebuild's click-recall on the probe set.
        live_recall: The serving index's click-recall on the SAME probe set,
            or ``None`` when nothing was live.
        decision: The gate's answer and its reason.
        retired: Versions past the retention depth.
        pruned: Whether ``retired`` was actually deleted from disk.
    """

    label: str
    path: Path
    candidate_recall: float
    live_recall: float | None
    decision: Promotion
    retired: tuple[str, ...]
    pruned: bool


def run_promotion(
    checkpoint: Path,
    *,
    root: Path = DEFAULT_ROOT,
    pointer: Path | None = None,
    kind: str = DEFAULT_KIND,
    moment: datetime | None = None,
    variant: str = CONTENT_VARIANTS[0],
    holdout_hours: int = 12,
    max_negs: int = 4,
    k: int = 100,
    tolerance: float = RECALL_TOLERANCE,
    keep: int = KEEP_VERSIONS,
    prune: bool = False,
    dry_run: bool = False,
) -> Outcome:
    """Encode, build, score against live, and swap the pointer if the gate allows.

    Args:
        checkpoint: The two-tower run whose item embeddings become the index.
            **Must be the same checkpoint ``make index-vectors`` used**: rung 2
            of ADR 0013 searches ``serving/artifacts/items.bin`` with a query
            embedding cached from this tower, and two checkpoints put the
            queries and the vectors in different spaces with nothing at runtime
            to notice.
        root: The artifacts directory holding ``v=*`` version directories.
        pointer: The CURRENT file. Defaults to ``root / "CURRENT"``.
        kind: ``flat``, ``hnsw`` or ``ivfpq``.
        moment: The instant the version label is built from. Defaults to now.
        variant: Which cached content vectors the towers were trained on.
        holdout_hours: The validation window the probe requests come from.
        max_negs: Slate negatives per row when loading the validation split.
        k: Depth at which click-recall is measured.
        tolerance: How far below live the candidate may fall.
        keep: Retention depth.
        prune: Actually delete retired versions. Off by default, because a
            deletion that happens as a side effect of a promotion is a bad way
            to discover the retention depth was wrong.
        dry_run: Build and score, report the decision, but never write the
            pointer.

    Returns:
        An :class:`Outcome`.
    """
    marker = pointer if pointer is not None else root / "CURRENT"

    probe = embed_catalogue(
        checkpoint, variant=variant, holdout_hours=holdout_hours, max_negs=max_negs
    )
    label = version_label(moment or datetime.now(UTC))
    written = write_version(build(probe.vectors, kind), root, label)

    candidate = probe_recall(load_version(root, label), probe, k)
    live_label = current(marker)
    live = probe_recall(load_version(root, live_label), probe, k) if live_label else None

    decision = gate(candidate, live, tolerance)
    if decision.allowed and not dry_run:
        promote(marker, label)

    # Retention is computed from what is on disk AFTER the swap, so the version
    # that just became CURRENT is inside the retained window by construction.
    # Only meaningful once the pointer moved: on a refusal the live index may be
    # an old one that retention would otherwise delete out from under serving.
    retired: tuple[str, ...] = ()
    if decision.allowed and not dry_run:
        retired = tuple(expired([path.name for path in root.glob("v=*")], keep))
        if prune:
            for stale in retired:
                shutil.rmtree(root / stale)

    return Outcome(
        label=label,
        path=written,
        candidate_recall=candidate,
        live_recall=live,
        decision=decision,
        retired=retired,
        pruned=prune and decision.allowed and not dry_run,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The CLI ``make index-promote`` drives."""
    parser = argparse.ArgumentParser(
        prog="python -m indexing.promote",
        description="Build a FAISS index from a checkpoint, gate it, and promote it.",
    )
    parser.add_argument("checkpoint", type=Path, help="two-tower checkpoint")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="artifacts directory")
    parser.add_argument("--pointer", type=Path, default=None, help="default: <root>/CURRENT")
    parser.add_argument("--kind", choices=("flat", "hnsw", "ivfpq"), default=DEFAULT_KIND)
    parser.add_argument("--variant", choices=CONTENT_VARIANTS, default=CONTENT_VARIANTS[0])
    parser.add_argument("--holdout-hours", type=int, default=12)
    parser.add_argument("--max-negs", type=int, default=4)
    parser.add_argument("-k", "--top-k", type=int, default=100, help="depth for click-recall")
    parser.add_argument("--tolerance", type=float, default=RECALL_TOLERANCE)
    parser.add_argument("--keep", type=int, default=KEEP_VERSIONS, help="retention depth")
    parser.add_argument(
        "--prune", action="store_true", help="delete retired versions instead of naming them"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="build and score, but never move the pointer"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Returns 0 on promotion, 1 on refusal.

    A refusal is a NON-ZERO exit on purpose: it is the expected outcome of a
    regression, and a caller that cannot tell it from a promotion would report
    a stale index as a fresh one.
    """
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        raise SystemExit(f"no such checkpoint: {args.checkpoint}")

    outcome = run_promotion(
        args.checkpoint,
        root=args.root,
        pointer=args.pointer,
        kind=args.kind,
        variant=args.variant,
        holdout_hours=args.holdout_hours,
        max_negs=args.max_negs,
        k=args.top_k,
        tolerance=args.tolerance,
        keep=args.keep,
        prune=args.prune,
        dry_run=args.dry_run,
    )

    live = "nothing live" if outcome.live_recall is None else f"{outcome.live_recall:.4f}"
    print(f"{outcome.label}: wrote {outcome.path}")
    print(f"  recall@{args.top_k}  {outcome.candidate_recall:.4f} candidate, {live} live")
    print(f"  gate        {outcome.decision.reason}")
    if args.dry_run:
        print("  pointer     UNCHANGED (--dry-run)")
    elif outcome.decision.allowed:
        print(f"  pointer     -> {outcome.label}")
    else:
        print(f"  pointer     unchanged, still {current(args.pointer or args.root / 'CURRENT')}")
    for stale in outcome.retired:
        print(f"  {'deleted' if outcome.pruned else 'retire'}     {stale}")

    return 0 if outcome.decision.allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
