"""``make content-vectors`` -- dump L2-normalised content vectors for MMR.

The re-ranker's diversity term needs item-to-item similarity, and the orchestrator
holds no content table.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from common.config import load_settings
from common.features import CONTENT_VARIANTS
from common.tracking import git_revision
from models.reranking.policies import normalise
from models.retrieval.dataloader.dataset import load_item_tables

# Row 0 holds the reserved OOV vector and is exactly zero. Asserted rather than
# assumed: `normalise` leaves zero rows alone, so a row 0 that had drifted
# non-zero would be normalised to a unit vector and become the most similar
# neighbour of whatever it happened to point at.
OOV_ROW = 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump L2-normalised content vectors for MMR.")
    parser.add_argument(
        "--variant",
        choices=CONTENT_VARIANTS,
        default=CONTENT_VARIANTS[0],
        help="Must match the variant the catalogue and the checkpoints used.",
    )
    parser.add_argument(
        "--stem",
        type=Path,
        default=Path("serving/artifacts/content"),
        help="Writes <stem>.bin and <stem>.json.",
    )
    parser.add_argument("--version", default=None)
    args = parser.parse_args(argv)

    items = load_item_tables(load_settings(), args.variant)
    raw = items.content.numpy()

    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 1:
        raise SystemExit(f"content table is {raw.shape}; expected [n_items + 1, dim]")
    if float(np.abs(raw[OOV_ROW]).max()) != 0.0:
        raise SystemExit(
            f"row {OOV_ROW} of the content table is not zero. It is the reserved OOV "
            "row; normalising a non-zero one turns it into a unit vector that is a "
            "plausible nearest neighbour of whatever it points at"
        )

    # `policies.normalise`, CALLED rather than reimplemented. It is the function
    # the offline arm used, and it leaves all-zero rows alone -- a plain
    # `vectors / norms` divides row 0 by zero and seeds NaNs that propagate
    # through every similarity in every slate that touches a padded candidate.
    unit = np.ascontiguousarray(normalise(raw), dtype="<f4")

    norms = np.linalg.norm(unit, axis=1)
    worst = float(np.abs(norms[1:] - 1.0).max()) if len(norms) > 1 else 0.0
    if worst > 1e-3:
        # The same tolerance indexing/build_index.py:assert_unit_norm uses, for
        # the same reason: MMR treats a dot product as a cosine, and an
        # un-normalised table makes it rank by magnitude while returning
        # perfectly plausible neighbours.
        raise SystemExit(f"rows are not unit-norm after normalise (worst deviation {worst:.4g})")
    if float(np.abs(norms[OOV_ROW])) != 0.0:
        raise SystemExit(f"row {OOV_ROW} gained a norm during normalisation")

    args.stem.parent.mkdir(parents=True, exist_ok=True)
    binary = args.stem.with_suffix(".bin")
    meta = args.stem.with_suffix(".json")

    binary.write_bytes(unit.tobytes())
    version = args.version or git_revision()["code_hash"]
    meta.write_text(
        json.dumps(
            {
                "rows": int(unit.shape[0]),
                "dim": int(unit.shape[1]),
                # ZERO, unlike items.json. Position p holds item p, including the
                # reserved row 0, because the serving path indexes this table by
                # item id directly. Stated in the artifact rather than assumed by
                # the reader, because an off-by-one here computes every
                # similarity against the neighbouring article.
                "base_index": 0,
                "variant": args.variant,
                "version": version,
            }
        )
        + "\n"
    )
    print(
        f"wrote {binary} and {meta}: {unit.shape[0]:,} x {unit.shape[1]} float32 "
        f"({binary.stat().st_size / 1024**2:.1f} MiB), variant {args.variant}, "
        f"version {version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
