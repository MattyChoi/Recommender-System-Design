"""``make index-vectors`` -- dump item embeddings for the orchestrator

**The same vectors, from the same checkpoint.** `indexing.build_index` drops
row 0 and hands FAISS items ``1..n`` at positions ``0..n-1``; this writes the
identical matrix with ``base_index: 1`` in the sidecar, so a position here
means what it means there. An artifact built from a different checkpoint than
the index would search a space the cached query vectors are not in, and return
real item ids in plausible order.

Two files, deliberately:

- ``<stem>.bin`` -- raw little-endian float32, row-major, no header. A header
  would be a second thing to keep in step with the sidecar that declares the
  shape.
- ``<stem>.json`` -- rows, dim, base_index, version. Read FIRST by the loader,
  which then checks the binary is exactly the declared size: a truncated file
  has a plausible size for a smaller matrix, and inferring the shape from it
  would silently serve the prefix.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from common.config import load_settings
from common.features import CONTENT_VARIANTS
from common.tracking import git_revision
from common.utils import load_tower
from indexing.build_index import assert_unit_norm, item_vectors
from models.retrieval.dataloader.dataset import load_item_tables


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump item vectors for in-process exact search.")
    parser.add_argument("checkpoint", type=Path, help="A .pt from models.retrieval.train.")
    parser.add_argument("--variant", choices=CONTENT_VARIANTS, default=CONTENT_VARIANTS[0])
    parser.add_argument(
        "--stem",
        type=Path,
        default=Path("serving/artifacts/items"),
        help="Writes <stem>.bin and <stem>.json.",
    )
    parser.add_argument("--version", default=None)
    args = parser.parse_args(argv)

    settings = load_settings()
    items = load_item_tables(settings, args.variant)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    if "user_norm.weight" not in state:
        raise SystemExit(f"{args.checkpoint} is not a two-tower checkpoint")
    n_user_feats = int(state["user_norm.weight"].shape[0])

    tower = load_tower(args.checkpoint, items, n_user_feats, torch.device("cpu"))
    tower.eval()

    # item_vectors drops row 0 and asserts unit norm. Asserted again after the
    # dtype cast below, because the check that matters is on the bytes that
    # actually get written -- inner product is cosine ONLY because the towers
    # normalise, and an un-normalised table ranks by magnitude while returning
    # perfectly plausible neighbours.
    vectors = item_vectors(tower)
    flat = np.ascontiguousarray(vectors, dtype="<f4")
    assert_unit_norm(flat)

    args.stem.parent.mkdir(parents=True, exist_ok=True)
    binary = args.stem.with_suffix(".bin")
    meta = args.stem.with_suffix(".json")

    binary.write_bytes(flat.tobytes())
    version = args.version or git_revision()["code_hash"]
    meta.write_text(
        json.dumps(
            {
                "rows": int(flat.shape[0]),
                "dim": int(flat.shape[1]),
                # Position p holds item p + 1: row 0 is the reserved OOV row
                # and never enters an index. Stated in the artifact rather than
                # assumed by the reader, because an off-by-one here returns
                # real ids for the wrong articles.
                "base_index": 1,
                "version": version,
            }
        )
        + "\n"
    )

    print(
        f"wrote {binary} ({flat.nbytes / 1e6:.1f} MB) and {meta}: "
        f"{flat.shape[0]:,} items x {flat.shape[1]}d, version {version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
