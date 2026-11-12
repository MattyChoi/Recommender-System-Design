"""``make catalogue`` -- dump the static per-item attributes for the Go server.

Category and subcategory are the ranker's last two columns and back the
per-category cap in the re-ranker.

Indexed by `item_idx`, position 0 reserved for OOV -- the same layout every other
artifact in this project uses.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from common.config import load_settings
from common.features import CONTENT_VARIANTS
from common.tracking import git_revision
from models.retrieval.dataloader.dataset import load_item_tables


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump per-item category indices for serving.")
    parser.add_argument("--variant", choices=CONTENT_VARIANTS, default=CONTENT_VARIANTS[0])
    parser.add_argument("--out", type=Path, default=Path("serving/artifacts/catalogue.json"))
    parser.add_argument("--version", default=None)
    args = parser.parse_args(argv)

    items = load_item_tables(load_settings(), args.variant)
    category = items.category.tolist()
    subcategory = items.subcategory.tolist()

    if len(category) != len(subcategory):
        raise SystemExit(
            f"category has {len(category)} rows and subcategory {len(subcategory)}; "
            "they index the same items and must agree"
        )
    if category[0] != 0 or subcategory[0] != 0:
        # Row 0 is the reserved OOV row. A real category sitting on it would be
        # attributed to every candidate a source padded with zeros.
        raise SystemExit(f"row 0 is not reserved: category {category[0]}, sub {subcategory[0]}")

    version = args.version or git_revision()["code_hash"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "version": version,
                # ARRAYS, positional by item_idx. An object keyed by index would
                # survive this dump's sort_keys and cost a parse.
                "category": category,
                "subcategory": subcategory,
                "n_categories": items.n_categories,
                "n_subcategories": items.n_subcategories,
            }
        )
        + "\n"
    )
    print(
        f"wrote {args.out}: {len(category):,} rows, "
        f"{items.n_categories} categories, {items.n_subcategories} subcategories, "
        f"version {version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
