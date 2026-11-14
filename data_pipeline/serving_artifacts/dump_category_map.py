"""``make category-map`` -- dump the category vocabulary for the Feast gateway.

`catalogue.json` holds each item's category as an INTEGER index, which is what
the ranker was fitted on. Feast keys `user_category_stats` on the category
STRING. This file is the mapping between them.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from common.config import Settings, load_settings
from common.tracking import git_revision


def load_vocabulary(settings: Settings, name: str) -> list[str]:
    """``{name}_map.parquet`` -> a positional array, index 0 reserved.

    Two columns, written by `_write_map` in `content_cache`: the string and
    its 1-based index. Read with pyarrow rather than Spark, for the reason every
    dumper in this directory does -- the serving path has no JVM.

    Args:
        settings: The root configuration object.
        name: `"category"` or `"subcategory"`.

    Returns:
        `[""] + [name at index 1, name at index 2, ...]`.

    Raises:
        SystemExit: If the map is missing, empty, claims index 0, or has a gap.
    """
    path = Path(settings.paths.gold) / "item_content" / f"{name}_map.parquet"
    if not path.exists():
        raise SystemExit(f"{path} does not exist -- run `make content` first")

    table = pq.read_table(path, columns=[name, f"{name}_idx"])
    pairs = dict(
        zip(
            table.column(f"{name}_idx").to_pylist(),
            table.column(name).to_pylist(),
            strict=True,
        )
    )

    if not pairs:
        raise SystemExit(f"{path} is empty -- `make content` wrote no {name} map")
    if 0 in pairs:
        # content_cache assigns from 1 and floors sub-threshold values to 0, so a
        # string ON index 0 means that floor was bypassed. Serving would then
        # resolve a real category name for every candidate that was floored.
        raise SystemExit(f"{path} assigns index 0 to {pairs[0]!r}; 0 is reserved")

    top = max(pairs)
    missing = sorted(set(range(1, top + 1)) - set(pairs))
    if missing:
        # A gap would leave "" at a REAL index, and the gateway would send an
        # empty category string to Feast -- a miss that reads as a cold user
        # rather than as a broken artifact.
        raise SystemExit(f"{path} is not dense over 1..{top}: missing {missing[:5]}")

    # Position 0 is the reserved bucket, and "" rather than a name because the
    # values content_cache floored to 0 are SEVERAL strings, not one.
    return [""] + [pairs[index] for index in range(1, top + 1)]


def check_against_catalogue(
    catalogue_path: Path, categories: list[str], subcategories: list[str]
) -> None:
    """Refuse a vocabulary that disagrees with the catalogue's own counts.

    `catalogue.json`'s `n_categories` is `int(category.max())` over the
    same content cache, so the two must agree. If they do not, they came from
    different `make content` runs and the gateway would resolve index 7 to a
    different name than the ranker was fitted with -- a feature that is in range,
    plausible, and wrong.

    Skipped when the catalogue has not been dumped yet, so the two dumpers can
    be run in either order; `make artifacts` runs catalogue first.
    """
    if not catalogue_path.exists():
        return

    catalogue = json.loads(catalogue_path.read_text())
    for key, vocabulary in (
        ("n_categories", categories),
        ("n_subcategories", subcategories),
    ):
        if key not in catalogue:
            continue
        if catalogue[key] != len(vocabulary) - 1:
            raise SystemExit(
                f"{catalogue_path} {key}={catalogue[key]} but the vocabulary has "
                f"{len(vocabulary) - 1}. One of the two is from an older "
                "`make content`; rebuild both."
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump the category vocabulary for serving.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("serving/artifacts/category_map.json"),
        help="Where to write. The Feast gateway loads this path at startup.",
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=Path("serving/artifacts/catalogue.json"),
        help="Cross-checked for agreeing category counts, when it exists.",
    )
    parser.add_argument("--version", default=None)
    args = parser.parse_args(argv)

    settings = load_settings()
    categories = load_vocabulary(settings, "category")
    subcategories = load_vocabulary(settings, "subcategory")
    check_against_catalogue(args.catalogue, categories, subcategories)

    version = args.version or git_revision()["code_hash"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "version": version,
                # ARRAYS, positional by index -- the same layout catalogue.json
                # uses, and for the same reason: an object keyed by index costs a
                # parse and carries no more information.
                "categories": categories,
                "subcategories": subcategories,
            }
        )
        + "\n"
    )
    print(
        f"wrote {args.out}: {len(categories) - 1} categories, "
        f"{len(subcategories) - 1} subcategories, version {version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
