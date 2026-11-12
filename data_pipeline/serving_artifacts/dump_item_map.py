"""``make item-map`` -- dump bronze's ``item_map`` for the Go server to load.

The serving path has to translate both ways: the API speaks the catalogue's
string ids (``"N12345"``), and every model in this project speaks the dense
integer index that :mod:`data_pipeline.transform.id_maps` assigned.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from common.config import Settings, load_settings
from common.schemas import OOV_IDX
from common.tracking import git_revision


def load_map(settings: Settings) -> dict[str, int]:
    """Read ``item_id`` to ``item_idx`` from bronze.

    Read with pyarrow rather than Spark: it is one small table, and starting a
    session to read 160K rows costs more than the read.
    """
    table = pq.read_table(settings.paths.bronze / "item_map", columns=["item_id", "item_idx"])
    ids = table.column("item_id").to_pylist()
    idx = table.column("item_idx").to_pylist()
    return {str(item): int(index) for item, index in zip(ids, idx, strict=True)}


def check(mapping: dict[str, int]) -> None:
    """Refuse a map the serving path cannot use.

    The Spark contract tests already assert item_id is unique and item_idx is
    dense, so none of this should ever fire. It is here because the failure it
    guards against is SILENT: the checks cost milliseconds, and the alternative
    is a server that mistranslates a fraction of its slates.
    """
    if not mapping:
        raise SystemExit("item_map is empty; run `make bronze` first")

    indices = list(mapping.values())
    if len(set(indices)) != len(indices):
        raise SystemExit(
            "two item_ids share an item_idx: the map cannot be inverted, so some "
            "slate would be served under another article's id"
        )
    if OOV_IDX in mapping.values():
        # Index 0 is the reserved out-of-vocabulary row that short candidate
        # lists pad with. A real item sitting on it would be served whenever a
        # source ran short.
        raise SystemExit(f"an item claims the reserved OOV index {OOV_IDX}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump bronze item_map as JSON for serving.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("serving/artifacts/item_map.json"),
        help="Where to write. The Go server loads this path at startup.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Snapshot identifier. Defaults to the working-tree hash, which "
        "changes whenever the code that built the map did.",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    mapping = load_map(settings)
    check(mapping)

    version = args.version or git_revision()["code_hash"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys for a stable diff between dumps. Safe HERE, unlike in the blend
    # fixtures where sorting destroyed the source-to-quota alignment: this is a
    # mapping, and its order carries nothing.
    args.out.write_text(
        json.dumps({"version": version, "items": mapping}, sort_keys=True, indent=0) + "\n"
    )
    print(f"wrote {args.out}: {len(mapping)} items, version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
