"""``make popular`` -- dump the popularity fallback's slate for the Go server.

The fallback is what answers when retrieval has produced nothing. The manual's
requirement is that it needs no user features, no index and no model; this project
adds **no Redis**, for the same reason. A fallback with a runtime dependency is a
fallback that can fail, on the one path whose entire job is to always return
something.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from common.config import Settings, load_settings
from common.tracking import git_revision
from common.utils import gold_location

DEFAULT_TOP_N = 500


def ranked_items(settings: Settings, item_map: dict[str, int], top_n: int) -> list[int]:
    """Internal item indices, most popular first."""
    location = gold_location(settings, "item_hourly_features", scheme="s3")
    filesystem = None
    if location.startswith(("s3://", "s3a://")):
        # The same s3fs wiring common.utils uses to probe for _SUCCESS, built from
        # conf/config.yml's endpoint -- so there is no second copy of the storage
        # config to drift. Credentials come from .env (UV_ENV_FILE).
        import s3fs

        filesystem = s3fs.S3FileSystem(
            client_kwargs={"endpoint_url": settings.storage.endpoint_url},
            skip_instance_cache=True,
        )
        location = location.removeprefix("s3a://").removeprefix("s3://")

    frame = pq.read_table(
        location,
        columns=["item_id", "feature_ts", "item_clicks_24h"],
        filesystem=filesystem,
    ).to_pandas()

    # The LATEST bucket per item, not a sum over buckets: the 24h column is
    # already a rolling window, so summing it would count each click up to 24
    # times and rank by how long an article has been live.
    latest = frame.sort_values("feature_ts").groupby("item_id", as_index=False).last()
    latest = latest.sort_values("item_clicks_24h", ascending=False)

    out: list[int] = []
    for item_id in latest["item_id"]:
        index = item_map.get(str(item_id))
        # Index 0 is the reserved OOV row and is not an article. An unmapped id
        # means this table and the item map are from different builds; SKIPPING is
        # right here because a fallback slate that is one item short beats a
        # fallback that refuses to load.
        if index and index > 0:
            out.append(int(index))
        if len(out) >= top_n:
            break
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump the popularity fallback slate.")
    parser.add_argument("--item-map", type=Path, default=Path("serving/artifacts/item_map.json"))
    parser.add_argument("--out", type=Path, default=Path("serving/artifacts/popular.json"))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--version", default=None)
    args = parser.parse_args(argv)

    payload = json.loads(args.item_map.read_text())
    item_map = {str(key): int(value) for key, value in (payload.get("items") or {}).items()}
    if not item_map:
        raise SystemExit(f"{args.item_map} is empty; run `make item-map` first")

    items = ranked_items(load_settings(), item_map, args.top_n)
    if not items:
        # An empty fallback loads fine and returns an empty slate on the one path
        # that exists to never do that.
        raise SystemExit("no popular items found; has `make gold` run?")

    version = args.version or git_revision()["code_hash"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"version": version, "items": items}) + "\n")
    print(f"wrote {args.out}: {len(items)} items, version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())