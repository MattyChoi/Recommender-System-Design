"""Frozen sentence-encoder vectors for every item in the catalogue.

The item tower reads content vectors on every forward pass, so recomputing them
per epoch would dominate training time. They are computed once here and cached.

**Two variants, deliberately.** ``vec_title`` matches exactly what F3's content
baseline tokenises, so the neural content tower can be compared against it with
the input held fixed. ``vec_title_abstract`` is the better representation and
what the model should actually use. Caching both turns "is the cold-item gap
architecture or is it the abstract?" from a caveat into a measurement, and on a
4090 the second pass costs about a minute.

**Code here, data in gold.** This needs torch, a downloaded encoder and ideally
a GPU, none of which ``make gold`` requires -- so it is built by ``make
content`` and lives beside the model that consumes it. The OUTPUT is a derived
table like any other and lands in the gold layer, which is why ``item_content``
is in ``GOLD_TABLES``. It is the one gold table ``make gold`` does not build.

Category and subcategory index maps are built here too, for the same reason the
vectors are: they are item-indexed inputs to the item tower, and splitting them
across two artefacts would let them disagree about what index 7 means. The two
maps are built from separate columns and share nothing: MIND reuses strings
across them (``tv`` and ``games`` are both category and subcategory names,
meaning different things), so index 7 in one has no relation to index 7 in the
other and the maps must never be joined on the string.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from common.config import Settings, load_settings
from common.spark import get_spark
from common.torch_env import describe, select_device
from common.utils import SPLITS, _is_built, read_news

DEFAULT_ENCODER = "BAAI/bge-base-en-v1.5"

# A subcategory with one item is an item-ID embedding wearing a different hat:
# a free parameter trained by that item alone, and reachable in the
# content-only arm of G1's ablation, which is meant to have none. 149 of 270
# subcategories sit below 25 and cover 1.32% of the catalogue; on the category
# side the cut takes games (1 item), northamerica (1), middleeast (2) and kids
# (22), with a clean gap to movies at 672.
MIN_INDEX_COUNT = 25


def _index_map(values: Iterable[str | None], min_count: int = 1) -> dict[str, int]:
    """Deterministic 1-based index per distinct value, 0 left reserved.

    Sorted rather than encounter-ordered so two builds of the same catalogue
    agree. Values seen fewer than ``min_count`` times get no index and fall to
    0, which ``padding_idx`` zeroes -- see :data:`MIN_INDEX_COUNT`. The default
    of 1 keeps every non-null value, which is the pre-floor behaviour.
    """
    counts = Counter(v for v in values if v)
    kept = sorted(v for v, n in counts.items() if n >= min_count)
    return {value: i for i, value in enumerate(kept, start=1)}


def _vector_column(matrix: np.ndarray) -> pa.FixedSizeListArray:
    """A ``[n, dim]`` float32 matrix as one fixed-size-list column.

    Built from the flat buffer rather than from a list of lists: the latter
    allocates 65,238 Python objects per variant and is minutes slower for a
    column that is already contiguous.
    """
    flat = pa.array(matrix.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, matrix.shape[1])


def encode(
    texts: Sequence[str],
    model_name: str,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode texts to unit-norm float32 vectors.

    ``normalize_embeddings=True`` because the item tower's output is L2
    normalised anyway and bge is trained for cosine -- normalising at the source
    keeps the tower's first layer seeing inputs on one scale.

    Args:
        texts: One string per item, in item order.
        model_name: A sentence-transformers model id.
        device: From :func:`common.torch_env.select_device`. Passed explicitly
            rather than defaulted, or sentence-transformers quietly picks CPU
            and a two-minute job becomes twenty.
        batch_size: Encoder batch size.

    Returns:
        ``[len(texts), dim]`` float32.
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name, device=str(device))
    vectors = encoder.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype="float32")


def _write_map(path: Path, mapping: dict[str, int], name: str) -> None:
    table = pa.table(
        {
            name: pa.array(list(mapping), type=pa.string()),
            f"{name}_idx": pa.array(list(mapping.values()), type=pa.int32()),
        }
    )
    pq.write_table(table, path)


def build(
    settings: Settings,
    splits: Sequence[str],
    model_name: str = DEFAULT_ENCODER,
    batch_size: int = 256,
    min_count: int = MIN_INDEX_COUNT,
) -> dict[str, int | str]:
    """Build the cache and write it to ``gold/item_content``.

    Returns:
        A summary for the build log.
    """
    spark = get_spark(settings, app="content-cache")
    try:
        catalogue = read_news(spark, settings, splits).select(
            "item_id", "category", "subcategory", "title", "abstract"
        )
        item_map = spark.read.parquet(str(settings.paths.bronze / "item_map"))
        frame = catalogue.join(item_map, on="item_id", how="inner").toPandas()
    finally:
        spark.stop()

    frame = frame.sort_values("item_idx").reset_index(drop=True)

    titles = frame["title"].fillna("").str.strip().tolist()
    abstracts = frame["abstract"].fillna("").str.strip().tolist()
    # An article with no abstract falls back to its title rather than to a
    # trailing separator: ~5% of MIND ships none, and ". " on its own is a token
    # the encoder would have to make sense of.
    combined = [f"{t}. {a}" if a else t for t, a in zip(titles, abstracts, strict=True)]

    categories = _index_map(frame["category"], min_count)
    subcategories = _index_map(frame["subcategory"], min_count)
    cat_idx = [categories.get(c, 0) for c in frame["category"]]
    sub_idx = [subcategories.get(s, 0) for s in frame["subcategory"]]

    device = select_device()
    print(f"  encoding {len(frame):,} items with {model_name} on {describe(device)}")
    vec_title = encode(titles, model_name, device, batch_size)
    vec_combined = encode(combined, model_name, device, batch_size)

    table = pa.table(
        {
            "item_idx": pa.array(frame["item_idx"].to_numpy(), type=pa.int32()),
            "category_idx": pa.array(cat_idx, type=pa.int32()),
            "subcategory_idx": pa.array(sub_idx, type=pa.int32()),
            "vec_title": _vector_column(vec_title),
            "vec_title_abstract": _vector_column(vec_combined),
        }
    )

    dest = Path(settings.paths.gold) / "item_content"
    dest.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dest / "part-00000.parquet")
    _write_map(dest / "category_map.parquet", categories, "category")
    _write_map(dest / "subcategory_map.parquet", subcategories, "subcategory")

    # Spark drops _SUCCESS only once a write commits, and `_is_built` reads that
    # marker. pyarrow writes no marker, so it is written here AFTER everything
    # else -- an interrupted build then reports unbuilt rather than half-built.
    (dest / "_SUCCESS").touch()

    return {
        "items": len(frame),
        "dim": vec_title.shape[1],
        "categories": len(categories),
        "subcategories": len(subcategories),
        "encoder": model_name,
        "with_abstract": sum(1 for a in abstracts if a),
        "min_count": min_count,
        "cat_folded": sum(1 for i in cat_idx if i == 0),
        "subcat_folded": sum(1 for i in sub_idx if i == 0),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Cache sentence vectors for the catalogue.")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["train", "dev"])
    parser.add_argument(
        "--model",
        default=DEFAULT_ENCODER,
        help="A sentence-transformers id. Swapping it is the encoder ablation; "
        "rebuild with --force after changing it.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--min-count",
        type=int,
        default=MIN_INDEX_COUNT,
        help="Categories and subcategories with fewer items than this get no "
        "index and fall to 0. Pass 1 to keep every value.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild an existing cache.")
    args = parser.parse_args(argv)

    settings = load_settings()

    missing = [s for s in args.splits if not _is_built(settings, "bronze", s)]
    if missing:
        print(f"error: bronze not built for {missing}. Run `make bronze` first.")
        return 1

    if _is_built(settings, "gold", "item_content") and not args.force:
        print("item_content: already built, skipping (--force to re-encode)")
        return 0

    print(f"item_content: catalogue -> {settings.paths.gold / 'item_content'}")
    summary = build(settings, args.splits, args.model, args.batch_size, args.min_count)
    print(
        f"  {summary['items']:,} items at dim {summary['dim']}, "
        f"{summary['with_abstract']:,} with an abstract, "
        f"{summary['categories']} categories / {summary['subcategories']} subcategories"
    )
    print(
        f"  floor {summary['min_count']}: {summary['cat_folded']:,} items on category 0, "
        f"{summary['subcat_folded']:,} on subcategory 0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
