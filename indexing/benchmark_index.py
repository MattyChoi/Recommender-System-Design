"""The recall/QPS curve, and the two different things called recall.

Metrics reported:

* ``agreement`` -- overlap with exact search's top-k. A property of the index.
* ``recall``    -- share of clicked articles found. The end metric, and the only
  one that has ever appeared in a results table here.

The gap between them is the finding. An index at 0.95 agreement and unchanged
recall has lost nothing that mattered; at 0.95 agreement and a real recall drop
it is discarding answers, and only the second is a reason to spend memory.

Latency is per request at **batch size 1**, because a search is one request.
A batched sweep divided by the batch size is throughput wearing a latency label.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from common.config import load_settings
from common.spark import get_spark
from common.torch_env import describe, select_device
from data_pipeline.features.user_history import MAX_HISTORY
from evaluation.offline.stats import PairedResult, paired_bootstrap, per_user_means
from indexing.build_index import (
    build_flat,
    build_hnsw,
    build_ivfpq,
    index_bytes,
    item_vectors,
    recommended_nlist,
    search,
)
from models.retrieval.dataloader.dataset import (
    CONTENT_VARIANTS,
    load_item_tables,
    load_train_and_validation,
)
from models.retrieval.evaluate import load_tower
from models.retrieval.sources import encode_users

EF_SEARCH = (16, 32, 64, 128, 256, 512)
NPROBE = (1, 4, 16, 32, 64, 128)
LATENCY_QUERIES = 512
WARMUP = 50


@dataclass(frozen=True)
class IndexRow:
    """One index at one setting.

    Attributes:
        delta: Click-recall against exact search, paired per user. ``None`` for
            exact itself. An approximation that differs from exact search by a
            few thousandths is not obviously worse OR better, and the interval
            is the only thing that distinguishes the two readings.
    """

    name: str
    param: str
    agreement: float
    recall: float
    p50_ms: float
    p99_ms: float
    qps: float
    megabytes: float
    delta: PairedResult | None = None


def agreement(candidate: npt.NDArray[np.int64], exact: npt.NDArray[np.int64]) -> float:
    """Mean overlap with exact search, per request.

    Padding (index 0) is excluded from both sides: an index that returned
    nothing would otherwise score agreement for matching the absence.
    """
    scores = []
    for row in range(len(exact)):
        truth = set(int(value) for value in exact[row] if value)
        got = set(int(value) for value in candidate[row] if value)
        scores.append(len(truth & got) / len(truth) if truth else float("nan"))
    return float(np.nanmean(scores))


def click_recall(candidate: npt.NDArray[np.int64], clicked: npt.NDArray[np.int64]) -> float:
    """Share of requests whose clicked article is in the returned list."""
    return float((candidate == clicked[:, None]).any(axis=1).mean())


def latency(index: Any, queries: npt.NDArray[np.float32], k: int) -> tuple[float, float]:
    """p50 and p99 milliseconds for a single-query search.

    Warm-up is discarded: the first searches on a graph index touch cold pages
    and pay one-off allocation, and including them moves the tail far more than
    any parameter under test.
    """
    one = np.ascontiguousarray(queries[:1], dtype="float32")
    for _ in range(WARMUP):
        index.search(one, k)

    samples = []
    for row in range(len(queries)):
        query = np.ascontiguousarray(queries[row : row + 1], dtype="float32")
        start = time.perf_counter_ns()
        index.search(query, k)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    ordered = sorted(samples)
    return ordered[len(ordered) // 2], ordered[int(0.99 * len(ordered)) - 1]


def throughput(index: Any, queries: npt.NDArray[np.float32], k: int) -> float:
    """Queries per second with the whole batch handed over at once."""
    batch = np.ascontiguousarray(queries, dtype="float32")
    index.search(batch[:WARMUP], k)
    start = time.perf_counter_ns()
    index.search(batch, k)
    return len(batch) / ((time.perf_counter_ns() - start) / 1e9)


def _hits(candidate: npt.NDArray[np.int64], clicked: npt.NDArray[np.int64]) -> list[float]:
    found: list[float] = (candidate == clicked[:, None]).any(axis=1).astype(float).tolist()
    return found


def measure(
    name: str,
    param: str,
    index: Any,
    queries: npt.NDArray[np.float32],
    exact: npt.NDArray[np.int64],
    clicked: npt.NDArray[np.int64],
    k: int,
    latency_queries: int,
    users: Sequence[str] | None = None,
) -> IndexRow:
    """One row of the table, with its difference from exact search bracketed."""
    got = search(index, queries, k)
    p50, p99 = latency(index, queries[:latency_queries], k)

    delta = None
    if users is not None:
        delta = paired_bootstrap(
            per_user_means(_hits(exact, clicked), users),
            per_user_means(_hits(got, clicked), users),
        )

    return IndexRow(
        name=name,
        param=param,
        agreement=agreement(got, exact),
        recall=click_recall(got, clicked),
        p50_ms=p50,
        p99_ms=p99,
        qps=throughput(index, queries, k),
        megabytes=index_bytes(index) / 1e6,
        delta=delta,
    )


def render(rows: Sequence[IndexRow], k: int) -> str:
    head = (
        f"{'index':>7}  {'param':>14}  {'agreement':>9}  {f'recall@{k}':>11}  "
        f"{'vs exact':>24}  {'p50 ms':>7}  {'p99 ms':>7}  {'QPS':>9}  {'MB':>7}"
    )
    lines = [head, "-" * len(head)]
    for row in rows:
        delta = row.delta
        if delta is None:
            cell = "--"
        else:
            mark = "*" if delta.significant else " "
            cell = f"{delta.difference:+.4f} [{delta.lo:+.4f},{delta.hi:+.4f}]{mark}"
        lines.append(
            f"{row.name:>7}  {row.param:>14}  {row.agreement:>9.4f}  {row.recall:>11.4f}  "
            f"{cell:>24}  {row.p50_ms:>7.3f}  {row.p99_ms:>7.3f}  "
            f"{row.qps:>9,.0f}  {row.megabytes:>7.1f}"
        )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--variant", choices=CONTENT_VARIANTS, default=CONTENT_VARIANTS[0])
    parser.add_argument("--max-history", type=int, default=MAX_HISTORY)
    parser.add_argument("--max-negs", type=int, default=4)
    parser.add_argument("--holdout-hours", type=int, default=12)
    parser.add_argument("--latency-queries", type=int, default=LATENCY_QUERIES)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--pq-m", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = load_settings()
    device = select_device()
    print(f"index benchmark: {describe(device)}")

    spark = get_spark(settings, app="index-benchmark")
    try:
        items = load_item_tables(settings, args.variant)
        train_split, val_split = load_train_and_validation(
            spark, settings, args.max_history, args.max_negs, args.holdout_hours
        )
    finally:
        spark.stop()

    tower = load_tower(args.checkpoint, items, train_split.user_feats.shape[1], device)
    vectors = item_vectors(tower)
    queries = encode_users(tower, val_split, device).numpy().astype("float32")
    clicked = val_split.item_ids.numpy()
    nlist = recommended_nlist(len(vectors))
    print(
        f"  {len(vectors):,} items x {vectors.shape[1]}d · {len(queries):,} queries · nlist={nlist}"
    )

    flat = build_flat(vectors)
    exact = search(flat, queries, args.k)
    users = [str(value) for value in val_split.user_ids.numpy()]

    rows = [measure("flat", "exact", flat, queries, exact, clicked, args.k, args.latency_queries)]

    hnsw = build_hnsw(vectors, m=args.hnsw_m)
    for ef in EF_SEARCH:
        hnsw.hnsw.efSearch = ef
        rows.append(
            measure(
                "hnsw",
                f"efSearch={ef}",
                hnsw,
                queries,
                exact,
                clicked,
                args.k,
                args.latency_queries,
                users,
            )
        )

    # `nprobe = nlist` searches every cell, so what is left is quantisation and
    # nothing else. Without it, a difference from exact search cannot be told
    # apart from having looked at only part of the catalogue.
    ivfpq = build_ivfpq(vectors, nlist=nlist, m=args.pq_m)
    for nprobe in (*NPROBE, nlist):
        ivfpq.nprobe = nprobe
        rows.append(
            measure(
                "ivfpq",
                f"nprobe={nprobe}",
                ivfpq,
                queries,
                exact,
                clicked,
                args.k,
                args.latency_queries,
                users,
            )
        )

    print()
    print(render(rows, args.k))
    print(
        "\n  `agreement` is overlap with exact search; `recall` is share of clicked\n"
        "  articles found. They are different questions and the gap between them is\n"
        "  what says whether the approximation costs anything.\n"
        f"  Latency is batch 1 over {args.latency_queries:,} queries; QPS is the whole\n"
        "  batch at once. The two cannot be derived from each other.\n"
        "  `vs exact` is the per-user paired difference in click-recall against\n"
        "  exact search, 95% interval; * excludes zero. The model is held fixed,\n"
        "  so this interval is over requests only -- not over training."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
