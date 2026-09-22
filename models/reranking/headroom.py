"""Is there anything for a re-ranking policy to recover? Answered before building one.

**The standing rule, applied before the measurement rather than after it:** check
that A and B *can* differ. A diversity policy reorders the ranker's candidates;
it cannot serve an item the retriever never proposed. So the coverage any policy
could possibly reach is bounded by the union of the candidate pools, and the
distance between that bound and what the ranker already serves is the entire
budget the rest of Part L is spending.

If that budget is small, the trade-off table is four near-identical rows and
reads as a finding. Part J built a sweep on an assumption that the planner's
decision could move, and had to test it first; this is the same shape.

Three ceilings, each a different denominator, and they are not interchangeable:

- **the catalogue** -- 65,238 items, the number a naive coverage metric divides
  by, and a fantasy: most of it is never clicked by anyone.
- **the clicked catalogue** -- distinct articles anyone clicks in the window.
  The honest denominator, established in ``docs/retrieval.md``.
- **the pool** -- distinct items the retriever actually proposed across every
  held-out request. **This is the re-ranker's ceiling**, and nothing below it
  is reachable by reordering.

Run: ``uv run python -m models.reranking.headroom <retrieval-checkpoint>``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from models.ranking.dataset import RankingRows
from models.ranking.pipeline import add_common_arguments, prepare

# The long-tail edge, DERIVED rather than chosen: `pop@100` is exactly 0.0000
# for every band below 26 training clicks, so 26 is where a popularity retriever
# stops existing. See docs/retrieval.md.
LONG_TAIL_EDGE = 26


def top_k_items(
    scores: npt.NDArray[np.float64], rows: RankingRows, k: int
) -> npt.NDArray[np.int64]:
    """The items each request would serve, at cutoff ``k``, concatenated.

    Ties are broken by ``argsort``'s stable order, which is candidate order --
    not by score, because there is nothing left to break them with. This
    project has already had a tie-break silently decide a metric once, so the
    rule is stated rather than inherited.
    """
    served: list[npt.NDArray[np.int64]] = []
    start = 0
    for size in rows.groups:
        end = start + int(size)
        order = np.argsort(-scores[start:end], kind="stable")[:k]
        served.append(rows.items[start:end][order])
        start = end
    return np.concatenate(served) if served else np.zeros(0, dtype=np.int64)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--top-k", type=int, default=10, help="Slots actually shown.")
    args = parser.parse_args(argv)

    prepared = prepare(args, "headroom")
    held = prepared.held

    pool = np.unique(held.items)
    clicked = np.unique(held.items[held.labels == 1])

    # What the ranker serves today, approximated by the order retrieval handed
    # over. No ranker is fitted here: this file answers whether the EXPERIMENT
    # is worth running, and fitting one to decide that would be the experiment.
    inherited = held.features[:, held.names.index("retrieval_score")].astype(float)
    served = top_k_items(inherited, held, args.top_k)
    distinct_served = len(np.unique(served))

    train_clicks = held.features[:, held.names.index("train_clicks")].astype(int)
    by_item = dict(zip(held.items.tolist(), train_clicks.tolist(), strict=True))

    def is_tail(item: int) -> bool:
        return bool(by_item.get(item, 0) < LONG_TAIL_EDGE)

    # DISTINCT items and IMPRESSIONS are different denominators and the answers
    # differ by two orders of magnitude. The first draft of this file printed a
    # distinct count for the pool beside an impression count for the served
    # list, under one heading -- 635 "items" in a top-10 that serves 206
    # distinct items in total. Both are reported, each labelled.
    tail_pool_distinct = sum(1 for item in pool.tolist() if is_tail(item))
    tail_served_distinct = sum(1 for item in np.unique(served).tolist() if is_tail(item))
    tail_served_impressions = sum(1 for item in served.tolist() if is_tail(item))

    print(f"\n  requests held out                     {len(held.groups):>8,}")
    print(f"  candidate rows                        {len(held.items):>8,}")
    print(f"  slots actually shown (k)              {args.top_k:>8,}")
    print(f"  impressions served                    {len(served):>8,}")
    print("\n  THREE CEILINGS, three denominators")
    print(f"    the catalogue                       {65_238:>8,}")
    print(f"    distinct items the POOL offers      {len(pool):>8,}   <- the re-ranker's ceiling")
    print(f"    distinct CLICKED items in the pool  {len(clicked):>8,}")
    print("\n  WHAT IS ALREADY SERVED, and what is left")
    print(f"    distinct items in the top {args.top_k:<2}          {distinct_served:>8,}")
    print(
        f"    share of the pool reached           {distinct_served / len(pool):>8.1%}"
        if len(pool)
        else ""
    )
    print(f"    HEADROOM a policy could recover     {len(pool) - distinct_served:>8,} items")
    print(f"\n  long tail (<{LONG_TAIL_EDGE} training clicks) -- TWO denominators")
    print(f"    distinct tail items in the pool     {tail_pool_distinct:>8,}")
    print(f"    distinct tail items served          {tail_served_distinct:>8,}")
    print(
        f"    tail SHARE of impressions served    {tail_served_impressions / len(served):>8.1%}"
        f"   ({tail_served_impressions:,} of {len(served):,})"
        if len(served)
        else ""
    )

    reachable = len(pool) - distinct_served
    print(
        "\n  Read this before believing any diversity gain below: a policy that\n"
        "  reorders candidates cannot serve what the retriever never proposed,\n"
        f"  so {len(pool):,} is the hard ceiling and the catalogue's 65,238 is not\n"
        "  a denominator any re-ranker could move."
    )
    if reachable < distinct_served * 0.1:
        print(
            "\n  ⚠️  HEADROOM IS UNDER 10% OF WHAT IS ALREADY SERVED. A diversity\n"
            "     table built on this would print near-identical rows and read as\n"
            "     a finding. Say so rather than building it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
