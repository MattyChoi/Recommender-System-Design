"""The trade-off table: what each policy costs in relevance and buys in diversity.

**One ranker, loaded from disk, shared by every arm.** Refitting per arm would
let model variation leak into a table that claims to measure policy, and this
project has already measured the size of that leak -- two runs of one neural
config differ by up to 0.0010 NDCG, which is a quarter of the largest effect
Part K found. A booster read from a file cannot drift between rows.

Four columns, because a diversity claim needs all four to mean anything:

- **NDCG@k** -- the relevance given up. On this corpus each request has ONE
  relevant item, so this column can only fall, and the table is a price list.
- **intra-list diversity** -- 1 minus mean pairwise cosine within a slate, over
  CONTENT vectors. Five write-ups of one breaking story are near-duplicates
  that ID embeddings cannot see, because all five are new.
- **catalogue coverage** -- distinct items served, against **the pool**, not
  the catalogue. A re-ranker cannot serve what retrieval never proposed, so
  65,238 is not a denominator it can move; ``models.reranking.headroom``
  measures the one it can.
- **long-tail share** -- share of served impressions on items with fewer than
  26 training clicks, the edge derived in ``docs/retrieval.md``.

Run: ``uv run python -m models.reranking.evaluate <retrieval-checkpoint>
--ranker data/checkpoints/ranking/<booster>.txt``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from common.spark import get_spark
from evaluation.offline.metrics import intra_list_diversity, ndcg_at_k
from evaluation.offline.stats import paired_bootstrap, per_user_means
from models.ranking.dataset import RankingRows
from models.ranking.pipeline import add_common_arguments, prepare
from models.reranking.headroom import LONG_TAIL_EDGE
from models.reranking.policies import (
    Slate,
    freshness_multiplier,
    normalise,
    select,
    slate_per_request,
)
from models.retrieval.dataloader.dataset import first_seen_hours, load_item_tables


@dataclass(frozen=True)
class Arm:
    """One configuration of the policy layer, and what it scored."""

    name: str
    ndcg: float
    diversity: float
    coverage: float
    distinct_items: int
    tail_share: float
    mean_propensity: float
    per_request: npt.NDArray[np.float64]


def score_slates(
    slates: list[Slate],
    rows: RankingRows,
    vectors: npt.NDArray[np.float32],
    train_clicks: npt.NDArray[np.int64],
    pool_size: int,
    k: int,
    name: str,
) -> Arm:
    """Every column of one row of the table, from one arm's slates."""
    ndcgs: list[float] = []
    diversities: list[float] = []
    served: list[int] = []
    propensities: list[float] = []

    for slate in slates:
        labels = rows.labels[slate.rows].tolist()
        # Descending position scores: the slate IS the order, so the metric is
        # told so explicitly rather than being handed the model's scores again,
        # which would re-sort and silently undo the policy.
        positions = list(range(len(labels), 0, -1))
        ndcgs.append(ndcg_at_k(labels, positions, k) if any(labels) else 0.0)

        if len(slate.items) > 1:
            keys = [str(item) for item in slate.items.tolist()]
            lookup = {
                key: vectors[item] for key, item in zip(keys, slate.items.tolist(), strict=True)
            }
            diversities.append(intra_list_diversity(keys, lookup))
        served.extend(slate.items.tolist())
        propensities.extend(slate.propensity.tolist())

    distinct = len(set(served))
    tail = sum(1 for item in served if train_clicks[item] < LONG_TAIL_EDGE)

    return Arm(
        name=name,
        ndcg=float(np.mean(ndcgs)) if ndcgs else float("nan"),
        diversity=float(np.nanmean(diversities)) if diversities else float("nan"),
        coverage=distinct / pool_size if pool_size else float("nan"),
        distinct_items=distinct,
        tail_share=tail / len(served) if served else float("nan"),
        mean_propensity=float(np.mean(propensities)) if propensities else float("nan"),
        per_request=np.asarray(ndcgs, dtype=float),
    )


def seen_filtered_slates(
    scores: npt.NDArray[np.float64],
    rows: RankingRows,
    k: int,
    url: str,
    capacity: int,
    rate: float,
    ttl: int,
) -> tuple[list[Slate], dict[str, float]]:
    """Replay the holdout IN ORDER, blocking what each user was already shown.

    **Sequential, not batched, and that is forced by the semantics.** A
    seen-list is a function of what this user was served EARLIER, so a request
    cannot be scored without first processing every earlier request of the same
    user. Every other arm in this table is order-independent; this one is not,
    and batching it would silently evaluate a filter that knew the future.

    The filter is populated with what was SHOWN, not what was clicked -- that is
    what the policy is for. Each user's key is flushed first so a rerun does not
    inherit the previous run's bits, which would make the arm's cost grow every
    time it was measured.
    """
    from models.reranking.seen_redis import RedisSeenList, connect

    client = connect(url)
    filter_ = RedisSeenList.sized_for(
        client, capacity=capacity, false_positive_rate=rate, ttl_seconds=ttl
    )

    for user in np.unique(rows.user_ids).tolist():
        client.delete(filter_.key(int(user)))

    slates: list[Slate] = []
    blocked_total = 0
    start = 0
    for request, size in enumerate(rows.groups.tolist()):
        end = start + size
        block = slice(start, end)
        user = int(rows.user_ids[request])
        items = rows.items[block]

        mask = filter_.seen_mask(user, items)
        blocked_total += int(mask.sum())

        local = select(scores[block], items, k, blocked=mask)
        slates.append(
            Slate(items=local.items, rows=local.rows + start, propensity=local.propensity)
        )
        filter_.add(user, local.items)
        start = end

    sample_user = int(rows.user_ids[0])
    stats = {
        "blocked_candidates": float(blocked_total),
        "blocked_share": blocked_total / len(rows.items),
        "load": filter_.load(sample_user),
        "bytes_redis": float(filter_.memory_bytes(sample_user)),
        "bits": float(filter_.bits),
    }
    return slates, stats


def render(arms: Sequence[Arm], baseline: Arm, users: list[str], pool_size: int) -> str:
    """The table, with each arm paired against the ranker-only row."""
    # 28 characters, because "+0.0002 [-0.0002,+0.0006]*" is 26 and a field too
    # narrow does not truncate in an f-string -- it overflows and shunts every
    # column after it, which is how a table stops lining up exactly when the
    # numbers get interesting.
    lines = [
        f"\n  {'config':<26}{'NDCG@10':>9}  {'vs base':<28}{'ILD':>7}"
        f"{'coverage':>10}{'items':>8}{'tail':>8}{'E[p]':>7}",
        "  " + "-" * 105,
    ]
    for arm in arms:
        if arm.name == baseline.name:
            delta = "--"
        else:
            result = paired_bootstrap(
                per_user_means(baseline.per_request.tolist(), users),
                per_user_means(arm.per_request.tolist(), users),
            )
            star = "*" if result.significant else ""
            delta = f"{result.difference:+.4f} [{result.lo:+.4f},{result.hi:+.4f}]{star}"
        lines.append(
            f"  {arm.name:<26}{arm.ndcg:>9.4f}  {delta:<28}{arm.diversity:>7.3f}"
            f"{arm.coverage:>10.1%}{arm.distinct_items:>8,}{arm.tail_share:>8.1%}"
            f"{arm.mean_propensity:>7.2f}"
        )
    lines += [
        "  " + "-" * 105,
        f"  coverage is against the POOL ({pool_size:,} distinct candidates), not the",
        "  catalogue -- a re-ranker cannot serve what retrieval never proposed.",
        "  E[p] is the mean logged propensity; 1.00 means the arm is deterministic",
        "  and carries no information for off-policy evaluation.",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--ranker", type=Path, required=True, help="A saved LightGBM booster.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.9, 0.7, 0.5])
    parser.add_argument("--cap", type=int, default=3, help="Max slots per category.")
    parser.add_argument("--half-life", type=float, default=24.0, help="Freshness, in hours.")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--explore-slots", type=int, default=2)
    parser.add_argument(
        "--seen",
        action="store_true",
        help="Add the seen-filter arm. Needs Redis (`make up`); the arm replays "
        "the holdout sequentially, so it is slower than the rest of the table.",
    )
    parser.add_argument("--seen-url", default="redis://localhost:6379/1")
    parser.add_argument("--seen-capacity", type=int, default=200)
    parser.add_argument("--seen-rate", type=float, default=0.01)
    parser.add_argument("--seen-ttl", type=int, default=7 * 24 * 3600)
    args = parser.parse_args(argv)

    prepared = prepare(args, "reranking")
    held = prepared.held

    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(args.ranker))
    scores = np.asarray(booster.predict(held.features), dtype=float)
    print(f"  ranker: {args.ranker.name}")

    items_table = load_item_tables(prepared.settings, args.variant)
    vectors = normalise(items_table.content.numpy())
    categories = items_table.subcategory.numpy()[held.items]

    spark = get_spark(prepared.settings, app="reranking")
    try:
        ages = first_seen_hours(
            spark, prepared.settings, args.holdout_hours, int(items_table.content.shape[0])
        ).numpy()
    finally:
        spark.stop()

    train_clicks = np.zeros(items_table.content.shape[0], dtype=np.int64)
    column = held.features[:, held.names.index("train_clicks")].astype(np.int64)
    train_clicks[held.items] = column

    pool_size = len(np.unique(held.items))
    users = [str(value) for value in held.user_ids]
    groups = held.groups.tolist()

    def arm(
        name: str,
        *,
        relevance: npt.NDArray[np.float64] | None = None,
        vectors_for_mmr: npt.NDArray[np.float32] | None = None,
        lambda_: float = 1.0,
        categories_for_caps: npt.NDArray[np.int64] | None = None,
        cap: int | None = None,
        epsilon: float = 0.0,
        explore_slots: int = 0,
        rng: np.random.Generator | None = None,
    ) -> Arm:
        """One row of the table. Arguments spelled out for the same reason
        `slate_per_request` spells its own out: a helper that forwards opaque
        keywords cannot be type-checked, and this one chooses between arrays
        whose lengths must agree."""
        slates = slate_per_request(
            scores if relevance is None else relevance,
            held.items,
            groups,
            args.top_k,
            vectors=vectors_for_mmr,
            lambda_=lambda_,
            categories=categories_for_caps,
            cap=cap,
            epsilon=epsilon,
            explore_slots=explore_slots,
            rng=rng,
        )
        return score_slates(slates, held, vectors, train_clicks, pool_size, args.top_k, name)

    arms = [arm("ranker only")]
    baseline = arms[0]

    candidate_vectors = vectors[held.items]
    for value in args.lambdas:
        arms.append(
            arm(
                f"+ MMR (lambda={value:g})",
                vectors_for_mmr=candidate_vectors,
                lambda_=value,
            )
        )

    arms.append(
        arm(
            f"+ MMR + caps (<={args.cap}/cat)",
            vectors_for_mmr=candidate_vectors,
            lambda_=args.lambdas[0],
            categories_for_caps=categories,
            cap=args.cap,
        )
    )

    # Freshness multiplies a POSITIVE score. The booster's raw output is not
    # positive, so it is squashed first; the ordering is unchanged by a monotone
    # map, which is what makes this legitimate.
    positive = 1.0 / (1.0 + np.exp(-scores))
    boosted = positive * freshness_multiplier(ages[held.items], args.half_life)
    arms.append(arm(f"+ freshness (hl={args.half_life:g}h)", relevance=boosted))

    arms.append(
        arm(
            f"+ exploration (eps={args.epsilon:g})",
            epsilon=args.epsilon,
            explore_slots=args.explore_slots,
            rng=np.random.default_rng(args.seed),
        )
    )

    if args.seen:
        slates, stats = seen_filtered_slates(
            scores,
            held,
            args.top_k,
            args.seen_url,
            args.seen_capacity,
            args.seen_rate,
            args.seen_ttl,
        )
        arms.append(
            score_slates(
                slates, held, vectors, train_clicks, pool_size, args.top_k, "+ seen filter"
            )
        )
        print(
            f"\n  seen filter: blocked {int(stats['blocked_candidates']):,} candidates "
            f"({stats['blocked_share']:.2%} of all rows)"
        )
        print(
            f"    {int(stats['bits']):,} bits/user = {stats['bits'] / 8:.0f} B of payload, "
            f"but {stats['bytes_redis']:.0f} B in Redis -- "
            f"{stats['bytes_redis'] / (stats['bits'] / 8):.1f}x, all key and object overhead"
        )
        print(f"    bit load on a sample user: {stats['load']:.1%}")

    print(render(arms, baseline, users, pool_size))
    print(
        "\n  Every row below the first is a COST in NDCG, by construction: each\n"
        "  request here has exactly one relevant item, so no reordering can find\n"
        "  a second. The table is an exchange rate, and whether the rate is worth\n"
        "  paying is a product decision that no offline number settles."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
