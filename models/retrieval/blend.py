"""Union the sources, attribute every candidate, and drop what earns nothing.

* ``union``  -- every source's full top-k. The CEILING of blending: what a
  perfect ranker downstream could reach if latency and slot count were free.
  Its pool is four times the two-tower's, so it must never be quoted beside a
  single source's Recall@100 as though the two were comparable.
* ``quota``  -- ``k // n_sources`` slots each, so the blend returns about k
  candidates in total, the same budget the two-tower alone gets.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from evaluation.offline.stats import PairedResult, paired_bootstrap, per_user_means
from models.retrieval.sources import RESULTS, SOURCE_NAMES, Retrieved, load_retrieved


@dataclass(frozen=True)
class Contribution:
    """One source's line in I3's table.

    Attributes:
        source: Which source.
        share: Fraction of the union's (request, item) pairs this source offers.
        unique: Fraction of the union's pairs **only** this source offers.
        alone: Recall@k with nothing else in the blend.
        found_only_here: Clicked items this source found and no other did, as a
            share of all requests. The number that decides whether a source
            earns its slot -- a source can contribute 30% of the candidates and
            none of the answers.
        loss: Recall lost by dropping it from the quota blend, paired per user.
    """

    source: str
    share: float
    unique: float
    alone: float
    found_only_here: float
    loss: PairedResult | None


def check_aligned(sources: Sequence[Retrieved]) -> None:
    """Every source must have answered the same requests, in the same order.

    Raises:
        ValueError: On any disagreement. Two sources built from different runs
            would still union row-for-row and still produce a table; the table
            would be about nothing. Same refusal ``ablation.check_aligned``
            makes, for the same reason.
    """
    first = sources[0]
    for other in sources[1:]:
        if len(other.item_ids) != len(first.item_ids):
            raise ValueError(
                f"{other.source} has {len(other.item_ids)} requests, "
                f"{first.source} has {len(first.item_ids)}"
            )
        if not np.array_equal(other.item_ids.numpy(), first.item_ids.numpy()):
            raise ValueError(f"{other.source} and {first.source} scored different clicked items")
        if not np.array_equal(other.user_ids.numpy(), first.user_ids.numpy()):
            raise ValueError(f"{other.source} and {first.source} scored different users")


def _quotas(sources: Sequence[Retrieved], quota: int | Sequence[int] | None) -> list[int]:
    """Slots per source: full lists, one number for all, or one number each.

    Unequal quotas are the point rather than a convenience. Equal slots assume
    the sources are equally good, and on this corpus they are not within an
    order of magnitude -- so an equal split measures the assumption, not the
    blend.
    """
    if quota is None:
        return [int(source.top.shape[1]) for source in sources]
    if isinstance(quota, int):
        return [quota] * len(sources)
    if len(quota) != len(sources):
        raise ValueError(f"{len(quota)} quotas for {len(sources)} sources")
    return [int(value) for value in quota]


def hit_matrix(
    sources: Sequence[Retrieved], quota: int | Sequence[int] | None = None
) -> npt.NDArray[np.bool_]:
    """``[n_sources, R]`` -- did each source find the click within its budget."""
    rows = []
    for source, slots in zip(sources, _quotas(sources, quota), strict=True):
        top = source.top[:, :slots]
        rows.append((top == source.item_ids.unsqueeze(1)).any(dim=1).numpy())
    return np.stack(rows)


def _pair_keys(
    sources: Sequence[Retrieved], quota: int | Sequence[int] | None
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Every (request, item) the blend offers, and which source offered it."""
    stride = int(max(int(source.top.max()) for source in sources)) + 1
    request = np.arange(len(sources[0].item_ids))

    keys: list[npt.NDArray[np.int64]] = []
    owner: list[npt.NDArray[np.int64]] = []
    for index, (source, slots) in enumerate(zip(sources, _quotas(sources, quota), strict=True)):
        top = source.top[:, :slots].numpy()
        flat = top.reshape(-1)
        rows = np.repeat(request, top.shape[1])
        keep = flat > 0
        keys.append(rows[keep].astype(np.int64) * stride + flat[keep].astype(np.int64))
        owner.append(np.full(int(keep.sum()), index, dtype=np.int64))
    return np.concatenate(keys), np.concatenate(owner)


def _distinct_per_request(
    sources: Sequence[Retrieved], quota: int | Sequence[int] | None
) -> npt.NDArray[np.int64]:
    """``[R]`` distinct items the blend puts forward per request."""
    keys, _ = _pair_keys(sources, quota)
    stride = int(max(int(source.top.max()) for source in sources)) + 1
    unique = np.unique(keys)
    return np.bincount(unique // stride, minlength=len(sources[0].item_ids)).astype(np.int64)


def attribute(
    sources: Sequence[Retrieved], quota: int | Sequence[int] | None = None
) -> dict[str, tuple[float, float]]:
    """Per source: share of the union's pairs, and share it alone offers."""
    keys, owner = _pair_keys(sources, quota)
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    total = len(unique)

    only = counts[inverse] == 1
    out: dict[str, tuple[float, float]] = {}
    for index, source in enumerate(sources):
        mine = owner == index
        held = len(np.unique(inverse[mine]))
        out[source.source] = (held / total, int((mine & only).sum()) / total)
    return out


def leave_one_out(
    sources: Sequence[Retrieved],
    user_ids: npt.NDArray[np.int64],
    k: int,
    mask: npt.NDArray[np.bool_] | None = None,
) -> dict[str, PairedResult]:
    """Recall lost by dropping each source, with the freed slots redistributed.

    The full blend gives every source ``k // n`` slots. Dropping one leaves
    ``n - 1`` sources sharing the same k, so each gets ``k // (n - 1)``. Holding
    the survivors at their old quota instead would charge the removed source for
    a budget cut it did not cause.
    """
    # One source cannot be left out of itself: there is no blend to compare the
    # remainder against, and k // 0 is the crash that says so.
    if len(sources) < 2:
        return {}

    names = [source.source for source in sources]
    rows = np.ones(len(user_ids), dtype=bool) if mask is None else mask
    users = [str(value) for value in user_ids[rows]]

    full = hit_matrix(sources, k // len(sources)).any(axis=0)[rows]
    baseline = per_user_means(full.astype(float).tolist(), users)

    out: dict[str, PairedResult] = {}
    for index, name in enumerate(names):
        rest = [source for position, source in enumerate(sources) if position != index]
        without = hit_matrix(rest, k // len(rest)).any(axis=0)[rows]
        # candidate - baseline, so a source that helps produces a NEGATIVE loss
        without_user = per_user_means(without.astype(float).tolist(), users)
        out[name] = paired_bootstrap(baseline, without_user)
    return out


def contributions(
    sources: Sequence[Retrieved],
    user_ids: npt.NDArray[np.int64],
    k: int,
    mask: npt.NDArray[np.bool_] | None = None,
) -> list[Contribution]:
    """I3's table, over the requests ``mask`` selects."""
    rows = np.ones(len(user_ids), dtype=bool) if mask is None else mask
    shares = attribute(sources)
    alone = hit_matrix(sources)
    losses = leave_one_out(sources, user_ids, k, mask)

    out = []
    for index, source in enumerate(sources):
        others = np.delete(alone, index, axis=0).any(axis=0)
        only = alone[index] & ~others
        share, unique = shares[source.source]
        out.append(
            Contribution(
                source=source.source,
                share=share,
                unique=unique,
                alone=float(alone[index][rows].mean()),
                found_only_here=float(only[rows].mean()),
                loss=losses.get(source.source),
            )
        )
    return out


@dataclass(frozen=True)
class Blend:
    """One allocation of the slot budget, and what it recalls.

    Attributes:
        label: How the slots were divided, spelled out -- the allocation IS the
            condition, so it travels with the number.
        recall: Recall@k of the union under that allocation.
        pool: Mean distinct candidates put forward per request.
    """

    label: str
    recall: float
    pool: float


def render(rows: Sequence[Contribution], k: int, blends: Sequence[Blend]) -> str:
    """I3's table."""
    head = (
        f"{'source':>10}  {'% of pool':>9}  {'% unique':>9}  {'alone':>8}  "
        f"{'only here':>9}  {'loss if dropped':>17}"
    )
    lines = [head, "-" * len(head)]
    for row in rows:
        loss = row.loss
        mark = "*" if loss is not None and loss.significant else " "
        cell = "--" if loss is None else f"{-loss.difference:+.4f} {mark}"
        lines.append(
            f"{row.source:>10}  {row.share:>9.3f}  {row.unique:>9.3f}  {row.alone:>8.4f}  "
            f"{row.found_only_here:>9.4f}  {cell:>17}"
        )
    lines.append("-" * len(head))
    for blend in blends:
        lines.append(f"  {blend.label:<44} pool {blend.pool:>5.0f}   recall {blend.recall:.4f}")
    lines += [
        "",
        "  `loss if dropped` is recall lost from the even-quota blend with the",
        "  freed slots redistributed; * marks an interval excluding zero, paired",
        "  per user. An allocation is a CONDITION, not a detail: a union over",
        "  full lists is a bigger pool than any single source and is not",
        f"  comparable to a Recall@{k}.",
    ]
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=RESULTS)
    parser.add_argument("--names", nargs="+", default=list(SOURCE_NAMES))
    parser.add_argument("--k", type=int, default=100, help="Total slot budget for the blend.")
    parser.add_argument(
        "--quota",
        type=int,
        nargs="+",
        default=None,
        help="Slots per source, in --names order. Default splits k evenly, which "
        "assumes the sources are equally good.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sources = [load_retrieved(name, args.sources) for name in args.names]
    check_aligned(sources)

    user_ids = sources[0].user_ids.numpy()

    def blend(label: str, quota: int | Sequence[int] | None) -> Blend:
        return Blend(
            label=label,
            recall=float(hit_matrix(sources, quota).any(axis=0).mean()),
            pool=float(_distinct_per_request(sources, quota).mean()),
        )

    even = args.k // len(sources)
    blends = [
        blend(f"union: every source's full top-{args.k}", None),
        blend(f"quota: {even} slots each", even),
    ]
    if args.quota:
        spelled = ", ".join(
            f"{name}={slots}" for name, slots in zip(args.names, args.quota, strict=True)
        )
        blends.append(blend(f"quota: {spelled}", args.quota))

    print(render(contributions(sources, user_ids, args.k), args.k, blends))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
