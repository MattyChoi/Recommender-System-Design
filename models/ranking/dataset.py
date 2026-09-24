"""Ranking rows built from what retrieval actually returns.

A ranker trained on random negatives and served retrieved candidates is the
most common way this stage goes wrong, and it is invisible offline: every
metric looks fine because the offline negatives are easy, and the model
degrades the moment it sees the candidates the retriever really produces. So
the training rows here are **replayed through retrieval** -- one row per
(request, candidate) that the sources actually put forward.

**Two things follow from that, and both are denominators.**

*The ranker's ceiling is the retriever's recall.* A request whose clicked
article was never retrieved cannot be fixed by any ordering. Those groups carry
no positive, so they teach a pairwise objective nothing and are dropped from
TRAINING -- but they are kept in EVALUATION, where they must count as zero.
Dropping them from both would report the ranker's skill on the subset retrieval
already solved.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from models.classes.dataset import ItemTables, SplitTensors
from models.retrieval.sources import Retrieved, encode_users
from models.retrieval.two_tower import TwoTower

# Columns, in the order the matrix carries them. Named because LightGBM reports
# importances positionally and an unnamed matrix makes that report unreadable.
FEATURES = (
    "retrieval_score",
    "two_tower_rank",
    "trending_rank",
    "n_sources",
    "prior_clicks",
    "train_clicks",
    "content_similarity",
    "history_length",
    "is_cold_item",
    "category_idx",
    "subcategory_idx",
)
CATEGORICAL = ("category_idx", "subcategory_idx")

# Rank assigned to a candidate a source did not propose. One past the cutoff, so
# "absent" is ordered worse than every rank a source could give -- and is a
# NUMBER rather than a NaN, because a tree splits on it either way but only the
# number is readable in a partial-dependence plot.
ABSENT = 10_000

# Buckets the user hash lands in. 1000 makes a 30% holdout exactly 300 of them.
_BUCKETS = 1000


@dataclass(frozen=True)
class RankingRows:
    """One flat table of (request, candidate) pairs.

    Attributes:
        names: Column names, in matrix order. Carried with the matrix rather
            than imported from :data:`FEATURES`, so a run that dropped a column
            cannot have its importances read against the full list.
        features: ``[N, len(FEATURES)]``.
        labels: ``[N]``, 1 where the candidate is the clicked article.
        groups: ``[R]`` candidates per request, in order. LightGBM's ranking
            objective needs candidates-per-query, **not** a per-row query id,
            and passing rows is the standard way to get a silently wrong model.
        items: ``[N]`` catalogue index of the candidate on each row. Carried
            because the ranker's own metrics never need it and every
            beyond-accuracy metric does: coverage, intra-list diversity and
            long-tail share are properties of WHICH items were served, not of
            how well they were ordered. A re-ranking stage cannot be evaluated
            without it.
        request: ``[N]`` index of the request each row belongs to.
        user_ids: ``[R]`` who made each request, so a comparison pairs on users.
        observed: ``[N]`` True where the candidate was in the user's real slate,
            so the zero is a decline rather than an absence of evidence.
        found: ``[R]`` whether retrieval surfaced the clicked article at all.
    """

    names: tuple[str, ...]
    features: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int64]
    groups: npt.NDArray[np.int64]
    items: npt.NDArray[np.int64]
    request: npt.NDArray[np.int64]
    user_ids: npt.NDArray[np.int64]
    observed: npt.NDArray[np.bool_]
    found: npt.NDArray[np.bool_]

    @property
    def recall_ceiling(self) -> float:
        """The best NDCG any ranker could reach: retrieval's own recall."""
        return float(self.found.mean())

    @property
    def observed_share(self) -> float:
        """Share of negatives the user actually had the chance to decline."""
        negatives = self.labels == 0
        return float(self.observed[negatives].mean()) if negatives.any() else float("nan")

    def select(self, requests: npt.NDArray[np.bool_]) -> RankingRows:
        """The subset of requests ``requests`` marks, with groups kept whole.

        Row masks and request masks are different things and mixing them is how
        a group boundary ends up describing a different set of rows than the
        rows it bounds. Everything here is derived from the request mask.
        """
        keep = np.repeat(requests, self.groups)
        return RankingRows(
            names=self.names,
            features=self.features[keep],
            labels=self.labels[keep],
            groups=self.groups[requests],
            items=self.items[keep],
            request=self.request[keep],
            user_ids=self.user_ids[requests],
            observed=self.observed[keep],
            found=self.found[requests],
        )

    def with_positives(self) -> RankingRows:
        """Only the requests retrieval solved -- the rows a pairwise loss can use."""
        return self.select(self.found)


def _bucket(user: int, seed: int) -> int:
    """A named hash, never ``hash()``.

    Python's ``hash`` on an int below 2**61 IS that int, so consecutive user
    indices would land in consecutive buckets and a 30% holdout would be a
    contiguous slice of the id space rather than a sample of it. On strings it
    is salted per process instead, so the split would differ between runs.
    """
    digest = hashlib.blake2b(f"{seed}:{user}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _BUCKETS


def holdout_mask(rows: RankingRows, holdout: float = 0.3, seed: int = 0) -> npt.NDArray[np.bool_]:
    """Which REQUESTS fall in the holdout, by hashing their user."""
    cut = int(holdout * _BUCKETS)
    return np.fromiter(
        (_bucket(int(user), seed) < cut for user in rows.user_ids),
        dtype=bool,
        count=len(rows.user_ids),
    )


def split_by_user(
    rows: RankingRows, holdout: float = 0.3, seed: int = 0
) -> tuple[RankingRows, RankingRows]:
    """Partition requests by USER, so no user appears on both sides.

    By user rather than by time, and the trade is worth naming. A temporal cut
    inside a twelve-hour window leaves the later half with barely any history
    the earlier half did not already have, and the row order that a time cut
    would rely on is a property of the collect, not of the data. A user cut is
    the same unit the bootstrap resamples and makes leakage through a repeated
    user impossible.

    **What it does not buy is retrieval hygiene.** Both halves come from the
    window the retriever early-stopped on, so its scores here are mildly
    optimistic for every request. The alternative -- generating ranker rows from
    the window the retriever was FITTED on -- is worse, because there the
    retrieval score is memorised rather than merely selected.
    """
    held = holdout_mask(rows, holdout, seed)
    return rows.select(~held), rows.select(held)


def shard_by_request(rows: RankingRows, rank: int, world: int) -> RankingRows:
    """One rank's slice, **split on requests and never inside a group**.

    The current loss is pointwise, so rows would shard fine on their own. A
    listwise loss would not: half a candidate list on each of two ranks is a
    ranking problem over a list neither rank can see, and it would train
    without complaining. Sharding on the request boundary now costs nothing and
    means the objective can change later without this becoming a silent bug.

    Strided rather than contiguous. Requests arrive in roughly time order, so
    contiguous blocks would hand each rank its own slice of the window -- one
    rank seeing only the small hours -- and the ranks would disagree about what
    the data looks like while DDP averaged their gradients.
    """
    if world < 1 or not 0 <= rank < world:
        raise ValueError(f"rank {rank} is not a member of a world of {world}")
    mask = np.zeros(len(rows.groups), dtype=bool)
    mask[rank::world] = True
    return rows.select(mask)


def blend_one(
    tops: dict[str, Sequence[int]], max_candidates: int, quotas: Sequence[int]
) -> tuple[list[int], dict[str, list[int]]]:
    """Blend ONE request's per-source lists into a candidate set with rank tags.

    Extracted from :func:`candidate_lists` so that the serving path has
    something to be equal TO. The Go orchestrator reimplements exactly this
    function, and `serving/go/internal/retrieval/blend_test.go` asserts the two
    agree on fixtures generated from this one -- which is only expressible if
    the behaviour lives in a function rather than inside a loop over requests.

    Three things here are behaviour, not detail, and each is a way the Go port
    could differ while looking right:

    1. **Insertion order is the output order.** ``seen`` is a dict and Python
       dicts keep insertion order, so the candidate list is ordered by the
       quota pass, then by the top-up pass. Go maps do NOT, so the port must
       carry an explicit slice.
    2. **Item 0 is the reserved OOV row and is never a candidate.** A source
       that returns a short list pads with 0, and serving a padded slot would
       put the reserved row in front of a user.
    3. **The top-up runs in source order after every quota**, so an
       under-filled quota is spent rather than lost.

    Args:
        tops: Source name to that source's ranked item ids, best first.
        max_candidates: Total slots.
        quotas: Slots per source, in ``tops`` order.

    Returns:
        ``(chosen, ranks)``. ``ranks[name][i]`` is that source's position for
        ``chosen[i]``, or :data:`ABSENT`.
    """
    names = list(tops)
    seen: dict[int, int] = {}
    for name, slots in zip(names, quotas, strict=True):
        taken = 0
        for item in tops[name]:
            if taken >= slots or len(seen) >= max_candidates:
                break
            if item > 0 and int(item) not in seen:
                seen[int(item)] = len(seen)
                taken += 1

    # Top up from whatever is left, in source order, so an under-filled quota
    # is spent rather than lost.
    for name in names:
        for item in tops[name]:
            if len(seen) >= max_candidates:
                break
            if item > 0:
                seen.setdefault(int(item), len(seen))

    chosen = list(seen)[:max_candidates]
    ranks: dict[str, list[int]] = {}
    for name in names:
        place = {int(item): position for position, item in enumerate(tops[name]) if item > 0}
        ranks[name] = [place.get(int(item), ABSENT) for item in chosen]
    return chosen, ranks


def candidate_lists(
    sources: dict[str, Retrieved], max_candidates: int, quotas: Sequence[int] | None = None
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], dict[str, npt.NDArray[np.int64]]]:
    """Fill the candidate slots by explicit per-source quota, keeping every rank.

    **Contributing candidates and contributing features are different jobs, and
    a source can do the second without the first.** Measured on this corpus, no
    equal-slot blend beats handing the whole budget to the strongest retriever:
    four sources sharing 100 slots recall 0.2947 where one source using all 100
    recalls 0.3776. So the default gives every slot to the first source, and the
    others still get a rank column for each chosen candidate -- "this item is
    also trending at position 4" is a feature, and it costs no slot.

    An equal split is still expressible, because the point is that the
    allocation is a stated decision rather than a property of loop order. The
    first version of this function concatenated each list in turn, which gave
    the whole budget to the first source by accident; the second round-robined,
    which gave every source an equal share and measurably lost.

    Args:
        sources: Name to that source's per-request candidates.
        max_candidates: Total slots per request.
        quotas: Slots per source, in ``sources`` order. ``None`` gives them all
            to the first. Short lists are topped up from later sources in order,
            so a source that cannot fill its quota does not waste it.

    Returns:
        ``(items [N], request [N], ranks)`` where ``ranks[name][i]`` is that
        source's position for row ``i``, or :data:`ABSENT`.

    Raises:
        ValueError: If ``quotas`` does not match the number of sources.
    """
    names = list(sources)
    if quotas is None:
        quotas = [max_candidates] + [0] * (len(names) - 1)
    if len(quotas) != len(names):
        raise ValueError(f"{len(quotas)} quotas for {len(names)} sources")

    rows = len(sources[names[0]].item_ids)
    items: list[npt.NDArray[np.int64]] = []
    owners: list[npt.NDArray[np.int64]] = []
    ranks: dict[str, list[npt.NDArray[np.int64]]] = {name: [] for name in names}

    for request in range(rows):
        tops = {name: sources[name].top[request].numpy().tolist() for name in names}

        picked, per_source = blend_one(tops, max_candidates, quotas)

        chosen = np.asarray(picked, dtype=np.int64)
        items.append(chosen)
        owners.append(np.full(len(chosen), request, dtype=np.int64))
        for name in names:
            ranks[name].append(np.asarray(per_source[name], dtype=np.int64))

    return (
        np.concatenate(items),
        np.concatenate(owners),
        {name: np.concatenate(values) for name, values in ranks.items()},
    )


def slate_membership(
    split: SplitTensors, items: npt.NDArray[np.int64], request: npt.NDArray[np.int64]
) -> npt.NDArray[np.bool_]:
    """Whether each candidate was in the user's real impression.

    Only the stored slate negatives are knowable, and the table keeps a prefix
    of them, so this UNDERCOUNTS: a candidate marked unobserved may still have
    been shown. Reported rather than corrected, because the correction would
    need the full impression and the gold table does not carry it.
    """
    negatives = split.neg_ids.numpy()
    mask = split.neg_mask.numpy().astype(bool)
    clicked = split.item_ids.numpy()

    shown = [set(row[keep].tolist()) for row, keep in zip(negatives, mask, strict=True)]
    for index, item in enumerate(clicked):
        shown[index].add(int(item))

    return np.fromiter(
        (int(item) in shown[row] for item, row in zip(items, request, strict=True)),
        dtype=bool,
        count=len(items),
    )


def feature_columns(
    *,
    retrieval_score: npt.NDArray[Any],
    ranks: Mapping[str, npt.NDArray[Any]],
    prior_clicks: npt.NDArray[Any],
    train_clicks: npt.NDArray[Any],
    content_similarity: npt.NDArray[Any],
    history_length: npt.NDArray[Any],
    category: npt.NDArray[Any],
    subcategory: npt.NDArray[Any],
) -> dict[str, npt.NDArray[Any]]:
    """The ranker's named columns, one value per candidate.

    Separated from :func:`build` so that the arithmetic which DEFINES each
    column can be called without a tower, a Spark split or a checkpoint --
    which is what lets ``scripts/dump_parity_fixtures.py`` record it, and so
    what lets the Go serving implementation be checked against it rather than
    against someone's reading of it. Everything above this line in ``build``
    is data assembly; everything this function does is the feature definition.

    Every argument is already indexed per candidate. Keyword-only, because
    eight same-shaped integer arrays in a row is an argument list where a
    transposition type-checks, runs, and trains a different model.

    Args:
        retrieval_score: Tower dot product per candidate.
        ranks: Per-source rank arrays, each aligned with the candidates.
            ``ABSENT`` where a source did not propose the item.
        prior_clicks: Cumulative clicks before the counting boundary.
        train_clicks: Cumulative clicks within it.
        content_similarity: Pooled-history content vector against the
            candidate's, the pooled side normalised and the candidate's not.
        history_length: Real history entries for the candidate's request.
        category: Category index per candidate.
        subcategory: Subcategory index per candidate.

    Returns:
        Column name to per-candidate values.
    """
    width = len(retrieval_score)

    # Summed over the sources RETRIEVAL ACTUALLY RAN, not over a configured
    # list: a build with one source makes this column constant, which is
    # correct and is what a ranker trained on that configuration saw. Started
    # from an explicit zero array so an empty `ranks` yields a column rather
    # than the integer 0, which would broadcast and silently produce a matrix
    # of the right shape.
    present = np.zeros(width, dtype=np.int64)
    for name in ranks:
        present = present + (ranks[name] < ABSENT).astype(np.int64)

    return {
        "retrieval_score": retrieval_score,
        "two_tower_rank": ranks.get("two_tower", np.full(width, ABSENT)),
        "trending_rank": ranks.get("trending", np.full(width, ABSENT)),
        "n_sources": present,
        "prior_clicks": prior_clicks,
        "train_clicks": train_clicks,
        "content_similarity": content_similarity,
        "history_length": history_length,
        # Derived, never fetched. A stored copy of this flag could disagree
        # with the count sitting next to it in the same row.
        "is_cold_item": (train_clicks == 0).astype(np.int64),
        "category_idx": category,
        "subcategory_idx": subcategory,
    }


def stack_features(
    columns: Mapping[str, npt.NDArray[Any]], names: Sequence[str]
) -> npt.NDArray[np.float32]:
    """Columns to a ``[N, len(names)]`` matrix, in ``names`` order.

    The order is the whole point and is why this is a lookup rather than a
    concatenation: the exported graph bakes in a standardiser fitted to these
    positions, and a permuted matrix has the right shape, the right dtype and
    scores a different world in silence.
    """
    unknown = set(names) - set(columns)
    if unknown:
        raise ValueError(f"no such feature(s): {sorted(unknown)}")
    return np.stack([columns[name] for name in names], axis=1).astype(np.float32)


def build(
    tower: TwoTower,
    items_table: ItemTables,
    split: SplitTensors,
    sources: dict[str, Retrieved],
    train_clicks: npt.NDArray[np.int64],
    prior_clicks: npt.NDArray[np.int64],
    device: torch.device,
    max_candidates: int = 50,
    quotas: Sequence[int] | None = None,
    features: Sequence[str] | None = None,
) -> RankingRows:
    """One ranking table from one split and its retrieved candidates.

    Precondition, currently guaranteed by the caller rather than by the code:
    every request must sit AFTER the boundary ``prior_clicks`` was counted at.
    That vector is a single snapshot taken at the validation cut, so it is
    stale-but-safe to the right of it and contains the row's own future to the
    left. Widening the window without a per-request point-in-time lookup turns
    it into an oracle, and a tree will find it immediately.
    """
    candidates, request, ranks = candidate_lists(sources, max_candidates, quotas)
    clicked = split.item_ids.numpy()
    labels = (candidates == clicked[request]).astype(np.int64)

    users = encode_users(tower, split, device)
    with torch.no_grad():
        catalogue = tower.precompute_items()
        scores = (
            (users[request].to(device) * catalogue[candidates].to(device)).sum(dim=-1).cpu().numpy()
        )

        content = items_table.content.to(device)
        history = split.history_ids.to(device)
        weight = split.history_mask.to(device).unsqueeze(-1).to(content.dtype)
        pooled = (content[history] * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        similarity = (
            (pooled[request].to(device) * content[candidates].to(device)).sum(dim=-1).cpu().numpy()
        )

    lengths = split.history_mask.sum(dim=1).numpy()

    columns = feature_columns(
        retrieval_score=scores,
        ranks=ranks,
        prior_clicks=prior_clicks[candidates],
        train_clicks=train_clicks[candidates],
        content_similarity=similarity,
        history_length=lengths[request],
        category=items_table.category.numpy()[candidates],
        subcategory=items_table.subcategory.numpy()[candidates],
    )

    chosen_features = tuple(FEATURES if features is None else features)
    return RankingRows(
        names=chosen_features,
        features=stack_features(columns, chosen_features),
        labels=labels,
        groups=np.bincount(request, minlength=len(clicked)).astype(np.int64),
        items=candidates,
        request=request,
        user_ids=split.user_ids.numpy(),
        observed=slate_membership(split, candidates, request),
        found=np.bincount(request, weights=labels, minlength=len(clicked)).astype(bool),
    )
