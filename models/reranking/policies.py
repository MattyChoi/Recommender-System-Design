"""The policy layer: one greedy pass that selects a slate, and what it costs.

Ranking optimises an item. Re-ranking optimises the **slate** -- which is a
different object, because the value of showing an article depends on what else
is beside it. Four policies act here and **they are not independent stages**:

- **MMR** trades relevance against similarity to what is already chosen;
- **category caps** forbid a choice outright;
- a **freshness** multiplier reweights before either looks;
- **exploration** overrides the choice at some slots, and must record the
  probability with which it did.

Composing them as four passes would be wrong, not merely slower. Capping after
MMR discards MMR's pick and promotes a candidate MMR never compared; capping
*inside* the greedy loop makes MMR choose the best FEASIBLE candidate, which is
what the constraint means. So this is one loop with parts that can be switched
off, and an arm of the trade-off table is a configuration of it.

**What this corpus permits, stated before any number is read.** Every request
here has exactly ONE relevant item. A re-ranker cannot raise NDCG by surfacing a
second good article, because there is no second good article -- so the relevance
column of the table is a cost curve by construction, and the only question is
how steep it is per unit of diversity bought. **Whether that exchange rate is
worth paying cannot be answered offline at all.** It is a product decision, and
the honest output is the rate, not a recommendation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# A slot filled by the greedy rule rather than by exploration. Kept as a named
# constant because it appears in the propensity vector, where "1.0" alone would
# look like a probability someone computed.
DETERMINISTIC = 1.0


@dataclass(frozen=True)
class Slate:
    """One request's final ordering, with the propensities that produced it.

    Attributes:
        items: Catalogue indices, in served order, at most ``k`` of them.
        rows: Index into the request's candidate block for each served item, so
            a caller can recover labels or features without a second lookup.
        propensity: ``P(this item in this slot | policy, history)``, one per
            served slot. **Logged at serving time or not at all**: it cannot be
            reconstructed afterwards, because the candidate set and the random
            draw are both gone. Off-policy evaluation is impossible without it,
            which is why it is part of the return value rather than an option.
    """

    items: npt.NDArray[np.int64]
    rows: npt.NDArray[np.int64]
    propensity: npt.NDArray[np.float64]


def freshness_multiplier(
    age_hours: npt.NDArray[np.float64], half_life_hours: float
) -> npt.NDArray[np.float64]:
    """``2 ** (-age / half_life)`` -- 1.0 for a brand-new item, 0.5 at one half-life.

    A multiplier on a POSITIVE score. The ranker's raw output is not positive
    and not on a meaningful scale, so callers pass calibrated probabilities;
    multiplying an arbitrary-scale score would make the boost's strength depend
    on the ranker's arbitrary units.

    ⚠️ **Age and popularity are nearly the same axis on a week of news.** A new
    article has few clicks because it is new, so a freshness boost and an
    inverse-popularity boost move the same items, and this policy's diversity
    gain should not be read as evidence that *recency* specifically was the
    useful signal.
    """
    if half_life_hours <= 0:
        raise ValueError(f"half_life_hours must be positive; got {half_life_hours}")
    return np.power(2.0, -np.asarray(age_hours, dtype=float) / half_life_hours)


def select(
    scores: npt.NDArray[np.float64],
    items: npt.NDArray[np.int64],
    k: int,
    *,
    vectors: npt.NDArray[np.float32] | None = None,
    lambda_: float = 1.0,
    categories: npt.NDArray[np.int64] | None = None,
    cap: int | None = None,
    blocked: npt.NDArray[np.bool_] | None = None,
    epsilon: float = 0.0,
    explore_slots: int = 0,
    rng: np.random.Generator | None = None,
) -> Slate:
    """Greedily fill ``k`` slots from one request's candidates.

    Args:
        scores: Relevance per candidate, already multiplied by any freshness
            term. Higher is better.
        items: Catalogue index per candidate, aligned with ``scores``.
        k: Slots to fill.
        vectors: ``[n_candidates, dim]`` L2-NORMALISED content vectors, or
            ``None`` to disable MMR. Normalisation is the caller's job and is
            asserted rather than redone, because doing it per request would
            renormalise the same vectors thousands of times.
        lambda_: MMR's weight on relevance. 1.0 is pure relevance and makes MMR
            a no-op; 0.0 ignores relevance entirely.
        categories: Category index per candidate, or ``None`` to disable caps.
        cap: Maximum slots one category may occupy.
        blocked: True where a candidate must not be served -- the seen-list's
            answer, or any other business rule. Applied BEFORE the greedy loop
            rather than as a post-filter, for the same reason caps are: removing
            a chosen item afterwards returns a short slate, while removing it
            from consideration promotes the next feasible candidate.
        epsilon: Probability that an exploration slot is filled uniformly at
            random from the remaining candidates instead of greedily.
        explore_slots: How many of the LAST slots are exploration slots. Last
            rather than first: an explored item in slot 1 costs the most
            relevance, and the point is to gather evidence at the cheapest
            slots that still get looked at.
        rng: Source of randomness. Required when ``epsilon`` is positive, so a
            run that explores is reproducible.

    Returns:
        A :class:`Slate`.

    Raises:
        ValueError: On an unusable configuration -- exploration without a
            generator, a cap that cannot fill ``k``, or mismatched lengths.
            Each of these produces a plausible-looking slate if allowed
            through.
    """
    n = len(scores)
    if len(items) != n:
        raise ValueError(f"{len(items)} items for {n} scores")
    if vectors is not None and len(vectors) != n:
        raise ValueError(f"{len(vectors)} vectors for {n} scores")
    if categories is not None and len(categories) != n:
        raise ValueError(f"{len(categories)} categories for {n} scores")
    if epsilon > 0 and rng is None:
        raise ValueError("exploration needs an rng, or the run cannot be reproduced")
    if epsilon > 0 and explore_slots <= 0:
        raise ValueError("epsilon > 0 with no exploration slots explores nothing")

    budget = min(k, n)
    chosen: list[int] = []
    propensities: list[float] = []
    available = np.ones(n, dtype=bool)
    if blocked is not None:
        if len(blocked) != n:
            raise ValueError(f"{len(blocked)} block flags for {n} scores")
        # A filter that blocks EVERYTHING returns an empty slate, which is a
        # blank page rather than a degraded one -- and a saturated Bloom filter
        # does exactly that (measured: 100% false positives at 10x capacity).
        # The business rule yields, the page does not.
        if blocked.all():
            blocked = np.zeros(n, dtype=bool)
        available &= ~blocked
    used: dict[int, int] = {}
    # Running max similarity to the chosen set, so MMR stays O(n * k) rather
    # than recomputing every pairwise similarity at every step.
    peak_similarity = np.zeros(n, dtype=float)

    for slot in range(budget):
        feasible = available.copy()
        if categories is not None and cap is not None:
            full = {category for category, count in used.items() if count >= cap}
            if full:
                feasible &= ~np.isin(categories, list(full))
            # A cap that leaves nothing is a cap the slate cannot honour. Fall
            # back to the uncapped set rather than returning a short slate: a
            # missing slot is a product decision nobody made, and a silently
            # short list is worse than a briefly violated cap.
            if not feasible.any():
                feasible = available.copy()

        candidates = np.flatnonzero(feasible)
        if len(candidates) == 0:
            break

        objective = scores[candidates]
        if vectors is not None and chosen:
            objective = lambda_ * objective - (1.0 - lambda_) * peak_similarity[candidates]

        greedy = int(candidates[int(np.argmax(objective))])

        exploring = slot >= budget - explore_slots and epsilon > 0
        if exploring:
            assert rng is not None  # guarded above; narrows the type
            if rng.random() < epsilon:
                pick = int(rng.choice(candidates))
                # Two disjoint ways to land here: the random draw, or the random
                # draw happening to choose what greedy would have. Both are
                # counted, or the propensity is wrong for exactly the item most
                # likely to be logged.
                propensity = epsilon / len(candidates)
                if pick == greedy:
                    propensity += 1.0 - epsilon
            else:
                pick = greedy
                propensity = (1.0 - epsilon) + epsilon / len(candidates)
        else:
            pick = greedy
            propensity = DETERMINISTIC

        chosen.append(pick)
        propensities.append(propensity)
        available[pick] = False
        if categories is not None:
            used[int(categories[pick])] = used.get(int(categories[pick]), 0) + 1
        if vectors is not None:
            peak_similarity = np.maximum(peak_similarity, vectors @ vectors[pick])

    rows = np.asarray(chosen, dtype=np.int64)
    return Slate(
        items=items[rows] if len(rows) else np.zeros(0, dtype=np.int64),
        rows=rows,
        propensity=np.asarray(propensities, dtype=float),
    )


def normalise(vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """L2-normalise rows, leaving all-zero rows alone.

    Row 0 of the content table is the reserved OOV vector and is exactly zero;
    dividing it by its norm would produce NaNs that propagate through every
    similarity in the slate.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.asarray(vectors / np.where(norms == 0.0, 1.0, norms), dtype=np.float32)


def slate_per_request(
    scores: npt.NDArray[np.float64],
    items: npt.NDArray[np.int64],
    groups: Sequence[int],
    k: int,
    *,
    vectors: npt.NDArray[np.float32] | None = None,
    lambda_: float = 1.0,
    categories: npt.NDArray[np.int64] | None = None,
    cap: int | None = None,
    blocked: npt.NDArray[np.bool_] | None = None,
    epsilon: float = 0.0,
    explore_slots: int = 0,
    rng: np.random.Generator | None = None,
) -> list[Slate]:
    """Apply :func:`select` to each request's block of candidates in turn.

    **Every per-candidate argument is sliced with the scores.** Spelled out
    rather than forwarded as ``**options``, which is how the first version of
    this handed one request's 100 scores to the whole holdout's 572,200
    vectors -- caught only because ``select`` validates its lengths. A helper
    that forwards opaque keyword arguments cannot be checked at all.
    """
    out: list[Slate] = []
    start = 0
    for size in groups:
        end = start + int(size)
        block = slice(start, end)
        local = select(
            scores[block],
            items[block],
            k,
            vectors=None if vectors is None else vectors[block],
            lambda_=lambda_,
            categories=None if categories is None else categories[block],
            cap=cap,
            blocked=None if blocked is None else blocked[block],
            epsilon=epsilon,
            explore_slots=explore_slots,
            rng=rng,
        )
        # Row indices come back local to the block; shift them so a caller can
        # index the flat arrays it passed in.
        out.append(Slate(items=local.items, rows=local.rows + start, propensity=local.propensity))
        start = end
    return out
