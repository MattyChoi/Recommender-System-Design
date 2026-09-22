"""Item-item co-visitation: what a user clicked near what else they clicked.

**Pairs come from a user's clicks, never from one impression.** MIND's
impression lists were assembled by Microsoft's own recommender, so counting
items that co-occur in a slate measures the incumbent system rather than user
intent -- build it that way and the model learns to imitate MSN. Same-impression
pairs are also degenerate here: ``to_events`` broadcasts one impression's single
``time`` to every row it explodes, so every intra-impression gap is exactly zero
and both the decay weight and the forward/backward asymmetry collapse.

**Fit on TRAIN ONLY.** A matrix computed over train plus dev leaks inside the
model, where no split test can see it: the rows are still correctly separated,
but the thing scoring them has already read the holdout.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f

# The backward edge is real but weaker: "clicked A then B" is not the same
# signal as "clicked B then A", and collapsing them throws away the ordering
# that the whole window exists to capture.
BACKWARD_DISCOUNT = 0.5

_SECONDS_PER_MINUTE = 60.0


def click_stream(train: DataFrame) -> DataFrame:
    """A user's clicks, ranked in time order.

    Clicks rather than impressions, for the reason ``click_counts`` gives: an
    item's impression count on MIND is how often the incumbent recommender chose
    to show it.

    The rank is what bounds the pair explosion downstream. It is assigned over
    clicks, so ``max_rank_gap`` counts *clicks* between two items, not rows.

    Args:
        train: Training rows with ``user_id``, ``item_id``, ``impression_id``,
            ``ts`` and ``clicked``.

    Returns:
        ``user_id``, ``item_id``, ``impression_id``, ``ts`` and ``rank``.
    """
    # impression_id is the tie-break, not the sort key: MIND's impression ids
    # are not chronological, but every row in an impression shares one ts, so
    # something deterministic is needed to order clicks that arrived together.
    ordered = Window.partitionBy("user_id").orderBy("ts", "impression_id", "item_id")
    return (
        train.filter(f.col("clicked"))
        .select("user_id", "item_id", "impression_id", "ts")
        .withColumn("rank", f.row_number().over(ordered))
    )


def build_covisitation(
    train: DataFrame,
    max_gap_seconds: float = 3600.0,
    max_rank_gap: int = 30,
    top_k: int = 50,
) -> DataFrame:
    """Weighted, asymmetric co-occurrence between items a user clicked near in time.

    Two caps bound the work, and they do different jobs. ``max_gap_seconds`` is
    the modelling parameter -- it says how long a click stays relevant to the
    next one, and it is the one worth sweeping. ``max_rank_gap`` is a compute
    guard: without it one heavy user's click run is O(n^2) pairs on its own.

    Args:
        train: Training rows. TRAIN ONLY -- see the module docstring.
        max_gap_seconds: Longest elapsed time between two clicks that still
            pairs them.
        max_rank_gap: Most clicks that may separate two paired clicks.
        top_k: Neighbours kept per item.

    Returns:
        ``item_id``, ``related_item_id``, ``weight`` and ``rank`` (1 is the
        strongest neighbour), at most ``top_k`` rows per ``item_id``.

    Raises:
        ValueError: If any bound is not positive. A zero window is a
            configuration error that would silently return an empty matrix,
            which reads downstream as "co-visitation does not work on news".
    """
    if max_gap_seconds <= 0:
        raise ValueError(f"max_gap_seconds must be positive, got {max_gap_seconds}")
    if max_rank_gap <= 0:
        raise ValueError(f"max_rank_gap must be positive, got {max_rank_gap}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    clicks = click_stream(train)
    earlier, later = clicks.alias("a"), clicks.alias("b")

    gap_seconds = f.col("b.ts").cast("long") - f.col("a.ts").cast("long")

    pairs = (
        earlier.join(
            later,
            on=(
                (f.col("a.user_id") == f.col("b.user_id"))
                # Strictly forward, and within the rank guard. Taking only b
                # after a means each unordered pair is visited once, so the
                # forward and backward edges below are emitted exactly once each.
                & (f.col("b.rank") > f.col("a.rank"))
                & (f.col("b.rank") <= f.col("a.rank") + max_rank_gap)
            ),
            how="inner",
        )
        # The gate: never pair within an impression.
        .filter(f.col("a.impression_id") != f.col("b.impression_id"))
        # An item re-clicked later is not its own neighbour.
        .filter(f.col("a.item_id") != f.col("b.item_id"))
        .filter(gap_seconds <= max_gap_seconds)
        # log2(2 + minutes) so a simultaneous pair weighs exactly 1.0 and the
        # curve decays without a discontinuity at zero.
        .withColumn(
            "weight",
            f.lit(1.0) / f.log2(f.lit(2.0) + gap_seconds / _SECONDS_PER_MINUTE),
        )
        .select(
            f.col("a.item_id").alias("from_item"),
            f.col("b.item_id").alias("to_item"),
            "weight",
        )
    )

    forward = pairs.select(
        f.col("from_item").alias("item_id"),
        f.col("to_item").alias("related_item_id"),
        "weight",
    )
    backward = pairs.select(
        f.col("to_item").alias("item_id"),
        f.col("from_item").alias("related_item_id"),
        (f.col("weight") * BACKWARD_DISCOUNT).alias("weight"),
    )

    edges = (
        forward.unionByName(backward)
        .groupBy("item_id", "related_item_id")
        .agg(f.sum("weight").alias("weight"))
    )

    # related_item_id breaks ties, so the matrix is byte-identical across runs.
    strongest = Window.partitionBy("item_id").orderBy(
        f.col("weight").desc(), f.col("related_item_id")
    )
    return (
        edges.withColumn("rank", f.row_number().over(strongest))
        .filter(f.col("rank") <= top_k)
        .select("item_id", "related_item_id", "weight", "rank")
    )


def _clicks(frame: DataFrame) -> DataFrame:
    """A frame's clicks, shaped as the prior-click context for scoring."""
    return frame.filter(f.col("clicked")).select(
        f.col("user_id").alias("prior_user"),
        f.col("item_id").alias("prior_item"),
        f.col("ts").alias("prior_ts"),
    )


def score_covisit(
    labels: DataFrame,
    train: DataFrame,
    context: DataFrame | None = None,
    max_gap_seconds: float = 3600.0,
    max_rank_gap: int = 30,
    top_k: int = 50,
) -> DataFrame:
    """Score each row by co-visitation from that user's earlier clicks.

    A candidate scores the summed edge weight from every click the user made
    **strictly before this label's own timestamp** to this candidate. That ``<``
    is the whole point-in-time contract, and it is the same rule the as-of join
    and the decayed baseline use. Because every row of an impression shares one
    ``ts``, it also excludes the user's clicks in the impression being scored --
    which is the leak this model would otherwise walk straight into.

    **Two different things are fitted on two different sets, deliberately.** The
    matrix is model parameters and comes from ``train`` alone: an edge built
    over the evaluation split carries the very click being predicted, so the
    ``<`` above would protect the lookup key while the looked-up value came from
    the future. The user's prior clicks are not model parameters -- they are
    part of the request, and production knows what this user clicked ten minutes
    ago whichever week it is. Withholding them would also be fatal here: 87.8%
    of dev impressions belong to users absent from train, so a train-only
    context scores 0.0 for almost the whole split and the result reads as
    "co-visitation fails on news" rather than "the context was withheld".

    Users with no knowable prior click score 0.0, not null: the same honest
    treatment ``score_most_popular`` gives cold items. Being unable to score a
    cold user is the finding, not something to patch.

    Args:
        labels: Rows to score, carrying ``user_id``, ``item_id``, ``ts`` and
            ``clicked``.
        train: Training rows. The matrix is built from these and only these.
        context: Where the user's prior clicks come from. Defaults to ``train``
            plus ``labels`` -- everything knowable, filtered by the ``<`` rule.
            Pass ``train`` alone to measure what the in-split context is worth.
        max_gap_seconds: Pairing window for the matrix.
        max_rank_gap: Compute guard for the matrix.
        top_k: Neighbours kept per item.

    Returns:
        ``labels`` plus ``score``, with the row count unchanged.
    """
    matrix = build_covisitation(train, max_gap_seconds, max_rank_gap, top_k)

    prior = _clicks(train).unionByName(_clicks(labels)) if context is None else _clicks(context)
    prior = prior.distinct()

    neighbours = matrix.select(
        f.col("item_id").alias("from_item"),
        f.col("related_item_id").alias("to_item"),
        "weight",
    )

    # Join order: attaching each label to its user's own prior clicks first
    # keeps the intermediate bounded by clicks-per-user, small on a one-week
    # corpus. Going through the matrix first would fan each label out by however
    # many items name it as a neighbour, which is bounded by nothing.
    with_history = labels.join(
        prior,
        on=(labels["user_id"] == f.col("prior_user")) & (f.col("prior_ts") < labels["ts"]),
        how="left",
    )

    scored = with_history.join(
        neighbours,
        on=(f.col("prior_item") == f.col("from_item")) & (labels["item_id"] == f.col("to_item")),
        how="left",
    )

    keys = list(labels.columns)
    return (
        scored.groupBy(*[labels[name] for name in keys])
        .agg(f.coalesce(f.sum("weight"), f.lit(0.0)).alias("score"))
        .select(*keys, "score")
    )
