"""ALS, the classic latent-factor baseline -- in two formulations that differ enormously.

Spark's implicit ALS learns a user factor and an item factor per id, and the
textbook score is their dot product. On MIND that textbook score is nearly
blind: **87.8% of dev impressions belong to users absent from train**, and a
user with no training interactions has no factor to dot with.

So this module ships both:

* :func:`score_als` -- the literal user-factor dot item-factor. What a reviewer
  expects, and a measured demonstration of the cold-user wall.
* :func:`score_als_item` -- the candidate's item factor against the factors of
  items the user clicked BEFORE this label. No user factor is needed, so it
  reaches dev-only users, and it is the same parameters-versus-request split

**`coldStartStrategy` is "nan", never "drop".** "drop" deletes rows whose user
or item was unseen, which silently removes labels from the evaluation and
changes the denominator of every metric on the card.
"""

from __future__ import annotations

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

from common.schemas import OOV_IDX

# Hu, Koren & Volinsky's implicit-feedback formulation: `alpha` scales the
# confidence a repeated click implies. 40 is their paper's value and a standard
# starting point; rank 32 is small because train holds ~230k clicks and a larger
# factorisation would fit noise.
DEFAULT_RANK = 32
DEFAULT_REG = 0.1
DEFAULT_ITERATIONS = 10
DEFAULT_ALPHA = 40.0


def fit_als(
    train: DataFrame,
    rank: int = DEFAULT_RANK,
    reg_param: float = DEFAULT_REG,
    max_iter: int = DEFAULT_ITERATIONS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> ALSModel:
    """Fit implicit-feedback ALS on train clicks.

    Clicks per (user, item) are the implicit "rating": how often this user
    clicked this item, never how often it was shown to them. An impression
    count would encode how often the incumbent recommender chose to surface it,
    which is the same trap ``click_counts`` avoids.

    ``OOV_IDX`` rows are dropped before fitting. Index 0 is a bucket, not an
    entity -- factorising it would learn one vector meaning "unknown", and every
    unknown id would then look similar to every other.

    Args:
        train: Training rows with ``user_idx``, ``item_idx`` and ``clicked``.
        rank: Latent dimensionality.
        reg_param: L2 regularisation.
        max_iter: ALS sweeps.
        alpha: Implicit-feedback confidence scaling.
        seed: Fixes initialisation so a rerun reproduces the card.

    Returns:
        The fitted model. ``coldStartStrategy`` is ``"nan"``
    """
    interactions = (
        train.filter(f.col("clicked"))
        .filter((f.col("user_idx") != OOV_IDX) & (f.col("item_idx") != OOV_IDX))
        .groupBy("user_idx", "item_idx")
        .agg(f.count("*").cast("double").alias("clicks"))
    )

    return ALS(
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="clicks",
        implicitPrefs=True,
        rank=rank,
        regParam=reg_param,
        maxIter=max_iter,
        alpha=alpha,
        seed=seed,
        # NEVER "drop": it deletes label rows and changes the denominator.
        coldStartStrategy="nan",
    ).fit(interactions)


def _dot(left: Column, right: Column) -> Column:
    """Dot product of two equal-length array<float> columns, in pure Spark SQL.

    A Python UDF here would serialise every row through the JVM-Python boundary;
    ``zip_with`` and ``aggregate`` keep it in the engine.
    """
    return f.aggregate(
        f.zip_with(left, right, lambda a, b: a * b),
        f.lit(0.0),
        lambda acc, value: acc + value,
    )


def score_als(labels: DataFrame, train: DataFrame, model: ALSModel | None = None) -> DataFrame:
    """Score by ``user_factor . item_factor`` -- the textbook formulation.

    Expect this to be largely blind on MIND. A user absent from train has no
    factor, ALS predicts NaN, and NaN becomes 0.0 here -- the same honest zero
    ``score_most_popular`` gives a cold item. The report card's
    ``gauc_ceiling`` is what makes the resulting blindness legible rather than
    reading as a weak model.

    Args:
        labels: Rows to score, carrying ``user_idx`` and ``item_idx``.
        train: Training rows, used only if ``model`` is not supplied.
        model: A pre-fitted model, so an ablation can share one fit.

    Returns:
        ``labels`` plus ``score``, with the row count unchanged.
    """
    model = fit_als(train) if model is None else model

    keys = list(labels.columns)
    return (
        model.transform(labels)
        .withColumn("score", f.coalesce(f.col("prediction").cast("double"), f.lit(0.0)))
        # NaN is not null: coalesce alone would let it through and poison every
        # comparison it takes part in, since NaN > x and NaN < x are both false.
        .withColumn("score", f.when(f.isnan("score"), f.lit(0.0)).otherwise(f.col("score")))
        .select(*keys, "score")
    )


def score_als_item(labels: DataFrame, train: DataFrame, model: ALSModel | None = None) -> DataFrame:
    """Score a candidate by its item factor against the user's earlier clicks.

    The latent-factor analogue of co-visitation: instead of a counted edge
    between two items, the affinity is the dot product of their learned
    factors. Summed over every click the user made **strictly before this
    label's own timestamp** -- the same ``<`` rule as the as-of join, which on
    MIND also excludes the user's clicks in the impression being scored,
    because every row of an impression shares one ``ts``.

    This needs no user factor, so it reaches the 87.8% of dev impressions whose
    users never appear in train. It still cannot reach an item absent from
    train: no interactions, no factor.

    Args:
        labels: Rows to score, carrying ``user_id``, ``item_idx``, ``ts`` and
            ``clicked``.
        train: Training rows, used only if ``model`` is not supplied.
        model: A pre-fitted model, so an ablation can share one fit.

    Returns:
        ``labels`` plus ``score``, with the row count unchanged.
    """
    model = fit_als(train) if model is None else model
    factors = model.itemFactors.select(
        f.col("id").alias("factor_idx"), f.col("features").alias("factor")
    )

    # Prior clicks come from the labels' own split as well as train: they are
    # the request, not the model.
    prior = (
        train.filter(f.col("clicked"))
        .select(
            f.col("user_id").alias("prior_user"),
            f.col("item_idx").alias("prior_idx"),
            f.col("ts").alias("prior_ts"),
        )
        .unionByName(
            labels.filter(f.col("clicked")).select(
                f.col("user_id").alias("prior_user"),
                f.col("item_idx").alias("prior_idx"),
                f.col("ts").alias("prior_ts"),
            )
        )
        # After the union, not inside each half: a click reachable through both
        # train and the labels is one event, and de-duplicating the halves
        # separately leaves the overlap intact and doubles every score.
        .distinct()
    )

    # The time test lives in the JOIN CONDITION. As a post-filter it would drop
    # the label outright whenever a user has clicks but none precede the label.
    with_history = labels.join(
        prior,
        on=(f.col("user_id") == f.col("prior_user")) & (f.col("prior_ts") < f.col("ts")),
        how="left",
    )

    candidate = factors.select(
        f.col("factor_idx").alias("cand_idx"), f.col("factor").alias("cand_factor")
    )
    paired = with_history.join(
        candidate, on=f.col("item_idx") == f.col("cand_idx"), how="left"
    ).join(factors, on=f.col("prior_idx") == f.col("factor_idx"), how="left")

    keys = list(labels.columns)
    return (
        paired.withColumn(
            "affinity",
            f.when(
                f.col("cand_factor").isNotNull() & f.col("factor").isNotNull(),
                _dot(f.col("cand_factor"), f.col("factor")),
            ),
        )
        .groupBy(*keys)
        .agg(f.coalesce(f.sum("affinity"), f.lit(0.0)).alias("score"))
        .select(*keys, "score")
    )
