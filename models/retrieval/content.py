"""Content similarity over titles -- the only baseline that can reach a cold item."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as f

# Terms carrying no topical signal. Deliberately short: an aggressive list
# starts removing words that matter in headlines ("no", "not", "new").
_STOPWORDS = (
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "they",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
)

# Single characters are almost always fragments of a split token rather than
# words, and they collide across unrelated titles.
_MIN_TERM_LENGTH = 2


def tokenize(frame: DataFrame, id_column: str = "item_id", text: str = "title") -> DataFrame:
    """Lowercase, split on non-alphanumerics, drop stopwords and single characters.

    Deliberately crude. A stemmer would help recall and would also be a second
    thing to justify; the point of this baseline is to be the simplest lexical
    reference that could work, so that anything beating it has beaten something
    honest.

    Args:
        frame: Rows carrying an id and a text column.
        id_column: Name of the id column.
        text: Name of the text column. Nulls are treated as empty.

    Returns:
        ``<id_column>`` and ``term``, one row per occurrence.
    """
    words = f.split(f.lower(f.coalesce(f.col(text), f.lit(""))), r"[^a-z0-9]+")
    return (
        frame.select(id_column, f.explode(words).alias("term"))
        .filter(f.length("term") >= _MIN_TERM_LENGTH)
        .filter(~f.col("term").isin(list(_STOPWORDS)))
    )


def term_weights(catalogue: DataFrame) -> tuple[DataFrame, DataFrame]:
    """TF-IDF weight per (item, term), and each item's vector norm.

    ``idf = log(N / df)`` unsmoothed, so a term appearing in every title weighs
    exactly zero rather than almost zero. That is the correct reading: a word
    every article uses distinguishes nothing.

    Args:
        catalogue: Distinct ``item_id`` and ``title`` for every addressable item.

    Returns:
        ``(item_id, term, weight)`` and ``(item_id, norm)``.
    """
    terms = tokenize(catalogue)
    n_items = catalogue.count()

    tf = terms.groupBy("item_id", "term").agg(f.count("*").cast("double").alias("tf"))
    df = tf.groupBy("term").agg(f.count_distinct("item_id").alias("df"))

    weights = (
        tf.join(f.broadcast(df), on="term", how="inner")
        .withColumn("idf", f.log(f.lit(float(n_items)) / f.col("df")))
        .select("item_id", "term", (f.col("tf") * f.col("idf")).alias("weight"))
    )
    norms = weights.groupBy("item_id").agg(
        f.sqrt(f.sum(f.col("weight") * f.col("weight"))).alias("norm")
    )
    return weights, norms


def _prior_clicks(labels: DataFrame, train: DataFrame) -> DataFrame:
    """Clicks knowable to a request, from train AND the split being scored.

    Every column is renamed so nothing collides with ``labels`` downstream --
    the discipline covisit._clicks follows, for the reasons given there.
    """

    def shape(frame: DataFrame) -> DataFrame:
        return frame.filter(f.col("clicked")).select(
            f.col("user_id").alias("prior_user"),
            f.col("item_id").alias("prior_item"),
            f.col("ts").alias("prior_ts"),
        )

    # After the union, never inside each half: one click reachable through both
    # sources is one event, and de-duplicating separately leaves the overlap.
    return shape(train).unionByName(shape(labels)).distinct()


def score_content(labels: DataFrame, train: DataFrame, catalogue: DataFrame) -> DataFrame:
    """Cosine between a candidate's title vector and the user's prior-click profile.

    The profile is built once per distinct ``(user_id, ts)`` rather than once
    per row: every candidate in a slate shares the user and the instant, so a
    per-row profile would rebuild the identical vector ~37 times.

    Users with no knowable prior click, and items whose title yields no terms,
    both score 0.0 -- the same honest zero the other baselines give. A cosine is
    undefined when either vector is the zero vector, and 0.0 is the right
    reading of "nothing in common" rather than a sentinel.

    Args:
        labels: Rows to score, carrying ``user_id``, ``item_id``,
            ``impression_id``, ``clicked`` and ``ts``.
        train: Training rows, for the prior-click history.
        catalogue: Distinct ``item_id`` and ``title`` over the addressable
            catalogue. See the module docstring on why this is not train-only.

    Returns:
        ``labels`` plus ``score``, with the row count unchanged.
    """
    weights, norms = term_weights(catalogue)
    prior = _prior_clicks(labels, train)

    profile_source = weights.select(
        f.col("item_id").alias("src_item"),
        f.col("term").alias("p_term"),
        f.col("weight").alias("src_weight"),
    )

    # One profile per request, not per row.
    requests = labels.select(
        f.col("user_id").alias("req_user"), f.col("ts").alias("req_ts")
    ).distinct()

    profile = (
        requests.join(
            prior,
            on=(f.col("req_user") == f.col("prior_user")) & (f.col("prior_ts") < f.col("req_ts")),
            how="inner",
        )
        .join(profile_source, on=f.col("prior_item") == f.col("src_item"), how="inner")
        .groupBy("req_user", "req_ts", "p_term")
        .agg(f.sum("src_weight").alias("p_weight"))
    )
    profile_norms = profile.groupBy("req_user", "req_ts").agg(
        f.sqrt(f.sum(f.col("p_weight") * f.col("p_weight"))).alias("p_norm")
    )

    candidate = weights.select(
        f.col("item_id").alias("c_item"),
        f.col("term").alias("c_term"),
        f.col("weight").alias("c_weight"),
    )

    # INNER joins here, then a left join back onto labels below. Only shared
    # terms contribute to a dot product, so carrying the non-matching ones
    # through the group-by would multiply the work by the vocabulary size for
    # nothing. The row count is restored by the left join, not preserved here.
    dot = (
        labels.select("impression_id", "item_id", "user_id", "ts")
        .join(candidate, on=f.col("item_id") == f.col("c_item"), how="inner")
        .join(
            profile,
            on=(f.col("user_id") == f.col("req_user"))
            & (f.col("ts") == f.col("req_ts"))
            & (f.col("c_term") == f.col("p_term")),
            how="inner",
        )
        .groupBy("impression_id", "item_id")
        .agg(f.sum(f.col("c_weight") * f.col("p_weight")).alias("dot"))
    )

    item_norms = norms.select(f.col("item_id").alias("n_item"), f.col("norm").alias("c_norm"))
    keys = list(labels.columns)
    return (
        labels.join(dot, on=["impression_id", "item_id"], how="left")
        .join(item_norms, on=f.col("item_id") == f.col("n_item"), how="left")
        .join(
            profile_norms,
            on=(f.col("user_id") == f.col("req_user")) & (f.col("ts") == f.col("req_ts")),
            how="left",
        )
        .withColumn(
            "score",
            f.when(
                f.col("dot").isNotNull() & (f.col("c_norm") > 0) & (f.col("p_norm") > 0),
                f.col("dot") / (f.col("c_norm") * f.col("p_norm")),
            ).otherwise(f.lit(0.0)),
        )
        .select(*keys, "score")
    )
