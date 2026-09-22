"""Light Gradient Boosting Machine over retrieved candidates.

**A baseline by placement, not by dismissal.** It sits here for the same reason
`popularity` and `covisit` sit under ``retrieval/baselines``: it is the thing a
learned model has to beat before it earns a second serving stack. On this corpus
it is also the thing that won, and a directory name is not a verdict.

It trains in minutes, has almost nothing to get wrong, and needs no feature
scaling -- a tree splits on order, so the rank sentinels and raw click counts
that the neural rankers must have transformed and standardised go in untouched.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from models.ranking.dataset import CATEGORICAL, RankingRows

DEFAULTS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    # Pairs are formed against the top of the list, where ordering is decided.
    # Above ~30 the gradients are dominated by comparisons nobody will see.
    "lambdarank_truncation_level": 30,
    "verbosity": -1,
}


def train(rows: RankingRows, validation: RankingRows, **overrides: Any) -> Any:
    """Fit a lambdarank model on the requests retrieval solved.

    Groups with no positive are dropped HERE and nowhere else: a pairwise
    objective needs something to rank above something, so they contribute no
    gradient, and keeping them only slows the fit. Evaluation keeps them.
    """
    import lightgbm as lgb

    usable, held = rows.with_positives(), validation.with_positives()
    model = lgb.LGBMRanker(**{**DEFAULTS, **overrides})
    model.fit(
        usable.features,
        usable.labels,
        # Candidates per query, NOT one query id per row. Passing rows is
        # accepted and produces a silently wrong model.
        group=usable.groups,
        eval_set=[(held.features, held.labels)],
        eval_group=[held.groups],
        eval_at=[10],
        # Indices, not names. A numpy matrix carries no column names, so naming
        # them at fit and not at predict makes sklearn warn on every call --
        # and `importances` recovers the names from the row set positionally
        # anyway, which is the same mapping this uses. Read off the ROWS, so a
        # run that dropped a column marks the right indices as categorical.
        categorical_feature=[rows.names.index(name) for name in CATEGORICAL if name in rows.names],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def importances(model: Any, names: Sequence[str]) -> list[tuple[str, float]]:
    """Feature importances by GAIN, largest first.

    Gain rather than split count: a categorical with many levels is split on
    constantly and contributes little, so the default count-based importance
    ranks it top and says nothing.

    ⚠️ **Gain does not separate signal from flexibility.** LightGBM splits a
    categorical by searching partitions of its levels, which is far more
    expressive than a numeric threshold, so a high-cardinality column can earn
    a large gain by fitting the training folds rather than by carrying
    information. A column's share here is a claim to be checked by dropping it
    and re-measuring, not a conclusion.
    """
    gains = model.booster_.feature_importance(importance_type="gain")
    total = float(gains.sum()) or 1.0
    ordered = sorted(zip(names, gains, strict=True), key=lambda pair: -pair[1])
    return [(name, float(value) / total) for name, value in ordered]
