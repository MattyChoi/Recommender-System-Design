"""The feature column order, and why it is declared here rather than inferred.

Feast returns a dict. Dicts have an order, and it is the order the features
were REQUESTED in -- which means the column order of every vector this gateway
produces is decided by a list literal somewhere. This module is that literal,
written down once, so the decision is visible instead of incidental.

**Order is the one part of the serving contract nothing downstream can check
by shape.** The ranker's ONNX graph was exported against a fixed column order
with the standardiser, the reciprocal on ranks and the log1p on counts baked
in. Hand it a permuted row and the shape is right, the dtype is right, Triton
is happy, the model returns plausible scores, and the slate is simply wrong.
The user tower has the same property one step earlier: a permuted feature
vector produces a believable embedding for a user who does not exist.

So the names travel with the values on the wire, and both consumers compare
them against what they were built for. This file is the single source both
sides are compared to.
"""

from __future__ import annotations

#: The user tower's static feature block, in `docs/design.md` §4b order.
#: This is the order `n_user_feats` counts and `user_norm` was fitted on.
USER_COLUMNS: tuple[str, ...] = (
    "user_impressions_24h",
    "user_clicks_24h",
    "user_ctr_smoothed",
    "user_tenure_hours",
)

#: Which Feast view each user column comes from. Kept beside the order rather
#: than derived, because Feast requests features as "view:name" and a column
#: moving between views is a rename this file should have to acknowledge.
USER_VIEW = "user_stats"

#: Per-item columns the ranker takes from the store.
#:
#: The ranker's full row also carries blend-derived columns (per-source ranks,
#: source count) and model-derived ones (`retrieval_score`,
#: `content_similarity`). This gateway owns neither: the first are the
#: orchestrator's, the second come from the retrieval sidecar, which already
#: holds the tower and the content table.
#:
#: `item_*_cum` are the live equivalents of the ranker's `prior_clicks` and
#: `train_clicks`, which are TOTALS over an article's life rather than windows.
#: They were added to `item_hourly_features` and the Feast schema for this;
#: serving `item_clicks_24h` in their place would have been a different
#: quantity of an entirely plausible size, and `is_cold_item` is derived from
#: one of them, so the flag would have flipped meaning too.
#:
#: ⚠️ Live counts are still not the SAME numbers the checkpoint was fitted on
#: -- those were a snapshot frozen at the training boundary, and these move.
#: The definition now matches; the values drift with traffic. That residual is
#: a row `docs/skew_report.md` owes, and it is small in a way the previous
#: version was not. See logs.txt [M].
ITEM_COLUMNS: tuple[str, ...] = (
    "item_impressions_24h",
    "item_clicks_24h",
    "item_impressions_cum",
    "item_clicks_cum",
    "item_ctr_smoothed",
    "item_age_hours",
)

#: Which gateway column feeds which of the ranker's FEATURES entries.
#:
#: Written down because the two vocabularies genuinely differ: the store names
#: columns by what they measure, `models/ranking/dataset.py` names them by what
#: the model was taught to call them. An undocumented mapping between two
#: orderings is how a permutation gets introduced by someone being helpful.
RANKER_COLUMN_SOURCE: dict[str, str] = {
    "prior_clicks": "item_impressions_cum",
    "train_clicks": "item_clicks_cum",
}

ITEM_VIEW = "item_stats"

#: Feast's realtime view holding the user's recent items, as EXTERNAL id
#: strings. Translated to internal indices before it leaves this process.
HISTORY_VIEW = "user_realtime"
HISTORY_COLUMN = "last_50_items"

#: Served when the store has no row for an entity.
#:
#: Zero, and that is a decision with a cost. The tower and the ranker were
#: fitted on standardised inputs, so zero is not "no information" -- it is a
#: specific point in feature space, roughly the population mean for a
#: standardised column and far from it for a raw count. The alternative,
#: training-set means, needs an artifact this gateway does not have and would
#: be one more thing to keep in step with a retrain.
#:
#: What makes it acceptable is that it is REPORTED: every response says whether
#: the row was found, so a rising miss rate is visible rather than being
#: absorbed as quietly average users.
MISSING = 0.0


def feature_refs(view: str, columns: tuple[str, ...]) -> list[str]:
    """Feast's "view:column" references, in this module's declared order."""
    return [f"{view}:{column}" for column in columns]
