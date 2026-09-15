"""Materialisation parity: does Redis hold what the gold series says it should?

NOT a skew report, and the distinction is the point. Both sides of this
comparison are our own offline reads: the gold Parquet, and the values Feast
materialised from it. It tests Feast, the TTLs and the entity-key encoding --
a renamed column, a narrowed type, a window that wrote nothing.

Skew is the gap between what a SERVING system computed and what training would
have computed for the same entity at the same instant. It lives in the serving
implementation's clock, its timezone handling and its fallbacks, none of which
exist yet. That report is M5, and it reads docs/skew_report.md. This one writes
docs/materialization_parity.md and must never be confused for it.

Usage:
    make parity                      # needs `make up` first
    uv run python -m data_pipeline.features.parity --sample 200
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from feast import FeatureStore, FeatureView, FileSource
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as f

from common.config import Settings, load_settings
from common.spark import get_spark
from common.utils import read_gold

FEATURE_REPO = Path("data_pipeline/features/recsys_store/feature_repo")
REPORT = Path("docs/materialization_parity.md")

# Names, not the objects from definition.py. The views are fetched from the
# REGISTRY instead, for two reasons.
#
# The first is that it is the only thing that works: a FeatureView built in
# Python has an empty `entity_columns` until apply() infers it, because
# entity_columns is populated from schema fields whose names match a join key --
# and our schemas list features only. So `join_keys` on a source-file view is
# [], silently, and `features` is everything.
#
# The second is that it is what we actually want to test. Redis was materialised
# from whatever was last applied. If the source file has moved on since, the
# registry is the honest account of what produced those keys.
VIEW_NAMES = ("item_stats", "user_stats", "user_category_stats")

# Floats round-trip through protobuf and Redis, so exact equality would report
# false mismatches on every rate. The largest observed delta goes in the report
# so the tolerance is visible rather than assumed.
_REL_TOL = 1e-9


def _gold_table(view: FeatureView) -> str:
    """The gold table a view reads, taken from the view itself.

    Deriving it from the source rather than a second hardcoded map means this
    cannot drift from the definitions the way a parallel table would.

    ``batch_source`` is typed as the abstract ``DataSource``; only file-backed
    sources carry a ``path``. The isinstance check is what makes that narrowing
    explicit rather than assumed -- and if a view is ever moved onto a warehouse
    source, this raises instead of failing somewhere less obvious.

    Args:
        view: A FeatureView, as fetched from the registry.

    Returns:
        The gold table name.

    Raises:
        TypeError: If the view does not read from a file source.
    """
    source = view.batch_source
    if not isinstance(source, FileSource):
        raise TypeError(
            f"{view.name} reads from {type(source).__name__}, which has no path; "
            "parity compares against gold Parquet and cannot follow it"
        )
    return source.path.rstrip("/").rsplit("/", 1)[-1]


def expected_rows(
    series: DataFrame, join_keys: Sequence[str], start: datetime, end: datetime
) -> DataFrame:
    """What materialisation should have written, per entity.

    ``feast materialize START END`` writes the latest row per entity inside the
    window. Rows before START are not written at all, so an entity whose last
    bucket predates START is legitimately absent online -- that is a match, not
    a miss, and the report counts it separately.

    Args:
        series: A gold feature series.
        join_keys: The view's entity columns.
        start: Materialisation window start, inclusive.
        end: Materialisation window end, inclusive.

    Returns:
        One row per entity: the bucket that should be in Redis.
    """
    window = Window.partitionBy(*join_keys).orderBy(
        f.col("feature_ts").desc(), f.col("created_ts").desc()
    )
    return (
        series.filter((f.col("feature_ts") >= start) & (f.col("feature_ts") <= end))
        .withColumn("_rank", f.row_number().over(window))
        .filter(f.col("_rank") == 1)
        .drop("_rank")
    )


def _matches(offline: Any, online: Any) -> tuple[bool, float]:
    """Compare one value, returning agreement and the absolute delta if numeric."""
    if offline is None or online is None:
        return offline is None and online is None, 0.0
    if isinstance(offline, float) or isinstance(online, float):
        delta = abs(float(offline) - float(online))
        return math.isclose(float(offline), float(online), rel_tol=_REL_TOL), delta
    return offline == online, 0.0


def compare_view(
    store: FeatureStore,
    view: FeatureView,
    sample: list[dict[str, Any]],
    population: int,
) -> dict[str, Any]:
    """Read one view's features online and diff them against the expected rows.

    Args:
        store: An initialised Feast store.
        view: The FeatureView under test.
        sample: Expected rows as dicts, each carrying the join keys and the
            feature values the gold series says should be online.
        population: Entities with a row in the window, of which ``sample`` is a
            subset. Without it "200 matched" is unreadable -- 200 of 210 and
            200 of 90,000 are very different claims.

    Returns:
        Per-feature counts, the largest float delta seen, and up to three
        example mismatches for the report.
    """
    features = [field.name for field in view.features]
    entity_rows = [{key: row[key] for key in view.join_keys} for row in sample]

    got = store.get_online_features(
        features=[f"{view.name}:{name}" for name in features],
        entity_rows=entity_rows,
    ).to_dict()

    matched = dict.fromkeys(features, 0)
    mismatched = dict.fromkeys(features, 0)
    missing_online = dict.fromkeys(features, 0)
    worst_delta = 0.0
    examples: list[str] = []

    for index, row in enumerate(sample):
        for name in features:
            offline, online = row[name], got[name][index]
            agrees, delta = _matches(offline, online)
            worst_delta = max(worst_delta, delta)

            if agrees:
                matched[name] += 1
            elif online is None:
                # The interesting bucket: present offline, absent online. TTL
                # expiry and a window that wrote nothing both land here.
                missing_online[name] += 1
            else:
                mismatched[name] += 1
                if len(examples) < 3:
                    keys = {key: row[key] for key in view.join_keys}
                    examples.append(
                        f"`{view.name}:{name}` {keys} offline={offline!r} online={online!r}"
                    )

    return {
        "view": view.name,
        "rows": len(sample),
        "population": population,
        "matched": matched,
        "mismatched": mismatched,
        "missing_online": missing_online,
        "worst_delta": worst_delta,
        "examples": examples,
    }


def _render(results: list[dict[str, Any]], start: datetime, end: datetime) -> str:
    lines = [
        "# Materialisation parity",
        "",
        "**This is not the skew report.** Both sides below are offline reads: the gold",
        "series, and what Feast materialised from it. It tests Feast, the TTLs and the",
        "entity-key encoding. The online/offline skew report needs a serving layer and",
        "its impression log; it is M5 and it writes `docs/skew_report.md`.",
        "",
        f"Window: `{start.isoformat()}` to `{end.isoformat()}` · "
        f"float tolerance: rel_tol={_REL_TOL:g}",
        "",
        "| view | feature | sampled | of | matched | mismatched | missing online |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        for name in result["matched"]:
            lines.append(
                f"| {result['view']} | `{name}` | {result['rows']} | "
                f"{result['population']:,} | "
                f"{result['matched'][name]} | {result['mismatched'][name]} | "
                f"{result['missing_online'][name]} |"
            )

    worst = max((r["worst_delta"] for r in results), default=0.0)
    lines += [
        "",
        f"Largest float delta observed: `{worst:.3e}`. A delta of exactly zero",
        "means the tolerance was never exercised, not that it is generous:",
        "doubles round-trip bit-exactly through protobuf and Redis.",
        "",
        "**`of`** is the number of entities with a row in the window, which is the",
        "population the sample is drawn from. Entities whose last bucket predates",
        "the window are not materialised and are not counted here.",
        "",
    ]

    examples = [line for result in results for line in result["examples"]]
    if examples:
        lines += ["## Example mismatches", ""] + [f"- {line}" for line in examples] + [""]
    return "\n".join(lines)


def run(
    spark: SparkSession, settings: Settings, sample_size: int, start: datetime, end: datetime
) -> list[dict[str, Any]]:
    """Sample entities, compare, and return one result per view."""
    store = FeatureStore(repo_path=str(FEATURE_REPO))
    results = []

    for name in VIEW_NAMES:
        view = store.get_feature_view(name)
        series = read_gold(spark, settings, _gold_table(view))
        expected = expected_rows(series, view.join_keys, start, end).cache()
        population = expected.count()
        # Seeded so a rerun compares the same entities; a moving sample makes
        # "did this get better" unanswerable.
        rows = expected.orderBy(f.hash(*view.join_keys)).limit(sample_size).collect()
        expected.unpersist()
        results.append(compare_view(store, view, [row.asDict() for row in rows], population))

    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=200, help="entities per view")
    parser.add_argument("--start", default="2019-11-09T00:00:00")
    parser.add_argument("--end", default="2019-11-16T00:00:00")
    parser.add_argument("--out", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    settings = load_settings()
    spark = get_spark(settings, app="parity")
    try:
        results = run(spark, settings, args.sample, start, end)
    finally:
        spark.stop()

    args.out.write_text(_render(results, start, end))
    print(f"wrote {args.out}")

    bad = sum(sum(r["mismatched"].values()) for r in results)
    for result in results:
        print(
            f"  {result['view']}: {sum(result['matched'].values())} matched, "
            f"{sum(result['mismatched'].values())} mismatched, "
            f"{sum(result['missing_online'].values())} missing online"
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
