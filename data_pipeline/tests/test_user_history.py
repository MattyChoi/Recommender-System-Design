"""The point-in-time guarantee on user history.

Every test here exists because the failure it describes is **silent**. A history
that contains the impression's own click produces no error, no null, and no
schema change -- it produces a two-tower that looks extraordinary offline and
collapses the moment it is served, which is the exact failure mode Part D says
point-in-time correctness exists to prevent.

The central case is two impressions at the same second. That is not
hypothetical on this corpus: 17 (user, ts) pairs in train and 5 in dev carry two
impressions each. A frame keyed on (user, ts), or one that admitted peers, would
have leaked across precisely those 44 impressions -- far too few to move a
metric, which is what would have made it permanent.

Timestamps here are naive and read as UTC because the root ``conftest.py`` sets
``TZ=UTC`` and calls ``time.tzset()`` before Spark starts (B4).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pyspark.sql import DataFrame, SparkSession

from data_pipeline.features.user_history import (
    assemble_history,
    in_window_history,
    point_in_time_history,
    snapshot_history,
)

_T = datetime(2019, 11, 11, 9, 0, 0)


def _at(seconds: int) -> datetime:
    return _T + timedelta(seconds=seconds)


def _clicks(spark: SparkSession, rows: list[tuple[str, int, datetime]]) -> DataFrame:
    return spark.createDataFrame(rows, "user_id string, item_idx int, ts timestamp")


def _requests(spark: SparkSession, rows: list[tuple[int, str, datetime]]) -> DataFrame:
    return spark.createDataFrame(rows, "impression_id long, user_id string, ts timestamp")


def _seen(frame: DataFrame) -> dict[int, list[int]]:
    return {row["impression_id"]: list(row["inwindow_idx"]) for row in frame.collect()}


class TestTheFrameExcludesPeers:
    def test_an_impression_never_sees_its_own_click(self, spark: SparkSession) -> None:
        """The whole point. The click being predicted must not be an input to
        the prediction."""
        clicks = _clicks(spark, [("U1", 11, _T)])
        requests = _requests(spark, [(1, "U1", _T)])

        assert _seen(in_window_history(clicks, requests)) == {1: []}

    def test_two_impressions_at_the_same_second_do_not_see_each_other(
        self, spark: SparkSession
    ) -> None:
        """Measured to occur 17 times in train and 5 in dev.

        Equal timestamps establish no ordering, so the conservative reading is
        that neither precedes the other. A frame keyed on (user, ts) would merge
        these two rows and hand each impression the other's click.
        """
        clicks = _clicks(spark, [("U1", 11, _T), ("U1", 22, _T)])
        requests = _requests(spark, [(1, "U1", _T), (2, "U1", _T)])

        assert _seen(in_window_history(clicks, requests)) == {1: [], 2: []}

    def test_one_second_earlier_is_visible(self, spark: SparkSession) -> None:
        """The other side of the boundary, so ``-1`` is pinned as exactly one
        second rather than as vague exclusion. Without this, widening the frame
        to an hour would pass every other test in this file."""
        clicks = _clicks(spark, [("U1", 11, _at(-1))])
        requests = _requests(spark, [(1, "U1", _T)])

        assert _seen(in_window_history(clicks, requests)) == {1: [11]}

    def test_a_later_click_is_never_visible(self, spark: SparkSession) -> None:
        """Reaching forward is the same bug wearing a different hat."""
        clicks = _clicks(spark, [("U1", 11, _at(60))])
        requests = _requests(spark, [(1, "U1", _T)])

        assert _seen(in_window_history(clicks, requests)) == {1: []}


class TestTheFrameIsPerUser:
    def test_another_users_clicks_are_invisible(self, spark: SparkSession) -> None:
        clicks = _clicks(spark, [("U2", 99, _at(-60))])
        requests = _requests(spark, [(1, "U1", _T)])

        assert _seen(in_window_history(clicks, requests)) == {1: []}

    def test_every_request_gets_a_row_even_with_no_clicks(self, spark: SparkSession) -> None:
        """88% of dev's users are new. An empty history is a state, not an
        absence, and a request that vanished here would silently shrink the
        training set."""
        clicks = _clicks(spark, [("U2", 99, _at(-60))])
        requests = _requests(spark, [(1, "U1", _T), (2, "U2", _T)])

        assert set(_seen(in_window_history(clicks, requests))) == {1, 2}

    def test_clicks_come_back_most_recent_first(self, spark: SparkSession) -> None:
        """The cap keeps a prefix, so the order decides what survives it."""
        clicks = _clicks(
            spark, [("U1", 11, _at(-300)), ("U1", 22, _at(-200)), ("U1", 33, _at(-100))]
        )
        requests = _requests(spark, [(1, "U1", _T)])

        assert _seen(in_window_history(clicks, requests)) == {1: [33, 22, 11]}


class TestSnapshot:
    @pytest.fixture
    def item_map(self, spark: SparkSession) -> DataFrame:
        return spark.createDataFrame(
            [("N1", 1), ("N2", 2), ("N3", 3)], "item_id string, item_idx int"
        )

    def _history(self, spark: SparkSession, rows: list[tuple[int, str, str]]) -> DataFrame:
        return spark.createDataFrame(rows, "impression_id long, user_id string, history string")

    def test_the_string_is_reversed_into_most_recent_first(
        self, spark: SparkSession, item_map: DataFrame
    ) -> None:
        """MIND gives no timestamps inside the snapshot, so this asserts the
        stated CONVENTION -- later in the string is more recent -- not a
        measured fact. Pinned so that changing it has to be deliberate."""
        history = self._history(spark, [(1, "U1", "N1 N2 N3")])

        row = snapshot_history(history, item_map).collect()[0]
        assert list(row["snapshot_idx"]) == [3, 2, 1]
        assert row["snapshot_len"] == 3

    def test_it_collapses_to_one_row_per_user(
        self, spark: SparkSession, item_map: DataFrame
    ) -> None:
        """bronze/history is one row per impression and C4 measured the string
        as constant per user, so this must not fan out -- a duplicated snapshot
        would multiply every downstream join."""
        history = self._history(spark, [(1, "U1", "N1 N2"), (2, "U1", "N1 N2"), (3, "U1", "N1 N2")])

        assert snapshot_history(history, item_map).count() == 1

    def test_unmappable_items_are_dropped_not_padded(
        self, spark: SparkSession, item_map: DataFrame
    ) -> None:
        """N9 is not in news.tsv. Mapping it to OOV_IDX would put padding_idx
        inside a sequence, and padding_idx has a permanently zero embedding --
        it would drag the mean-pool toward the origin in proportion to how many
        a user happens to have. Dropping shortens the history, which
        snapshot_len then reports honestly."""
        history = self._history(spark, [(1, "U1", "N1 N9 N3")])

        row = snapshot_history(history, item_map).collect()[0]
        assert list(row["snapshot_idx"]) == [3, 1]
        assert row["snapshot_len"] == 2

    def test_a_user_with_no_snapshot_produces_no_row(
        self, spark: SparkSession, item_map: DataFrame
    ) -> None:
        """Which is why assemble_history joins LEFT and coalesces."""
        history = self._history(spark, [(1, "U1", None), (2, "U2", "   ")])

        assert snapshot_history(history, item_map).count() == 0


class TestAssembly:
    @pytest.fixture
    def parts(self, spark: SparkSession) -> tuple[DataFrame, DataFrame, DataFrame]:
        requests = _requests(spark, [(1, "U1", _T), (2, "U2", _T)])
        in_window = spark.createDataFrame(
            [(1, [33, 22], 2)],
            "impression_id long, inwindow_idx array<int>, inwindow_len int",
        )
        snapshot = spark.createDataFrame(
            [("U1", [7, 8, 9], 3)],
            "user_id string, snapshot_idx array<int>, snapshot_len int",
        )
        return requests, in_window, snapshot

    def test_in_window_clicks_lead_the_snapshot(
        self, parts: tuple[DataFrame, DataFrame, DataFrame]
    ) -> None:
        """C4 measured the snapshot as predating the log window, so anything
        in-window is newer than all of it. Ordering them the other way would
        make the cap discard the freshest signal first."""
        got = {
            r["impression_id"]: list(r["history_idx"]) for r in assemble_history(*parts).collect()
        }

        assert got[1] == [33, 22, 7, 8, 9]

    def test_the_cap_keeps_the_most_recent(
        self, parts: tuple[DataFrame, DataFrame, DataFrame]
    ) -> None:
        got = {
            r["impression_id"]: list(r["history_idx"])
            for r in assemble_history(*parts, max_len=3).collect()
        }

        assert got[1] == [33, 22, 7]

    def test_a_request_with_neither_source_survives(
        self, parts: tuple[DataFrame, DataFrame, DataFrame]
    ) -> None:
        """U2 has no in-window clicks and no snapshot."""
        got = {r["impression_id"]: r for r in assemble_history(*parts).collect()}

        assert list(got[2]["history_idx"]) == []
        assert got[2]["has_history"] is False
        assert got[2]["history_len"] == 0

    def test_lengths_report_each_source_separately(
        self, parts: tuple[DataFrame, DataFrame, DataFrame]
    ) -> None:
        """Training sees ~3x the in-window history evaluation does, so the two
        sources have to stay separable on the row rather than being summed into
        one number that hides the asymmetry."""
        got = {r["impression_id"]: r for r in assemble_history(*parts).collect()}

        assert (got[1]["inwindow_len"], got[1]["snapshot_len"]) == (2, 3)
        assert (got[2]["inwindow_len"], got[2]["snapshot_len"]) == (0, 0)

    def test_one_row_per_request(self, parts: tuple[DataFrame, DataFrame, DataFrame]) -> None:
        """The grain is the request. A join that fanned out would multiply
        training rows silently."""
        frame = assemble_history(*parts)

        assert frame.count() == frame.select("impression_id").distinct().count() == 2


class TestEndToEnd:
    def test_the_composed_transform_agrees_with_its_parts(self, spark: SparkSession) -> None:
        """One case through ``point_in_time_history`` so the wiring itself is
        covered -- the parts can each be right while the composition passes the
        wrong frame to the wrong argument."""
        requests = _requests(spark, [(1, "U1", _T)])
        clicks = _clicks(spark, [("U1", 33, _at(-100)), ("U1", 44, _T)])
        history = spark.createDataFrame(
            [(1, "U1", "N1 N2")], "impression_id long, user_id string, history string"
        )
        item_map = spark.createDataFrame([("N1", 1), ("N2", 2)], "item_id string, item_idx int")

        row = point_in_time_history(requests, clicks, history, item_map).collect()[0]

        # 44 was clicked AT the request instant and must not appear.
        assert list(row["history_idx"]) == [33, 2, 1]
        assert (row["inwindow_len"], row["snapshot_len"], row["history_len"]) == (1, 2, 3)
        assert row["has_history"] is True
