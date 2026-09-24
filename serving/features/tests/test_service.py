"""The feature gateway.

Feast is faked. What is under test is not Feast -- it is the translation layer
around it, which is where this gateway can be wrong in ways nothing downstream
detects: a column order that drifts from what the tower was fitted on, a
history entry silently zero-filled to the reserved OOV row, a miss served as a
plausible average user with no record that it happened.
"""

from __future__ import annotations

from typing import Any

import pytest

from common.pb import features_pb2
from serving.features.columns import ITEM_COLUMNS, MISSING, USER_COLUMNS
from serving.features.service import FeaturesServicer, ItemIndex, Loaded


class AbortedError(Exception):
    """Raised by the fake context, because the real `abort` raises.

    The servicer has no `return` after an abort and relies on that. A fake that
    only recorded the call would let execution run on into code production
    never reaches.
    """

    def __init__(self, code: Any, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class FakeContext:
    def abort(self, code: Any, detail: str) -> None:
        raise AbortedError(code, detail)


class FakeStore:
    """Returns whatever dict it was handed, and records what was asked for.

    Feast's shape matters here: `to_dict()` gives one LIST per feature name,
    aligned with entity_rows, and a missing entity comes back as None inside
    that list rather than as an absent key.
    """

    def __init__(self, payload: dict[str, list[Any]] | None = None) -> None:
        self.payload = payload or {}
        self.calls: list[dict[str, Any]] = []

    def get_online_features(self, features: list[str], entity_rows: list[dict[str, Any]]) -> Any:
        self.calls.append({"features": list(features), "entity_rows": list(entity_rows)})
        payload = self.payload

        class Result:
            @staticmethod
            def to_dict() -> dict[str, list[Any]]:
                return payload

        return Result()


ITEM_MAP = {"N1": 1, "N2": 2, "N3": 3}


def make_servicer(payload: dict[str, list[Any]] | None = None) -> FeaturesServicer:
    return FeaturesServicer(
        Loaded(store=FakeStore(payload), items=ItemIndex(dict(ITEM_MAP), "v=test"))
    )


def user_payload(**overrides: Any) -> dict[str, list[Any]]:
    payload: dict[str, list[Any]] = {
        "user_impressions_24h": [10.0],
        "user_clicks_24h": [2.0],
        "user_ctr_smoothed": [0.2],
        "user_tenure_hours": [48.0],
        "last_50_items": [["N2", "N1"]],
    }
    payload.update(overrides)
    return payload


# --- The item map ------------------------------------------------------------


def test_an_uninvertible_map_is_refused_at_construction() -> None:
    """Two ids on one index means GetItems has no answer.

    Whichever won, the gateway would fetch a DIFFERENT article's features for
    that candidate -- a correctly-shaped row describing the wrong item, which
    the ranker scores without complaint.
    """
    with pytest.raises(RuntimeError, match="invertible"):
        ItemIndex({"N1": 1, "N2": 1}, "broken")


def test_unknown_history_ids_are_dropped_and_counted() -> None:
    """Dropped, not zero-filled.

    Index 0 is the reserved OOV row. Pooled into a user's history it would drag
    their vector toward a row that is not an article, and it would do it more
    for users whose history the map knows least about.
    """
    index = ItemIndex(dict(ITEM_MAP), "v=test")

    got, unmapped = index.translate(["N1", "N404", "N3", "N999"])

    assert got == [1, 3]
    assert unmapped == 2


# --- GetUser -----------------------------------------------------------------


def test_the_user_vector_is_in_declared_column_order() -> None:
    """The order is the contract. Nothing downstream can check it by shape.

    Built from a payload whose values are all distinct so a permutation cannot
    coincidentally pass.
    """
    servicer = make_servicer(user_payload())

    response = servicer.GetUser(features_pb2.GetUserRequest(user_id="U1"), FakeContext())

    assert list(response.names) == list(USER_COLUMNS)
    assert list(response.values) == pytest.approx([10.0, 2.0, 0.2, 48.0])  # float32 on the wire


def test_the_names_travel_with_the_values() -> None:
    # The consumer compares these against what the tower was fitted on. A
    # response carrying values alone would make that check impossible.
    servicer = make_servicer(user_payload())

    response = servicer.GetUser(features_pb2.GetUserRequest(user_id="U1"), FakeContext())

    assert len(response.names) == len(response.values)


def test_history_comes_back_as_internal_indices_most_recent_first() -> None:
    servicer = make_servicer(user_payload(last_50_items=[["N2", "N1", "N3"]]))

    response = servicer.GetUser(features_pb2.GetUserRequest(user_id="U1"), FakeContext())

    # Order preserved: Feast stores most-recent-first and the tower pools in
    # that order. Sorting here would be invisible and wrong.
    assert list(response.history) == [2, 1, 3]
    assert response.unmapped_history == 0


def test_history_is_truncated_to_the_requested_limit() -> None:
    servicer = make_servicer(user_payload(last_50_items=[["N1", "N2", "N3"]]))

    response = servicer.GetUser(
        features_pb2.GetUserRequest(user_id="U1", max_history=2), FakeContext()
    )

    # The head, because most-recent-first.
    assert list(response.history) == [1, 2]


def test_a_missing_user_is_served_with_defaults_and_reported() -> None:
    """The miss has to be visible.

    Zero is not "no information" to a model fitted on standardised inputs -- it
    is a specific point in feature space. Absorbing a miss as a plausible
    average user with no record is how a stopped materialisation looks healthy.
    """
    servicer = make_servicer({name: [None] for name in USER_COLUMNS} | {"last_50_items": [None]})

    response = servicer.GetUser(features_pb2.GetUserRequest(user_id="U404"), FakeContext())

    assert response.found is False
    assert list(response.values) == [MISSING] * len(USER_COLUMNS)
    assert list(response.history) == []


def test_a_user_with_one_null_column_still_counts_as_found() -> None:
    """A row with a null tenure is not the same event as a user the store has
    never seen, and collapsing them makes the miss rate unreadable."""
    servicer = make_servicer(user_payload(user_tenure_hours=[None]))

    response = servicer.GetUser(features_pb2.GetUserRequest(user_id="U1"), FakeContext())

    assert response.found is True
    assert response.values[-1] == MISSING


def test_an_empty_user_id_is_refused() -> None:
    servicer = make_servicer(user_payload())

    with pytest.raises(AbortedError, match="user_id is required"):
        servicer.GetUser(features_pb2.GetUserRequest(user_id=""), FakeContext())


def test_the_user_call_asks_feast_for_history_in_the_same_request() -> None:
    """One call, not two.

    Split across two RPCs, the retriever and the ranker could see different
    snapshots of the same user mid-request.
    """
    servicer = make_servicer(user_payload())
    store: Any = servicer.loaded.store

    servicer.GetUser(features_pb2.GetUserRequest(user_id="U1"), FakeContext())

    assert len(store.calls) == 1
    assert "user_realtime:last_50_items" in store.calls[0]["features"]


# --- GetItems ----------------------------------------------------------------


def item_payload(rows: int) -> dict[str, list[Any]]:
    return {name: [float(position) for position in range(rows)] for name in ITEM_COLUMNS}


def test_item_rows_are_flattened_in_request_order() -> None:
    servicer = make_servicer(item_payload(2))

    response = servicer.GetItems(features_pb2.GetItemsRequest(items=[1, 2]), FakeContext())

    # Row-major: item 1's four columns, then item 2's. Column-major would be
    # the same length and the same numbers, and would score every candidate
    # against another candidate's features.
    assert list(response.values) == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert list(response.names) == list(ITEM_COLUMNS)


def test_the_gateway_translates_indices_back_to_catalogue_ids() -> None:
    servicer = make_servicer(item_payload(2))
    store: Any = servicer.loaded.store

    servicer.GetItems(features_pb2.GetItemsRequest(items=[3, 1]), FakeContext())

    # Feast is keyed on the external id; the orchestrator speaks indices.
    assert store.calls[0]["entity_rows"] == [{"item_id": "N3"}, {"item_id": "N1"}]


def test_an_unmappable_item_index_is_refused() -> None:
    """Loud, because it means two different item maps are in play.

    Serving defaults instead would return a well-formed row of zeros for a
    real candidate, and the ranker would simply score it low -- a quality bug
    with no error anywhere.
    """
    servicer = make_servicer(item_payload(1))

    with pytest.raises(AbortedError) as caught:
        servicer.GetItems(features_pb2.GetItemsRequest(items=[99]), FakeContext())

    assert "different maps" in caught.value.detail
    assert "[99]" in caught.value.detail, "the error should name the offending index"


def test_an_empty_candidate_set_is_not_an_error() -> None:
    # A legal state after filtering, and the orchestrator will not call the
    # ranker either.
    servicer = make_servicer()
    store: Any = servicer.loaded.store

    response = servicer.GetItems(features_pb2.GetItemsRequest(items=[]), FakeContext())

    assert list(response.values) == []
    assert list(response.names) == list(ITEM_COLUMNS)
    assert store.calls == [], "an empty request is a round trip nobody needs"


def test_missing_items_are_reported_per_item() -> None:
    """Per-item, because the interpretation differs from users.

    A cold article is normal on a news corpus. ALL of them being cold is a
    broken materialisation, and only a per-item flag can tell those apart.
    """
    payload = {name: [1.0, None] for name in ITEM_COLUMNS}
    servicer = make_servicer(payload)

    response = servicer.GetItems(features_pb2.GetItemsRequest(items=[1, 2]), FakeContext())

    assert list(response.found) == [True, False]
    assert list(response.values)[4:] == [MISSING] * len(ITEM_COLUMNS)


# --- Health ------------------------------------------------------------------


def test_health_does_a_probe_read_rather_than_trusting_the_sdk() -> None:
    """A flag saying the SDK constructed says nothing about whether Redis is
    reachable, and this endpoint exists to answer exactly that."""
    servicer = make_servicer(user_payload())
    store: Any = servicer.loaded.store

    response = servicer.Health(features_pb2.FeaturesHealthRequest(), FakeContext())

    assert response.ready is True
    assert store.calls, "health must actually read the store"
    assert response.item_map_version == "v=test"


def test_health_reports_an_unreadable_store_as_an_answer() -> None:
    class Broken(FakeStore):
        def get_online_features(
            self, features: list[str], entity_rows: list[dict[str, Any]]
        ) -> Any:
            raise ConnectionError("redis is down")

    servicer = FeaturesServicer(Loaded(store=Broken(), items=ItemIndex(dict(ITEM_MAP), "v=test")))

    response = servicer.Health(features_pb2.FeaturesHealthRequest(), FakeContext())

    # An answer, not a gRPC error: the latter is indistinguishable from an
    # unreachable gateway.
    assert response.ready is False
    assert "redis is down" in response.detail
