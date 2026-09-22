"""The banding rule, and the three ways it could be wrong without erroring.

The first is an off-by-one in the edges. Every band shifts by one, the table
still prints, the plot still slopes, and the caption describes ranges the rows
are not in. The property test below checks each count against the text of its
own label rather than against a second hand-written table, which would just be
the same mistake twice.

The second is a label that drifts from its edge. Labels are derived here for
that reason, and the test holds the derivation to the edges it was given.

The third is ``load_tower`` inferring the ablation flags from state-dict key
names. Rename an attribute on ``TwoTower`` and the inference silently returns
False, scoring one arm through the other's architecture -- so the flags are
round-tripped through a real save/load rather than asserted against a constant.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch

from common.torch_env import select_device
from models.classes.dataset import ItemTables, SplitTensors
from models.classes.train import Hits
from models.retrieval.dataloader.batching import make_loader
from models.retrieval.evaluate import (
    BAND_EDGES,
    BandRow,
    band_labels,
    band_of,
    load_tower,
    popularity_hits,
    render,
    summarise,
)
from models.retrieval.train import retrieval_hits
from models.retrieval.two_tower import TwoTower

N_ITEMS = 12
CONTENT_DIM = 4
N_USER_FEATS = 3
HISTORY = 4
NEGS = 2
ROWS = 32
BATCH = 8


def _range_of(label: str) -> tuple[int, int]:
    """Parse a band label back into the range it claims to cover."""
    if label.endswith("+"):
        return int(label[:-1]), 2**62
    if "-" in label:
        low, high = label.split("-")
        return int(low), int(high)
    return int(label), int(label)


class TestTheBandingRule:
    def test_every_count_lands_in_the_band_its_label_describes(self) -> None:
        """The off-by-one test. Checking band_of against a second hand-written
        table would encode the same mistake twice; checking it against the text
        of the label it produces cannot."""
        labels = band_labels()
        counts = np.arange(0, 1200, dtype=np.int64)

        for count, index in zip(counts, band_of(counts), strict=True):
            low, high = _range_of(labels[index])
            assert low <= count <= high, f"{count} landed in band {labels[index]}"

    def test_the_bands_are_contiguous_and_start_at_zero(self) -> None:
        """A gap between bands would silently drop rows from the report; an
        overlap would count them twice."""
        ranges = [_range_of(label) for label in band_labels()]

        assert ranges[0][0] == 0
        for (_, previous_high), (low, _) in pairwise(ranges):
            assert low == previous_high + 1

    def test_there_is_one_more_band_than_edge(self) -> None:
        """Anything past the last edge needs somewhere to go."""
        assert len(band_labels()) == len(BAND_EDGES) + 1

    @pytest.mark.parametrize(
        "count,expected",
        [
            *[(0, "0"), (1, "1-2"), (2, "1-2"), (3, "3-5"), (5, "3-5"), (6, "6-10")],
            *[(100, "51-100"), (101, "101-500"), (500, "101-500"), (501, "501+")],
        ],
    )
    def test_the_boundaries_themselves(self, count: int, expected: str) -> None:
        """Both sides of every edge, spelled out, because an upper-inclusive
        rule read as lower-inclusive passes every aggregate check."""
        assert band_labels()[band_of(np.array([count]))[0]] == expected

    def test_a_cold_item_is_its_own_band(self) -> None:
        """Zero must not be swept in with 1-2. It is a quarter of the traffic
        and the only band a counting retriever cannot reach at all."""
        assert band_labels()[band_of(np.array([0]))[0]] == "0"
        assert band_labels()[band_of(np.array([1]))[0]] != "0"

    def test_labels_follow_the_edges_they_are_given(self) -> None:
        """Derived, not written beside the edges, so moving an edge moves the
        label with it."""
        assert band_labels((0, 3)) == ("0", "1-3", "4+")


class TestThePopularityReference:
    def test_it_selects_exactly_k_items(self) -> None:
        prior = torch.tensor([0, 9, 8, 7, 6, 5, 0, 0])
        items = np.arange(8)

        assert int(popularity_hits(prior, items, k=3).sum()) == 3

    def test_an_item_with_no_recent_clicks_cannot_be_reached(self) -> None:
        """The structural claim behind the cold band: a counting retriever
        ranks by a count, so a count of zero is unreachable whenever the
        top k is filled by items with positive counts."""
        prior = torch.tensor([0, 5, 4, 3, 0, 0])
        cold = np.array([4, 5])

        assert not popularity_hits(prior, cold, k=3).any()

    def test_the_reserved_row_never_wins_a_slot(self) -> None:
        """Index 0 is the OOV bucket and gets no clicks, so it must not be
        ranked into the top k just by sorting."""
        prior = torch.tensor([0, 3, 2, 1])

        assert not popularity_hits(prior, np.array([0]), k=3)[0]


class TestTheReport:
    @pytest.fixture
    def hits(self) -> Hits:
        return Hits(
            hit=torch.tensor([True, False, True, True, False, False]),
            item_ids=torch.tensor([1, 1, 2, 3, 4, 4]),
            user_ids=torch.tensor([7, 7, 8, 8, 9, 9]),
        )

    def test_the_bands_partition_the_rows(self, hits: Hits) -> None:
        """Every row in exactly one band, and the overall row counting all of
        them -- which is what makes the aggregate cross-checkable against what
        training reported."""
        band = np.array([0, 0, 1, 1, 2, 8])
        popularity = np.zeros(6, dtype=bool)

        rows = summarise(hits, band, popularity)
        banded = [row for row in rows if row.label != "overall"]
        overall = next(row for row in rows if row.label == "overall")

        assert sum(row.rows for row in banded) == overall.rows == 6

    def test_the_overall_recall_is_the_pooled_mean(self, hits: Hits) -> None:
        """NOT the mean of the band recalls, which would weight a 500-row band
        the same as a 5-row one."""
        band = np.array([0, 0, 1, 1, 2, 8])
        popularity = np.zeros(6, dtype=bool)

        overall = next(r for r in summarise(hits, band, popularity) if r.label == "overall")

        assert overall.recall == pytest.approx(3 / 6)

    def test_distinct_items_are_counted_not_rows(self, hits: Hits) -> None:
        """rows/item is what says whether a band is many items clicked once or
        few clicked often -- which decides whether it needs splitting."""
        band = np.array([0, 0, 0, 0, 0, 0])
        popularity = np.zeros(6, dtype=bool)

        first = summarise(hits, band, popularity)[0]

        assert first.rows == 6 and first.items == 4

    def test_an_empty_band_does_not_crash_the_report(self, hits: Hits) -> None:
        """Bands are fixed, so a corpus that happens to have none of some
        popularity is normal and must still render."""
        band = np.zeros(6, dtype=np.int64)
        popularity = np.zeros(6, dtype=bool)

        rows = summarise(hits, band, popularity)

        assert any(row.rows == 0 for row in rows)
        assert "nan" in render(rows, 100)

    def test_the_table_has_a_line_per_band_plus_overall(self, hits: Hits) -> None:
        rows = [BandRow("0", 1, 1, 0.5, 0.0), BandRow("overall", 1, 1, 0.5, 0.0)]

        assert len(render(rows, 100).splitlines()) == 5  # header, rule, band, rule, overall


def _items() -> ItemTables:
    return ItemTables(
        content=torch.randn(N_ITEMS + 1, CONTENT_DIM),
        category=torch.arange(N_ITEMS + 1) % 3,
        subcategory=torch.arange(N_ITEMS + 1) % 5,
        n_categories=2,
        n_subcategories=4,
    )


def _model(items: ItemTables, *, use_id: bool = True, use_content: bool = True) -> TwoTower:
    return TwoTower(
        content=items.content,
        item_category=items.category,
        item_subcategory=items.subcategory,
        n_user_feats=N_USER_FEATS,
        n_categories=items.n_categories,
        n_subcategories=items.n_subcategories,
        use_id=use_id,
        use_content=use_content,
    )


class TestRebuildingFromACheckpoint:
    @pytest.mark.parametrize(
        "use_id,use_content",
        [(True, True), (True, False), (False, True)],
        ids=["both", "id-only", "content-only"],
    )
    def test_the_arm_is_recovered_from_the_state_dict(
        self, tmp_path: Path, use_id: bool, use_content: bool
    ) -> None:
        """The flags are inferred from key names, so this fails the moment
        TwoTower renames item_id_emb or content -- which would otherwise
        silently score every arm as if it were the smallest one."""
        items = _items()
        path = tmp_path / "arm.pt"
        saved = _model(items, use_id=use_id, use_content=use_content)
        torch.save({"model": saved.state_dict()}, path)

        loaded = load_tower(path, items, N_USER_FEATS, select_device("cpu"))

        assert loaded.use_id is use_id
        assert loaded.use_content is use_content

    def test_a_mismatched_checkpoint_is_refused(self, tmp_path: Path) -> None:
        """strict=True is the guard standing in for the dimensions that are
        left at their defaults rather than read back."""
        items = _items()
        path = tmp_path / "wrong.pt"
        state = _model(items).state_dict()
        state["user_tower.net.0.weight"] = torch.randn(512, N_USER_FEATS + 99)
        torch.save({"model": state}, path)

        with pytest.raises(RuntimeError, match="size mismatch"):
            load_tower(path, items, N_USER_FEATS, select_device("cpu"))


class TestPerRowHits:
    @pytest.fixture
    def split(self) -> SplitTensors:
        generator = torch.Generator().manual_seed(7)
        history = torch.randint(1, N_ITEMS + 1, (ROWS, HISTORY), generator=generator)
        return SplitTensors(
            user_feats=torch.randn(ROWS, N_USER_FEATS, generator=generator),
            history_ids=history,
            history_mask=torch.ones(ROWS, HISTORY, dtype=torch.long),
            item_ids=history[:, 0].clone(),
            impression_ids=torch.arange(ROWS),
            user_ids=torch.arange(ROWS) // 4,
            neg_ids=torch.randint(1, N_ITEMS + 1, (ROWS, NEGS), generator=generator),
            neg_mask=torch.ones(ROWS, NEGS, dtype=torch.long),
        )

    def test_every_validation_row_is_scored_exactly_once(self, split: SplitTensors) -> None:
        """drop_last is False on the validation loader; a True would discard the
        final short batch and quietly shrink the denominator."""
        device = select_device("cpu")
        loader = make_loader(split, N_ITEMS, 5, device, training=False, history_dropout=0.0)

        hits = retrieval_hits(_model(_items()).to(device), loader, device, k=3)

        assert len(hits.hit) == ROWS

    def test_the_user_id_arrives_with_its_own_hit(self, split: SplitTensors) -> None:
        """What the paired comparison depends on. A hit attributed to the wrong
        user pairs two unrelated rows and the interval around the difference
        comes back clean anyway."""
        device = select_device("cpu")
        loader = make_loader(split, N_ITEMS, 5, device, training=False, history_dropout=0.0)

        hits = retrieval_hits(_model(_items()).to(device), loader, device, k=3)

        assert torch.equal(hits.item_ids, split.item_ids)
        assert torch.equal(hits.user_ids, split.user_ids)

    # "validate's recall is the mean of these rows" lives in test_train.py, with
    # the function it constrains. Asserting it here as well would be one contract
    # pinned in two files, which drift independently.

    def test_scoring_leaves_the_model_in_training_mode(self, split: SplitTensors) -> None:
        """Dropout left off improves the training loss, so the damage reads as
        success. Same trap validate already guards; retrieval_hits is where the
        guard now lives."""
        device = select_device("cpu")
        model = _model(_items()).to(device)
        model.train()
        loader = make_loader(split, N_ITEMS, 5, device, training=False, history_dropout=0.0)

        retrieval_hits(model, loader, device, k=3)

        assert model.training
