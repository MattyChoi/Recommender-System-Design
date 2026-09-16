"""The loop, on a synthetic corpus small enough to run 100 steps in a test.

§15's list for model tests: loss decreases, shapes and ranges hold, determinism
given a seed, no NaNs. Three additions that are specific to this design and
would each be silent:

``set_epoch`` -- without it ``DistributedSampler`` draws the identical
permutation every epoch, so the in-batch negatives never change composition and
the effective negative count is a fraction of the batch size.

Validation must not apply history dropout. It would shorten exactly the
histories the metric is meant to measure, and the number would still look fine.

Validation must not leave the model in eval mode. Dropout off improves training
loss, so the damage reads as success.

No Spark and no GPU: the tensors are built by hand.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader

from common.torch_env import deterministic, grad_scaler_for, select_device
from models.retrieval.batching import Batch
from models.retrieval.dataset import SplitTensors
from models.retrieval.sampling import StreamingLogQ
from models.retrieval.train import (
    Counters,
    _arm,
    _parser,
    fit,
    make_loader,
    run_epoch,
    train_step,
    validate,
)
from models.retrieval.two_tower import TwoTower

N_ITEMS = 12
CONTENT_DIM = 4
N_USER_FEATS = 3
HISTORY = 4
NEGS = 2
ROWS = 32
BATCH = 8


@pytest.fixture
def device() -> torch.device:
    return select_device("cpu")


@pytest.fixture
def split() -> SplitTensors:
    """A corpus where the clicked item is a function of the history.

    Learnable on purpose: a loss that fails to fall on random noise says nothing
    about whether the loop works.
    """
    generator = torch.Generator().manual_seed(7)
    history = torch.randint(1, N_ITEMS + 1, (ROWS, HISTORY), generator=generator)
    return SplitTensors(
        user_feats=torch.randn(ROWS, N_USER_FEATS, generator=generator),
        history_ids=history,
        history_mask=torch.ones(ROWS, HISTORY, dtype=torch.long),
        # The answer is the first history entry, so there is a signal to find.
        item_ids=history[:, 0].clone(),
        impression_ids=torch.arange(ROWS),
        neg_ids=torch.randint(1, N_ITEMS + 1, (ROWS, NEGS), generator=generator),
        neg_mask=torch.ones(ROWS, NEGS, dtype=torch.long),
    )


def _model() -> TwoTower:
    return TwoTower(
        content=torch.randn(N_ITEMS + 1, CONTENT_DIM),
        item_category=torch.arange(N_ITEMS + 1) % 3,
        item_subcategory=torch.arange(N_ITEMS + 1) % 5,
        n_user_feats=N_USER_FEATS,
        n_categories=2,
        n_subcategories=4,
        id_dim=8,
        cat_dim=4,
        subcat_dim=4,
        out_dim=8,
    )


def _counters() -> Counters:
    return Counters(StreamingLogQ(N_ITEMS, half_life=64.0), StreamingLogQ(N_ITEMS, half_life=64.0))


def _loader(
    split: SplitTensors, device: torch.device, *, training: bool, dropout: float = 0.0
) -> DataLoader[Batch]:
    return make_loader(
        split,
        N_ITEMS,
        BATCH,
        device,
        training=training,
        history_dropout=dropout,
        generator=torch.Generator().manual_seed(0),
    )


class TestTheStep:
    def test_a_hundred_steps_reduce_the_loss(
        self, split: SplitTensors, device: torch.device
    ) -> None:
        """§15's headline check. Compared over windows rather than first against
        last, so one lucky batch cannot decide it."""
        with deterministic(0):
            model, counters = _model(), _counters()
            optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
            scaler = grad_scaler_for(device)
            loader = _loader(split, device, training=True)

            losses = []
            for _ in range(25):  # 25 epochs x 4 batches = 100 steps
                for batch in loader:
                    losses.append(
                        float(
                            train_step(model, batch, counters, optimiser, scaler, device, N_ITEMS)
                        )
                    )

        assert sum(losses[:10]) / 10 > sum(losses[-10:]) / 10

    def test_no_step_produces_a_nan(self, split: SplitTensors, device: torch.device) -> None:
        """An -inf mask meeting a zero is the way this loss makes one, and a
        single NaN poisons every parameter on the next step."""
        with deterministic(0):
            model, counters = _model(), _counters()
            optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
            scaler = grad_scaler_for(device)

            for batch in _loader(split, device, training=True):
                loss = train_step(model, batch, counters, optimiser, scaler, device, N_ITEMS)
                assert torch.isfinite(loss)

            assert all(torch.isfinite(p).all() for p in model.parameters())

    def test_the_counters_see_the_batch_only_after_it_is_used(
        self, split: SplitTensors, device: torch.device
    ) -> None:
        """Query before update. Updating first would let a batch's own positives
        inflate their own correction."""
        with deterministic(0):
            model, counters = _model(), _counters()
            before = counters.positives.observed.clone()

            batch = next(iter(_loader(split, device, training=True)))
            train_step(
                model,
                batch,
                counters,
                torch.optim.Adam(model.parameters()),
                grad_scaler_for(device),
                device,
                N_ITEMS,
            )

        assert not torch.equal(counters.positives.observed, before)
        assert float(counters.positives.observed.sum()) == pytest.approx(float(BATCH))

    def test_a_seed_reproduces_the_run(self, split: SplitTensors, device: torch.device) -> None:
        """An ablation whose arms differ by initialisation is not an ablation."""
        results = []
        for _ in range(2):
            with deterministic(0):
                model, counters = _model(), _counters()
                optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
                scaler = grad_scaler_for(device)
                results.append(
                    [
                        float(train_step(model, b, counters, optimiser, scaler, device, N_ITEMS))
                        for b in _loader(split, device, training=True)
                    ]
                )

        assert results[0] == pytest.approx(results[1])


class TestValidation:
    def test_recall_is_a_fraction(self, split: SplitTensors, device: torch.device) -> None:
        with deterministic(0):
            got = validate(_model(), _loader(split, device, training=False), device, k=5)

        assert 0.0 <= got <= 1.0

    def test_a_full_k_finds_everything(self, split: SplitTensors, device: torch.device) -> None:
        """k = the whole catalogue must recall every held-out click. If it does
        not, the top-k indices are being mapped back to item ids wrongly -- the
        off-by-one that the reserved row 0 invites."""
        with deterministic(0):
            got = validate(_model(), _loader(split, device, training=False), device, k=N_ITEMS)

        assert got == 1.0

    def test_it_leaves_the_model_in_the_mode_it_found(
        self, split: SplitTensors, device: torch.device
    ) -> None:
        """Returning in eval would disable dropout for the rest of training, and
        the resulting lower loss reads as the validation loop working."""
        with deterministic(0):
            model = _model()
            model.train()
            validate(model, _loader(split, device, training=False), device, k=5)

        assert model.training

    def test_dropout_is_a_property_of_the_loader_not_of_the_caller(
        self, split: SplitTensors, device: torch.device
    ) -> None:
        """Two loaders, two settings, nothing to remember at the call site.

        The training one must shorten histories and the validation one must not
        -- otherwise the metric is computed on inputs that were deliberately
        degraded, and the number still looks perfectly reasonable.
        """
        with deterministic(0):
            trimmed = next(iter(_loader(split, device, training=True, dropout=1.0)))
            intact = next(iter(_loader(split, device, training=False)))

        assert int(trimmed.history_mask.sum()) < trimmed.history_mask.numel()
        assert torch.equal(intact.history_mask, split.history_mask[: len(intact.item_ids)])

    def test_the_validation_loader_keeps_every_row(
        self, split: SplitTensors, device: torch.device
    ) -> None:
        """drop_last on validation would silently discard held-out requests."""
        seen = sum(len(b.item_ids) for b in _loader(split, device, training=False))

        assert seen == ROWS


class TestTheLogQAblation:
    """G2's gate is "with and without the correction", so "without" has to be
    reachable and has to mean exactly zero."""

    def test_disabling_it_gives_every_column_zero(self, device: torch.device) -> None:
        """Zero, not "some small number": a per-row constant is invisible to
        softmax, so zeros ARE no correction, exactly."""
        counters = Counters(StreamingLogQ(N_ITEMS), StreamingLogQ(N_ITEMS), corrected=False)
        positive, negative = counters.log_q_for(
            torch.tensor([1, 2]),
            torch.tensor([3, 4, 5, 6]),
            torch.ones(4, dtype=torch.bool),
            N_ITEMS,
            device,
        )

        assert torch.equal(positive, torch.zeros(2))
        assert torch.equal(negative, torch.zeros(4))

    def test_disabling_it_stops_the_counters_moving(self, device: torch.device) -> None:
        """Nothing reads them in this arm, so counting would be work with no
        consumer -- and two gathers per step of it."""
        counters = Counters(StreamingLogQ(N_ITEMS), StreamingLogQ(N_ITEMS), corrected=False)
        counters.observe(
            torch.tensor([1, 2]), torch.tensor([3, 4]), torch.ones(2, dtype=torch.bool)
        )

        assert float(counters.positives.observed.sum()) == 0.0

    def test_the_uniform_fills_take_the_closed_form(self, device: torch.device) -> None:
        """A slate that ran short is topped up with a uniform draw, and that
        column's q is exactly 1/n_items rather than the slate counter's guess."""
        counters = Counters(StreamingLogQ(N_ITEMS), StreamingLogQ(N_ITEMS))
        _, negative = counters.log_q_for(
            torch.tensor([1]),
            torch.tensor([2, 3]),
            torch.tensor([True, False]),
            N_ITEMS,
            device,
        )

        assert float(negative[1]) == pytest.approx(-math.log(N_ITEMS))
        assert float(negative[0]) != pytest.approx(-math.log(N_ITEMS))


class TestTheCommandLine:
    """Every remaining Part G deliverable is one invocation with different flags,
    so the flags reaching the right place is worth a test of its own."""

    def test_the_defaults_are_the_full_model_with_the_correction(self) -> None:
        args = _parser().parse_args([])

        assert args.use_id and args.use_content and args.logq

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            ([], "both-logq-n4u0"),
            (["--no-logq"], "both-nologq-n4u0"),
            (["--no-use-content"], "id-logq-n4u0"),
            (["--no-use-id"], "content-logq-n4u0"),
            (["--max-negs", "0", "--uniform-negs", "4"], "both-logq-n0u4"),
        ],
    )
    def test_each_arm_gets_a_distinct_name(self, flags: list[str], expected: str) -> None:
        """The run name reaches MLflow and the checkpoint filename, so two arms
        sharing one would overwrite each other's artifact."""
        assert _arm(_parser().parse_args(flags)) == expected


class TestFit:
    def test_it_records_one_entry_per_epoch(
        self, split: SplitTensors, device: torch.device
    ) -> None:
        with deterministic(0):
            history = fit(
                _model(),
                _loader(split, device, training=True),
                _loader(split, device, training=False),
                _counters(),
                device,
                N_ITEMS,
                epochs=3,
                patience=99,
                k=5,
            )

        assert len(history) == 3
        assert all(math.isfinite(row["loss"]) for row in history)
        assert all(0.0 <= row["recall@5"] <= 1.0 for row in history)

    def test_it_stops_when_recall_stops_improving(
        self, split: SplitTensors, device: torch.device
    ) -> None:
        """Early stopping is on RECALL, not loss: loss keeps falling well past
        the point retrieval quality stops."""
        with deterministic(0):
            history = fit(
                _model(),
                _loader(split, device, training=True),
                _loader(split, device, training=False),
                _counters(),
                device,
                N_ITEMS,
                epochs=50,
                patience=1,
                k=5,
            )

        assert len(history) < 50

    def test_an_epoch_runs_without_a_distributed_sampler(
        self, split: SplitTensors, device: torch.device
    ) -> None:
        """Single-process there is no sampler to hand the epoch to, so what this
        pins is that ``run_epoch`` tolerates its absence. That the epoch reaches
        a real ``DistributedSampler`` is a multi-rank property and belongs with
        the multi-rank tests.
        """
        with deterministic(0):
            model = _model()
            loss = run_epoch(
                model,
                _loader(split, device, training=True),
                _counters(),
                torch.optim.Adam(model.parameters()),
                grad_scaler_for(device),
                device,
                N_ITEMS,
                epoch=3,
            )

        assert math.isfinite(loss)
