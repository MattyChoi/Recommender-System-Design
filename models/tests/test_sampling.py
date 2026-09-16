"""The frequency counter, and what its defaults quietly assume.

Two properties are load-bearing and neither is visible in a training curve.

The estimate must be a real distribution -- ``exp(log_q)`` summing to one over
the whole table -- because the correction subtracts it from a logit. An estimate
off by a constant factor is a per-row shift that softmax cancels, so a broken
normaliser looks exactly like no correction at all.

And the prior must survive the decay. Decaying the Laplace mass along with the
observations means a never-sampled item drifts toward ``log q = -inf``,
collecting an unbounded correction it earned no evidence for. On this corpus
that is not hypothetical: C2 measured ~63% of the item table never appearing in
an impression.
"""

from __future__ import annotations

import pytest
import torch

from common.torch_env import select_device
from models.retrieval.sampling import StreamingLogQ, uniform_log_q, uniform_negatives

N_ITEMS = 8


@pytest.fixture
def counter() -> StreamingLogQ:
    return StreamingLogQ(N_ITEMS, half_life=100.0).to(select_device("cpu"))


class TestTheDistribution:
    def test_it_normalises_over_the_whole_table(self, counter: StreamingLogQ) -> None:
        counter.update(torch.tensor([1, 1, 2, 5]))
        mass = counter.log_q(torch.arange(N_ITEMS + 1)).exp().sum()
        assert torch.allclose(mass, torch.ones(()), atol=1e-6)

    def test_a_frequent_item_outranks_a_rare_one(self, counter: StreamingLogQ) -> None:
        counter.update(torch.tensor([1, 1, 1, 1, 2]))
        seen = counter.log_q(torch.tensor([1, 2]))
        assert seen[0] > seen[1]

    def test_an_unseen_item_gets_exactly_the_prior(self, counter: StreamingLogQ) -> None:
        """Finite and bounded below, which is the whole reason the prior is held
        out of the decay."""
        counter.update(torch.tensor([1, 1, 1]))
        total = counter.prior * (N_ITEMS + 1) + counter.observed.sum()
        expected = torch.log(torch.tensor(counter.prior) / total)
        assert torch.allclose(counter.log_q(torch.tensor([7]))[0], expected)

    def test_the_prior_does_not_decay_away(self) -> None:
        """A short half-life and a long stream: the floor must hold.

        With the manual's shared decay this value falls without bound.
        """
        counter = StreamingLogQ(N_ITEMS, half_life=1.0)
        for _ in range(50):
            counter.update(torch.tensor([1, 2]))
        unseen = counter.log_q(torch.tensor([7]))[0]
        assert torch.isfinite(unseen)
        assert unseen > torch.log(torch.tensor(1e-4))


class TestTheDecay:
    def test_it_is_per_example_not_per_call(self) -> None:
        """The same number of examples must fade an observation by the same
        factor however they are batched, or the half-life means something
        different at every batch size -- which is the defect in the manual's
        per-step ``decay``.

        Note this is NOT the same as two half-batches leaving the same state as
        one whole batch: the first half decays before the second is counted, so
        the counts themselves are order-dependent by construction. What has to
        be invariant is the decay applied to what came before.
        """
        whole = StreamingLogQ(N_ITEMS, half_life=10.0)
        split = StreamingLogQ(N_ITEMS, half_life=10.0)
        whole.update(torch.tensor([1]))
        split.update(torch.tensor([1]))

        whole.update(torch.tensor([2, 2, 2, 2]))
        split.update(torch.tensor([2, 2]))
        split.update(torch.tensor([2, 2]))

        assert float(whole.observed[1]) == pytest.approx(float(split.observed[1]), rel=1e-6)

    def test_an_old_observation_fades(self) -> None:
        counter = StreamingLogQ(N_ITEMS, half_life=4.0)
        counter.update(torch.tensor([1]))
        before = float(counter.observed[1])
        counter.update(torch.tensor([2, 2, 2, 2]))
        assert float(counter.observed[1]) == pytest.approx(before * 0.5, rel=1e-5)

    def test_a_nonpositive_half_life_raises(self) -> None:
        with pytest.raises(ValueError, match="half_life"):
            StreamingLogQ(N_ITEMS, half_life=0.0)


class TestTheOrdering:
    def test_querying_first_does_not_see_the_batch(self, counter: StreamingLogQ) -> None:
        """The point-in-time rule, applied to the counter: a batch must not
        condition its own correction on its own labels."""
        batch = torch.tensor([3, 3, 3])
        before = counter.log_q(batch)
        counter.update(batch)
        assert not torch.allclose(before, counter.log_q(batch))


class TestUniformNegatives:
    def test_the_reserved_index_is_never_drawn(self) -> None:
        """Index 0 is OOV and padding (B2). Drawn as a negative it would teach
        the model to push the padding vector away from every user, which is a
        direction with no meaning."""
        device = select_device("cpu")
        drawn = uniform_negatives((64, 8), N_ITEMS, device)

        assert int(drawn.min()) >= 1
        assert int(drawn.max()) <= N_ITEMS

    def test_a_seeded_generator_repeats(self) -> None:
        device = select_device("cpu")
        first = uniform_negatives(
            (16, 4), N_ITEMS, device, torch.Generator(device=device).manual_seed(0)
        )
        second = uniform_negatives(
            (16, 4), N_ITEMS, device, torch.Generator(device=device).manual_seed(0)
        )

        assert torch.equal(first, second)

    def test_the_log_q_is_exact_rather_than_estimated(self) -> None:
        """A uniform draw's probability is known in closed form, so running it
        through the streaming counter would estimate a constant -- badly, early
        in training, when the counter has seen almost nothing."""
        device = select_device("cpu")
        got = uniform_log_q((3, 2), N_ITEMS, device)

        assert torch.allclose(got.exp(), torch.full((3, 2), 1.0 / N_ITEMS), atol=1e-6)


class TestTheState:
    def test_the_counts_ride_the_checkpoint(self, counter: StreamingLogQ) -> None:
        """A bare attribute would restore a resumed run to a flat prior."""
        counter.update(torch.tensor([1, 1, 4]))
        restored = StreamingLogQ(N_ITEMS, half_life=100.0)
        restored.load_state_dict(counter.state_dict())
        assert torch.allclose(restored.observed, counter.observed)

    def test_the_counts_follow_a_device_move(self, counter: StreamingLogQ) -> None:
        """Registered as a buffer so `.to()` moves it, which is what removes the
        two host round-trips the manual's version does on every step."""
        assert "observed" in dict(counter.named_buffers())
        device = select_device("cpu")
        assert counter.to(device).observed.device.type == device.type
