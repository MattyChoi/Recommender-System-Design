"""A simulated A/B test: sessions in, per-arm outcomes out.

⚠️ **This is a simulator, and a simulator cannot falsify its own assumptions.**
Everything below encodes a belief about how users behave -- that examination
decays with slot, that relevance is multiplicative, that a better ranker raises
relevance uniformly. A result produced here is a statement about the MODEL,
and its only honest use is to exercise the decision machinery: the SRM guard,
the variance reduction, the confidence interval, the power calculation and the
launch rule. It is not evidence that a policy is better.

Two of the assumptions are known to be wrong for this corpus and are kept on
purpose, because the decision machinery still has to run somewhere:

* **The examination curve.** MIND shuffles its impression lists, so the logged
  ``slot`` carries no position bias (the standing departure that made K4
  replace §12.4's correction). A decaying examination curve is therefore
  imported from the literature, not measured here. Changing ``gamma`` changes
  how clicks spread across slots and barely moves the session CTR, which is
  calibrated separately -- so conclusions about the PRIMARY metric are robust
  to it and conclusions about slot-level behaviour are not.
* **Uniform relevance lift.** A real ranker improvement helps some users and
  hurts others; this one multiplies every user's relevance by the same factor.
  That makes the simulated effect cleaner than any real one, so the power
  figures here are optimistic.

What IS calibrated from the repo's own data: the slate size, the session click
rate, and the item popularity distribution the slates are drawn from -- all
read from the served impression log rather than invented. See
``scripts/writeups/simulate_ab.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

#: Users drawn when integrating the click equation over the population. Fixed
#: count and fixed seed, so calibration is reproducible and cacheable.
_CALIBRATION_USERS = 10_000
_CALIBRATION_SEED = 0x5EED

#: Slot examination decay. ``1 / (slot + 1) ** GAMMA``, slots 0-indexed.
#: Imported, not measured -- see the module docstring.
DEFAULT_GAMMA = 0.8

#: Spread of per-user propensity to click, as the sigma of a lognormal that is
#: then normalised to mean 1. Heterogeneity is what makes CUPED worth anything:
#: with identical users there is no pre-period variance to strip.
DEFAULT_USER_SIGMA = 0.6

#: How faithfully the pre-period covariate reflects a user's true propensity.
#: 1.0 would be a covariate that predicts the outcome perfectly, which no real
#: pre-period does.
DEFAULT_COVARIATE_FIDELITY = 0.75


@dataclass(frozen=True)
class Corpus:
    """Structural parameters, calibrated from the served log.

    Attributes:
        slate_size: Items per impression.
        session_ctr: Probability a control session produces at least one
            click. The quantity the primary metric measures.
        item_weights: Relative exposure per item under the control policy,
            summing to 1. Drawn from the real log's item frequencies.
        head_size: Items counted as the head when computing long-tail share.
        latency_p50_ms: Median request latency under control.
        latency_sigma: Lognormal sigma giving the measured p99.
    """

    slate_size: int
    session_ctr: float
    item_weights: npt.NDArray[np.float64]
    head_size: int
    latency_p50_ms: float
    latency_sigma: float

    def __post_init__(self) -> None:
        if self.slate_size < 1:
            raise ValueError(f"slate_size must be >= 1, got {self.slate_size}")
        if not 0.0 < self.session_ctr < 1.0:
            raise ValueError(f"session_ctr must be in (0, 1), got {self.session_ctr}")
        if self.item_weights.ndim != 1 or self.item_weights.size < self.slate_size:
            raise ValueError("item_weights must be a 1-D array at least slate_size long")
        if not np.isclose(self.item_weights.sum(), 1.0):
            raise ValueError(f"item_weights sum to {self.item_weights.sum()}, not 1")

    def head_mask(self) -> npt.NDArray[np.bool_]:
        """True for the ``head_size`` most-exposed items, by POPULARITY RANK.

        Not by item index. Indices come from ``item_map``, which orders items
        by nothing at all -- so comparing an index against a count asks
        whether an arbitrary identifier happens to be small, and on a 65,238
        item catalogue the answer is "almost never", which reads as a
        long-tail share of 1.000 and a guardrail that cannot move.
        """
        order = np.argsort(self.item_weights)[::-1]
        mask = np.zeros(self.item_weights.size, dtype=bool)
        mask[order[: self.head_size]] = True
        return mask


@dataclass(frozen=True)
class Policy:
    """One arm's behaviour, as a departure from control.

    Attributes:
        name: Arm label, as the assignment service would report it.
        relevance_lift: Multiplier on relevance. 0.0 is control.
        epsilon: Share of slots filled uniformly at random rather than by the
            popularity order. Drives the propensities the off-policy
            estimators need, and widens exposure.
        latency_ms_shift: Added to the median latency. A model that ranks
            better and costs 3 ms is a real trade and the guardrail has to see
            it.
    """

    name: str
    relevance_lift: float = 0.0
    epsilon: float = 0.0
    latency_ms_shift: float = 0.0


class Arm(NamedTuple):
    """Everything one arm produced, per session.

    Attributes:
        name: Arm label.
        clicked: Whether each session produced at least one click.
        latency_ms: Per-session request latency.
        pre_ctr: The pre-experiment covariate, for CUPED. Measured BEFORE
            assignment by construction -- it is a function of the user's own
            propensity and nothing the policy touched.
        items: Slates, shape ``(sessions, slate_size)``.
        logging_prob: Probability THIS policy chose each slot's item.
        counterfactual_prob: Probability the OTHER policy would have chosen
            that same item, so the off-policy estimators have a target.
    """

    name: str
    clicked: npt.NDArray[np.bool_]
    latency_ms: npt.NDArray[np.float64]
    pre_ctr: npt.NDArray[np.float64]
    items: npt.NDArray[np.int64]
    logging_prob: npt.NDArray[np.float64]
    counterfactual_prob: npt.NDArray[np.float64]

    @property
    def sessions(self) -> int:
        return int(self.clicked.size)

    @property
    def ctr(self) -> float:
        return float(self.clicked.mean())

    def distinct_items(self) -> int:
        return int(np.unique(self.items).size)

    def long_tail_share(self, head: npt.NDArray[np.bool_]) -> float:
        """Share of impressions landing outside the most-exposed items.

        Takes the MASK rather than a count, so the caller cannot accidentally
        compare an item index against a number of items -- which is a
        well-typed expression that means nothing. See ``Corpus.head_mask``.
        """
        return float(np.count_nonzero(~head[self.items]) / self.items.size)


def examination(slate_size: int, gamma: float = DEFAULT_GAMMA) -> npt.NDArray[np.float64]:
    """Probability each slot is looked at, by position."""
    return (1.0 / (np.arange(slate_size, dtype=np.float64) + 1.0)) ** gamma


def _user_factors(user_sigma: float, size: int, seed: int) -> npt.NDArray[np.float64]:
    """Per-user click propensity, lognormal and normalised to mean 1."""
    rng = np.random.default_rng(seed)
    factors = rng.lognormal(0.0, user_sigma, size)
    return factors / factors.mean()


@cache
def calibrate_relevance(
    session_ctr: float,
    slate_size: int,
    gamma: float = DEFAULT_GAMMA,
    user_sigma: float = DEFAULT_USER_SIGMA,
) -> float:
    """The per-item relevance that reproduces an observed session click rate.

    Solves ``mean_u[1 - prod(1 - clip(e_k * r * u))] = session_ctr`` for r by
    bisection, so the simulator's headline rate IS the measured one rather
    than something near it.

    **Integrated over the user distribution, not evaluated at the average
    user.** ``P(click | u)`` is concave in u, so the population mean sits below
    the value at the mean -- Jensen. Calibrating on the mean user overstates
    the realised rate by about four points at a 50% CTR and by almost nothing
    at 2%, which is precisely the error that passes every test written at a
    low baseline and surfaces the first time someone raises it.

    The integration uses a fixed sample at a fixed seed, so this stays a pure
    function of its arguments rather than of whichever generator the caller
    happens to be holding. Cached, because ``simulate_arm`` asks for the same
    answer on every call.

    Raises:
        ValueError: If no relevance reaches the target. With heterogeneous
            users there IS a real ceiling below 1.0 -- a user with a low
            factor cannot be made to click reliably however relevant the
            items are -- so this fires rather than silently returning r = 1.
    """
    if not 0.0 < session_ctr < 1.0:
        raise ValueError(f"session_ctr must be in (0, 1), got {session_ctr}")

    weights = examination(slate_size, gamma)[None, :]
    users = _user_factors(user_sigma, _CALIBRATION_USERS, _CALIBRATION_SEED)[:, None]

    def ctr_at(relevance: float) -> float:
        per_slot = np.clip(weights * relevance * users, 0.0, 1.0)
        return float((1.0 - np.prod(1.0 - per_slot, axis=1)).mean())

    low, high = 0.0, 1.0
    ceiling = ctr_at(high)
    if ceiling < session_ctr:
        raise ValueError(
            f"a session CTR of {session_ctr:.4f} is unreachable with {slate_size} slots "
            f"at gamma={gamma} and user_sigma={user_sigma}; the ceiling is {ceiling:.4f}"
        )
    for _ in range(60):
        middle = (low + high) / 2.0
        if ctr_at(middle) < session_ctr:
            low = middle
        else:
            high = middle
    return high


def simulate_arm(
    policy: Policy,
    other: Policy,
    corpus: Corpus,
    sessions: int,
    rng: np.random.Generator,
    gamma: float = DEFAULT_GAMMA,
    user_sigma: float = DEFAULT_USER_SIGMA,
    covariate_fidelity: float = DEFAULT_COVARIATE_FIDELITY,
) -> Arm:
    """Run one arm.

    Args:
        policy: The arm being simulated.
        other: The arm it is being compared against, so each logged slot can
            carry the probability the OTHER policy would have chosen it --
            which is what the off-policy estimators consume.
        corpus: Calibrated structure.
        sessions: Sessions in this arm.
        rng: Seeded generator. The caller owns the seed so a whole experiment
            is reproducible from one number.
        gamma: Examination decay.
        user_sigma: Spread of per-user click propensity.
        covariate_fidelity: Correlation between the pre-period covariate and
            the user's true propensity.

    Returns:
        The arm's per-session record.
    """
    if sessions < 1:
        raise ValueError(f"sessions must be >= 1, got {sessions}")

    catalogue = corpus.item_weights.size
    slate = corpus.slate_size

    # --- users: heterogeneous propensity, and a pre-period view of it -------
    #
    # The covariate is built from the SAME latent factor and nothing else, so
    # it is pre-assignment by construction. A simulator that let the treatment
    # touch it would be modelling the mistake CUPED exists to avoid.
    latent = rng.lognormal(0.0, user_sigma, sessions)
    latent /= latent.mean()  # mean 1, so the calibrated relevance is the scale
    noise = rng.lognormal(0.0, user_sigma, sessions)
    noise /= noise.mean()
    pre_ctr = covariate_fidelity * latent + (1.0 - covariate_fidelity) * noise

    # --- slates: popularity, diluted by this policy's exploration -----------
    exploit = rng.choice(catalogue, size=(sessions, slate), p=corpus.item_weights)
    explore = rng.integers(0, catalogue, size=(sessions, slate))
    explored = rng.random((sessions, slate)) < policy.epsilon
    items = np.where(explored, explore, exploit)

    # Propensity of the item actually shown, under each policy. Exploitation
    # contributes the item's own weight; exploration contributes 1/catalogue.
    weights_of_items = corpus.item_weights[items]
    uniform = 1.0 / catalogue

    def probability(epsilon: float) -> npt.NDArray[np.float64]:
        return (1.0 - epsilon) * weights_of_items + epsilon * uniform

    logging_prob = probability(policy.epsilon)
    counterfactual_prob = probability(other.epsilon)

    # --- clicks -------------------------------------------------------------
    # user_sigma is passed through: the realised rate depends on how spread
    # the population is, so a calibration that ignored it would be solving a
    # different equation from the one this function then simulates.
    relevance = calibrate_relevance(corpus.session_ctr, slate, gamma, user_sigma)
    per_slot = examination(slate, gamma)[None, :] * relevance
    per_slot = per_slot * latent[:, None] * (1.0 + policy.relevance_lift)
    clicks = rng.random((sessions, slate)) < np.clip(per_slot, 0.0, 1.0)

    # --- latency ------------------------------------------------------------
    median = corpus.latency_p50_ms + policy.latency_ms_shift
    latency = rng.lognormal(np.log(median), corpus.latency_sigma, sessions)

    return Arm(
        name=policy.name,
        clicked=clicks.any(axis=1),
        latency_ms=latency,
        pre_ctr=pre_ctr,
        items=items.astype(np.int64),
        logging_prob=logging_prob,
        counterfactual_prob=counterfactual_prob,
    )


def simulate(
    control: Policy,
    treatment: Policy,
    corpus: Corpus,
    sessions_per_arm: int,
    seed: int = 0,
    gamma: float = DEFAULT_GAMMA,
) -> tuple[Arm, Arm]:
    """Run both arms from one seed.

    Allocation is exactly equal rather than randomised: an SRM check belongs
    on REAL assignment counts, and simulating an imbalance here would be
    testing the random number generator. ``evaluation.experiments.srm`` is run
    against the live ``recsys_experiment_assignments_total`` instead.
    """
    rng = np.random.default_rng(seed)
    return (
        simulate_arm(control, treatment, corpus, sessions_per_arm, rng, gamma),
        simulate_arm(treatment, control, corpus, sessions_per_arm, rng, gamma),
    )
