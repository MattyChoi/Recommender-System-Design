"""Off-policy evaluation: what a NEW policy would have scored on an OLD log.

Every offline metric in this repo is computed on impressions some other policy
chose to show. That mismatch is the fundamental problem of offline recommender
evaluation, and importance weighting is the honest answer to it: reweight each
logged outcome by how much more likely the target policy was to take that
action than the policy that actually ran.

The weight is ``target_prob / logging_prob``, and three things about it decide
whether the number means anything:

* **Common support.** A zero logging probability means the action was
  impossible under the logged policy, so the log carries no evidence about it
  and the estimate is undefined -- not large, undefined. The usual formulation
  (``np.clip(target / logging, 0, clip)``) turns that division into ``inf`` and
  then clips it to the MAXIMUM weight, handing the most influence to exactly
  the rows that contain the least information. This module refuses instead.
* **Clipping is a bias/variance trade, and the bias is one-directional.**
  Capping weights at 10 bounds the variance and systematically understates any
  action the target policy favours far more than the logger did. So the cap is
  not enough: the share of weight mass sitting AT the cap is reported, because
  an estimate with a third of its mass clipped is not the estimator anyone
  thinks they are reading.
* **Effective sample size.** Ten thousand rows whose weight is concentrated on
  forty of them is a forty-row experiment. ESS is returned with every estimate
  rather than offered as a separate call, because the manual's advice to
  "report it alongside EVERY SNIPS number" is advice a caller can forget.

⚠️ **A deterministic log cannot support this at all.** If every logged action
had probability 1.0 the policy never explored, so the log holds no evidence
about any action it did not take and SNIPS returns the naive average of the
rewards -- the same number, wearing an IPS label. This repo logged exactly that
until the reranker's exploration flags were wired up (see
``claude/part-o-monitoring.md``), so the condition is detected and reported
rather than assumed away. Note that this is narrower than "the propensities are
all equal": a 50/50 randomisation logs 0.5 throughout and supports IPS fine.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

#: Default weight cap. Conventional rather than derived: 10 means a single
#: logged row may stand in for at most ten, which bounds the variance without
#: discarding the tail entirely.
DEFAULT_CLIP = 10.0

#: Below this share of rows, the estimate is driven by a handful of logs and
#: should not be quoted as a point estimate. A rule of thumb, reported rather
#: than enforced.
MIN_ESS_FRACTION = 0.1


class OffPolicy(NamedTuple):
    """An off-policy estimate and everything needed to distrust it.

    Attributes:
        snips: Self-normalised IPS. The headline number.
        ips: Vanilla IPS over the same weights, for comparison. Unbiased but
            higher variance; a large gap between the two means the weights are
            poorly behaved.
        effective_sample_size: Kish's ESS of the weights actually used.
        ess_fraction: ESS over rows used.
        rows: Rows that contributed.
        dropped_unsupported: Rows removed for a non-positive logging
            probability, when dropping was requested.
        clipped: Weights that hit the cap.
        clipped_mass: Share of total weight sitting at the cap, in [0, 1].
        deterministic_logging: Every logged action had probability 1.0, so the
            policy never explored and the log holds no evidence about any
            action it did not take. NOT merely "the propensities are equal" --
            a 50/50 randomisation logs 0.5 throughout and supports IPS fine.
    """

    snips: float
    ips: float
    effective_sample_size: float
    ess_fraction: float
    rows: int
    dropped_unsupported: int
    clipped: int
    clipped_mass: float
    deterministic_logging: bool

    @property
    def reliable(self) -> bool:
        """Whether this is worth quoting as a point estimate."""
        return (
            self.ess_fraction >= MIN_ESS_FRACTION
            and not self.deterministic_logging
            and self.clipped_mass < 0.25
        )

    def summary(self) -> str:
        warnings = []
        if self.deterministic_logging:
            warnings.append("DETERMINISTIC LOG: no counterfactual evidence")
        if self.ess_fraction < MIN_ESS_FRACTION:
            warnings.append(f"ESS {self.ess_fraction:.1%} of rows")
        if self.clipped_mass >= 0.25:
            warnings.append(f"{self.clipped_mass:.0%} of weight mass at the cap")
        if self.dropped_unsupported:
            warnings.append(f"{self.dropped_unsupported:,} unsupported rows dropped")

        line = (
            f"SNIPS={self.snips:.4f} (IPS={self.ips:.4f}) "
            f"ESS={self.effective_sample_size:.0f}/{self.rows:,}"
        )
        return line if not warnings else f"{line} -- " + "; ".join(warnings)


def effective_sample_size(weights: npt.ArrayLike) -> float:
    """Kish's effective sample size: ``sum(w)^2 / sum(w^2)``.

    Equal weights give back the row count; weight concentrated on one row gives
    1.0. This is the number that says whether an estimate over ten thousand
    logged impressions is really an estimate over forty of them.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return 0.0
    denominator = float((w**2).sum())
    if denominator == 0.0:
        return 0.0
    return float(w.sum() ** 2 / denominator)


def snips(
    rewards: npt.ArrayLike,
    logging_probs: npt.ArrayLike,
    target_probs: npt.ArrayLike,
    clip: float = DEFAULT_CLIP,
    drop_unsupported: bool = False,
) -> OffPolicy:
    """Estimate the target policy's value from a log written by another policy.

    Args:
        rewards: Observed outcome per logged row -- a click, a dwell, a
            objective value.
        logging_probs: Probability the LOGGING policy took the action it took.
            For this repo that is ``ImpressionContext.propensity``, which is
            ``rerank.Deterministic`` on exploited slots and ``epsilon / n`` on
            explored ones.
        target_probs: Probability the TARGET policy would take that same
            action.
        clip: Weight cap. See :data:`DEFAULT_CLIP`.
        drop_unsupported: Drop rows whose logging probability is non-positive
            instead of refusing. The count is reported either way, and this
            changes the estimand to the logging policy's support -- which is
            a real choice, not a cleanup.

    Returns:
        The estimate and its diagnostics.

    Raises:
        ValueError: On a length mismatch, an empty input, a non-positive clip,
            a negative target probability, or an unsupported row when
            ``drop_unsupported`` is False.

    Note:
        A non-positive logging probability is refused rather than clipped. The
        usual formulation divides by it, gets ``inf``, and clips that to the
        maximum weight -- so a row recording something the logging policy could
        never have done ends up with more influence than any row that actually
        happened. The failure is silent and it inflates whatever the target
        policy happens to favour.
    """
    reward = np.asarray(rewards, dtype=np.float64)
    logging = np.asarray(logging_probs, dtype=np.float64)
    target = np.asarray(target_probs, dtype=np.float64)

    if not (reward.shape == logging.shape == target.shape):
        raise ValueError(
            f"rewards {reward.shape}, logging_probs {logging.shape}, "
            f"target_probs {target.shape} must match"
        )
    if reward.ndim != 1:
        raise ValueError(f"expected one value per logged row, got shape {reward.shape}")
    if reward.size == 0:
        raise ValueError("no rows to estimate from")
    if clip <= 0.0:
        raise ValueError(f"clip must be positive, got {clip}")
    if np.any(target < 0.0):
        raise ValueError("target_probs contains a negative probability")

    unsupported = int(np.count_nonzero(logging <= 0.0))
    if unsupported and not drop_unsupported:
        raise ValueError(
            f"{unsupported:,} of {logging.size:,} rows have a logging probability <= 0. "
            "The logging policy could not have taken those actions, so the log carries no "
            "evidence about them and the estimate is undefined. Pass drop_unsupported=True "
            "to estimate over the logging policy's support instead -- a different estimand, "
            "deliberately chosen."
        )

    if unsupported:
        keep = logging > 0.0
        reward, logging, target = reward[keep], logging[keep], target[keep]
        if reward.size == 0:
            raise ValueError("every row was unsupported; nothing to estimate from")

    raw = target / logging
    weights = np.minimum(raw, clip)

    total_weight = float(weights.sum())
    if total_weight == 0.0:
        # Every target probability is zero: the target policy would never have
        # done any of this. Not an error, but not an estimate either.
        return OffPolicy(
            snips=float("nan"),
            ips=0.0,
            effective_sample_size=0.0,
            ess_fraction=0.0,
            rows=int(reward.size),
            dropped_unsupported=unsupported,
            clipped=0,
            clipped_mass=0.0,
            deterministic_logging=bool(np.all(logging >= 1.0)),
        )

    clipped_mask = raw > clip
    ess = effective_sample_size(weights)

    return OffPolicy(
        snips=float((weights * reward).sum() / total_weight),
        ips=float((weights * reward).mean()),
        effective_sample_size=ess,
        ess_fraction=ess / reward.size,
        rows=int(reward.size),
        dropped_unsupported=unsupported,
        clipped=int(np.count_nonzero(clipped_mask)),
        clipped_mass=float(weights[clipped_mask].sum() / total_weight),
        # 1.0 throughout means the logger never explored. Equal-but-not-one
        # propensities are a working randomisation and are fine.
        deterministic_logging=bool(np.all(logging >= 1.0)),
    )
