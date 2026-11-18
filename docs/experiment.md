# The launch decision

A simulated A/B test of a ranker change, and the recommendation it supports.
Regenerate with `make simulate`. The block below is generated and must not be
hand-edited.

## What this is, and what it is not

**It is a simulator, and a simulator cannot falsify its own assumptions.** The
response model encodes beliefs about how users behave — that examination decays
with slot, that relevance is multiplicative, that a better ranker lifts every
user's relevance equally. A result here is a statement about that model. Its
honest use is to exercise the decision machinery end to end: the SRM guard, the
variance reduction, the interval, the power calculation and the launch rule.

Two assumptions are known to be wrong for this corpus and are kept anyway,
because the machinery has to run somewhere:

- **The examination curve is imported, not measured.** MIND shuffles its
  impression lists, so the logged `slot` carries no position bias — the
  standing departure that made K4 replace §12.4's correction. Varying the decay
  moves how clicks spread across slots and barely moves the session rate, which
  is calibrated separately, so conclusions about the **primary metric** survive
  the assumption and conclusions about **slot behaviour** do not.
- **The treatment lifts every user equally.** A real ranker change helps some
  users and hurts others. That makes the simulated effect cleaner than any real
  one, so **the power figures here are optimistic** — a real experiment of the
  same size would detect less.

What is calibrated from real data: the slate size and the item popularity
distribution come from this orchestrator's own impression log, so the control
arm draws only from the items the policy actually reaches and exploration is
the only thing that touches the rest of the catalogue. That is the
reachability gap [docs/exposure.md](exposure.md) measured, carried in rather
than smoothed away. The baseline click rate comes from the corpus (MIND's
4.04% per item) because `impression_context` carries no clicks — there is no
client sending them.

## Why off-policy evaluation does not appear below

`evaluation/counterfactual/ips.py` is built and tested, and it is deliberately
**not** used to estimate this effect. Importance weighting reweights the
distribution over **which items are shown**; it assumes the reward for a given
(context, action) is a property of the world, not of the policy. The treatment
simulated here changes how *relevant* the shown items are, which is a change to
the reward function itself. SNIPS cannot see that, and running it anyway would
produce a confident number for a quantity it is not estimating.

It applies cleanly to a change in the action distribution — a different
exploration rate, a different blend, a different candidate pool — and that is
where it should be pointed. The distinction is worth more than the number
would have been.

<!-- experiment:begin -->
## 🛑 DO NOT LAUNCH

> Guardrail breached: latency (ms). A primary-metric win that costs one of these is a trade, and this one is past the tolerance agreed in advance.

### Design

- **77,258 sessions per arm**, equal allocation.
- Powered for a **+2.0%** relative lift; this run could have detected **+1.90%** (power 84% at the target).
- Catalogue 65,238 items, slate 10, head 120 items carry 80% of exposure.
- Baseline session CTR 33.79%, seed 0.

### Sample ratio

no SRM: chi2=0.000 df=1 p=1.00e+00

### Primary metric

| Metric | Control | Treatment | Change | 95% CI on the difference |
|---|---:|---:|---:|---|
| session CTR | 0.3381 | 0.3674 | +8.66% | [+0.02470, +0.03406] |
| session CTR (CUPED) | 0.3381 | 0.3674 | +8.66% | [+0.02469, +0.03384] |

CUPED removed **10.1%** of the variance, which is why the adjusted interval is the narrower one. The decision reads the adjusted row.

### Guardrails

| Guardrail | Control | Treatment | Change | 95% CI | Tolerance | Verdict |
|---|---:|---:|---:|---|---:|---|
| latency (ms) | 8.202 | 9.727 | +18.59% | [+1.504, +1.544] | 5% | BREACHED |
| long-tail share | 0.1995 | 0.2791 | +39.88% | [+0.07821, +0.08087] | 2% | ok |

_Generated 2026-11-18T01:33:45+00:00 from `9d4799492a67`, calibrated on `/tmp/served.jsonl`._
<!-- experiment:end -->
