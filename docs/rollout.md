# Rollout

How a new model or a new binary reaches traffic, and what has to hold at each
step before the next one starts.

**This system has no users.** MIND is a fixed 2019 corpus, so rungs 2–4 below
are a procedure with its mechanism built and exercised against synthetic
traffic, not a thing that has been run for real. Rung 1 is the one that runs.
Said once here so that nothing further down has to be hedged.

Not duplicated here: the probe and draining semantics
(`infra/k8s/README.md`), shadow mode's design and its known flaw
(`claude/n4-shadow-mode.md`), the gate's reasoning (`evaluation/offline/gate.py`).

| rung | mechanism | state |
|---|---|---|
| 0 · offline gate | `evaluation/offline/gate.py`, `make quality-gate` | built, gates every candidate |
| 1 · shadow | `-shadow-triton-model`, a second `ranking.Client` | built, runs |
| 2 · canary | a variant at a small percent, or a second Deployment | mechanism built; **no auto-rollback** |
| 3 · A/B | `experiments.Assign`, `-experiment-id`, `-variants` | assignment and SRM inputs built; analysis partial |
| 4 · ramp + holdback | the same percents, moved; `Unassigned` as the holdback | mechanism built; bookkeeping not |

---

## Rung 0 — the offline gate

No candidate reaches traffic without clearing `evaluation/offline/gate.py`:
an **absolute floor** and a **paired bootstrap over users**, per cohort, on
recall, NDCG@10 and GAUC. Fail-closed, and the reason is recorded when it
passes as well as when it refuses.

Absolute rather than the 2% relative figure the reference suggests: recall@100
≈ 0.3776, so 2% relative is 0.0075 — 1.5× looser than the gate already running
on index promotion, and a model is harder to roll back than an index.

⚠️ `GAUC_TOLERANCE` is inherited from recall's, not derived. The paired test
does not earn it: a significance test catches a floor that is too loose and is
silent about one that is too tight, which refuses good candidates and leaves no
evidence it was the threshold's fault.

⚠️ The paired half of GAUC needs the per-slate rows in
`evaluation/results/slates/`, which only a re-score produces. A card without
them reports `paired UNAVAILABLE` and the floor stands alone.

## Rung 1 — shadow

A second `ranking.Client` on `-shadow-triton-model`, scoring the **same feature
matrix** as the live ranker, served never, logged into `ImpressionContext`
fields 11–12. Zero user risk: it starts after `Features.Build` and runs
concurrently, so the gateway is read once and both models see one world.

What it catches: score-distribution skew between a candidate and what is
serving, and the candidate's latency under real feature vectors rather than
under a benchmark's.

What it cannot catch: anything about a different **feature set**. A shadow with
its own columns needs a second `Features.Build` and doubles per-request gateway
load. This rung tests a different model, not a different pipeline.

⚠️ **The shadow column is indicative, not a measurement.** The read is
non-blocking and a slow shadow logs nothing, so which requests carry a shadow
score is correlated with how hard they were — missing not at random. Comparing
the two columns as if the shadow were observed on a random sample overstates it.
The fix, emitting the shadow as its own record keyed by `request_id`, is a
stretch goal.

## Rung 2 — canary

**Decide first whether the change is a new model or a new binary.** They take
different mechanisms, and conflating them is how a canary ends up measuring
nothing.

**A new model** is a variant inside the running Deployment:
`-variants 'control=99,canary=1'` with `-experiment-id`. Assignment is per
user, evaluated in-process, so every pod splits traffic the same way and the
canary's share is exactly its percent.

**A new binary** is a second Deployment behind the same Service, and here the
obvious approach is wrong:

⚠️ **A Kubernetes Service load-balances connections, not requests.** A gRPC
client holds one long-lived HTTP/2 connection and multiplexes every RPC over
it, so one canary pod in ten does **not** receive 10% of requests — it receives
whatever share of *clients* happened to connect to it. With a handful of
clients that is wildly off; with one client it is 0% or 100%. Request-level
splitting needs a proxy that balances per-RPC, or a headless Service with
client-side round-robin. Until one of those exists here, a binary canary is
a smoke test on real traffic, not a measured comparison.

**The guardrails exist as alerts.** `serving/monitoring/alerts.yml` defines ten,
and the Grafana dashboard mirrors every threshold so a red panel and a firing
alert cannot disagree:

| alert | for |
|---|---|
| `RecsysP99OverBudget` | 5m |
| `RecsysStageOverBudget` | 10m |
| `RecsysErrorRate` | 5m |
| `RecsysClientErrorSpike` | 15m |
| `RecsysFallbackRate` | 5m |
| `RecsysRankerDegraded` | 5m |
| `RecsysFeatureGatewayDegraded` | 5m |
| `RecsysRetrievalOnRungTwo` | 5m |
| `RecsysSeenFilterSaturated` | 10m |
| `RecsysExperimentImbalance` | 30m |

⚠️ **Nothing acts on them.** These are the right guardrails and they are
wired to no actuator: auto-rollback needs something watching them and calling
`kubectl rollout undo`, and that does not exist. A canary here is a human
reading a dashboard. The reference calls for "automated guardrail checks,
auto-rollback on breach"; this is the one place the ladder is instrumented but
not automated, and it is deliberate — an actuator that can roll back
production is not something to write without a cluster worth rolling back.

⚠️ A degraded rung-2 reading is easy to misread: `scored` equals `retrieved`
whether or not the ranker ran, because every degrade path in `score()` returns
a full-length synthetic vector. Read `degraded_sources` and the degradation
panel, never the funnel, to decide whether the canary's pipeline was intact.

## Rung 3 — A/B

Assignment is `serving/go/internal/experiments/assign.go`: FNV-1a over
`experiment_id + ":" + user_id`, modulo `Buckets = 10000`, walking variants in
allocation order.

Three properties that make it analysable:

- **Deterministic, therefore sticky.** A user who flipped arms mid-experiment
  would contaminate both. `TestAssignmentIsDeterministic` pins it, and
  `server_test.go` asserts the same user never gets two variants.
- **Salted by experiment id.** Without the salt two experiments over the same
  users produce the *same* split, so their effects correlate and neither is
  interpretable. `main.go` refuses `-variants` without `-experiment-id` for
  exactly this reason.
- **Under-allocation is refused, not truncated.** `-variants` totalling less
  than 100 would silently shrink the last arm; `main.go` errors instead.
  `Assign` itself returns `Unassigned` outside every allocation, which is what
  makes rung 4's holdback expressible.

**The SRM input is instrumented.** `recsys_experiment_assignments_total` is
counted by variant, and the dashboard's "Assignments per arm" panel counts the
`none` arm **on purpose** — a holdback that is not counted makes the chi-square
compare arms against a denominator it cannot see. The imbalance panel tickets
past a 1.2 max/min ratio over 30m, and if it fires the experiment is invalid
whatever the primary metric did.

What is missing is the analysis, not the plumbing: a chi-square SRM test rather
than a ratio alert, CUPED for variance reduction, and a power analysis to decide
how long to run before looking. All three are §17 work.

**Read differences from the pairing, never from two intervals side by side.**
`evaluation/offline/stats.paired_bootstrap` resamples **users**, not
impressions. This project has already made the mistake once and written it down:
a paired interval measured 3.5× narrower than either marginal and excluded zero
where the marginals overlapped heavily.

⚠️ **Propensity is what makes an early read possible, and it has a hole.**
Every served slot logs `propensity`; `rerank.Deterministic` marks a slot that
carries no information for an off-policy estimator. §17.4's SNIPS clips weights
at 10, so a **zero** propensity silently receives the *maximum* weight rather
than erroring or being dropped. `tests/test_serving_ranked.py` asserts
`propensity > 0` on the served path, which protects the producer; the estimator
accepts whatever an older record hands it. Guard it before trusting an
off-policy canary read.

## Rung 4 — ramp and holdback

The ramp is the same percents, moved: 1 → 5 → 10 → 50. Nothing new is needed.

The **holdback** is the `Unassigned` arm. Allocate less than 100% and
`Experiment.Names()` includes `Unassigned` in what an analysis should expect to
see, so a long-term holdback is a deliberate under-allocation rather than a
separate mechanism.

What is not built is the bookkeeping: which experiment a holdback belongs to,
when it started, and when it is allowed to end. A holdback nobody remembers
creating is a slice of users permanently on an old model.

---

## What would have to be true to run this for real

1. Users, and a reason for them to click.
2. An actuator on the ten guardrail alerts.
3. Request-level traffic splitting, if a *binary* canary is ever wanted — the
   per-user variant path does not need it.
4. The SRM test, CUPED and a power analysis from §17.
5. The SNIPS zero-propensity guard.
6. The unbiased shadow record, so rung 1 produces a measurement rather than an
   indication.
