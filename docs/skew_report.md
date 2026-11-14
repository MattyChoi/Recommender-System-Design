# Training/serving skew report

**Status: run 2026-11-11, against traffic put through the orchestrator with
`-kafka` set. The Measured table below is from that run.** The
orchestrator now logs the feature vectors it served (`ImpressionContext.user_feats`
on the `impression_context` topic, with `user_found` beside them), and
`scripts/writeups/skew_report.py` compares them against what training read. The
table below is filled by the job, between the markers, and by nothing else:

```bash
make up && make schemas                 # broker + registry
# run the orchestrator with -kafka, put traffic through it, then:
make consume TOPIC=impression_context N=20000 > /tmp/served.jsonl
make skew SERVED=/tmp/served.jsonl
```

Generated, never hand-written — a hand-maintained skew report is a claim, not a
measurement. Method (in the script's docstring): one row per request on both
sides; store misses go to the coverage line instead of the distributions; requests
whose feature fetch failed are dropped; PSI on training's deciles decides the
status (<0.1 OK, 0.1–0.25 WATCH, >0.25 SKEW), with KS reported beside it.

## Measured

<!-- skew-report:begin -->
Served: 12,418 requests parsed; 12,417 with a stored user row; 0 excluded because the feature fetch failed; 0 malformed. Training: `train`, 156,795 requests (one per impression).

| Feature | Train mean | Serve mean | KS-stat | PSI | Status |
|---|---:|---:|---:|---:|---|
| user.user_impressions_24h | 79.91 | 58.43 | 0.121 | 0.094 | OK |
| user.user_clicks_24h | 3.412 | 2.342 | 0.170 | 0.171 | WATCH |
| user.user_ctr_smoothed | 0.0522 | 0.05109 | 0.027 | 0.010 | OK |
| user.user_tenure_hours | 28.98 | 24.23 | 0.206 | 0.247 | WATCH |

**Coverage** (`has_user_features` / `user_found`): train 66.7%, served 100.0%. Rows without a stored user are excluded above, so a coverage gap shows here and not as a shifted mean.
<!-- skew-report:end -->

## What this does not cover yet

**Item features.** The log carries the served *user* block only. `item_age_hours`
(below) is the row most likely to skew, and it is not measurable from this log:
`ImpressionContext` would need the per-candidate item rows the ranker scored, which
is `k x 6` floats per request rather than 4. Worth adding when this report first
runs clean on the user block — not before, since it multiplies the topic's volume.

## The row to expect trouble on

`item_age_hours`. It is computed offline from an article's **first impression in the log**,
because MIND ships no publication timestamp (see `docs/design.md`). A serving layer has no
such history to hand and will reach for ingestion time instead — so the same article gets
two different ages, diverging by however long the pipeline lags. That is a real, plausible
bug rather than a hypothetical, and it is the first thing to check once the orchestrator
can log.

## Two more places skew can enter, neither caught by a parity test

**Freshness, not logic.** Training reads a feature bucket that closed at the impression
instant; serving reads Redis, which holds whatever was last materialised — up to the
FeatureView's `ttl` old (2h for items, 6h for users, 24h for the cross view). The values
can agree perfectly and still describe different moments. Fixes: lag the training features
to match serving staleness, or move to feature logging and train on what was actually
served.

**Coverage.** `has_user_features` is true for 67.2% of train rows and 37.1% of dev's. If
the online store's coverage differs from that, the model meets a different missingness
pattern in production than it trained on — and missingness is itself a feature here, since
the flags are columns.
