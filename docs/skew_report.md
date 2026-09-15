# Training/serving skew report

**Status: blocked on Part M, deliberately.** Skew is measured by logging the feature
vectors a *serving* layer actually used and recomputing offline what training would have
produced for the same entities at the same timestamps. There is no serving layer yet, so
there is nothing to log and nothing to compare against. This file is sequenced, not
forgotten.

Generated, never hand-written — a hand-maintained skew report is a claim, not a
measurement.

## What it will contain

```
Feature                      Train mean   Serve mean   KS-stat   Status
user.user_impressions_24h        42.3         41.8       0.011      OK
item.item_ctr_smoothed            0.031        0.029      0.024      OK
item.item_age_hours               6.2         11.4       0.290    SKEW
```

## The row to expect trouble on

`item_age_hours`. It is computed offline from an article's **first impression in the log**,
because MIND ships no publication timestamp (see `docs/design.md`). A serving layer has no
such history to hand and will reach for ingestion time instead — so the same article gets
two different ages, diverging by however long the pipeline lags. That is a real, plausible
bug rather than a hypothetical, and it is the first thing to check once Part M can log.

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
