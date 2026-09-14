# Materialisation parity

**This is not the skew report.** Both sides below are offline reads: the gold
series, and what Feast materialised from it. It tests Feast, the TTLs and the
entity-key encoding. The online/offline skew report needs a serving layer and
its impression log; it is M5 and it writes `docs/skew_report.md`.

Window: `2019-11-09T00:00:00` to `2019-11-16T00:00:00` · float tolerance: rel_tol=1e-09

| view | feature | sampled | of | matched | mismatched | missing online |
|---|---|---|---|---|---|---|
| item_stats | `item_impressions_24h` | 200 | 22,417 | 200 | 0 | 0 |
| item_stats | `item_clicks_24h` | 200 | 22,417 | 200 | 0 | 0 |
| item_stats | `item_ctr_smoothed` | 200 | 22,417 | 200 | 0 | 0 |
| item_stats | `cat_expanding_ctr` | 200 | 22,417 | 200 | 0 | 0 |
| item_stats | `item_age_hours` | 200 | 22,417 | 200 | 0 | 0 |
| item_stats | `category` | 200 | 22,417 | 200 | 0 | 0 |
| user_stats | `user_impressions_24h` | 200 | 93,646 | 200 | 0 | 0 |
| user_stats | `user_clicks_24h` | 200 | 93,646 | 200 | 0 | 0 |
| user_stats | `user_ctr_smoothed` | 200 | 93,646 | 200 | 0 | 0 |
| user_stats | `user_tenure_hours` | 200 | 93,646 | 200 | 0 | 0 |
| user_category_stats | `user_cat_impressions_cum` | 200 | 987,797 | 200 | 0 | 0 |
| user_category_stats | `user_cat_clicks_cum` | 200 | 987,797 | 200 | 0 | 0 |
| user_category_stats | `user_cat_affinity` | 200 | 987,797 | 200 | 0 | 0 |

Largest float delta observed: `0.000e+00`. A delta of exactly zero
means the tolerance was never exercised, not that it is generous:
doubles round-trip bit-exactly through protobuf and Redis.

**`of`** is the number of entities with a row in the window, which is the
population the sample is drawn from. Entities whose last bucket predates
the window are not materialised and are not counted here.
