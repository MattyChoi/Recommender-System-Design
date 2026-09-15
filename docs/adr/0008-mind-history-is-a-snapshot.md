# ADR 0008 — MIND's `history` is a snapshot, so C4's leak test was replaced

**Status:** accepted · **Date:** 2026-09-12

## Context

The guide's C4 specifies a data contract it calls "THE MIND-specific leak, with no
MovieLens equivalent":

> Every item in a user's `history` must have been clicked BEFORE the impression it is
> attached to.

Implemented literally, it explodes `history`, joins to clicked events, and asserts no
history item has a click timestamp at or after the impression it is attached to.

**That test cannot fail on this dataset, and it took reading the data to see why.** MIND's
`history` is a single snapshot Microsoft computed over the four weeks *preceding* the log
window, and it is attached unchanged to every one of that user's impressions. It does not
advance as the user clicks through the log. So the items it names are not in the impression
log's timeline at all, and there is nothing for the join to order them against. The
assertion is structurally green: it passes on correct data, and it passes equally on data
where the ordering has been destroyed.

A test that cannot go red is worse than no test. It occupies the place where a real check
would go, and it reports success.

Measured on MIND-small, confirming the snapshot reading: **0.4% of train's in-log clicked
`(user, item)` pairs appear in that user's history, and 0.2% of dev's.** An end-of-window
snapshot would produce something near 100%. A per-impression running history would produce
a rising fraction. 0.4% is the overlap of two nearly disjoint time ranges.

## Decision

**Delete the literal C4 test and replace it with two invariants that can fail**, both in
`data_pipeline/tests/test_contracts.py`:

1. **`test_history_is_one_snapshot_per_user`** — a user's `history` string is identical
   across all of their impressions. This is the assumption every sequence feature in Part H
   rests on. If MIND ever ships a running history, this goes red and the code that treats
   history as static must be revisited.
2. **`test_history_predates_the_log_window`** — the share of in-log clicked pairs that also
   appear in history stays below `_MAX_HISTORY_OVERLAP = 0.05`. The ceiling sits well above
   the measured 0.4% / 0.2% and far below what any end-of-window or running snapshot would
   produce, so it catches the failure mode the original test was reaching for — history
   contaminated by the evaluation window — without asserting an ordering that does not
   exist.

## Consequences — accepted downsides

1. **A departure from the guide that must be explained, not hidden.** Anyone comparing this
   repo to the guide will find C4's named test missing. That is what this ADR is for.
2. **The 0.05 ceiling is a judgement, not a derivation.** It is an order of magnitude above
   the measured value and an order of magnitude below a contaminated one. If MIND-large
   differs materially, the number needs re-measuring rather than raising.
3. **Neither replacement tests ordering**, because ordering is not observable here. If a
   future dataset carries per-impression history with timestamps, the original test becomes
   meaningful and should be added back alongside these.
4. **`history` is stale for later impressions by construction**, and that is now recorded in
   the test rather than discovered during Part H. A user's last impression reads a profile
   that knows nothing of the days between.

## Alternatives considered

- **Keep the test as written.** Rejected: it is green regardless of the data, which is the
  specific property that makes a test worthless.
- **Keep it but mark it `xfail`.** Rejected: `xfail` says "this should pass and does not".
  The truth is "this cannot be evaluated on this dataset", which is a different statement
  and belongs in an ADR rather than a marker.
- **Synthesise a running history to test the ordering.** Rejected: it would test the
  synthesiser. The fixture generator already produces MIND-shaped data; making it produce
  data MIND does not ship would validate an invariant the real corpus cannot violate.
