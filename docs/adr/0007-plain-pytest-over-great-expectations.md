# ADR 0007 — Plain pytest for data contracts, not Great Expectations

**Status:** accepted · **Date:** 2026-09-12

## Context

The guide's §3 layout names Great Expectations as the data-contract tool, and §5.5 asks
for assertions that fail the pipeline rather than warnings that scroll past. Those are two
separate requirements: *what* the checks assert, and *what framework* expresses them.

The checks themselves are not in question. `data_pipeline/tests/test_contracts.py` holds 34
of them — row-count conservation from raw through silver, click-implies-impression,
referential integrity against the catalogue, the CTR plausibility band that catches an
inverted label parse, the history-snapshot invariants, and the as-of leakage gate.

What Great Expectations would add on top is a suite store, a checkpoint configuration, a
validation-result store, and Data Docs. What it would cost is a second vocabulary, a second
runner, and a second place for a failure to be configured into a warning.

## Decision

**Express data contracts as pytest tests**, run by the same `uv run pytest` that runs
everything else, gated by the same CI step.

The deciding argument is that the contracts and the unit tests have the same job here:
stop a bad build. Splitting them across two frameworks means two ways to be green, two
ways to be skipped, and a reviewer who has to learn a DSL before they can read the
assertion. A contract expressed as `assert orphans == 0, f"{orphans} clicks without a
matching impression"` needs no manual.

Great Expectations earns its weight when non-engineers author expectations, when profiling
results need to be browsable, or when validation results must be retained and compared
across runs by a platform team. None of those apply to a single-author repository.

## Consequences — accepted downsides

1. **No Data Docs.** There is no browsable HTML report of expectations and their results.
   The pytest output is the report, which is fine in CI and worse for showing someone.
2. **No expectation profiling.** GE can suggest expectations from a sample; every check
   here was written by hand, which is slower and biased toward what the author thought to
   check.
3. **No validation history.** A GE store retains results across runs, so drift in a
   contract's pass rate is visible. Here a check is green or red, with no series behind it.
   Part O's monitoring covers drift for features, not for contracts.
4. **Hand-rolled equivalents.** `_MAX_HISTORY_OVERLAP`, the CTR plausibility band and the
   row-count tolerances are thresholds GE ships as built-ins. They are five lines each
   here, and they will stay hand-rolled as the suite grows.

## Alternatives considered

- **Great Expectations as §3 specifies.** Rejected above. Worth restating that this is a
  deliberate departure from the guide rather than an omission.
- **Both: GE for data, pytest for code.** Rejected as the worst option — it pays GE's
  setup cost and still leaves contract failures in a second place to check.
- **Pandera.** Lighter than GE, schema-first, integrates with pytest. Genuinely close, and
  would have been reasonable. Rejected because the interesting contracts here are not
  schema shape — Spark already enforces that from `common/schemas.py` — but relationships
  between tables, which Pandera expresses no more naturally than a plain assertion.
