# ADR 0005 — The item embedding table spans the whole catalogue

**Status:** accepted · **Date:** 2026-09-10

## Context

`item_map` is built from the union of every split's `news` table, so it indexes every
article MIND ships. `user_map` is built from `events`, so it indexes only users who
actually appear. Measured on MIND-small:

| | rows |
|---|---|
| `item_map` (catalogue) | 65,238 |
| distinct items appearing in any impression | ~24,600 |
| `user_map` | 94,057 |

So roughly **63% of the item embedding table addresses articles that were never shown to
anyone.** Those rows receive no gradient during training and keep their random
initialisation for the life of the model. If retrieval ever surfaces such an item, it is
ranked on noise.

Index 0 is reserved for OOV (`common.schemas.OOV_IDX`), so real indices run 1..N and the
embedding table must be sized N + 1.

## Decision

**Keep indexing the full catalogue.** No frequency threshold, no restriction to items that
appear in `events`.

The deciding argument is not accuracy, it is stability. `build_id_maps` now refuses to
overwrite existing maps without `--force`, because renumbering invalidates every checkpoint
trained against the old indices. A map derived from the *catalogue* changes only when the
catalogue changes. A map derived from *events* changes whenever ingest changes — a new
split, a filter, a sampled subset — which is exactly the class of routine action that
should not force a retrain.

## Consequences — accepted downsides

1. **2.6× the ID-embedding parameters**, ~63% of them untrained. Costs memory in the
   training loop, in the serving container, and in the ANN export.
2. **Untrained vectors are reachable.** Any retrieval source that can propose an unshown
   item will propose it on the strength of a random vector. Mitigation: the content-based
   sources carry cold-start items; ID embeddings are not the only signal in the tower.
3. **Offline evaluation cannot see the problem.** The dead rows are never exercised by
   held-out impressions, so no offline metric degrades. This is a known blind spot, not a
   clean bill of health.

## Alternatives considered

- **Index only items appearing in `events`.** Every row trained, 2.6× smaller. Rejected:
  couples the map to the event set, so re-ingesting renumbers everything and invalidates
  checkpoints — the failure this project just spent an ADR's worth of effort preventing.
  Also makes ~40,600 articles permanently unrecommendable through any ID path.
- **Frequency threshold (index items with ≥ k impressions).** Targets the sharper problem —
  an item seen three times has an embedding trained on three examples, which is noise
  wearing a trained coat. Rejected for now: k has no principled value, and C6 already
  measured and *rejected* min-impression filtering elsewhere in the pipeline. Applying one
  here after declining it there needs a justification this project does not yet have.
- **Drop item ID embeddings entirely; go content-only.** MIND ships title, abstract,
  category and subcategory for every article, and news items live for hours — rarely long
  enough for an ID embedding to accumulate useful exposures. This is the strongest
  alternative and may well win. Rejected *for now* only because it should be decided by
  measurement rather than by argument.

## Revisit when

**Part G, at the two-tower ablation.** The tower is already content-weighted, so run it with
and without the item ID embedding and read the difference in recall@k. Three outcomes:

- ID embedding earns little → adopt the content-only alternative, and this ADR is superseded
  by evidence rather than by opinion.
- ID embedding earns a lot on head items only → revisit the frequency threshold, with `k`
  chosen from the ablation curve instead of guessed.
- ID embedding earns a lot broadly → keep this decision and record the number.

Do not revisit before then. The width question is cheap to change and expensive to argue
about without data.
