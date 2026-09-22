# Results

An index. Every measurement this project has made lives in one of the five
documents below, and they are split by **what question was asked**, not by when
the work was done.

| document | the question | author |
| --- | --- | --- |
| [baselines.md](baselines.md) | Given the ~37 items MSN already chose, can a model reorder them? | generated |
| [retrieval.md](retrieval.md) | Given the 65,239-row catalogue, can a model find the clicked article? | hand-written |
| [ranking.md](ranking.md) | Given the ~100 candidates OUR retriever produced, can a model order them? | hand-written |
| [benchmarks.md](benchmarks.md) | What does it cost -- time, memory, latency? | hand-written |
| [evaluation.md](evaluation.md) | How is any of this measured, and why that way? | hand-written |

## The one thing to get right

**Three documents, three candidate sets, three denominators -- and the failure
mode is quiet.** `baselines.md` and `retrieval.md` both report a
`Recall@10`-shaped quantity. In `baselines.md` the denominator is one impression
-- did the clicked item reach the top ten of the ~37 shown. In `retrieval.md` it
is the whole catalogue. The ranking numbers are the larger ones and they are the
easier question; putting them beside each other in a sentence produces a claim
nobody made.

`ranking.md` is a third question again: order the ~100 candidates **this
project's own retriever** produced. That distinction is not pedantry -- it is
the central requirement of the stage, since a ranker trained on a candidate
distribution it will not be served degrades while every offline metric still
looks healthy.

This is why they are separate files rather than sections of one. `baselines.md`
and `retrieval.md` were a single file until the split, and the first line of the
retrieval half had to be a warning not to read across the boundary.

## Why `baselines.md` is generated and the rest are not

`evaluation/offline/results_table.py` rebuilds `baselines.md` from the report
cards in `evaluation/results/` on every `make results`. It rewrites the whole
file, so nothing hand-written can survive in it -- which is the point: a table of
numbers should have exactly one author, and that author should be the code that
read the cards.

It refuses to write to a file that does not open with the marker it stamps into
its own output, so pointing `--out` at any of the other three documents fails
instead of deleting them. That guard exists because this file used to carry both
halves and the only thing protecting the hand-written one was a comment asking
people not to run the command.

```
make results    # rebuild docs/baselines.md from evaluation/results/*.json
make baselines  # re-score every baseline on one commit, then the above
```

## Status

- **Slate reordering** -- complete for the counting and content baselines. See
  `baselines.md`, and ADRs [0010](adr/0010-covisitation-is-near-uninformative-on-mind.md)
  and [0011](adr/0011-the-decay-optimum-is-29-minutes.md).
- **Retrieval** -- two-tower, Parts G and H. `evaluate_retrieval` has not yet
  been run against the ranking baselines, so no model appears in both documents.
- **Ranking over retrieved candidates** -- complete. LightGBM `lambdarank` beats
  both neural arms significantly; see `ranking.md` and
  [ADR 0003](adr/0003-lightgbm-baseline-before-dcnv2.md). Position debiasing is
  built and deliberately unwired -- MIND shuffles impression order, so the
  correction's own verification would pass by construction.
- **Cost** -- sharding, hashing, distributed training and per-request user-encode
  latency are measured; end-to-end latency, index and diversity are still TODO in
  `benchmarks.md`.
