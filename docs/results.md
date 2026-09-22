# Results

An index. Every measurement this project has made lives in one of the four
documents below, and they are split by **what question was asked**, not by when
the work was done.

| document | the question | author |
| --- | --- | --- |
| [baselines.md](baselines.md) | Given the ~37 items MSN already chose, can a model reorder them? | generated |
| [retrieval.md](retrieval.md) | Given the 65,239-row catalogue, can a model find the clicked article? | hand-written |
| [benchmarks.md](benchmarks.md) | What does it cost -- time, memory, latency? | hand-written |
| [evaluation.md](evaluation.md) | How is any of this measured, and why that way? | hand-written |

## The one thing to get right

**A number from `baselines.md` and a number from `retrieval.md` are not
comparable, and the failure mode is quiet.** Both report a `Recall@10`-shaped
quantity. In `baselines.md` the denominator is one impression -- did the clicked
item reach the top ten of the ~37 shown. In `retrieval.md` the denominator is the
whole catalogue. The ranking numbers are the larger ones and they are the easier
question; putting them beside each other in a sentence produces a claim nobody
made.

This is why the two are separate files rather than two halves of one. They were
one file until the split, and the first line of the retrieval half had to be a
warning not to read across the boundary.

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

- **Ranking** -- complete for the counting and content baselines. See
  `baselines.md`, and ADRs [0010](adr/0010-covisitation-is-near-uninformative-on-mind.md)
  and [0011](adr/0011-the-decay-optimum-is-29-minutes.md).
- **Retrieval** -- two-tower, Parts G and H. `evaluate_retrieval` has not yet
  been run against the ranking baselines, so no model appears in both documents.
- **Cost** -- sharding, hashing, distributed training and per-request user-encode
  latency are measured; end-to-end latency, index and diversity are still TODO in
  `benchmarks.md`.
