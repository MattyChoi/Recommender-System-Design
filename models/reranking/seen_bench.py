"""What the seen-list costs, and what it gets wrong, at sizes this corpus implies.

**No Spark and no checkpoint, deliberately.** A Bloom filter's false-positive
rate is a function of ``(bits, hashes, items held)`` and of nothing else -- not
of which items, not of their popularity, not of the corpus. So loading real
seen-lists would not make the rate more real; it would only make the benchmark
slower and hide that the rate is arithmetic. What the corpus decides is the one
parameter that matters, the **capacity**, and those come from measured numbers:

- a MIND click history holds a mean of **28.9** items, median 27, with 31.6%
  truncated at the 50-slot cap (measured over 512 sampled validation requests);
- the serving seen-list is bigger than the history, because it holds everything
  SHOWN rather than everything clicked: 10 slots per request, and a week at a
  couple of requests a day is a few hundred.

So the sweep runs 50 / 200 / 1000, and the last row is the one to read for a
real deployment.

Run: ``uv run python -m models.reranking.seen_bench``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from models.reranking.seen import measure_false_positives, sizing

# MEASURED, not estimated: a live 1,918-bit key (240 B of payload) reported 504 B
# to Redis MEMORY USAGE, so ~264 B is key name, object header, SDS header and
# dict entry. Per KEY, not per bit, which is why it changes the shape of the
# saving curve rather than shifting it.
REDIS_KEY_OVERHEAD_BYTES = 264

# Bytes for one item id stored exactly, which is what the filter is competing
# against. An int64, or a Redis set member of similar width.
EXACT_BYTES_PER_ITEM = 8

# Users, for the memory column. Round, and stated rather than implied: the whole
# argument for the structure is what happens at a scale this project does not
# have, and quoting a total without its multiplier hides that.
USERS = 10_000_000

PROBES = list(range(1_000_000, 1_050_000))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacities", type=int, nargs="+", default=[50, 200, 1000])
    parser.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.01, 0.001])
    parser.add_argument("--users", type=int, default=USERS)
    args = parser.parse_args(argv)

    header = (
        f"  {'held':>6}{'target':>9}{'measured':>10}{'bits':>8}{'B/user':>9}"
        f"{'load':>8}{'exact B':>9}{'saving':>9}{'at ' + f'{args.users:,}':>16}"
    )
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    for capacity in args.capacities:
        for rate in args.rates:
            got = measure_false_positives(
                seen=list(range(capacity)),
                probes=PROBES,
                capacity=capacity,
                target_rate=rate,
            )
            exact = capacity * EXACT_BYTES_PER_ITEM
            bytes_per_user = got["bytes_per_user"]
            total_gb = bytes_per_user * args.users / 1024**3
            print(
                f"  {capacity:>6,}{rate:>9.3f}{got['measured_rate']:>10.4f}"
                f"{int(got['bits']):>8,}{bytes_per_user:>9.1f}{got['load']:>8.1%}"
                f"{exact:>9,}{1 - bytes_per_user / exact:>9.1%}{total_gb:>13.1f} GB"
            )

    print("\n  The saving column is against storing the ids exactly at 8 bytes each,")
    print("  which is the honest rival -- not against storing nothing.")
    print(
        "\n  Saving is CONSTANT down each target-rate group because bits per item is\n"
        "  -log2(p) / ln 2, a function of the error rate ALONE: 1% costs 9.6 bits =\n"
        "  1.2 bytes per item however many items a user holds.\n"
        "\n  ⚠️  THAT PROPERTY IS ARITHMETIC AND DOES NOT SURVIVE DEPLOYMENT. A real\n"
        f"  Redis key costs ~{REDIS_KEY_OVERHEAD_BYTES} bytes beyond its payload -- key name,\n"
        "  object header, SDS header, dict entry -- measured with MEMORY USAGE on a\n"
        "  live key. Overhead is per KEY, so it does not scale with capacity, and\n"
        "  the saving therefore stops being constant and starts growing with it:"
    )

    print(f"\n  {'held':>6}{'payload B':>11}{'+ overhead':>12}{'exact B':>9}{'real saving':>13}")
    print("  " + "-" * 51)
    for capacity in args.capacities:
        bits, _ = sizing(capacity, 0.01)
        payload = bits / 8.0
        real = payload + REDIS_KEY_OVERHEAD_BYTES
        exact = capacity * EXACT_BYTES_PER_ITEM
        print(f"  {capacity:>6,}{payload:>11.0f}{real:>12.0f}{exact:>9,}{1 - real / exact:>13.1%}")
    print(
        "\n  At a 50-item seen-list the filter barely pays for itself; the structure\n"
        "  earns its place at hundreds of items per user, not tens. A bits-only\n"
        "  estimate claims 85% at every row and is wrong at exactly the sizes where\n"
        "  someone might reach for a plain set instead."
    )
    print(
        "\n  The error is ONE-SIDED and that is the whole argument. A false positive\n"
        "  hides a fresh item and the user sees the next-best one instead, which\n"
        "  nobody notices. A false negative would re-show an item the user just\n"
        "  saw, which everybody notices -- and the structure cannot produce one.\n"
        "  Choosing this filter is accepting the measured rate above in exchange\n"
        "  for making the visible failure impossible."
    )

    print("\n  OVERFILL -- what happens when the capacity estimate is wrong")
    print(f"  {'held':>6}{'sized for':>11}{'measured':>10}{'load':>8}")
    print("  " + "-" * 33)
    for held in (100, 200, 400, 1000):
        got = measure_false_positives(
            seen=list(range(held)), probes=PROBES, capacity=100, target_rate=0.01
        )
        print(f"  {held:>6,}{100:>11,}{got['measured_rate']:>10.1%}{got['load']:>8.0%}")

    print(
        "\n  This is NOT graceful degradation past a point, and the last row is why.\n"
        "  At ten times its capacity every bit is set, so the filter answers 'seen'\n"
        "  to EVERY candidate. A seen-filter that hides everything does not return\n"
        "  a slightly worse slate -- it returns an empty one. The one-sided\n"
        "  guarantee still holds and is worth nothing, because the useful half of\n"
        "  the answer is gone.\n"
        "  So capacity is not a number to set once: it is a number to monitor, with\n"
        "  bit LOAD as the alarm. Load is observable per user in Redis and moves\n"
        "  long before the rate does."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
