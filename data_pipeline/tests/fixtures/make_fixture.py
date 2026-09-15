"""Generate the synthetic MIND-shaped corpus the contract tests run against.

    uv run python -m data_pipeline.tests.fixtures.make_fixture

Why synthetic rather than sampled. MIND is gated -- Microsoft's terms have to
be accepted by a human before download -- so committing real rows to a public
repository is redistribution, permanently, in git history. These rows are
invented. They are shaped to satisfy every contract in ``test_contracts.py``
for the same reasons the real corpus does, which makes this file a readable
statement of what those contracts actually assume.

The output is RAW TSV, not Parquet, on purpose: the tests build bronze and
silver from it with the real pipeline, so the contracts exercise the ingest
code rather than tables assembled by hand here. A fixture that bypassed the
pipeline could not catch a parse bug, which is most of what the contracts are
for.

Regenerate after changing anything about the shape, and commit the result.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

FIXTURE_ROOT = Path(__file__).resolve().parent / "raw"
SEED = 20191109

# MIND's real vocabulary shape: a handful of single lowercase tokens. The
# closed-vocabulary contract asserts no category contains a space, because a
# shifted delimiter fills this column with titles.
CATEGORIES = ("news", "sports", "finance", "travel", "lifestyle", "health", "video", "tv")
SUBCATEGORIES = (
    "newsworld",
    "newspolitics",
    "football_nfl",
    "basketball_nba",
    "markets",
    "traveltips",
    "lifestyleroyals",
    "healthnews",
    "medical",
    "tvnews",
)

# ~4% click-through, matching the real corpus closely enough that
# test_ctr_is_plausible's 1-15% band means the same thing here as there.
P_IMPRESSION_HAS_CLICK = 0.80
ITEMS_PER_IMPRESSION = (5, 35)

# Roughly one article in twenty ships no abstract. Spark's CSV reader turns an
# empty field into null, which is what the "mostly present" ceiling checks --
# so this must be an empty field, not the string "null".
P_ABSTRACT_MISSING = 0.05

SPLITS = {
    # split:  (users, impressions, own articles, id offset, first day)
    "train": (200, 800, 1500, 10_000, datetime(2019, 11, 9)),
    "dev": (150, 400, 1200, 60_000, datetime(2019, 11, 15)),
}
# History is drawn from here, and these ids appear in NO catalogue and no
# impression. That is the point: MIND's history is a snapshot of clicks made
# BEFORE the log window, so its items are largely absent from the log itself.
# Measured at 0.4% overlap on the real corpus; exactly 0% here.
ARCHIVE_IDS = tuple(f"N{200_000 + i}" for i in range(400))
# Articles present in both splits' news files. The cross-split agreement
# contract only has something to check when this is non-zero, and
# read_news()'s dropDuplicates is only safe because these rows are identical.
SHARED_CATALOGUE = 600


def _mind_timestamp(when: datetime) -> str:
    """Format as MIND does: M/d/yyyy h:mm:ss AM, no zero padding on M/d/h."""
    hour = when.hour % 12 or 12
    meridiem = "AM" if when.hour < 12 else "PM"
    return (
        f"{when.month}/{when.day}/{when.year} {hour}:{when.minute:02d}:{when.second:02d} {meridiem}"
    )


def _news_row(item_id: str) -> str:
    """One article, derived ONLY from its id.

    Seeding per item rather than drawing from the caller's stream is what makes
    an article shared between train and dev come out byte-identical in both
    files. Draw from a shared stream and the two copies diverge, because the
    stream is at a different position when each split is written -- which would
    break the cross-split agreement contract and, worse, make read_news()'s
    dropDuplicates genuinely unsafe.
    """
    rng = random.Random(f"article:{item_id}")
    category = rng.choice(CATEGORIES)
    subcategory = rng.choice(SUBCATEGORIES)
    title = f"Synthetic headline {item_id} about {category}"
    # An EMPTY field, which the reader nulls -- not the word "null", which it
    # would happily keep as a five-character abstract.
    abstract = "" if rng.random() < P_ABSTRACT_MISSING else f"Body text for {item_id}."
    url = f"https://example.invalid/{item_id}"
    return "\t".join([item_id, category, subcategory, title, abstract, url, "[]", "[]"])


def build() -> None:
    """Write both splits' behaviors.tsv and news.tsv under FIXTURE_ROOT."""
    shared = [f"N{90_000 + i}" for i in range(SHARED_CATALOGUE)]

    for split, (n_users, n_impressions, n_items, offset, day_zero) in SPLITS.items():
        out = FIXTURE_ROOT / split
        out.mkdir(parents=True, exist_ok=True)
        # Seeded per split, so regenerating one split cannot disturb the other.
        rng = random.Random(f"{SEED}:{split}")

        # Explicit id ranges rather than hash() -- hash() of a str is salted by
        # PYTHONHASHSEED, so a "deterministic" fixture built that way differs
        # between runs and between machines.
        catalogue = sorted(set(shared + [f"N{offset + i}" for i in range(n_items)]))
        users = [f"U{10_000 + i}" for i in range(n_users)]

        # One history string per user, seeded by the user id: MIND's history is
        # a snapshot, identical across every impression that user makes
        # (verified at zero exceptions over 100,000 real users). Seeding per
        # user also keeps a user who appears in both splits consistent.
        histories = {}
        for user in users:
            user_rng = random.Random(f"history:{user}")
            histories[user] = " ".join(user_rng.sample(ARCHIVE_IDS, user_rng.randint(0, 40)))

        (out / "news.tsv").write_text(
            "\n".join(_news_row(item) for item in catalogue) + "\n", encoding="utf-8"
        )

        lines = []
        for impression_id in range(1, n_impressions + 1):
            user = rng.choice(users)
            when = day_zero + timedelta(seconds=rng.randint(0, 5 * 24 * 3600))
            shown = rng.sample(catalogue, rng.randint(*ITEMS_PER_IMPRESSION))

            # At most one click per impression keeps CTR near 4% without
            # having to tune a per-item probability against list length.
            clicked = rng.randrange(len(shown)) if rng.random() < P_IMPRESSION_HAS_CLICK else -1
            impressions = " ".join(
                f"{item}-{1 if i == clicked else 0}" for i, item in enumerate(shown)
            )
            lines.append(
                "\t".join(
                    [str(impression_id), user, _mind_timestamp(when), histories[user], impressions]
                )
            )

        (out / "behaviors.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{split}: {n_impressions} impressions, {len(catalogue)} articles -> {out}")


if __name__ == "__main__":
    build()
