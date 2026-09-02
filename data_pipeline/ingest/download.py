"""Download and unpack the MIND dataset into paths.raw.

That repository is *gated*: files are listed publicly but fetching them requires
a Hugging Face account that has accepted the dataset's conditions. Accept them
once at https://huggingface.co/datasets/yjw1029/MIND , then authenticate with
``hf auth login`` (or export ``HF_TOKEN``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

from common.config import load_settings

REPO_ID = "yjw1029/MIND"
DATASET_URL = f"https://huggingface.co/datasets/{REPO_ID}"

EXPECTED_MEMBERS = frozenset(
    {"behaviors.tsv", "news.tsv", "entity_embedding.vec", "relation_embedding.vec"}
)
SPLITS = ("train", "dev", "test")

_AUTH_HELP = (
    f"\n{REPO_ID} is a gated huggingface dataset.\n"
    f"  1. Sign in and accept the conditions at {DATASET_URL}\n"
    "  2. Authenticate locally:  hf auth login   (or export HF_TOKEN=...)\n"
)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Hash a file by chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def fetch_split(size: str, split: str, raw_root: Path, *, force: bool = False) -> None:
    """Download one MIND split and unpack it into ``raw_root/<split>``.

    Args:
        size: ``small`` or ``large``.
        split: ``train``, ``dev``, or ``test`` (``test`` is large-only and its
            click labels are withheld -- it was the leaderboard set).
        raw_root: Destination root, normally ``settings.paths.raw``.
        force: Re-download and re-extract even when the split is present.

    Raises:
        RuntimeError: If the archive is corrupt, is missing an expected member,
            or the dataset's terms have not been accepted.
    """
    dest = raw_root
    marker = dest / split / "behaviors.tsv"

    # For idempotency, we check for the presence of one of the expected members.
    if marker.is_file() and marker.stat().st_size > 0 and not force:
        print(f"{split}: already present, skipping")
        return

    name = f"MIND{size}_{split}.zip"
    print(f"{split}: {REPO_ID}/{name}")

    try:
        archive = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=name,
                repo_type="dataset",
                force_download=force,
            )
        )
    # HfHubHTTPError is the superclass of GatedRepoError, RepositoryNotFoundError
    # and EntryNotFoundError, so 401/403/404 all land here with guidance rather
    # than a raw traceback.
    except (HfHubHTTPError, LocalTokenNotFoundError) as exc:
        raise RuntimeError(f"cannot download {name}.{_AUTH_HELP}") from exc

    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise RuntimeError(f"{name}: CRC failure, archive is corrupt")
        missing = EXPECTED_MEMBERS - {Path(n).name for n in zf.namelist()}
        if missing:
            raise RuntimeError(f"{name}: missing {sorted(missing)}")
        dest.mkdir(parents=True, exist_ok=True)
        zf.extractall(dest)

    zf_dir = dest / Path(name).stem
    if (dest / split).is_dir():
        shutil.rmtree(dest / split)
    zf_dir.rename(dest / split)

    (dest / split / "_manifest.json").write_text(
        json.dumps(
            {
                "repo_id": REPO_ID,
                "filename": name,
                "sha256": _sha256(archive),  # gives each download a fingerprint to track changes
                "size": size,
                "downloaded_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        A process exit code; 1 if a split could not be fetched.
    """
    parser = argparse.ArgumentParser(description="Download and unpack the MIND dataset.")
    parser.add_argument(
        "--size",
        choices=("small", "large"),
        default="small",
        help="MIND variant. Develop on small; run final numbers on large.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=["train", "dev"],
        help="Splits to fetch. 'test' is large-only and has no click labels.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Destination root. Defaults to paths.raw from conf/config.yml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when the split is already present.",
    )
    args = parser.parse_args(argv)

    raw_root: Path = args.raw if args.raw is not None else load_settings().paths.raw

    try:
        for split in args.splits:
            fetch_split(args.size, split, raw_root, force=args.force)
    except RuntimeError as exc:
        print(f"\nerror: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
