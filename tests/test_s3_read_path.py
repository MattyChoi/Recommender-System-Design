"""The client stack Feast reads gold through, pinned by behaviour not by hope.

Spark writes the gold series through Hadoop's S3A client; Feast reads them back
through pyarrow and s3fs. ``tests/test_spark_s3a.py`` pins the write side. This
pins the read side, which broke first and more quietly:

``uv add s3fs`` resolved s3fs 0.4.2 -- released in 2020, and the only version
whose requirements are loose enough to coexist with a pinned botocore. Nothing
announced it. It failed much later, inside ``feast materialize``, on an ETag.
"""

from __future__ import annotations

import fsspec
import pytest
import s3fs

# Verbatim from the failure: every object S3A's magic committer writes is
# completed as a multipart upload, and a multipart ETag carries a "-<parts>"
# suffix that a plain hex parse cannot read. Single-part uploads do not, which
# is why the default committer would have hidden this.
MULTIPART_ETAG = '"50aaa750191fe7fe37042ac66c3dc06d-1"'
SINGLE_PART_ETAG = '"50aaa750191fe7fe37042ac66c3dc06d"'


def _calver(version: str) -> tuple[str, str]:
    """The year.month pair, which is what s3fs and fsspec release in lockstep."""
    year, month, *_ = version.split(".")
    return year, month


def test_s3fs_is_not_the_ancient_line() -> None:
    """0.x is the 2020 series. Anything on calver is not it."""
    assert not s3fs.__version__.startswith("0."), (
        f"s3fs {s3fs.__version__} predates fsspec's calver scheme; it cannot "
        "read a multipart ETag. Check whether boto3's pinned botocore is "
        "forcing the resolver back onto it."
    )


def test_s3fs_and_fsspec_are_released_in_lockstep() -> None:
    """Modern s3fs pins an exact fsspec, so a mismatch means one was overridden."""
    assert _calver(s3fs.__version__) == _calver(fsspec.__version__), (
        f"s3fs {s3fs.__version__} against fsspec {fsspec.__version__}"
    )


@pytest.mark.parametrize("etag", [MULTIPART_ETAG, SINGLE_PART_ETAG])
def test_checksum_reads_both_etag_shapes(etag: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The behaviour, not the version string.

    A version assertion goes stale the moment s3fs changes its scheme again.
    This calls the installed ``checksum`` with the ETag shape that actually
    broke, so it keeps testing the thing we care about.

    No network: the filesystem is never connected, only its lookup replaced.
    """
    fs = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)

    # _info, not info. Modern s3fs is async underneath: `checksum` is a sync
    # facade over `_checksum`, which awaits `self._info`. Patching the public
    # sync name leaves the real lookup in place, and the call goes out to the
    # network -- where anon=True earns a 403 that looks like a credentials
    # problem and is nothing of the kind.
    async def _info(path: str, **kwargs: object) -> dict[str, str]:
        return {"type": "file", "ETag": etag}

    monkeypatch.setattr(fs, "_info", _info)

    assert fs.checksum("recsys/gold/item_hourly_features/part-0.parquet") == int(
        "50aaa750191fe7fe37042ac66c3dc06d", 16
    )
