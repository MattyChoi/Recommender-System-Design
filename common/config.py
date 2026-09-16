"""Typed configs for the recommender model.

Every threshold, path and tuning knob the pipeline uses lives here and in
``conf/config.yml``

Values may be overridden from the environment without editing a file, using the
``RECSYS_`` prefix and ``__`` to descend into nested models::

    RECSYS_SPLIT__HOLDOUT_DAYS=1 uv run python -m data_pipeline.ingest.bronze

That is how CI runs the whole pipeline over a single day of data as a smoke
test while a laptop runs the real thing.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Where the YAML lives, as a ContextVar rather than a module global so a nested
# or concurrent load cannot see another caller's value.
#
# The default is relative, which is correct for every process started from the
# repo root. The Feast CLI is the exception: it chdirs into feature_repo/ before
# importing definition.py, so that file passes an absolute root to
# load_settings() instead of duplicating the values it needs.
_DEFAULT_YAML = Path("conf/config.yml")
_yaml_file: ContextVar[Path] = ContextVar("_yaml_file", default=_DEFAULT_YAML)


class Paths(BaseModel):
    """Filesystem roots for the training data following the medallion layer architecture.

    ``raw`` is the MIND TSVs as downloaded (train/ and dev/; MIND-large's test/
    archive has its labels withheld)
    The other three are derived and can be rebuilt from it, which is
    why all four are gitignored but only ``raw`` is expensive to lose.

    Attributes:
        raw: MIND's ``train/``, ``dev/`` and ``test/`` directories as shipped.
        bronze: Raw-but-typed Parquet -- one row per item shown, partitioned by date.
        silver: Bronze joined to article metadata and integer ID indices.
        gold: Point-in-time-correct feature tables, ready for training.
    """

    raw: Path
    bronze: Path
    silver: Path
    gold: Path


class StorageConfig(BaseModel):
    """Where the gold layer is written and read from.

    Two schemes address the same bucket. Spark writes through Hadoop's S3A
    client, which registers ``s3a://``; Feast reads through pyarrow, which knows
    only ``s3://``. They are separate clients with separate credential plumbing,
    so both get configured, and their agreement is an assumption this config
    makes rather than one it can enforce.

    Attributes:
        backend: ``local`` resolves against :attr:`Paths.gold`; ``s3`` resolves
            against ``{bucket}/{gold_prefix}``. The default is s3

            The two places that must NOT inherit it pin it themselves rather
            than relying on the environment: the synthetic test corpus
            (``data_pipeline/tests/conftest.py``) and ``make demo``, both of
            which have to run with no container up.
        bucket: Bucket name. Created once by hand; nothing creates it on demand,
            and a missing bucket surfaces as a 404 from the S3 client rather
            than as anything self-explanatory.
        endpoint_url: MinIO's S3 API. The host-visible address, because Spark
            and the Feast CLI both run on the host -- a process inside the
            compose network would use ``http://minio:9000``.
        gold_prefix: Key prefix under the bucket, mirroring the local layout so
            the two backends are diffable.
    """

    backend: Literal["local", "s3"] = "s3"
    bucket: str = "recsys"
    endpoint_url: str = "http://localhost:9000"
    gold_prefix: str = "gold"


class SparkConfig(BaseModel):
    """Spark runtime tuning.

    Defaults target a single developer machine, not a cluster. Keeping them here
    means pointing at real infrastructure later is a YAML edit rather than a code
    change.

    Attributes:
        master: Spark master URL. ``local[*]`` runs in-process with one worker
            thread per core; ``local[1]`` removes concurrency when debugging
            something non-deterministic.
        driver_memory: JVM heap for the driver. In local mode there are no
            separate executors, so the driver does all the work and this is the
            entire memory budget. The Spark default of 1g will not survive
            the MIND-large dataset.
        shuffle_partitions: Partitions produced by each shuffle (every groupBy
            and every non-broadcast join). Spark's default of 200 is tuned for a
            cluster; locally it yields near-empty partitions whose scheduling
            costs more than the work. Two to four times the core count is the
            rule of thumb.
    """

    master: str = "local[*]"
    driver_memory: str = "8g"
    shuffle_partitions: int = 32


class SplitConfig(BaseModel):
    """Temporal split parameters.

    Used for ABLATIONS only, over a single split's timeline. The project's
    primary evaluation is MIND's official train/dev boundary, which already
    satisfies the protocol and is not produced from these values.

    Attributes:
        holdout_days: Length of the held-out window at the end of the corpus,
            applied TWICE -- test is the last ``holdout_days``, validation the
            ``holdout_days`` before it. Boundaries are snapped to midnight so
            they align with the ``dt`` partitioning.
        min_user_impressions: Impressions a user must have in *train* to enter
            the test set, guarding against measuring cold-start and calling it
            personalization. Zero disables the filter
    """

    holdout_days: int = 1
    min_user_impressions: int = 3


class SessionConfig(BaseModel):
    """Sessionization parameters.

    A session groups one user's impressions into a burst of activity, split
    wherever they go quiet for longer than ``gap_minutes``. This is COARSER
    than MIND's ``impression_id``, which already groups the items shown
    together on a single page view -- sessions group several page views into
    one sitting.

    Nothing in the batch pipeline needs sessions; they exist for the streaming
    session-window aggregates in Part Q and the recently-viewed retriever in
    Part I.

    Attributes:
        gap_minutes: Inactivity gap that ends a session. Guide 5.2 suggests 30,
            which is a web-analytics convention rather than a measurement --
            and arguably long for news, where reading is short and bursty.
            Measure the inter-impression gap distribution and set this from the
            elbow rather than inheriting the default.
    """

    gap_minutes: int = 30


class ReplayConfig(BaseModel):
    """Kafka replay harness parameters (``data_pipeline/replay``).

    The harness reads bronze and produces it to Kafka as though it were
    arriving now, so the streaming jobs have something to consume.

    Attributes:
        topic: Destination topic. Must match the Flink DDL's ``'topic'``.
        bootstrap_servers: ``localhost:9092`` from the host, ``kafka:29092``
            from inside the compose network -- the broker advertises both.
        speed: Event-time seconds per wall-clock second. 3600 replays an hour
            per second, which walks MIND's week in about three minutes. 0 is
            unthrottled, which is what backfills and tests want.
        max_lateness_seconds: Upper bound on injected delay. THIS MUST NOT
            EXCEED the allowed lateness in the consumer's watermark -- the
            Flink table declares ``ts - INTERVAL '30' SECOND``. Raise one
            without the other and the extra records are dropped as too late
            rather than handled as late, which looks like data loss.
        late_fraction: Share of records to delay. A few percent is enough to
            exercise the path; more turns every window into a straggler.
        seed: Fixes the draw sequence, so a window replays identically.
    """

    topic: str = "impressions"
    bootstrap_servers: str = "localhost:9092"
    speed: float = 3600.0
    max_lateness_seconds: int = 30
    late_fraction: float = 0.02
    seed: int = 0


class FilterConfig(BaseModel):
    """Filter thresholds that were measured and REJECTED. Nothing reads these.

    * ``min_item_impressions = 5`` discards 0.3% of train rows but **50.2% of
      dev rows**, because 32.9% of the evaluation catalogue never appears in
      training.
    * ``max_impressions_per_user = 500`` never binds: the busiest user in
      MIND-small has 62 impressions.

    The pipeline is therefore lossless from raw to silver, and the row-count
    contracts in ``data_pipeline/tests/test_contracts.py`` assert exactly that.

    Attributes:
        min_item_impressions: Minimum impressions for an article to be
            considered learnable. Not enforced; see above.
        max_impressions_per_user: Cap on impressions retained per user, to stop
            heavy users dominating row-averaged metrics. Not enforced, and
            unreachable at this dataset size.
    """

    min_item_impressions: int = 5
    max_impressions_per_user: int = 500


class MLflowConfig(BaseModel):
    """Where training runs are recorded.

    Attributes:
        enabled: False skips tracking entirely. The server is a compose service,
            so anything that must run without ``make up`` -- CI, every test --
            turns it off rather than depending on Docker.
        tracking_uri: The server's host-visible address. A process inside the
            compose network would use ``http://mlflow:9001``.
        experiment: Experiment name. The manual's verify step is
            ``mlflow runs list --experiment-name two_tower``.
    """

    enabled: bool = True
    tracking_uri: str = "http://localhost:5001"
    experiment: str = "two_tower"


class Settings(BaseSettings):
    """Root configuration object, assembled from YAML and the environment.

    Precedence is highest-first: explicit keyword arguments, then environment
    variables, then ``conf/config.yml``. The ``RECSYS_`` prefix avoids
    collisions and ``__`` descends into nested models, so
    ``RECSYS_SPARK__SHUFFLE_PARTITIONS=8`` reaches
    :attr:`SparkConfig.shuffle_partitions` and beats the file.

    Attributes:
        paths: Required
        storage: Optional; gold-layer location and object-store endpoint.
        spark: Optional; spark runtime tuning.
        split: Optional; temporal split parameters.
        session: Optional; sessionization parameters.
        replay: Optional; Kafka replay harness parameters.
        filter: Optional; corpus filters.
        mlflow: Optional; where training runs are recorded.
    """

    model_config = SettingsConfigDict(
        env_prefix="RECSYS_",
        env_nested_delimiter="__",
    )

    paths: Paths
    storage: StorageConfig = StorageConfig()
    spark: SparkConfig = SparkConfig()
    split: SplitConfig = SplitConfig()
    session: SessionConfig = SessionConfig()
    replay: ReplayConfig = ReplayConfig()
    filter: FilterConfig = FilterConfig()
    mlflow: MLflowConfig = MLflowConfig()

    def gold_uri(self, name: str, scheme: str = "s3a") -> str:
        """Resolve a gold table to a location the calling client can open.

        Args:
            name: Table name under the gold layer, such as
                ``item_hourly_features``. May contain separators, so
                ``training_examples/train`` resolves the same way on both
                backends.
            scheme: ``s3a`` for Spark, ``s3`` for pyarrow and Feast. Ignored
                when the backend is local, where there is no scheme to choose.

        Returns:
            A local filesystem path, or a URI in the requested scheme.
        """
        if self.storage.backend == "local":
            return str(self.paths.gold / name)
        return f"{scheme}://{self.storage.bucket}/{self.storage.gold_prefix}/{name}"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order the configuration sources, highest priority first.

        ``deep_merge=True`` lets an environment variable override one nested
        field without discarding the rest of the block it belongs to: setting
        ``RECSYS_SPLIT__HOLDOUT_DAYS`` must not wipe ``min_user_impressions``.

        The YAML path is read from ``_yaml_file`` at call time rather than
        fixed in ``model_config``, which is what lets :func:`load_settings`
        accept a root for callers that cannot rely on the working directory.

        Args:
            settings_cls: The settings class being built.
            init_settings: Values passed as keyword arguments.
            env_settings: Values read from the environment.
            dotenv_settings: Values read from a dotenv file. Unused.
            file_secret_settings: Values read from a secrets directory. Unused.

        Returns:
            The active sources in descending priority order.
        """
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=_yaml_file.get(), deep_merge=True),
        )


def load_settings(root: Path | None = None) -> Settings:
    """Load and validate configuration.

    Reads ``conf/config.yml`` relative to the process working directory, then
    applies any ``RECSYS_``-prefixed environment variables over it.

    Args:
        root: Repository root to resolve the configuration against. Omit it in
            anything launched from the repo root, which is everything except
            the Feast CLI -- that chdirs into the feature repo before importing
            its definitions, so ``definition.py`` passes its own anchored root
            rather than keeping a second copy of the values it needs.

            This anchors the DATA PATHS as well as the YAML file. Both are
            relative by default, so resolving only the file would hand back
            ``data/gold`` to a caller whose working directory is not the repo
            root -- which fails as a missing directory underneath it, not as a
            configuration error.

    Returns:
        A validated :class:`Settings`.

    Raises:
        pydantic.ValidationError: If a required key is missing or a value has
            the wrong type. A missing or unreadable ``conf/config.yml`` surfaces
            this way too -- as a complaint about the absent ``paths`` block
            rather than as ``FileNotFoundError``.
    """
    # mypy sees `paths` as a required argument because pydantic synthesises
    # __init__ from the field declarations. At runtime it is supplied by the
    # YAML source, which mypy cannot see. This is the documented friction
    # between BaseSettings and strict mode, not a real call error.
    if root is None:
        return Settings()  # type: ignore[call-arg]

    root = Path(root).resolve()

    # reset() rather than leaving the value set: a load with an explicit root
    # must not change what the next default load reads.
    token = _yaml_file.set(root / "conf" / "config.yml")
    try:
        settings = Settings()  # type: ignore[call-arg]
    finally:
        _yaml_file.reset(token)

    # An absolute path in the YAML is taken as deliberate and left alone.
    settings.paths = Paths(
        **{name: value if value.is_absolute() else root / value for name, value in settings.paths}
    )
    return settings
