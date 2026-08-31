"""Typed configs for the recommender model.

Every threshold, path and tuning knob the pipeline uses lives here and in
``conf/config.yml``

Values may be overridden from the environment without editing a file, using the
``RECSYS_`` prefix and ``__`` to descend into nested models::

    RECSYS_SPLIT__HOLDOUT_DAYS=1 uv run python -m data_pipeline.ingest.mind

That is how CI runs the whole pipeline over a single day of data as a smoke
test while a laptop runs the real thing.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


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

    Attributes:
        holdout_days: Length of the held-out window at the end of the corpus.
        min_user_impressions: Impressions a user must have in *train* to enter
            the test set.
    """

    holdout_days: int = 1
    min_user_impressions: int = 3


class FilterConfig(BaseModel):
    """Corpus filters applied between bronze and silver.

    Each of these moves your metrics, so each belongs in ``docs/evaluation.md``
    with its measured effect on row count. An unstated filter is how offline
    numbers stop meaning anything.

    Note:
        There is NO filter for zero-click impressions. These are real observations
        -- a user was shown five articles and wanted none -- and they carry most
        of the calibration signal.

    Attributes:
        min_item_impressions: Articles shown fewer times than this are dropped;
            below this threshold nothing is learnable about them.
        max_impressions_per_user: Cap on impressions retained per user. A handful
            of very heavy users would otherwise dominate any row-averaged metric
            while representing nobody.
    """

    min_item_impressions: int = 5
    max_impressions_per_user: int = 500


class Settings(BaseSettings):
    """Root configuration object, assembled from YAML and the environment.

    Precedence is highest-first: explicit keyword arguments, then environment
    variables, then ``conf/config.yml``. The ``RECSYS_`` prefix avoids
    collisions and ``__`` descends into nested models, so
    ``RECSYS_SPARK__SHUFFLE_PARTITIONS=8`` reaches
    :attr:`SparkConfig.shuffle_partitions` and beats the file.

    Attributes:
        paths: Required
        spark: Optional; spark runtime tuning.
        split: Optional; temporal split parameters.
        filter: Optional; corpus filters.
    """

    model_config = SettingsConfigDict(
        env_prefix="RECSYS_",
        env_nested_delimiter="__",
        yaml_file="conf/config.yml",
    )

    paths: Paths
    spark: SparkConfig = SparkConfig()
    split: SplitConfig = SplitConfig()
    filter: FilterConfig = FilterConfig()

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
            YamlConfigSettingsSource(settings_cls, deep_merge=True),
        )


def load_settings() -> Settings:
    """Load and validate configuration.

    Reads ``conf/config.yml`` relative to the process working directory, then
    applies any ``RECSYS_``-prefixed environment variables over it.

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
    return Settings()  # type: ignore[call-arg]
