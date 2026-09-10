from pyspark.sql import SparkSession

from common.config import Settings


def get_spark(settings: Settings, app: str = "recsys") -> SparkSession:
    """Create a Spark session with the configured tuning.

    Args:
        settings: The root configuration object.
        app: Name for the Spark application.

    Returns:
        A Spark session.
    """
    return (
        SparkSession.builder.appName(app)
        .master(settings.spark.master)
        .config("spark.driver.memory", settings.spark.driver_memory)
        .config("spark.sql.shuffle.partitions", settings.spark.shuffle_partitions)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
    )
