import pytest
from pyspark.sql import SparkSession


def create_spark_or_skip(master: str, app_name: str) -> SparkSession:
    try:
        return (
            SparkSession.builder
            .master(master)
            .appName(app_name)
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
    except Exception as exc:
        pytest.skip(f"Spark unavailable in this environment: {exc}")


@pytest.fixture(scope="session")
def spark():
    """Provides a Spark session for the whole test session."""
    spark = create_spark_or_skip("local[2]", "spark-ai-tests")
    yield spark
    spark.stop()
