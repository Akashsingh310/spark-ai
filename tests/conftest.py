import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark():
    """Provides a Spark session for the whole test session."""
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("spark-ai-tests")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    yield spark
    spark.stop()
