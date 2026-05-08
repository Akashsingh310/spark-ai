from spark_ai import AI
from spark_ai.config import AIConfig
from tests.conftest import create_spark_or_skip


def test_sentiment_udf_serializes_on_local_cluster():
    spark = create_spark_or_skip("local-cluster[2,1,1024]", "spark-ai-serialization-test")
    try:
        ai = AI(AIConfig(batch_size=8))
        df = spark.createDataFrame(
            [("I love this!",), ("I hate this.",), (None,)],
            ["text"],
        ).repartition(2)

        out = df.withColumn("sentiment", ai.sentiment("text")).collect()
        assert len(out) == 3
        assert all(row["sentiment"] in ("POSITIVE", "NEGATIVE") for row in out)
    finally:
        spark.stop()
