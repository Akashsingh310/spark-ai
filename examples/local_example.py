from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, slice

from spark_ai import AI

spark = (
    SparkSession.builder.appName("spark-ai-demo")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)

df = spark.createDataFrame(
    [
        ("Great product, works perfectly.",),
        ("Terrible delivery and no support.",),
        ("Production is down, need help now.",),
    ],
    ["review"],
)

ai = AI()
result = (
    df.withColumn("sentiment", ai.sentiment("review"))
    .withColumn("topic", ai.classify("review", labels=["urgent", "complaint", "praise"]))
    .withColumn("summary", ai.summarize("review"))
    .withColumn("embedding", ai.embed("review"))
    .withColumn("embedding_dims", size("embedding"))
    .withColumn("embedding_preview", slice(col("embedding"), 1, 3))
)

result.select(
    "review",
    "sentiment",
    "topic",
    "summary",
    "embedding_dims",
    "embedding_preview",
).show(truncate=60)
spark.stop()
