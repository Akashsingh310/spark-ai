from pyspark.sql import SparkSession
from spark_ai import AI

spark = SparkSession.builder \
    .appName("spark-ai-demo") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

df = spark.createDataFrame(
    [
        ("I am very happy with the results!",),
        ("This is the worst experience ever.",),
        ("Need a callback in 10 minutes",),
    ],
    ["review"]
)

ai = AI()
result = (
    df.withColumn("sentiment", ai.sentiment("review"))
    .withColumn("topic", ai.classify("review", labels=["urgent", "complaint", "praise"]))
)
result.show(truncate=False)
spark.stop()
