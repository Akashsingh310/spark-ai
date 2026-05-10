from pyspark.sql import SparkSession
from spark_ai import AI

spark = SparkSession.builder \
    .appName("spark-ai-demo") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

df = spark.createDataFrame(
    [
        ("I am very happy with the results! The product worked exactly as expected.",),
        ("This is the worst experience ever. The delivery was late and support never replied.",),
        ("Need a callback in 10 minutes. The issue impacts production and blocks checkout.",),
    ],
    ["review"]
)

ai = AI()
result = (
    df.withColumn("sentiment", ai.sentiment("review"))
    .withColumn("topic", ai.classify("review", labels=["urgent", "complaint", "praise"]))
    .withColumn("summary", ai.summarize("review"))
)
result.show(truncate=False)
spark.stop()
