from pyspark.sql.types import StringType, StructType, StructField
from spark_ai import AI

def test_positive_sentiment(spark):
    ai = AI()
    df = spark.createDataFrame([("I love this product!",)], ["text"])
    result = df.withColumn("sentiment", ai.sentiment("text"))
    assert result.collect()[0]["sentiment"] == "POSITIVE"

def test_negative_sentiment(spark):
    ai = AI()
    df = spark.createDataFrame([("This is awful.",)], ["text"])
    result = df.withColumn("sentiment", ai.sentiment("text"))
    assert result.collect()[0]["sentiment"] == "NEGATIVE"

def test_null_text(spark):
    ai = AI()
    # Provide explicit schema so Spark knows the column type even with null
    schema = StructType([StructField("text", StringType(), True)])
    df = spark.createDataFrame([(None,)], schema=schema)
    result = df.withColumn("sentiment", ai.sentiment("text"))
    row = result.collect()[0]
    # The UDF should handle null gracefully (filled as empty string)
    assert row["sentiment"] is not None

def test_empty_string(spark):
    ai = AI()
    df = spark.createDataFrame([("",)], ["text"])
    result = df.withColumn("sentiment", ai.sentiment("text"))
    row = result.collect()[0]
    assert isinstance(row["sentiment"], str)

def test_multiple_rows(spark):
    ai = AI()
    data = [("Great job!",), ("Could be better.",), ("Terrible!",)]
    df = spark.createDataFrame(data, ["text"])
    result = df.withColumn("sentiment", ai.sentiment("text"))
    rows = result.collect()
    assert len(rows) == 3
    assert all(r["sentiment"] in ("POSITIVE", "NEGATIVE") for r in rows)
