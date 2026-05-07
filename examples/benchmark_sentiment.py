import time

from pyspark.sql import SparkSession
from spark_ai import AI


def run_benchmark(rows: int = 20000, partitions: int = 8) -> None:
    spark = (
        SparkSession.builder.appName("spark-ai-benchmark")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        .getOrCreate()
    )

    base = [
        ("I love this product and would buy it again.",),
        ("This is terrible and I want a refund.",),
        ("Pretty good overall, but shipping was slow.",),
        ("Worst experience ever.",),
        ("Amazing quality and quick delivery!",),
    ]
    data = [base[i % len(base)] for i in range(rows)]

    df = spark.createDataFrame(data, ["review"]).repartition(partitions)
    ai = AI()

    t0 = time.perf_counter()
    out = df.withColumn("sentiment", ai.sentiment("review"))
    count = out.count()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    print(f"rows={count}")
    print(f"partitions={partitions}")
    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"rows_per_second={count / elapsed:.1f}")

    spark.stop()


if __name__ == "__main__":
    run_benchmark()
