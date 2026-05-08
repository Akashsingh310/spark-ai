# spark-infer-ai

Distributed AI inference for PySpark DataFrames.

[![Tests](https://github.com/Akashsingh310/spark-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/Akashsingh310/spark-ai/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/spark-infer-ai.svg)](https://pypi.org/project/spark-infer-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

`spark-ai` brings model-powered text processing directly into Spark transformations using a simple API.  

It is designed for portability and works anywhere Spark runs: local development, EMR, Dataproc, Kubernetes, or on-prem clusters.

## Features

- Spark-native sentiment analysis API
- Vectorized execution with `pandas_udf` for better throughput than row-wise Python UDFs
- Hugging Face Transformers backend
- Null-safe text handling for production pipelines
- Clean package structure for extension with additional AI tasks/backends

## Installation

```bash
pip install spark-infer-ai
```

## Requirements

- Python 3.10+
- Apache Spark 3.5+
- Java runtime compatible with your Spark distribution

Core dependencies are installed automatically:

- `pyspark`
- `pandas`
- `pyarrow`
- `transformers`
- `torch`

## Quick Start

```python
from pyspark.sql import SparkSession
from spark_ai import AI

spark = (
    SparkSession.builder
    .appName("spark-ai-demo")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)

df = spark.createDataFrame(
    [
        ("I love this product!",),
        ("This is the worst experience ever.",),
    ],
    ["review"],
)

ai = AI()
result = df.withColumn("sentiment", ai.sentiment("review"))
result.show(truncate=False)
```

Expected sentiment labels are typically `POSITIVE` / `NEGATIVE` (model-dependent).

## API

### `AI`

Primary interface for DataFrame AI transformations.

#### `AI.sentiment(column_name: str)`

Applies sentiment analysis to a text column and returns a Spark `Column`.

```python
result = df.withColumn("sentiment", ai.sentiment("review"))
```

## Performance Notes

`spark-ai` uses a vectorized Pandas UDF and batched Hugging Face inference internally.

For best performance in production:

- Enable Arrow:
  - `spark.sql.execution.arrow.pyspark.enabled=true`
- Tune Spark partitions to match your cluster resources
- Tune `batch_size` for your hardware, or enable `auto_tune_batch_size=True`
- Run benchmarks on representative text lengths and data sizes

Model-loading behavior:

- Spark may run multiple Python workers per executor
- Each Python worker keeps its own singleton model instance
- That means model reuse is per worker process, not globally shared across all workers

You can use the included benchmark script:

```bash
python examples/benchmark_sentiment.py
```

Example benchmark output:

```text
rows=20000
partitions=8
elapsed_seconds=6.383
rows_per_second=3133.1
```

## Logging and Runtime Warnings

Common Spark startup warnings like:

- `NativeCodeLoader: Unable to load native-hadoop library...`
- JDK incubator module notices

are typically informational in local environments and do not indicate a failure.

## Development

Clone and install in editable mode:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```


## Project Structure

```text
src/spark_ai/
  ai.py                  # Public API
  config.py              # Central configuration
  udf/sentiment_udf.py   # Vectorized Spark UDF
  backends/              # Inference backend implementations
tests/unit/              # Unit tests
examples/                # Usage and benchmark scripts
```

