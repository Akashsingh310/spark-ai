from typing import Any, Callable
from pyspark.sql.functions import col as spark_col
from spark_ai.config import AIConfig
from spark_ai.udf.classify_udf import build_classify_udf
from spark_ai.udf.sentiment_udf import build_sentiment_udf

class AI:
    """Public API for AI-powered DataFrame transformations."""

    def __init__(self, config: AIConfig | None = None):
        self._config = config or AIConfig()
        self._sentiment_udf = build_sentiment_udf(self._config)
        self._classify_udf_cache: dict[
            tuple[str, ...], Callable[[Any], Any]
        ] = {}

    def sentiment(self, column_name: str):
        """Apply sentiment analysis on a column.

        Args:
            column_name: Name of the text column.

        Returns:
            pyspark.sql.Column with POSITIVE/NEGATIVE labels.
        """
        return self._sentiment_udf(spark_col(column_name))

    def classify(self, text_col: str, labels: list[str]):
        """Classify free text into custom labels using zero-shot classification."""
        if not labels:
            raise ValueError("labels must not be empty")
        cache_key = tuple(labels)
        classify_udf = self._classify_udf_cache.get(cache_key)
        if classify_udf is None:
            classify_udf = build_classify_udf(self._config, labels)
            self._classify_udf_cache[cache_key] = classify_udf
        return classify_udf(spark_col(text_col))
