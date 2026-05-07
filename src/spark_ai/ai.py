from pyspark.sql.functions import col as spark_col
from spark_ai.config import AIConfig
from spark_ai.udf.sentiment_udf import build_sentiment_udf

class AI:
    """Public API for AI-powered DataFrame transformations."""

    def __init__(self, config: AIConfig | None = None):
        self._config = config or AIConfig()
        self._sentiment_udf = build_sentiment_udf(self._config)

    def sentiment(self, column_name: str):
        """Apply sentiment analysis on a column.

        Args:
            column_name: Name of the text column.

        Returns:
            pyspark.sql.Column with POSITIVE/NEGATIVE labels.
        """
        return self._sentiment_udf(spark_col(column_name))
