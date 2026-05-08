import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
from pyspark.sql.pandas.functions import PandasUDFType

from spark_ai.backends.huggingface_backend import HuggingFaceBackend
from spark_ai.config import AIConfig
from spark_ai.exceptions import InferenceError
from spark_ai.logging_config import configure_logger

logger = configure_logger(__name__)

def build_sentiment_udf(config: AIConfig):
    """Create a vectorized sentiment UDF bound to a specific config."""
    # This backend instance is created once per Python worker process.
    backend_instance: HuggingFaceBackend | None = None

    def _get_backend() -> HuggingFaceBackend:
        nonlocal backend_instance
        if backend_instance is None:
            backend_instance = HuggingFaceBackend(config)
        return backend_instance

    @pandas_udf(StringType(), functionType=PandasUDFType.SCALAR)
    def _sentiment_udf(texts: pd.Series) -> pd.Series:
        """Vectorized sentiment UDF that batches inference."""
        try:
            cleaned = texts.fillna("").astype(str)
            backend = _get_backend()
            predictions = backend.predict(cleaned.tolist())
            return pd.Series(predictions)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise InferenceError("UDF execution failed") from e

    return _sentiment_udf


# Backward-compatible default UDF.
sentiment_udf = build_sentiment_udf(AIConfig())
