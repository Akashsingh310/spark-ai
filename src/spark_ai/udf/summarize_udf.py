import pandas as pd
from spark_ai.config import AIConfig
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
from spark_ai.exceptions import InferenceError
from spark_ai.logging_config import configure_logger

logger = configure_logger(__name__)


def build_summarize_udf(config: AIConfig):
    backend_instance = None

    def _get_backend():
        nonlocal backend_instance
        if backend_instance is None:
            from spark_ai.backends.huggingface_backend import HuggingFaceBackend

            backend_instance = HuggingFaceBackend(config)
        return backend_instance

    @pandas_udf(StringType())  # type: ignore[call-overload]
    def summarize_udf(texts: pd.Series) -> pd.Series:
        backend = _get_backend()
        cleaned = texts.fillna("").astype(str)
        try:
            return pd.Series(backend.summarize(cleaned.tolist()))
        except Exception as e:
            logger.error(f"Summarization inference failed: {e}")
            raise InferenceError("UDF execution failed") from e

    return summarize_udf
