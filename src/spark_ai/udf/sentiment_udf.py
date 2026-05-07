import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

from spark_ai.backends.huggingface_backend import HuggingFaceBackend
from spark_ai.config import AIConfig
from spark_ai.exceptions import InferenceError
from spark_ai.logging_config import configure_logger

logger = configure_logger(__name__)

# This module-level backend is instantiated only once per executor worker
_backend_instance = None

def _get_backend() -> HuggingFaceBackend:
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = HuggingFaceBackend(AIConfig())
    return _backend_instance

@pandas_udf(StringType())
def sentiment_udf(texts: pd.Series) -> pd.Series:
    """Vectorized sentiment UDF that batches inference."""
    try:
        # Fill NA and convert to strings
        cleaned = texts.fillna("").astype(str)
        backend = _get_backend()
        predictions = backend.predict(cleaned.tolist())
        return pd.Series(predictions)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise InferenceError("UDF execution failed") from e
