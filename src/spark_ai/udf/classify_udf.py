import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
from pyspark.sql.pandas.functions import PandasUDFType

from spark_ai.backends.huggingface_backend import HuggingFaceBackend
from spark_ai.config import AIConfig
from spark_ai.exceptions import InferenceError
from spark_ai.logging_config import configure_logger

logger = configure_logger(__name__)


def build_classify_udf(config: AIConfig, labels: list[str]):
    """Create a vectorized zero-shot classification UDF for fixed labels."""
    backend_instance: HuggingFaceBackend | None = None

    def _get_backend() -> HuggingFaceBackend:
        nonlocal backend_instance
        if backend_instance is None:
            backend_instance = HuggingFaceBackend(config)
        return backend_instance

    @pandas_udf(StringType(), functionType=PandasUDFType.SCALAR)
    def classify_udf(texts: pd.Series) -> pd.Series:
        try:
            backend = _get_backend()
            cleaned = texts.fillna("").astype(str)
            predictions = backend.classify(cleaned.tolist(), labels)
            return pd.Series([p["label"] for p in predictions])
        except Exception as e:
            logger.error(f"Zero-shot classify inference failed: {e}")
            raise InferenceError("UDF execution failed") from e

    return classify_udf
