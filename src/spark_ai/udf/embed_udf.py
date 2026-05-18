import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.pandas.functions import PandasUDFType
from pyspark.sql.types import ArrayType, DoubleType

from spark_ai.backends.huggingface_backend import HuggingFaceBackend
from spark_ai.config import AIConfig
from spark_ai.exceptions import InferenceError
from spark_ai.logging_config import configure_logger

logger = configure_logger(__name__)


def build_embed_udf(config: AIConfig):
    """Create a vectorized embedding UDF bound to a specific config."""
    backend_instance: HuggingFaceBackend | None = None

    def _get_backend() -> HuggingFaceBackend:
        nonlocal backend_instance
        if backend_instance is None:
            backend_instance = HuggingFaceBackend(config)
        return backend_instance

    @pandas_udf(ArrayType(DoubleType()), functionType=PandasUDFType.SCALAR)
    def embed_udf(texts: pd.Series) -> pd.Series:
        try:
            backend = _get_backend()
            cleaned = texts.fillna("").astype(str)
            embeddings = backend.embed(cleaned.tolist())
            return pd.Series(embeddings)
        except Exception as e:
            logger.error(f"Embedding inference failed: {e}")
            raise InferenceError("UDF execution failed") from e

    return embed_udf
