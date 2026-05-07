from transformers import pipeline
from spark_ai.exceptions import ModelLoadError
from spark_ai.logging_config import configure_logger
from spark_ai.config import AIConfig

logger = configure_logger(__name__)

class HuggingFaceBackend:
    """Hugging Face transformers pipeline backend.
    
    IMPORTANT: The pipeline is loaded lazily and kept as a class-level
    singleton so that it is only loaded once per executor worker.
    """
    _pipeline = None
    _config = None

    def __init__(self, config: AIConfig):
        self._config = config

    def _load_pipeline(self):
        if HuggingFaceBackend._pipeline is None:
            logger.info(f"Loading Hugging Face model: {self._config.model_name}")
            try:
                HuggingFaceBackend._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self._config.model_name,
                    device=self._config.device,
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise ModelLoadError(f"Could not load model '{self._config.model_name}'") from e

    def predict(self, texts: list[str]) -> list[str]:
        self._load_pipeline()
        # Batch inference
        results = HuggingFaceBackend._pipeline(texts, truncation=True, max_length=self._config.max_length)
        return [r["label"] for r in results]
