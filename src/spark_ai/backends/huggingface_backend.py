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
        self._resolved_batch_size: int | None = None

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

    def _resolve_batch_size(self, texts: list[str]) -> int:
        if self._resolved_batch_size is not None:
            return self._resolved_batch_size

        configured = self._config.batch_size
        if not self._config.auto_tune_batch_size and configured > 0:
            self._resolved_batch_size = configured
            return self._resolved_batch_size

        sample = texts[: min(len(texts), 256)]
        avg_len = (sum(len(t) for t in sample) / len(sample)) if sample else 0.0
        # Conservative heuristic: shorter text => larger batches.
        # GPU can usually handle higher throughput than CPU.
        candidate = 128 if self._config.device >= 0 else 48
        if avg_len > 350:
            candidate //= 4
        elif avg_len > 180:
            candidate //= 2

        candidate = max(8, min(256, candidate))
        if configured > 0:
            candidate = min(candidate, configured)

        self._resolved_batch_size = candidate
        logger.info(
            "Using inference batch size=%s (auto_tune=%s, avg_text_len=%.1f)",
            self._resolved_batch_size,
            self._config.auto_tune_batch_size,
            avg_len,
        )
        return self._resolved_batch_size

    def predict(self, texts: list[str]) -> list[str]:
        self._load_pipeline()
        batch_size = self._resolve_batch_size(texts)
        # Batch inference with tuned/configured batch size.
        results = HuggingFaceBackend._pipeline(
            texts,
            truncation=True,
            max_length=self._config.max_length,
            batch_size=batch_size,
        )
        return [r["label"] for r in results]
