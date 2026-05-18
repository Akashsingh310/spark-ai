from typing import Any, ClassVar

import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from spark_ai.exceptions import ModelLoadError
from spark_ai.logging_config import configure_logger
from spark_ai.config import AIConfig

logger = configure_logger(__name__)


class HuggingFaceBackend:
    """Hugging Face transformers pipeline backend.

    IMPORTANT: The pipeline is loaded lazily and kept as a class-level
    singleton so that it is only loaded once per executor worker.
    """

    _pipeline: ClassVar[Any | None] = None
    _classifier: ClassVar[Any | None] = None
    _summarizer_model: ClassVar[Any | None] = None
    _summarizer_tokenizer: ClassVar[Any | None] = None
    _zero_shot_model_loaded: ClassVar[str | None] = None
    _summarization_model_loaded: ClassVar[str | None] = None
    _embedder: ClassVar[Any | None] = None
    _embedder_tokenizer: ClassVar[Any | None] = None
    _embedding_model_loaded: ClassVar[str | None] = None
    _AUTO_TUNE_WARMUP_BATCHES = 3

    def __init__(self, config: AIConfig):
        self._config = config
        self._resolved_batch_size: int | None = None
        self._auto_tune_candidates: list[int] = []
        self._batch_size_locked = False

    def _load_pipeline(self) -> None:
        if HuggingFaceBackend._pipeline is None:
            logger.info(f"Loading Hugging Face model: {self._config.model_name}")
            try:
                HuggingFaceBackend._pipeline = pipeline(
                    "text-classification",
                    model=self._config.model_name,
                    device=self._config.device,
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise ModelLoadError(
                    f"Could not load model '{self._config.model_name}'"
                ) from e

    def _resolve_batch_size(self, texts: list[str]) -> int:
        configured = self._config.batch_size
        if not self._config.auto_tune_batch_size and configured > 0:
            return configured
        if self._batch_size_locked and self._resolved_batch_size is not None:
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

        if not self._config.auto_tune_batch_size:
            self._resolved_batch_size = candidate
            logger.info(
                "Using inference batch size=%s (auto_tune=%s, avg_text_len=%.1f)",
                self._resolved_batch_size,
                self._config.auto_tune_batch_size,
                avg_len,
            )
            return self._resolved_batch_size

        if self._auto_tune_candidates and self._auto_tune_candidates[-1] != candidate:
            logger.warning(
                "Auto-tune candidate changed from %s to %s before lock-in (avg_text_len=%.1f)",
                self._auto_tune_candidates[-1],
                candidate,
                avg_len,
            )
        self._auto_tune_candidates.append(candidate)

        if len(self._auto_tune_candidates) >= self._AUTO_TUNE_WARMUP_BATCHES:
            # Use the median of warm-up candidates to avoid locking onto an outlier first batch.
            sorted_candidates = sorted(self._auto_tune_candidates)
            self._resolved_batch_size = sorted_candidates[len(sorted_candidates) // 2]
            self._batch_size_locked = True
        else:
            # During warm-up, use the current candidate and continue adapting.
            self._resolved_batch_size = candidate

        logger.info(
            "Using inference batch size=%s (auto_tune=%s, avg_text_len=%.1f, warmup_batches=%s/%s)",
            self._resolved_batch_size,
            self._config.auto_tune_batch_size,
            avg_len,
            len(self._auto_tune_candidates),
            self._AUTO_TUNE_WARMUP_BATCHES,
        )
        return self._resolved_batch_size

    def predict(self, texts: list[str]) -> list[str]:
        self._load_pipeline()
        batch_size = self._resolve_batch_size(texts)
        pipeline_obj = HuggingFaceBackend._pipeline
        if pipeline_obj is None:
            raise ModelLoadError(f"Could not load model '{self._config.model_name}'")
        # Batch inference with tuned/configured batch size.
        results = pipeline_obj(
            texts,
            truncation=True,
            max_length=self._config.max_length,
            batch_size=batch_size,
        )
        return [r["label"] for r in results]

    def classify(
        self, texts: list[str], labels: list[str]
    ) -> list[dict[str, float | str]]:
        z_model = self._config.zero_shot_model_name
        if (
            HuggingFaceBackend._classifier is None
            or HuggingFaceBackend._zero_shot_model_loaded != z_model
        ):
            try:
                logger.info(f"Loading zero-shot classifier model: {z_model}")
                HuggingFaceBackend._classifier = pipeline(
                    "zero-shot-classification",
                    model=z_model,
                    device=self._config.device,
                )
                HuggingFaceBackend._zero_shot_model_loaded = z_model
            except Exception as e:
                logger.error(f"Failed to load zero-shot classifier: {e}")
                raise ModelLoadError(
                    f"Could not load zero-shot classification model '{z_model}'"
                ) from e

        classifier = HuggingFaceBackend._classifier
        if classifier is None:
            raise ModelLoadError("Could not load zero-shot classification model")

        results = classifier(texts, labels, multi_label=False)
        return [
            {"label": r["labels"][0], "score": float(r["scores"][0])} for r in results
        ]

    def _load_summarizer(self) -> None:
        s_model = self._config.summarization_model_name
        if (
            HuggingFaceBackend._summarizer_model is None
            or HuggingFaceBackend._summarization_model_loaded != s_model
        ):
            try:
                logger.info(f"Loading summarization model: {s_model}")
                HuggingFaceBackend._summarizer_tokenizer = (
                    AutoTokenizer.from_pretrained(
                        s_model,
                        cache_dir=self._config.cache_dir,
                    )
                )
                HuggingFaceBackend._summarizer_model = (
                    AutoModelForSeq2SeqLM.from_pretrained(
                        s_model,
                        cache_dir=self._config.cache_dir,
                    )
                )
                HuggingFaceBackend._summarizer_model.to(self._embedder_device())
                HuggingFaceBackend._summarizer_model.eval()
                HuggingFaceBackend._summarization_model_loaded = s_model
                logger.info("Summarization model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load summarization model: {e}")
                raise ModelLoadError(
                    f"Could not load summarization model '{s_model}'"
                ) from e

    def summarize(self, texts: list[str]) -> list[str]:
        self._load_summarizer()
        model = HuggingFaceBackend._summarizer_model
        tokenizer = HuggingFaceBackend._summarizer_tokenizer
        if model is None or tokenizer is None:
            raise ModelLoadError("Could not load summarization model")

        batch_size = self._resolve_batch_size(texts)
        device = self._embedder_device()
        summaries: list[str] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **encoded,
                    max_length=self._config.summarization_max_length,
                    min_length=self._config.summarization_min_length,
                    num_beams=4,
                )
            summaries.extend(
                tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            )

        return summaries

    def _embedder_device(self) -> torch.device:
        if self._config.device >= 0 and torch.cuda.is_available():
            return torch.device(f"cuda:{self._config.device}")
        return torch.device("cpu")

    def _load_embedder(self) -> None:
        model_name = self._config.embedding_model_name
        if (
            HuggingFaceBackend._embedder is None
            or HuggingFaceBackend._embedding_model_loaded != model_name
        ):
            try:
                logger.info(f"Loading embedding model: {model_name}")
                HuggingFaceBackend._embedder_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self._config.cache_dir,
                )
                HuggingFaceBackend._embedder = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=self._config.cache_dir,
                )
                HuggingFaceBackend._embedder.to(self._embedder_device())
                HuggingFaceBackend._embedder.eval()
                HuggingFaceBackend._embedding_model_loaded = model_name
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise ModelLoadError(
                    f"Could not load embedding model '{model_name}'"
                ) from e

    @staticmethod
    def _mean_pool(
        token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load_embedder()
        model = HuggingFaceBackend._embedder
        tokenizer = HuggingFaceBackend._embedder_tokenizer
        if model is None or tokenizer is None:
            raise ModelLoadError(
                f"Could not load embedding model '{self._config.embedding_model_name}'"
            )

        batch_size = self._resolve_batch_size(texts)
        device = self._embedder_device()
        vectors: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
            pooled = self._mean_pool(
                outputs.last_hidden_state, encoded["attention_mask"]
            )
            vectors.extend(pooled.cpu().tolist())

        return vectors
