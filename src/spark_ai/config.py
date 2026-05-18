from dataclasses import dataclass


@dataclass(slots=True)
class AIConfig:
    """Central configuration for spark-ai."""

    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    # Used by zero-shot classify (`AI.classify`); MNLI-style models work best here.
    zero_shot_model_name: str = "valhalla/distilbart-mnli-12-3"
    summarization_model_name: str = "sshleifer/distilbart-cnn-12-6"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    # -1 = CPU, 0 = first GPU
    device: int = -1
    # If True, batch size is automatically selected per worker from the first batch.
    # A positive batch_size still acts as an upper bound.
    auto_tune_batch_size: bool = False
    max_length: int = 512
    summarization_max_length: int = 130
    summarization_min_length: int = 30
    log_level: str = "INFO"
    cache_dir: str | None = None
