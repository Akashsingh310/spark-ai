from dataclasses import dataclass

@dataclass(slots=True)
class AIConfig:
    """Central configuration for spark-ai."""
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    batch_size: int = 32
    device: int = -1              
    max_length: int = 512
    log_level: str = "INFO"
    cache_dir: str | None = None  
