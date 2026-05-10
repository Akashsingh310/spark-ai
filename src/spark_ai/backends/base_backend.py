from typing import Protocol

class InferenceBackend(Protocol):
    """Protocol that all model backends must implement."""
    def predict(self, texts: list[str]) -> list[str]:
        """Run inference on a list of texts and return predictions."""
        ...

    def classify(self, texts: list[str], labels: list[str]) -> list[dict[str, float | str]]:
        """Run zero-shot classification for provided candidate labels."""
        ...

    def summarize(self, texts: list[str]) -> list[str]:
        """Summarize a list of texts."""
        ...
