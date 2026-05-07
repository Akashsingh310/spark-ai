from typing import Protocol

class InferenceBackend(Protocol):
    """Protocol that all model backends must implement."""
    def predict(self, texts: list[str]) -> list[str]:
        """Run inference on a list of texts and return predictions."""
        ...
