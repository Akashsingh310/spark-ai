class SparkAIError(Exception):
    """Base exception for all spark-ai errors."""
    pass

class ModelLoadError(SparkAIError):
    """Raised when a model backend fails to load."""
    pass

class InferenceError(SparkAIError):
    """Raised when inference fails on the executor."""
    pass

class InvalidColumnError(SparkAIError):
    """Raised when the provided column is invalid."""
    pass
