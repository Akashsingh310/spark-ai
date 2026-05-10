from unittest.mock import patch

import pytest

from spark_ai.backends.huggingface_backend import HuggingFaceBackend
from spark_ai.config import AIConfig
from spark_ai.exceptions import ModelLoadError


@pytest.fixture(autouse=True)
def reset_backend_singleton():
    HuggingFaceBackend._pipeline = None
    HuggingFaceBackend._classifier = None
    HuggingFaceBackend._summarizer = None
    HuggingFaceBackend._zero_shot_model_loaded = None
    HuggingFaceBackend._summarization_model_loaded = None
    yield
    HuggingFaceBackend._pipeline = None
    HuggingFaceBackend._classifier = None
    HuggingFaceBackend._summarizer = None
    HuggingFaceBackend._zero_shot_model_loaded = None
    HuggingFaceBackend._summarization_model_loaded = None


def test_predict_extracts_labels_and_uses_configured_batch_size():
    calls = []

    def fake_pipeline_factory(*args, **kwargs):
        def fake_pipeline(texts, **inference_kwargs):
            calls.append(inference_kwargs)
            return [{"label": "POSITIVE"} for _ in texts]

        return fake_pipeline

    config = AIConfig(batch_size=12, auto_tune_batch_size=False)
    backend = HuggingFaceBackend(config)

    with patch("spark_ai.backends.huggingface_backend.pipeline", side_effect=fake_pipeline_factory):
        labels = backend.predict(["good", "great"])

    assert labels == ["POSITIVE", "POSITIVE"]
    assert calls[0]["batch_size"] == 12
    assert calls[0]["max_length"] == config.max_length
    assert calls[0]["truncation"] is True


def test_auto_tune_locks_after_warmup_batches():
    def fake_pipeline_factory(*args, **kwargs):
        def fake_pipeline(texts, **inference_kwargs):
            return [{"label": "NEGATIVE"} for _ in texts]

        return fake_pipeline

    config = AIConfig(batch_size=64, auto_tune_batch_size=True, device=-1)
    backend = HuggingFaceBackend(config)

    warmup_batches = [
        ["x" * 500],  # long -> small candidate
        ["x" * 220],  # medium candidate
        ["x" * 20],   # short candidate
    ]

    with patch("spark_ai.backends.huggingface_backend.pipeline", side_effect=fake_pipeline_factory):
        for texts in warmup_batches:
            backend.predict(texts)
        locked = backend._resolved_batch_size
        backend.predict(["x" * 500])

    # CPU candidates for lengths above are: 12, 24, 48 -> median lock should be 24.
    assert locked == 24
    assert backend._resolved_batch_size == 24


def test_model_load_error_is_wrapped():
    config = AIConfig()
    backend = HuggingFaceBackend(config)

    with patch("spark_ai.backends.huggingface_backend.pipeline", side_effect=RuntimeError("boom")):
        with pytest.raises(ModelLoadError):
            backend.predict(["text"])


def test_zero_shot_classify_returns_top_label_and_score():
    def fake_pipeline_factory(*args, **kwargs):
        def fake_pipeline(texts, labels, multi_label=False):
            return [
                {"labels": ["urgent", "normal"], "scores": [0.93, 0.07]}
                for _ in texts
            ]

        return fake_pipeline

    backend = HuggingFaceBackend(AIConfig())
    with patch("spark_ai.backends.huggingface_backend.pipeline", side_effect=fake_pipeline_factory):
        predictions = backend.classify(["Need this now"], ["urgent", "normal"])

    assert predictions == [{"label": "urgent", "score": 0.93}]


def test_summarize_returns_summary_text_and_honors_config_lengths():
    calls = []

    def fake_pipeline_factory(task, *args, **kwargs):
        if task != "summarization":
            raise AssertionError("expected summarization pipeline")

        def fake_pipeline(texts, **inference_kwargs):
            calls.append(inference_kwargs)
            return [{"summary_text": "short summary"} for _ in texts]

        return fake_pipeline

    config = AIConfig(
        summarization_max_length=77,
        summarization_min_length=20,
        batch_size=10,
        auto_tune_batch_size=False,
    )
    backend = HuggingFaceBackend(config)
    with patch("spark_ai.backends.huggingface_backend.pipeline", side_effect=fake_pipeline_factory):
        summaries = backend.summarize(["very long review"])

    assert summaries == ["short summary"]
    assert calls[0]["max_length"] == 77
    assert calls[0]["min_length"] == 20
    assert calls[0]["batch_size"] == 10
