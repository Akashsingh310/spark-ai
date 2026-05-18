from unittest.mock import patch

import pytest
import torch

from spark_ai.backends.huggingface_backend import HuggingFaceBackend
from spark_ai.config import AIConfig
from spark_ai.exceptions import ModelLoadError


@pytest.fixture(autouse=True)
def reset_backend_singleton():
    HuggingFaceBackend._pipeline = None
    HuggingFaceBackend._classifier = None
    HuggingFaceBackend._summarizer_model = None
    HuggingFaceBackend._summarizer_tokenizer = None
    HuggingFaceBackend._zero_shot_model_loaded = None
    HuggingFaceBackend._summarization_model_loaded = None
    HuggingFaceBackend._embedder = None
    HuggingFaceBackend._embedder_tokenizer = None
    HuggingFaceBackend._embedding_model_loaded = None
    yield
    HuggingFaceBackend._pipeline = None
    HuggingFaceBackend._classifier = None
    HuggingFaceBackend._summarizer_model = None
    HuggingFaceBackend._summarizer_tokenizer = None
    HuggingFaceBackend._zero_shot_model_loaded = None
    HuggingFaceBackend._summarization_model_loaded = None
    HuggingFaceBackend._embedder = None
    HuggingFaceBackend._embedder_tokenizer = None
    HuggingFaceBackend._embedding_model_loaded = None


def test_predict_extracts_labels_and_uses_configured_batch_size():
    calls = []

    def fake_pipeline_factory(*args, **kwargs):
        def fake_pipeline(texts, **inference_kwargs):
            calls.append(inference_kwargs)
            return [{"label": "POSITIVE"} for _ in texts]

        return fake_pipeline

    config = AIConfig(batch_size=12, auto_tune_batch_size=False)
    backend = HuggingFaceBackend(config)

    with patch(
        "spark_ai.backends.huggingface_backend.pipeline",
        side_effect=fake_pipeline_factory,
    ):
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
        ["x" * 20],  # short candidate
    ]

    with patch(
        "spark_ai.backends.huggingface_backend.pipeline",
        side_effect=fake_pipeline_factory,
    ):
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

    with patch(
        "spark_ai.backends.huggingface_backend.pipeline",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(ModelLoadError):
            backend.predict(["text"])


def test_zero_shot_classify_returns_top_label_and_score():
    def fake_pipeline_factory(*args, **kwargs):
        def fake_pipeline(texts, labels, multi_label=False):
            return [
                {"labels": ["urgent", "normal"], "scores": [0.93, 0.07]} for _ in texts
            ]

        return fake_pipeline

    backend = HuggingFaceBackend(AIConfig())
    with patch(
        "spark_ai.backends.huggingface_backend.pipeline",
        side_effect=fake_pipeline_factory,
    ):
        predictions = backend.classify(["Need this now"], ["urgent", "normal"])

    assert predictions == [{"label": "urgent", "score": 0.93}]


def test_summarize_returns_summary_text_and_honors_config_lengths():
    generate_calls = []

    class FakeModel:
        def parameters(self):
            yield torch.tensor(0.0)

        def to(self, _device):
            return self

        def eval(self):
            return None

        def generate(self, **kwargs):
            generate_calls.append(kwargs)
            batch_size = kwargs["input_ids"].shape[0]
            return torch.zeros(batch_size, 4, dtype=torch.long)

    class FakeTokenizer:
        def __call__(self, batch, **kwargs):
            return {
                "input_ids": torch.ones(len(batch), 3, dtype=torch.long),
                "attention_mask": torch.ones(len(batch), 3, dtype=torch.long),
            }

        def batch_decode(self, output_ids, skip_special_tokens=True):
            return ["short summary"] * len(output_ids)

    config = AIConfig(
        summarization_max_length=77,
        summarization_min_length=20,
        batch_size=10,
        auto_tune_batch_size=False,
    )
    backend = HuggingFaceBackend(config)
    with (
        patch(
            "spark_ai.backends.huggingface_backend.AutoTokenizer.from_pretrained",
            return_value=FakeTokenizer(),
        ),
        patch(
            "spark_ai.backends.huggingface_backend.AutoModelForSeq2SeqLM.from_pretrained",
            return_value=FakeModel(),
        ),
    ):
        summaries = backend.summarize(["very long review"])

    assert summaries == ["short summary"]
    assert generate_calls[0]["max_length"] == 77
    assert generate_calls[0]["min_length"] == 20


def test_embed_returns_mean_pooled_vectors():
    class FakeOutputs:
        def __init__(self, hidden_state: object):
            self.last_hidden_state = hidden_state

    class FakeModel:
        def __init__(self):
            self._device = "cpu"

        def parameters(self):
            yield torch.tensor(0.0)

        def to(self, _device):
            return self

        def eval(self):
            return None

        def __call__(self, **encoded):
            batch_size, seq_len = encoded["input_ids"].shape
            hidden_dim = 4
            hidden = torch.ones(batch_size, seq_len, hidden_dim)
            return FakeOutputs(hidden)

    class FakeTokenizer:
        def __call__(self, batch, **kwargs):
            return {
                "input_ids": torch.ones(len(batch), 3, dtype=torch.long),
                "attention_mask": torch.ones(len(batch), 3, dtype=torch.long),
            }

    config = AIConfig(batch_size=2, auto_tune_batch_size=False, max_length=128)
    backend = HuggingFaceBackend(config)

    with (
        patch(
            "spark_ai.backends.huggingface_backend.AutoTokenizer.from_pretrained",
            return_value=FakeTokenizer(),
        ),
        patch(
            "spark_ai.backends.huggingface_backend.AutoModel.from_pretrained",
            return_value=FakeModel(),
        ),
    ):
        vectors = backend.embed(["hello", "world", "spark"])

    assert len(vectors) == 3
    assert all(len(vector) == 4 for vector in vectors)
    assert vectors[0] == [1.0, 1.0, 1.0, 1.0]


def test_embed_model_load_error_is_wrapped():
    config = AIConfig()
    backend = HuggingFaceBackend(config)

    with patch(
        "spark_ai.backends.huggingface_backend.AutoTokenizer.from_pretrained",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(ModelLoadError):
            backend.embed(["text"])
