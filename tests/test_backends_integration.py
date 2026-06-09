"""
Integration tests for LLM backends.

These tests make real API calls or load real models and are skipped by default.
Run explicitly with:

    pytest -m api           # Anthropic and OpenAI (requires API keys in .env)
    pytest -m gpu           # HuggingFace (requires a GPU)
    pytest -m integration   # all of the above
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.api
@pytest.mark.integration
class TestAnthropicBackendAPI:

    def test_generate_returns_non_empty_string(self):
        from actantial.backends.anthropic import AnthropicBackend

        backend = AnthropicBackend(model_name="claude-haiku-4-5", temperature=0)
        result = backend.generate("Reply with one word: hello.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_model_raises_at_init(self):
        from actantial.backends.anthropic import AnthropicBackend

        with pytest.raises(ValueError, match="not found"):
            AnthropicBackend(model_name="not-a-real-model")

    def test_temperature_out_of_range_raises(self):
        from actantial.backends.anthropic import AnthropicBackend

        with pytest.raises(ValueError, match="temperature"):
            AnthropicBackend(model_name="claude-haiku-4-5", temperature=2.0)


@pytest.mark.api
@pytest.mark.integration
class TestOpenAIBackendAPI:

    def test_generate_returns_non_empty_string(self):
        from actantial.backends.openai import OpenAIBackend

        backend = OpenAIBackend(model_name="gpt-4.1-nano", temperature=0)
        result = backend.generate("Reply with one word: hello.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_model_raises_at_init(self):
        from actantial.backends.openai import OpenAIBackend

        with pytest.raises(ValueError, match="not found"):
            OpenAIBackend(model_name="not-a-real-model")

    def test_temperature_out_of_range_raises(self):
        from actantial.backends.openai import OpenAIBackend

        with pytest.raises(ValueError, match="temperature"):
            OpenAIBackend(model_name="gpt-4.1-nano", temperature=3.0)

    def test_temperature_none_skips_parameter(self):
        from actantial.backends.openai import OpenAIBackend

        backend = OpenAIBackend(model_name="gpt-4.1-nano", temperature=None)
        result = backend.generate("Reply with one word: hello.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unsupported_temperature_warns(self):
        from actantial.backends.openai import OpenAIBackend

        with pytest.warns(UserWarning, match="temperature"):
            backend = OpenAIBackend(model_name="o4-mini", temperature=0)
        assert not backend._supports_temperature


class TestHuggingFaceBackendUnit:
    """Unit tests for HuggingFaceBackend — no GPU required, model loading is mocked."""

    def _make_backend(self, **kwargs):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = None

        with patch("actantial.backends.huggingface.AutoTokenizer") as mock_tok_cls, \
             patch("actantial.backends.huggingface.AutoModelForCausalLM") as mock_model_cls, \
             patch("builtins.print"):
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer
            mock_model_cls.from_pretrained.return_value = mock_model

            from actantial.backends.huggingface import HuggingFaceBackend
            backend = HuggingFaceBackend(
                repository="google",
                model_name="gemma-3-270m-it",
                **kwargs,
            )

        return backend

    def test_default_sampling_params(self):
        backend = self._make_backend()
        assert backend.temperature is None
        assert backend.do_sample is False
        assert backend.top_p is None
        assert backend.top_k is None

    def test_sampling_params_stored_at_init(self):
        backend = self._make_backend(temperature=0.7, do_sample=True, top_p=0.9, top_k=50)
        assert backend.temperature == 0.7
        assert backend.do_sample is True
        assert backend.top_p == 0.9
        assert backend.top_k == 50

    def test_model_path_constructed_correctly(self):
        backend = self._make_backend()
        assert backend.model_path == "google/gemma-3-270m-it"
        assert backend.repository == "google"
        assert backend.model_name == "gemma-3-270m-it"


@pytest.mark.gpu
@pytest.mark.integration
class TestHuggingFaceBackendGPU:

    def test_generate_returns_non_empty_string(self):
        from actantial.backends.huggingface import HuggingFaceBackend

        backend = HuggingFaceBackend(
            repository="google",
            model_name="gemma-3-270m-it",
            quantisation=False,
        )
        result = backend.generate("List three animals.", max_new_tokens=100)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_sampling_params(self):
        from actantial.backends.huggingface import HuggingFaceBackend

        backend = HuggingFaceBackend(
            repository="google",
            model_name="gemma-3-270m-it",
            quantisation=False,
            temperature=0.7,
            do_sample=True,
        )
        result = backend.generate("List three animals.", max_new_tokens=100)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_cleanup_frees_model(self):
        from actantial.backends.huggingface import HuggingFaceBackend

        backend = HuggingFaceBackend(
            repository="google",
            model_name="gemma-3-270m-it",
            quantisation=False,
        )
        backend.cleanup()
        assert not hasattr(backend, "model")
        assert not hasattr(backend, "tokenizer")
