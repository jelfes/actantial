"""
Integration tests for LLM backends.

These tests make real API calls or load real models and are skipped by default.
Run explicitly with:

    pytest -m api           # Anthropic and OpenAI (requires API keys in .env)
    pytest -m gpu           # HuggingFace (requires a GPU)
    pytest -m integration   # all of the above
"""

import pytest


@pytest.mark.api
@pytest.mark.integration
class TestAnthropicBackendAPI:

    def test_generate_returns_non_empty_string(self):
        from actantial.backends.anthropic import AnthropicBackend

        backend = AnthropicBackend(model_name="claude-haiku-4-5")
        result = backend.generate("Reply with one word: hello.", temperature=0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_model_raises_at_init(self):
        from actantial.backends.anthropic import AnthropicBackend

        with pytest.raises(ValueError, match="not found"):
            AnthropicBackend(model_name="not-a-real-model")

    def test_temperature_out_of_range_raises(self):
        from actantial.backends.anthropic import AnthropicBackend

        backend = AnthropicBackend(model_name="claude-haiku-4-5")
        with pytest.raises(ValueError, match="temperature"):
            backend.generate("hello", temperature=2.0)


@pytest.mark.api
@pytest.mark.integration
class TestOpenAIBackendAPI:

    def test_generate_returns_non_empty_string(self):
        from actantial.backends.openai import OpenAIBackend

        backend = OpenAIBackend(model_name="gpt-4.1-nano")
        result = backend.generate("Reply with one word: hello.", temperature=0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_model_raises_at_init(self):
        from actantial.backends.openai import OpenAIBackend

        with pytest.raises(ValueError, match="not found"):
            OpenAIBackend(model_name="not-a-real-model")

    def test_temperature_out_of_range_raises(self):
        from actantial.backends.openai import OpenAIBackend

        backend = OpenAIBackend(model_name="gpt-4.1-nano")
        with pytest.raises(ValueError, match="temperature"):
            backend.generate("hello", temperature=3.0)


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
