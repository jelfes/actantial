from .base import LLMBackend
from dotenv import load_dotenv
from openai import OpenAI


class OpenAIBackend(LLMBackend):
    """Backend for OpenAI models."""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None):
        """
        Initialize OpenAI backend.

        Args:
            model_name: OpenAI model identifier (e.g., "gpt-4o")
            api_key: API key for OpenAI service, if None, will be fetched from environment variable 'OPENAI_API_KEY'
        """
        super().__init__(model_name)
        self.model_name = model_name

        if api_key is None:
            load_dotenv()

            self.client = OpenAI()
        else:
            self.client = OpenAI(api_key=api_key)

    def generate(
        self, prompt: str, max_new_tokens: int = 2048, temperature: float = 0, **kwargs
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 for deterministic)
            **kwargs: Additional generation parameters

        Returns:
            Generated text (excluding prompt)
        """
        response = self.client.responses.create(
            model=self.model_name,
            instructions="You are a helpful assistant.",
            input=prompt,
            max_output_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

        return response.output_text
