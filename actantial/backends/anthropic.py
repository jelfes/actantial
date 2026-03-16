from .base import LLMBackend
from dotenv import load_dotenv  # type: ignore
from anthropic import Anthropic, NotFoundError


class AnthropicBackend(LLMBackend):
    """Backend for Anthropic models."""

    def __init__(
        self,
        model_name: str = "claude-haiku-4-5",
        system_prompt: str = "You are a helpful assistant.",
        api_key: str = None,
    ):
        """
        Initialize Anthropic backend.

        Args:
            model_name: Anthropic model identifier (e.g., "claude-haiku-4-5")
            system_prompt: System prompt to use for all requests
            api_key: API key for Anthropic service, if None, will be fetched from environment variable 'ANTHROPIC_API_KEY'
        """
        super().__init__(model_name)
        self.model_name = model_name

        if api_key is None:
            load_dotenv()
            self.client = Anthropic()
        else:
            self.client = Anthropic(api_key=api_key)

        self.system_prompt = system_prompt

        try:
            self.client.models.retrieve(model_name)
        except NotFoundError:
            raise ValueError(
                f"Model '{self.model_name}' was not found. "
                "Please check that you are using a valid Anthropic model identifier "
                "(e.g. 'claude-haiku-4-5', 'claude-sonnet-4-6', 'claude-opus-4-6')."
            )

    def generate(
        self, prompt: str, max_new_tokens: int = 2048, temperature: float = 0, **kwargs
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature in [0, 1]. 0 is deterministic, 1 is maximum randomness.
            **kwargs: Additional generation parameters

        Returns:
            Generated text (excluding prompt)
        """
        if not 0 <= temperature <= 1:
            raise ValueError(
                f"temperature must be between 0 and 1 for the Anthropic API (got {temperature})."
            )

        response = self.client.messages.create(
            model=self.model_name,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

        return response.content[0].text
