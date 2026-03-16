from .base import LLMBackend
from dotenv import load_dotenv  # type: ignore
from openai import OpenAI, NotFoundError


class OpenAIBackend(LLMBackend):
    """Backend for OpenAI models."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful assistant.",
        api_key: str = None,
    ):
        """
        Initialize OpenAI backend.

        Args:
            model_name: OpenAI model identifier (e.g., "gpt-4o")
            system_prompt: System prompt to use for all requests
            api_key: API key for OpenAI service, if None, will be fetched from environment variable 'OPENAI_API_KEY'
        """
        super().__init__(model_name)
        self.model_name = model_name

        if api_key is None:
            load_dotenv()
            self.client = OpenAI()
        else:
            self.client = OpenAI(api_key=api_key)

        self.system_prompt = system_prompt

        try:
            self.client.models.retrieve(model_name)
        except NotFoundError:
            raise ValueError(
                f"Model '{self.model_name}' was not found. "
                "Please check that you are using a valid OpenAI model identifier "
                "(e.g. 'gpt-4o-mini', 'gpt-4o', 'gpt-4.1-nano')."
            )

    def generate(
        self, prompt: str, max_new_tokens: int = 2048, temperature: float = 0, **kwargs
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature in [0, 2]. 0 is deterministic, higher values increase randomness.
            **kwargs: Additional generation parameters

        Returns:
            Generated text (excluding prompt)
        """
        if not 0 <= temperature <= 2:
            raise ValueError(
                f"temperature must be between 0 and 2 for the OpenAI API (got {temperature})."
            )

        response = self.client.responses.create(
            model=self.model_name,
            instructions=self.system_prompt,
            input=prompt,
            max_output_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

        return response.output_text
