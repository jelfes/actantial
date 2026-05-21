from .base import LLMBackend
from dotenv import load_dotenv
from openai import OpenAI, NotFoundError
from typing import Any, Optional


class OpenAIBackend(LLMBackend):
    """
    Backend for OpenAI API models.

    Wraps the OpenAI Responses API to provide text generation compatible with
    the actantial pipeline.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful assistant.",
        api_key: Optional[str] = None,
    ):
        """
        Initialise the OpenAI backend and validate the model.

        Args:
            model_name: OpenAI model identifier (e.g., ``"gpt-4o"``, ``"gpt-4o-mini"``).
            system_prompt: System-level instruction passed to the model on every request.
            api_key: OpenAI API key. If ``None``, read from the ``OPENAI_API_KEY``
                environment variable.
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
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt string.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature in [0, 2]; higher values increase randomness.
                Defaults to 0 for deterministic output.
            **kwargs: Additional parameters passed to the OpenAI Responses API.

        Returns:
            The generated text string, excluding the input prompt.
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
