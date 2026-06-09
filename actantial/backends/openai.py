import warnings

from .base import LLMBackend
from dotenv import load_dotenv
from openai import OpenAI, NotFoundError, BadRequestError
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
        temperature: Optional[float] = 0,
        api_key: Optional[str] = None,
    ):
        """
        Initialise the OpenAI backend, validate the model, and run a validation call.

        A lightweight validation call (``max_output_tokens=16``) is made at construction
        time to verify connectivity and model availability. When ``temperature`` is a float,
        the call also checks whether the model accepts that parameter; if not, a warning is
        issued and temperature is omitted from all subsequent calls.

        Args:
            model_name: OpenAI model identifier (e.g., ``"gpt-4o"``, ``"gpt-4o-mini"``).
            system_prompt: System-level instruction passed to the model on every request.
            temperature: Sampling temperature in [0, 2]; higher values increase randomness.
                Defaults to 0 for deterministic output. Pass ``None`` to omit the parameter
                entirely (e.g. for reasoning models that do not accept it). If a float is
                given but the model rejects it, a warning is issued and it is ignored.
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

        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError(
                f"temperature must be between 0 and 2 for the OpenAI API (got {temperature})."
            )
        self.temperature = temperature

        try:
            self.client.models.retrieve(model_name)
        except NotFoundError:
            raise ValueError(
                f"Model '{self.model_name}' was not found. "
                "Please check that you are using a valid OpenAI model identifier "
                "(e.g. 'gpt-4o-mini', 'gpt-4o', 'gpt-4.1-nano')."
            )

        self._supports_temperature = self._validate_api_call()

    def _validate_api_call(self) -> bool:
        """
        Run a lightweight API call to verify connectivity and temperature support.

        Returns True if temperature should be included in generation calls, False otherwise.
        """
        call_kwargs: dict[str, Any] = dict(
            model=self.model_name,
            instructions=self.system_prompt,
            input="test",
            max_output_tokens=16,
        )
        if self.temperature is not None:
            call_kwargs["temperature"] = self.temperature

        try:
            self.client.responses.create(**call_kwargs)
        except BadRequestError as e:
            if e.param == "temperature":
                msg = (
                    f"Model '{self.model_name}' does not support the 'temperature' parameter "
                    "— it will be ignored for all calls with this backend."
                )
                warnings.warn(msg, UserWarning, stacklevel=3)
                return False
            raise

        return self.temperature is not None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt string.
            max_new_tokens: Maximum number of tokens to generate.
            **kwargs: Additional parameters passed to the OpenAI Responses API.

        Returns:
            The generated text string, excluding the input prompt.
        """

        call_kwargs: dict[str, Any] = dict(
            model=self.model_name,
            instructions=self.system_prompt,
            input=prompt,
            max_output_tokens=max_new_tokens,
            **kwargs,
        )
        if self._supports_temperature:
            call_kwargs["temperature"] = self.temperature

        response = self.client.responses.create(**call_kwargs)
        return response.output_text
