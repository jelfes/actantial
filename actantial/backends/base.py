from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

from actantial.template_utils import resolve_template_path, list_templates_for_model


class LLMBackend(ABC):
    """
    Abstract base class for all LLM backends in the actantial pipeline.

    Concrete backends (Anthropic, OpenAI, HuggingFace) inherit from this class
    and implement [`generate`][actantial.backends.base.LLMBackend.generate]. Shared template utilities are defined here
    so they are available to all backends.
    """

    def __init__(self, model_name: str, **kwargs: Any):
        """
        Set the model name and store any extra backend-specific config.

        Args:
            model_name: Model identifier used by the underlying LLM service
                or framework (e.g. an API model name or a HuggingFace path).
            **kwargs: Backend-specific configuration stored in ``self.config``.
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Generation parameters (e.g. ``temperature``, ``max_new_tokens``).

        Returns:
            The generated text string, excluding the input prompt.
        """
        pass

    def list_templates(
        self,
        templates_dir: Union[str, Path] = Path(__file__).parent.parent / "templates",
    ) -> dict[str, list[str]]:
        """
        List the prompt templates available for this backend's model.

        Args:
            templates_dir: Root directory containing per-model template
                subdirectories and the shared ``default/`` directory.
                Defaults to the built-in ``templates/`` folder.

        Returns:
            A dict with keys ``"model_specific"`` and ``"default"``, each
                mapping to a sorted list of template names available for
                this model, without the ``.txt`` extension.
        """

        return list_templates_for_model(templates_dir, self.model_name)

    def show_template(
        self,
        template: str,
        templates_dir: Union[str, Path] = Path(__file__).parent.parent / "templates",
    ) -> None:
        """
        Print the raw source of a prompt template.

        Args:
            template: Name of the template to display, with or without
                the ``.txt`` extension.
            templates_dir: Root directory containing per-model template
                subdirectories and the shared ``default/`` directory.
                Defaults to the built-in ``templates/`` folder.
        """

        template = template if template.endswith(".txt") else template + ".txt"
        template_path, source_dir = resolve_template_path(
            templates_dir, self.model_name, template
        )

        with open(template_path) as f:
            source = f.read()

        print(
            f"Template '{template}' for model '{self.model_name}' "
            f"(source: {source_dir}):\n"
        )
        print(source)

    def cleanup(self):
        """Clean up resources (unload model, close connections, etc.). No-op by default."""
        pass
