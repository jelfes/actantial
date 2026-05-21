from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union
from jinja2 import Environment, FileSystemLoader


class LLMBackend(ABC):
    """
    Abstract base class for all LLM backends in the actantial pipeline.

    Concrete backends (Anthropic, OpenAI, HuggingFace) inherit from this class
    and implement :meth:`generate`. Shared template utilities are defined here
    so they are available to all backends. Backends support use as context
    managers, calling :meth:`cleanup` on exit.
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
    ) -> list:
        """
        List the prompt templates available for this backend's model.

        Args:
            templates_dir: Root directory containing per-model template
                subdirectories. Defaults to the built-in ``templates/`` folder.

        Returns:
            List of template names available for this model, without the
                ``.txt`` extension.
        """

        templates_dir = Path(templates_dir) / self.model_name
        if not templates_dir.exists():
            raise FileNotFoundError(f"Directory not found: {templates_dir}.")
        return [f.stem for f in templates_dir.glob("*.txt")]

    def show_template(
        self,
        template_name: str,
        templates_dir: Union[str, Path] = Path(__file__).parent.parent / "templates",
    ) -> None:
        """
        Print a prompt template with placeholder values substituted.

        Renders the template with dummy values so the structure is visible
        without requiring real input data.

        Args:
            template_name: Name of the template to display, with or without
                the ``.txt`` extension.
            templates_dir: Root directory containing per-model template
                subdirectories. Defaults to the built-in ``templates/`` folder.
        """

        template_name = (
            template_name if template_name.endswith(".txt") else template_name + ".txt"
        )
        environment = Environment(loader=FileSystemLoader(templates_dir))

        try:
            template = environment.get_template(
                str(Path(self.model_name, template_name))
            )
        except Exception as e:
            error_message = f"Error loading template {e}. Please ensure that the template exists in the templates/{self.model_name} directory and is named correctly."
            raise FileNotFoundError(error_message)

        print(f"Template '{template_name}' for model '{self.model_name}':\n")
        print(
            template.render(
                text="[TEXT]",
                actor_labels="[ACTOR_LABELS]",
                object_labels="[OBJECT_LABELS]",
            )
        )

    def cleanup(self):
        """Clean up resources (unload model, close connections, etc.). No-op by default."""
        pass

