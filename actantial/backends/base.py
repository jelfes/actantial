from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
from jinja2 import Environment, FileSystemLoader


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the backend.

        Args:
            model_name: Model identifier (HF model path, API model name, etc.)
            **kwargs: Backend-specific configuration
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt
            **kwargs: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text string
        """
        pass

    def list_templates(
        self,
        templates_dir: Union[str, Path] = Path(__file__).parent.parent / "templates",
    ) -> list:
        """
        List available templates for this backend.

        Args:
            templates_dir: Base directory where templates are stored. Should contain subdirectories for the model.

        Returns:
            List of template names (without .txt extension)
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
        Display the content of a template with placeholders.

        Args:
            template_name: Name of the template to display (with or without .txt extension)
            templates_dir: Base directory where templates are stored. Should contain subdirectories for the model.
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

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
