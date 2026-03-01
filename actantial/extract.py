from jinja2 import Template
from typing import Dict

from actantial.backends.base import LLMBackend


def extract_actants(
    input_text: str, backend: LLMBackend, prompt_template: Template, actor_labels: list = None, object_labels: list = None
) -> Dict:
    """
    Extract actantial roles from a text file.

    Args:
        input_text: The text to process
        backend: An initialized LLMBackend instance
        prompt_template: A template for the prompt
        actor_labels: Optional list of actor labels for annotation with predefined labels. Requires a template that supports labels.
        object_labels: Optional list of object labels for annotation with predefined labels. Requires a template that supports labels.

    Returns:
        actant_dict: Extracted actantial roles as a dictionary
    """

    # Render prompt from template
    prompt = prompt_template.render(text=input_text, actor_labels=actor_labels, object_labels=object_labels)

    # Generate with backend
    output = backend.generate(prompt)

    return output
