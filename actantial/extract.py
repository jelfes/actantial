from jinja2 import Template
from typing import Dict

from actantial.backends.base import LLMBackend


def extract_actants(
    input_text: str, backend: LLMBackend, prompt_template: Template
) -> Dict:
    """
    Extract actantial roles from a text file.

    Args:
        input_text: The text to process
        backend: An initialized LLMBackend instance
        prompt_template: A template for the prompt

    Returns:
        actant_dict: Extracted actantial roles as a dictionary
    """

    # Render prompt from template
    prompt = prompt_template.render(text=input_text)

    # Generate with backend
    output = backend.generate(prompt)

    return output
