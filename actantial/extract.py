from typing import Optional
from jinja2 import Template

from actantial.backends.base import LLMBackend


def extract_actants(
    input_text: str, backend: LLMBackend, prompt_template: Template, actor_labels: Optional[list] = None, object_labels: Optional[list] = None
) -> str:
    """
    Extract actantial roles from a text using an LLM backend.

    Renders the prompt template with the input text and optional label
    constraints, then passes the result to the backend for generation.
    Central function of the pipeline, called by the runner for each text.

    Args:
        input_text: The text to annotate.
        backend: An initialised [`LLMBackend`][actantial.backends.base.LLMBackend]
            instance.
        prompt_template: A Jinja2 template that accepts a ``text`` variable
            at minimum. For closed-set annotation, the template can also
            accept ``actor_labels`` and ``object_labels``.
        actor_labels: Predefined list of actor labels for closed-set
            annotation. Only used if the template supports this variable.
        object_labels: Predefined list of object labels for closed-set
            annotation. Only used if the template supports this variable.

    Returns:
        The raw text output from the backend, typically a JSON string
            mapping each actant role to its extracted value.
    """

    # Render prompt from template
    prompt = prompt_template.render(text=input_text, actor_labels=actor_labels, object_labels=object_labels)

    # Generate with backend
    output = backend.generate(prompt)

    return output
