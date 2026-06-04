"""
actantial: a toolkit for analysing narratives using Greimas' Actantial Model.

Provides LLM-based extraction of actant roles from text, along with utilities
for loading results and computing inter-annotator agreement. API-based backends
(Anthropic, OpenAI) are available by default. The HuggingFace backend for
local GPU inference requires a separate install: ``pip install actantial[huggingface]``.
"""

__version__ = "0.1.0"

from actantial.runner import run_extract
from actantial.io import load_annotations, load_actors
from actantial.validation import compare_annotations
from actantial.backends.anthropic import AnthropicBackend
from actantial.backends.openai import OpenAIBackend

__all__ = [
    "run_extract",
    "load_annotations",
    "load_actors",
    "compare_annotations",
    "AnthropicBackend",
    "OpenAIBackend",
]
