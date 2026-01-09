from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, model: str, **kwargs):
        """
        Initialize the backend.

        Args:
            model: Model identifier (HF model path, API model name, etc.)
            **kwargs: Backend-specific configuration
        """
        self.model = model
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

    @abstractmethod
    def cleanup(self):
        """Clean up resources (unload model, close connections, etc.)"""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
