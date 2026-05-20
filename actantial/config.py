"""
Package-wide constants for the actantial annotation pipeline.

Defines the supported LLM backends, the default random seed, and the
canonical list of actant roles from Greimas's Actantial Model.
"""

BACKENDS = ["anthropic", "huggingface", "openai"]

SEED = 815
ACTANTS = ["Subject", "Object", "Sender", "Receiver", "Helper", "Opponent"]
