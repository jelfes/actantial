import torch

BACKENDS = ["anthropic", "huggingface", "openai"]

SEED = 815

GENERATION_DEFAULTS = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "max_new_tokens": 2048,
}

DTYPE_MAP = {
    "auto": "auto",
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
