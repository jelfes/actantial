# actantial/backends/huggingface.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .base import LLMBackend
from actantial.config import DTYPE_MAP


class HuggingFaceBackend(LLMBackend):
    """Backend for local HuggingFace models.

    Quantization via bits-and-bytes is supported, but only when a CUDA GPU
    is available. Attempting to enable quantisation on MPS or a CPU-only
    environment will raise a :class:`RuntimeError`.
    """

    def __init__(
        self,
        model_name: str,
        quantisation: bool = False,
        torch_dtype: str = "float16",
        **kwargs,
    ):
        """
        Initialize HuggingFace backend.

        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3-8B")
            torch_dtype: Data type ("auto", "float16", "bfloat16")
            **kwargs: Additional model/tokenizer arguments

        Raises:
            RuntimeError: if ``quantisation`` is requested but no CUDA GPU is
                detected (bits-and-bytes only supports CUDA devices).
        """
        super().__init__(model_name, **kwargs)
        self.model_name = model_name.split("/")[-1]

        # Convert torch_dtype string to actual dtype
        torch_dtype = DTYPE_MAP.get(torch_dtype, "float16")

        # Set quantization configuration if needed
        quant_config = None
        if quantisation:
            # BitsAndBytes quantization currently only works with CUDA backends.
            # If we are running on Apple MPS or CPU-only, inform the user early.
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Quantisation via bits-and-bytes requires a CUDA GPU. "
                    "Detected no CUDA device (MPS or CPU only). "
                    "Disable quantisation or switch to a CUDA-capable machine."
                )
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        # Load model and tokenizer
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config,
            **kwargs,
        )

        print("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        **kwargs,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling (FALSE for deterministic output)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated text (excluding prompt)
        """

        # Tokenize input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs.to(self.model.device)

        # Run model inference
        with torch.no_grad():
            model_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                **kwargs,
            )

        # Decode generated tokens to text
        output = self.tokenizer.decode(
            model_outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return output

    def cleanup(self):
        """Unload model and free GPU memory."""
        # TODO: assess use and revise
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Model unloaded")
