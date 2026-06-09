# actantial/backends/huggingface.py

import gc
from typing import Any, Optional
from .base import LLMBackend

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    raise ImportError(
        "HuggingFace dependencies are not installed. "
        "Run: pip install actantial[huggingface]"
    )


class HuggingFaceBackend(LLMBackend):
    """
    Backend for locally loaded HuggingFace models.

    Loads the model and tokenizer from the HuggingFace Hub at initialisation.
    Quantisation via bitsandbytes (4-bit) is supported, but requires a CUDA GPU.
    """

    def __init__(
        self,
        repository: str,
        model_name: str,
        quantisation: bool = False,
        torch_dtype: str = "auto",
        temperature: Optional[float] = None,
        do_sample: bool = False,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Load the model and tokenizer from the HuggingFace Hub.

        Args:
            repository: HuggingFace repository name (e.g., ``deepseek-ai``).
            model_name: Model identifier within the repository (e.g., ``DeepSeek-R1-Distill-Qwen-32B``).
            quantisation: If ``True``, load the model in 4-bit precision using
                bitsandbytes. Requires a CUDA GPU.
            torch_dtype: Floating-point precision passed to ``from_pretrained``.
                Accepts ``"auto"`` (default), ``"float16"``, or ``"bfloat16"``.
            temperature: Sampling temperature; higher values increase randomness.
            do_sample: If ``True``, use sampling; defaults to ``False`` for
                deterministic (greedy) output.
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling parameter.
            **kwargs: Additional arguments passed to ``AutoModelForCausalLM.from_pretrained``.
        """

        model_path = "/".join([repository, model_name])

        super().__init__(model_path)
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.top_k = top_k
        self.model_name = model_name
        self.repository = repository
        self.model_path = model_path
        self.quantisation = quantisation

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
            # BitsAndBytesConfig requires a concrete dtype, not "auto"
            bnb_dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else torch.float16
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=bnb_dtype,
                bnb_4bit_use_double_quant=True,
            )

        # Load model and tokenizer
        print(f"Loading model {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch_dtype,
            quantization_config=quant_config,
            trust_remote_code=True,
            **kwargs,
        )

        # Switch to inference mode: disables dropout and batch normalisation.
        self.model.eval()

        print("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt string.
            max_new_tokens: Maximum number of tokens to generate.
            **kwargs: Additional parameters passed to the model's ``generate`` method.

        Returns:
            The generated text string, excluding the input prompt.
        """

        # Tokenize input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs.to(self.model.device)

        # Run model inference
        with torch.no_grad():
            model_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode generated tokens to text
        output = self.tokenizer.decode(
            model_outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return output

    def cleanup(self):
        """Unload model and free GPU memory."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Model unloaded")
