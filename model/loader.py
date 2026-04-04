"""
loader.py — Model Loading for HuggingFace and Ollama
=====================================================
Handles model loading for both backends with sensible defaults
optimized for 8GB RAM machines.

For HuggingFace:
  - Loads with 4-bit weight quantization (bitsandbytes) by default
  - Stacking weight quantization + TurboQuant KV cache compression
    gives maximum memory savings:
      8B model weights: ~4GB (4-bit) instead of ~16GB (FP16)
      KV cache: ~6x smaller via TurboQuant
    → Total: run a 7-8B model on a machine with 8GB RAM

For Ollama:
  - Uses Ollama's REST API (model must be pulled first)
  - TurboQuant applied to Ollama's exposed KV representations
  - Simpler setup, less control over internals
"""

import torch
import logging
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """
    Auto-detect the best available compute device.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        device = "cuda"
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)} ({vram_gb:.1f}GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Apple Silicon MPS detected")
    else:
        device = "cpu"
        logger.info("No GPU detected — running on CPU (will be slow for large models)")
    return device


def load_huggingface_model(
    model_id: str,
    bits: int = 4,
    device: str = "auto",
    hf_token: Optional[str] = None,
    context_length: int = 4096,
) -> Tuple[Any, Any]:
    """
    Load a HuggingFace model with optional 4-bit weight quantization.

    Best models for 8GB RAM (after 4-bit weight quant + TurboQuant KV):
      "microsoft/Phi-3-mini-4k-instruct"    — 3.8B, ~2.3GB, very fast
      "google/gemma-2-2b-it"                — 2B,   ~1.5GB, good quality
      "mistralai/Mistral-7B-Instruct-v0.3"  — 7B,   ~4GB,   best quality
      "meta-llama/Llama-3.2-3B-Instruct"   — 3B,   ~2GB,   needs HF token

    Args:
        model_id:       HuggingFace model ID
        bits:           Weight quantization (4 recommended for 8GB machines)
        device:         "auto", "cuda", "mps", or "cpu"
        hf_token:       HuggingFace token for gated models
        context_length: Max context (reduce if OOM)

    Returns:
        (model, tokenizer)
    """
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError:
        raise ImportError(
            "transformers not installed. Run: pip install transformers"
        )

    if device == "auto":
        device = detect_device()

    logger.info(f"Loading {model_id} on {device}...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit weight quantization config (bitsandbytes)
    # This compresses MODEL WEIGHTS — separate from TurboQuant KV cache compression
    quantization_config = None
    if bits == 4 and device == "cuda":
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,    # nested quantization
                bnb_4bit_quant_type="nf4",         # NormalFloat4 — best for LLMs
            )
            logger.info("4-bit weight quantization enabled (bitsandbytes NF4)")
        except Exception as e:
            logger.warning(f"bitsandbytes not available: {e}. Loading in FP16.")

    # Load model
    model_kwargs = {
        "token": hf_token,
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        if device == "mps":
            model_kwargs["device_map"] = {"": device}
        elif device == "cuda":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = {"": "cpu"}

    # ── Fix rope_scaling compatibility ──────────────────────────────
    # Newer transformers (≥4.45) populates rope_scaling={"rope_type": "default"}
    # where older model custom code (e.g. Phi-3) expects rope_scaling=None for
    # standard RoPE. Additionally, the key was renamed "type" → "rope_type".
    # We fix both issues here by pre-loading the config.
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        model_id, token=hf_token, trust_remote_code=True,
    )
    if getattr(config, "rope_scaling", None) is not None:
            rs = config.rope_scaling
            scaling_type = rs.get("rope_type", rs.get("type", None))

            if scaling_type in ("default", None):
                # Only clear rope_scaling for models that use custom modeling code
                # (e.g. Phi-3). Standard transformers models (LLaMA, TinyLlama)
                # derive rope_parameters from rope_scaling — clearing it breaks them.
                if getattr(config, "model_type", "") in ("phi3",):
                    config.rope_scaling = None
                    logger.info("Cleared rope_scaling (type='default' → None for standard RoPE)")
                else:
                    # Ensure both key names exist for standard models
                    rs["type"] = scaling_type or "default"
                    rs["rope_type"] = scaling_type or "default"
                    config.rope_scaling = rs
            else:
                if "type" not in rs:
                    rs["type"] = scaling_type
                if "rope_type" not in rs:
                    rs["rope_type"] = scaling_type
                config.rope_scaling = rs
                logger.info(f"Patched rope_scaling: ensured both keys for type='{scaling_type}'")
    # ────────────────────────────────────────────────────────────────

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    # Report memory usage
    if device == "cuda":
        mem_gb = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Model loaded. GPU memory used: {mem_gb:.2f}GB")

    logger.info(f"Model ready: {model_id}")
    return model, tokenizer


def load_ollama_model(model_name: str) -> Any:
    """
    Connect to a local Ollama instance.

    Prerequisites:
      1. Install Ollama: https://ollama.ai
      2. Pull a model:  ollama pull mistral
      3. Start server:  ollama serve

    Good models to pull for 8GB RAM:
      ollama pull mistral        (7B, ~4GB)
      ollama pull phi3           (3.8B, ~2.3GB)
      ollama pull gemma2:2b      (2B, ~1.5GB)
      ollama pull llama3.2:3b    (3B, ~2GB)

    Args:
        model_name: Ollama model name (e.g., "mistral", "phi3")

    Returns:
        OllamaClient wrapper
    """
    try:
        import ollama
    except ImportError:
        raise ImportError(
            "ollama not installed. Run: pip install ollama\n"
            "Also ensure Ollama is running: ollama serve"
        )

    # Verify the model is available
    try:
        models = ollama.list()
        available = [m.model for m in models.models]

        # Exact match
        if model_name in available:
            resolved_name = model_name

        else:
            # 🔍 Try prefix match (e.g., "llama3" → "llama3.1:8b")
            matches = [m for m in available if m.startswith(model_name)]

            if matches:
                resolved_name = matches[0]
                logger.info(f"Resolved '{model_name}' → '{resolved_name}'")
            else:
                logger.warning(
                    f"Model '{model_name}' not found in Ollama.\n"
                    f"Available: {available}\n"
                    f"Pull it with: ollama pull {model_name}"
                )
                resolved_name = model_name  # fallback (will likely error later)

    except Exception as e:
        logger.warning(f"Could not verify Ollama model: {e}")
        resolved_name = model_name  # fallback

    logger.info(f"Ollama model ready: {resolved_name}")
    return OllamaClient(resolved_name)


class OllamaClient:
    """Simple wrapper around the Ollama Python client."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def chat(self, messages: list, stream: bool = True, **kwargs):
        """
        Send a chat request to Ollama.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            stream:   Whether to stream the response
        """
        import ollama
        return ollama.chat(
            model=self.model_name,
            messages=messages,
            stream=stream,
            **kwargs,
        )

    def __repr__(self):
        return f"OllamaClient(model={self.model_name})"
