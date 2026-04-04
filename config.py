"""
TurboQuant Configuration
========================
All settings live here. Edit this file to switch models,
change bit-width, or toggle compression on/off.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class TurboQuantConfig:
    # ── Compression settings ──────────────────────────────────────────
    bits: int = 4               # 4-bit: 6x memory reduction, zero accuracy loss
    rotation_seed: int = 42
    enabled: bool = True

    # ── Model settings ────────────────────────────────────────────────
    backend: Literal["ollama", "huggingface"] = "ollama"

    # Ollama: make sure you've run `ollama pull llama3`
    ollama_model: str = "llama3"

    # HuggingFace fallback (good for Apple Silicon)
    hf_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_token: Optional[str] = None

    # Apple Silicon: no bitsandbytes — use MPS + FP16 instead
    load_in_4bit: bool = False   # bitsandbytes not supported on MPS

    # ── Inference settings ────────────────────────────────────────────
    max_new_tokens: int = 512
    temperature: float = 0.7

    # Apple Silicon unified memory — 8GB machine can handle 8K context
    # with TurboQuant reducing KV cache pressure
    context_length: int = 8192

    # ── Device ────────────────────────────────────────────────────────
    # MPS = Metal Performance Shaders (Apple Silicon GPU)
    device: str = "auto"   # auto-detects: MPS on Mac, CUDA on NVIDIA


# Singleton — import this everywhere
CONFIG = TurboQuantConfig()
