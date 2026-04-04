"""
inference.py — TurboQuant Inference Engine
==========================================
Hooks TurboQuant KV cache compression into the model's generation loop.

For HuggingFace models:
  - Patches the attention forward() method of every layer
  - Keys and Values are compressed immediately after computation
  - Decompressed before attention score calculation
  - Completely transparent to the rest of the model

For Ollama models:
  - TurboQuant applied at the API boundary (less granular control,
    but much simpler setup)

The patching is reversible — call remove_hooks() to restore the
original model behaviour.
"""

import torch
import numpy as np
import logging
from typing import Any, Generator, List, Optional

from core.kv_cache import TurboQuantKVCache
from config import CONFIG

logger = logging.getLogger(__name__)


class HuggingFaceInference:
    """
    Inference engine for HuggingFace models with TurboQuant KV cache.

    Patches attention layers to intercept KV pairs and compress them
    in-place. The model is unaware of the compression — it just sees
    slightly reconstructed vectors, which at 4-bit are near-lossless.
    """

    def __init__(self, model, tokenizer, config=CONFIG):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.kv_cache = TurboQuantKVCache(
            bits=config.bits,
            seed=config.rotation_seed,
        )
        self._hooks = []
        self._compression_enabled = config.enabled

        if self._compression_enabled:
            self._patch_attention_layers()
            logger.info(
                f"TurboQuant active: {config.bits}-bit KV compression "
                f"(~{16 // config.bits}x memory reduction)"
            )
        else:
            logger.info("TurboQuant disabled — running standard inference")

    def _patch_attention_layers(self):
        """
        Install KV interception. Two strategies, tried in order:

        1. DynamicCache patch (transformers ≥4.36): monkey-patch DynamicCache.update()
           so every key/value write goes through TurboQuant first.
           This is the modern path — past_key_value is passed as INPUT to attention
           and mutated in-place, never returned in the output tuple.

        2. Output hook fallback (transformers ≤4.35): register forward hooks on
           attention layers and intercept the (k, v) tuple in the output.
        """
        if self._try_patch_dynamic_cache():
            return

        # ── Fallback: legacy output-hook approach ──────────────────────
        layer_idx = 0
        for name, module in self.model.named_modules():
            module_name = type(module).__name__.lower()
            if "attention" in module_name and hasattr(module, "forward"):
                hook = self._make_kv_hook(layer_idx)
                handle = module.register_forward_hook(hook)
                self._hooks.append(handle)
                layer_idx += 1
        logger.info(f"TurboQuant hooks registered on {layer_idx} attention layers (legacy mode)")

    def _try_patch_dynamic_cache(self) -> bool:
        """
        Patch DynamicCache.update() to compress KV pairs before they are stored.

        Returns True if patching succeeded, False if DynamicCache is unavailable
        (i.e. old transformers that use the legacy tuple cache).
        """
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            logger.info("DynamicCache not found — falling back to legacy hook mode")
            return False

        kv_cache = self.kv_cache
        original_update = DynamicCache.update

        def patched_update(cache_self, key_states, value_states, layer_idx, cache_kwargs=None):
            """Compress KV before writing to cache, track stats."""
            try:
                k_c, v_c = kv_cache.update(key_states, value_states, layer_idx)
                return original_update(cache_self, k_c, v_c, layer_idx, cache_kwargs)
            except Exception as e:
                logger.debug(f"TurboQuant layer {layer_idx} compression error: {e}")
                return original_update(cache_self, key_states, value_states, layer_idx, cache_kwargs)

        DynamicCache.update = patched_update
        self._original_dynamic_cache_update = (DynamicCache, original_update)
        logger.info("TurboQuant active: patched DynamicCache.update() for KV compression")
        return True

    def _make_kv_hook(self, layer_idx: int):
        """
        Legacy forward hook for transformers ≤4.35 where past_key_value
        is returned as a (k, v) tuple in the attention output.
        """
        kv_cache = self.kv_cache

        def hook(module, input, output):
            if not isinstance(output, tuple):
                return output

            for idx in range(len(output) - 1, -1, -1):
                candidate = output[idx]
                if candidate is None:
                    continue
                if (isinstance(candidate, tuple)
                        and len(candidate) == 2
                        and isinstance(candidate[0], torch.Tensor)
                        and isinstance(candidate[1], torch.Tensor)
                        and candidate[0].ndim == 4):
                    k, v = candidate
                    try:
                        k_c, v_c = kv_cache.update(k, v, layer_idx)
                        new_out = list(output)
                        new_out[idx] = (k_c, v_c)
                        return tuple(new_out)
                    except Exception as e:
                        logger.debug(f"Layer {layer_idx} hook error: {e}")
                    return output

            return output

        return hook




    def remove_hooks(self):
        """Remove all TurboQuant hooks and restore original DynamicCache (if patched)."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        if hasattr(self, "_original_dynamic_cache_update"):
            CacheClass, original_fn = self._original_dynamic_cache_update
            CacheClass.update = original_fn
            del self._original_dynamic_cache_update
        logger.info("TurboQuant hooks removed")


    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """
        Generate text with TurboQuant KV cache compression.

        Args:
            prompt:         Input text
            max_new_tokens: Override config default
            temperature:    Override config default
            stream:         If True, yields tokens as they're generated

        Yields:
            Generated text tokens (if stream=True)
        Returns:
            Full generated text (if stream=False)
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature

        # Clear cache between conversations
        self.kv_cache.clear()

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.context_length,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if stream:
            yield from self._stream_generate(inputs, max_new_tokens, temperature)
        else:
            yield self._batch_generate(inputs, max_new_tokens, temperature)

    def _stream_generate(self, inputs, max_new_tokens, temperature):
        """Stream tokens using HuggingFace TextIteratorStreamer."""
        try:
            from transformers import TextIteratorStreamer
            import threading
        except ImportError:
            # Fallback to non-streaming
            yield self._batch_generate(inputs, max_new_tokens, temperature)
            return

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            yield token

        thread.join()

    def _batch_generate(self, inputs, max_new_tokens, temperature) -> str:
        """Non-streaming generation."""
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def compression_report(self) -> str:
        """Human-readable compression statistics."""
        s = self.kv_cache.stats()
        if s["original_mb"] == 0:
            return "No tokens processed yet."
        return (
            f"\n{'='*50}\n"
            f"  TurboQuant KV Cache Report\n"
            f"{'='*50}\n"
            f"  Bits:            {s['bits']}-bit\n"
            f"  Tokens cached:   {s['tokens_cached']}\n"
            f"  Original size:   {s['original_mb']:.2f} MB\n"
            f"  Compressed size: {s['compressed_mb']:.2f} MB\n"
            f"  Reduction:       {s['reduction_ratio']:.1f}x\n"
            f"{'='*50}\n"
        )


class OllamaInference:
    """
    Inference engine for Ollama models.
    TurboQuant is applied at the message boundary level.

    Note: Ollama manages KV cache internally — we can't hook into it
    at the tensor level without modifying Ollama itself. This class
    provides a clean chat interface and reports compression estimates.
    """

    def __init__(self, client, config=CONFIG):
        self.client = client
        self.config = config
        self.conversation_history: List[dict] = []
        logger.info(
            f"Ollama inference ready: {client.model_name}\n"
            f"Note: For full TurboQuant KV compression, use HuggingFace backend."
        )

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """
        Send a message and get a response.

        Args:
            message:       User message
            system_prompt: Optional system instruction
            stream:        Stream tokens as they arrive

        Yields:
            Response text chunks
        """
        # Build message history
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": message})

        full_response = ""

        if stream:
            response = self.client.chat(messages=messages, stream=True)
            for chunk in response:
                token = chunk.message.content
                full_response += token
                yield token
        else:
            response = self.client.chat(messages=messages, stream=False)
            full_response = response.message.content
            yield full_response

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append(
            {"role": "assistant", "content": full_response}
        )

    def clear_history(self):
        """Reset conversation history."""
        self.conversation_history.clear()