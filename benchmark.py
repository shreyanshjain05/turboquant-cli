"""
benchmark.py — TurboQuant Benchmark
=====================================
Run this to get REAL numbers on your machine.

Tests three things:
  1. Synthetic vectors (random gaussian with outlier spikes)
     — runs without any model, instant results
  2. Real KV vectors extracted from Ollama/HuggingFace model
     — requires model to be running, gives production-accurate numbers
  3. Memory projection for Llama 3 8B on your specific RAM

Usage:
    # Quick synthetic test (no model needed):
    python benchmark.py

    # Full test with real Llama 3 KV vectors via Ollama:
    python benchmark.py --real --backend ollama --model llama3.1:8B

    # Full test with HuggingFace model:
    python benchmark.py --real --backend huggingface \
        --model microsoft/Phi-3-mini-4k-instruct

    python benchmark.py --real --backend huggingface \
        --model facebook/opt-125m
        
"""

import argparse
import time
import numpy as np
import sys

# ── Formatting helpers ────────────────────────────────────────────────────────

def header(title: str):
    print(f"\n{'='*58}")
    print(f"  {title}")
    print(f"{'='*58}")

def row(label: str, value: str):
    print(f"  {label:<30} {value}")

def note(msg: str):
    print(f"\n  ⚠️  {msg}")

# ── Test 1: Synthetic vectors ─────────────────────────────────────────────────

def run_synthetic_benchmark():
    """
    Test the compression pipeline on synthetic random vectors.
    These are NOT real LLM KV vectors but give a fast sanity check
    that the math is working correctly.
    """
    from core.turboquant import TurboQuantizer

    header("TEST 1: Synthetic Vectors (No Model Required)")
    note("These are random numpy vectors, NOT real LLM KV vectors.")
    print("  Real LLM vectors will give better cosine similarity.")
    print()

    np.random.seed(42)

    configs = [
        {"n": 64,   "dim": 128,  "label": "Small  (64 vec,  dim=128)"},
        {"n": 512,  "dim": 128,  "label": "Medium (512 vec, dim=128)"},
        {"n": 2048, "dim": 128,  "label": "Large  (2K vec,  dim=128)"},
    ]

    for cfg in configs:
        n, dim = cfg["n"], cfg["dim"]
        x = np.random.randn(n, dim).astype(np.float32)
        # Realistic outlier spikes (common in transformer KV vectors)
        spike_coords = np.random.choice(dim, size=4, replace=False)
        for coord in spike_coords:
            x[:, coord] *= np.random.uniform(5, 12)

        print(f"\n  [{cfg['label']}]")
        print(f"  {'Bits':<6} {'Cosine Sim':>12} {'Reduction':>12} {'Time (ms)':>12}")
        print(f"  {'-'*46}")

        for bits in [4, 3, 2]:
            tq = TurboQuantizer(dim=dim, bits=bits, seed=42)

            t0 = time.perf_counter()
            state = tq.compress(x)
            t1 = time.perf_counter()

            stats = tq.compression_stats(x, state)
            ms = (t1 - t0) * 1000

            print(
                f"  {bits}-bit  "
                f"  {stats['cosine_similarity']:>10.4f}  "
                f"  {stats['reduction_ratio']:>9.1f}x  "
                f"  {ms:>9.1f}ms"
            )


# ── Test 2: Real KV vectors from Ollama ──────────────────────────────────────

def extract_kv_from_ollama(model_name: str, prompt: str) -> dict:
    """
    Extract real KV vectors from an Ollama model by hooking into
    a HuggingFace version of the same architecture.

    Note: Ollama doesn't expose raw KV tensors via its API.
    We use the HuggingFace version of the model instead,
    which gives us identical KV vectors (same weights, same math).
    """
    raise NotImplementedError(
        "Ollama doesn't expose raw KV tensors via its REST API.\n"
        "Use --backend huggingface to extract real KV vectors.\n"
        "The HuggingFace model will use the same weights as Ollama."
    )


def extract_kv_from_huggingface(model_id: str, prompt: str) -> dict:
    """
    Run a forward pass through a HuggingFace model and capture
    the actual Key and Value tensors from each attention layer.

    Returns a dict of {layer_idx: {"keys": np.array, "values": np.array}}
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("pip install transformers torch")

    print(f"\n  Loading {model_id} for KV extraction...")
    print("  (This may take a minute on first load)\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if getattr(config, "rope_scaling", None) is not None:
        rs = config.rope_scaling
        scaling_type = rs.get("rope_type", rs.get("type", None))
        if scaling_type in ("default", None):
            config.rope_scaling = None
        else:
            if "type" not in rs:
                rs["type"] = scaling_type
            if "rope_type" not in rs:
                rs["rope_type"] = scaling_type
            config.rope_scaling = rs

    # Load in FP32 for accurate baseline (no quantization during benchmark)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=False, use_cache=True)

    if not hasattr(outputs, "past_key_values") or outputs.past_key_values is None:
        raise ValueError("Model did not return past_key_values. KV cache extraction failed.")

    captured_kv = {}
    past_kv = outputs.past_key_values

    # Handle different cache APIs across transformers versions
    if hasattr(past_kv, "layers"):
        # Transformers >= 5.x DynamicCache with .layers list
        for layer_idx, layer in enumerate(past_kv.layers):
            captured_kv[layer_idx] = {
                "keys": layer.keys.detach().cpu().float().numpy(),
                "values": layer.values.detach().cpu().float().numpy(),
            }
    elif hasattr(past_kv, "key_cache"):
        # Transformers 4.x DynamicCache with key_cache/value_cache
        for layer_idx in range(len(past_kv.key_cache)):
            captured_kv[layer_idx] = {
                "keys": past_kv.key_cache[layer_idx].detach().cpu().float().numpy(),
                "values": past_kv.value_cache[layer_idx].detach().cpu().float().numpy(),
            }
    else:
        # Legacy Tuple(Tuple(Tensor))
        for layer_idx, kv_tuple in enumerate(past_kv):
            # Handle both 2-tuples and 3-tuples (keys, values, [sliding_window])
            k = kv_tuple[0]
            v = kv_tuple[1]
            # If k/v are tensors, use them directly; otherwise they may be layer objects
            if hasattr(k, 'detach'):
                captured_kv[layer_idx] = {
                    "keys": k.detach().cpu().float().numpy(),
                    "values": v.detach().cpu().float().numpy(),
                }
            else:
                raise TypeError(f"Unexpected KV cache format: layer contains {type(k)}")

    del model  # free memory
    return captured_kv


def run_real_kv_benchmark(backend: str, model: str):
    """
    Benchmark TurboQuant on REAL KV vectors from an actual model.
    This is the production-accurate test.
    """
    from core.turboquant import TurboQuantizer

    header("TEST 2: Real LLM KV Vectors")

    # Use a multi-sentence prompt to get realistic KV diversity
    test_prompt = (
        "The transformer architecture relies on attention mechanisms "
        "that compute key-value pairs for every token in the context. "
        "As context length grows, the KV cache becomes the dominant "
        "memory consumer during inference, often exceeding the model "
        "weights themselves at long context lengths."
    )

    print(f"\n  Backend: {backend}")
    print(f"  Model:   {model}")
    print(f"  Prompt:  {len(test_prompt.split())} words")

    if backend == "ollama":
        note("Ollama doesn't expose raw KV tensors. Use --backend huggingface.")
        print("  Falling back to synthetic vectors for this test.")
        run_synthetic_benchmark()
        return

    try:
        captured_kv = extract_kv_from_huggingface(model, test_prompt)
    except Exception as e:
        import traceback
        print(f"\n  [Error extracting KV vectors]: {e}")
        traceback.print_exc()
        print("\n  Falling back to synthetic benchmark.")
        run_synthetic_benchmark()
        return

    if not captured_kv:
        print("  No KV vectors captured. Check model compatibility.")
        return

    print(f"\n  Captured KV from {len(captured_kv)} attention layers")

    # Test compression on first 4 layers (representative sample)
    test_layers = sorted(captured_kv.keys())[:4]

    print(f"\n  {'Layer':<8} {'Shape':<20} {'4-bit CosSim':>14} {'4-bit Ratio':>12}")
    print(f"  {'-'*58}")

    all_cosine = []
    all_ratios = []

    for layer_idx in test_layers:
        kv = captured_kv[layer_idx]
        keys = kv["keys"]   # (batch, n_heads, seq_len, head_dim)

        # Reshape to (N, head_dim) for compression
        b, h, s, d = keys.shape
        keys_flat = keys.reshape(-1, d)

        tq = TurboQuantizer(dim=d, bits=4, seed=42)
        state = tq.compress(keys_flat)
        stats = tq.compression_stats(keys_flat, state)

        cos = stats["cosine_similarity"]
        ratio = stats["reduction_ratio"]
        all_cosine.append(cos)
        all_ratios.append(ratio)

        print(
            f"  Layer {layer_idx:<3}  "
            f"  {str(keys.shape):<18}  "
            f"  {cos:>12.4f}  "
            f"  {ratio:>10.1f}x"
        )

    print(f"\n  Average cosine similarity: {np.mean(all_cosine):.4f}")
    print(f"  Average reduction ratio:   {np.mean(all_ratios):.1f}x")
    print()

    if np.mean(all_cosine) > 0.97:
        print("  ✅ Excellent — matches paper's near-lossless claim at 4-bit")
    elif np.mean(all_cosine) > 0.90:
        print("  ✓  Good — slight quality loss, acceptable for most tasks")
    else:
        print("  ⚠️  Lower than expected — check model architecture compatibility")


# ── Test 3: Memory projection ─────────────────────────────────────────────────

def run_memory_projection():
    """
    Project KV cache memory usage for Llama 3 8B on your specific machine.
    Shows what context lengths become feasible with TurboQuant.
    """
    import platform
    import subprocess

    header("TEST 3: Theoretical Memory Projection (Llama 3 8B Reference)")
    print("\n  Note: This calculates theoretical limits for an 8B class model")
    print("  on your current machine, regardless of the test model above.")

    # Detect available RAM
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / 1e9
    except ImportError:
        total_ram_gb = 8.0  # assume 8GB if psutil not available
        note("Install psutil for accurate RAM detection: pip install psutil")

    print(f"\n  Detected RAM: {total_ram_gb:.1f} GB")
    print(f"  Machine:      {platform.machine()} / {platform.system()}")

    # Llama 3 8B architecture constants
    n_layers   = 32
    n_kv_heads = 8
    head_dim   = 128
    model_size_gb = 4.7   # Llama 3 8B in FP16 / Ollama Q4

    available_for_kv = total_ram_gb - model_size_gb - 1.0  # 1GB system overhead
    available_for_kv_mb = available_for_kv * 1000

    print(f"\n  Llama 3 8B model size:  ~{model_size_gb} GB")
    print(f"  Available for KV cache: ~{available_for_kv:.1f} GB")

    print(f"\n  {'Context':>10}  {'FP16 KV':>10}  {'TurboQuant':>12}  {'Feasible?':>12}")
    print(f"  {'-'*52}")

    contexts = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

    for ctx in contexts:
        # Keys + Values, both heads, all layers, FP16 (2 bytes)
        fp16_mb = 2 * n_layers * n_kv_heads * ctx * head_dim * 2 / 1e6
        tq_mb   = fp16_mb / 6.0   # 4-bit TurboQuant: 6x reduction

        fp16_feasible = "✅" if fp16_mb < available_for_kv_mb else "❌ OOM"
        tq_feasible   = "✅" if tq_mb   < available_for_kv_mb else "❌ OOM"

        ctx_label = f"{ctx//1024}K" if ctx >= 1024 else str(ctx)
        print(
            f"  {ctx_label:>8}   "
            f"  {fp16_mb:>7.0f} MB  "
            f"  {tq_mb:>9.0f} MB  "
            f"  FP16: {fp16_feasible}  TQ: {tq_feasible}"
        )

    print(f"\n  TurboQuant unlocks {6}x longer contexts within the same RAM budget.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TurboQuant Benchmark")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Run benchmark on real LLM KV vectors (requires model)",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "huggingface"],
        default="huggingface",
        help="Backend for real KV extraction (default: huggingface)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Model for real KV extraction",
    )
    args = parser.parse_args()

    print("\n  TurboQuant Benchmark")
    print("  " + "─" * 54)
    print("  All results are measured on THIS machine.")
    print("  Paper results: Llama-3.1-8B, NVIDIA A100, ICLR 2026.")

    # Always run synthetic (fast, no model needed)
    run_synthetic_benchmark()

    # Real KV test (optional, requires model)
    if args.real:
        run_real_kv_benchmark(args.backend, args.model)
    else:
        header("TEST 2: Real LLM KV Vectors")
        print("\n  Skipped. Run with --real to test on actual model KV vectors.")
        print("  Example:")
        print("    python benchmark.py --real --backend huggingface \\")
        print("      --model microsoft/Phi-3-mini-4k-instruct")

    # Memory projection (always runs)
    run_memory_projection()

    print(f"\n{'='*58}")
    print("  Benchmark complete.")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
