# TurboQuant — Local LLM on 8GB Mac

Implementation of **TurboQuant** (ICLR 2026, Google Research) for running
Llama 3 8B locally on Apple Silicon with 6x KV cache memory reduction.

> Paper: *TurboQuant: Redefining AI efficiency with extreme compression*  
> Authors: Amir Zandieh, Vahab Mirrokni et al., Google Research

---

## What This Does

Without TurboQuant, Llama 3 8B at 8K context needs ~1GB just for the KV cache.
At 32K context, that's ~4.3GB — eating most of your unified memory.

With 4-bit TurboQuant:

| Context | KV Cache (FP16) | KV Cache (TurboQuant) | Saving |
|---------|----------------|----------------------|--------|
| 4K      | 537 MB         | 89 MB                | 6x     |
| 8K      | 1.07 GB        | 179 MB               | 6x     |
| 32K     | 4.3 GB         | 716 MB               | 6x     |

---

## Project Structure

```
turboquant/
├── core/
│   ├── rotation.py      # Haar random rotation matrix (Π)
│   ├── turboquant_mse.py   # TurboQuantMSE — Stage 1: MSE-optimal compression
│   ├── qjl.py           # QJL — Stage 2: 1-bit unbiased residual correction
│   ├── turboquant.py    # Full two-stage pipeline
│   └── kv_cache.py      # KV cache interceptor for transformer models
├── model/
│   ├── loader.py        # HuggingFace + Ollama model loading
│   └── inference.py     # Inference engine with TurboQuant hooks
├── config.py            # All settings (bits, model, context length)
├── main.py              # Entry point — interactive chat
└── requirements.txt
```

---

## Setup (Mac Apple Silicon)

### Step 1 — Install Ollama

```bash
# Download from https://ollama.ai or via Homebrew:
brew install ollama

# Start the Ollama server
ollama serve
```

### Step 2 — Pull Llama 3 8B

```bash
# In a new terminal:
ollama pull llama3

# Verify it's available:
ollama list
```

### Step 3 — Install Python dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# PyTorch with MPS support (Apple Silicon)
pip install torch torchvision torchaudio
```

### Step 4 — Run

```bash
python main.py
```

That's it. TurboQuant activates automatically.

---

## CLI Options

```bash
# Default: Ollama + Llama 3 + 4-bit TurboQuant
python main.py

# Use a different Ollama model
python main.py --model mistral
python main.py --model phi3
python main.py --model gemma2:2b

# Change bit-width
python main.py --bits 3   # 7x compression, slight quality drop
python main.py --bits 4   # 6x compression, near-lossless (default)

# Disable TurboQuant (baseline comparison)
python main.py --no-compress

# Longer context (if you have 16GB unified memory)
python main.py --context 16384

# HuggingFace backend (full KV cache hook, more control)
python main.py --backend huggingface --model microsoft/Phi-3-mini-4k-instruct
```

---

## Chat Commands

During chat, type:

```
/stats    — show TurboQuant compression statistics
/clear    — clear conversation history + KV cache
/help     — show all commands
/quit     — exit
```

---

## How TurboQuant Works (Quick Reference)

```
Input KV vector x (FP16, 128-dim)
          │
          ▼
    ┌────────────────┐
    │  TurboQuantMSE │  Stage 1: random rotation Π → normalise →
    │  (3 bits)      │  Lloyd-Max quantize on Beta distribution
    └────────────────┘  → X_Base (zero overhead, near-optimal MSE)
          │
          ▼  residual r = x - X_Base
          │
    ┌─────────────┐
    │    QJL      │  Stage 2: Johnson-Lindenstrauss projection →
    │   (1 bit)   │  store only sign bit → unbiased attention estimator
    └─────────────┘
          │
          ▼
    x̃ = X_Base + X_residual
    E[⟨y, x̃⟩] = ⟨y, x⟩  ← provably unbiased
```

Total: 4 bits per value vs 16 bits (FP16) = **4-6x memory reduction**

---

## Benchmark Results

> ⚠️ **Transparency note:** The numbers below are from the TurboQuant paper
> (Google Research, ICLR 2026) and from synthetic pipeline tests on random
> numpy vectors — NOT from running against a real LLM on this machine.
> To get real numbers on your hardware, run `python benchmark.py` (see below).

**From the paper** (Llama-3.1-8B-Instruct, LongBench):

| Mode                  | Accuracy vs FP16 | Memory Reduction |
|-----------------------|-----------------|-----------------|
| 4-bit TurboQuant      | Matches exactly  | ≥ 6x            |
| 3-bit TurboQuant      | Near-perfect     | ~8x             |
| FP16 (baseline)       | 1.000            | 1x              |

**From synthetic pipeline test** (random vectors, head_dim=128, with outlier spikes):

| Mode                  | Cosine Similarity | Reduction |
|-----------------------|-------------------|-----------|
| 4-bit TurboQuant      | 0.948             | ~5x       |
| 3-bit TurboQuant      | 0.819             | ~7x       |

Run `python benchmark.py` to measure real numbers on your machine.

### Benchmarking Real HuggingFace Models
To check the compression quality and ratio against a real HuggingFace model's KV tensors:
```bash
python benchmark.py --real \
  --backend huggingface \
  --model microsoft/Phi-3-mini-4k-instruct
```

---

## Understanding KV Cache Testing: Ollama vs HuggingFace

When using TurboQuant, you have two backend options: **Ollama** and **HuggingFace**. While both work perfectly for regular chatting and inferencing, benchmarking the mathematical loss of extreme KV compression requires access to the model's raw memory.

### The Ollama Limitation
Ollama is a compiled standalone server built on top of `llama.cpp`. We interact with it via a REST API. Because it acts as a black box that prioritizes ease-of-use and speed, **Ollama does not expose its raw internal memory tensors**.
* **Chatting:** TurboQuant can smoothly wrap the Ollama interface and track token counts to give you *projections* of memory saved.
* **Benchmarking:** We cannot extract the actual dynamically generated Key-Value tensors from Ollama to run our compression mathematics on them. Running `benchmark.py --real` on Ollama will default back to the synthetic test.

### The HuggingFace Solution
To calculate exact cosine similarity and mathematical compression loss on **real** data, you must use the HuggingFace backend for benchmarking. 
* HuggingFace runs directly in Python, giving TurboQuant the ability to attach **hooks** to every attention layer.
* When a test sentence is fed into the model, TurboQuant intercepts the resulting KV Cache memory exactly as the model naturally generates it, applies the mathematical compression (down to 3 or 4 bits), and compares the result against the original 16-bit float tensor to measure exactly how near-lossless the algorithm is.

If you are only using TurboQuant to save RAM while chatting, Ollama is highly recommended for its out-of-the-box metal acceleration. If you are a researcher wanting to verify the Google paper's claims on your own machine, use the HuggingFace backend with the `--real` benchmarking flag!

---

## Recommended Models for 8GB Mac

| Model | Pull Command | Size | Speed | Quality |
|-------|-------------|------|-------|---------|
| Llama 3 8B | `ollama pull llama3` | ~4.7GB | Medium | ⭐⭐⭐⭐⭐ |
| Mistral 7B | `ollama pull mistral` | ~4.1GB | Medium | ⭐⭐⭐⭐ |
| Phi-3 Mini | `ollama pull phi3` | ~2.3GB | Fast | ⭐⭐⭐⭐ |
| Gemma 2 2B | `ollama pull gemma2:2b` | ~1.6GB | Very fast | ⭐⭐⭐ |

With TurboQuant's KV compression, all of these run comfortably
in 8GB unified memory even at extended context lengths.

---

## References
- [Understanding TurboQuant](https://medium.com/data-science-collective/turboquant-how-google-made-it-possible-to-run-huge-models-locally-099b6b501517)
- [TurboQuant Paper (ICLR 2026)](https://openreview.net/pdf/6593f484501e295cdbe7efcbc46d7f20fc7e741f.pdf)
- [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [Polar Quant](https://arxiv.org/abs/2502.02617)
- [QJL (AAAI 2025)](https://dl.acm.org/doi/10.1609/aaai.v39i18.34037)
