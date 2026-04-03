"""
turboquant.py — Full TurboQuant Pipeline
=========================================
Combines TurboQuantMSE (Stage 1) and QJL (Stage 2) into the complete
two-stage compression pipeline described in the ICLR 2026 paper.

Pipeline:
  COMPRESS:
    x  →  [TurboQuantMSE]  →  X_Base (b-1 bits)
    r = x - X_Base
    r  →  [QJL]         →  X_residual (1 bit per projection)
    Store: (X_Base indices, X_residual signs)

  DECOMPRESS / ATTENTION:
    X_Base = TurboQuantMSE.decompress(indices)
    ⟨q, x̃⟩ = ⟨q, X_Base⟩ + QJL.estimate_inner_product(q, residual_state)

Total budget: (b-1) bits (TurboQuantMSE) + 1 bit (QJL) = b bits
Result: unbiased inner product estimator at near-Shannon compression.

Benchmark results (from paper, Llama-3.1-8B on LongBench):
  4-bit TurboQuant: matches full FP16 precision
  3-bit TurboQuant: near-perfect on most tasks
  Memory reduction: ≥ 6x vs FP16 KV cache
  Speed on H100:    8x faster attention logits at 4-bit
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.turboquant_mse import TurboQuantMSE, TurboQuantMSEState
from core.qjl import QJL, QJLState


@dataclass
class TurboQuantState:
    """
    Complete compressed representation of a batch of KV vectors.
    Everything needed for reconstruction or attention estimation.
    """
    # Stage 1: TurboQuantMSE
    pq_state: TurboQuantMSEState

    # Stage 2: QJL residual
    qjl_state: QJLState

    # Metadata
    bits: int
    original_shape: tuple
    dtype: np.dtype


class TurboQuantizer:
    """
    The full TurboQuant pipeline — compress and decompress KV vectors.

    This is the main class you interact with. It manages both stages
    internally and exposes a clean compress/decompress API.

    Usage:
        tq = TurboQuantizer(dim=128, bits=4, seed=42)

        # Compress
        state = tq.compress(key_vectors)

        # Estimate attention score (preferred — no decompression needed)
        score = tq.attention_score(query, state)

        # Full reconstruct (when you need the actual vector)
        reconstructed = tq.decompress(state)

        # Check memory savings
        print(tq.compression_stats(key_vectors, state))
    """

    def __init__(self, dim: int, bits: int = 4, seed: int = 42):
        """
        Args:
            dim:  Head dimension of KV vectors (e.g., 128 for Llama 3).
            bits: Total bit budget per value.
                  TurboQuantMSE gets (bits-1), QJL gets 1.
                  Recommended: 4 (best quality) or 3 (max compression).
            seed: Reproducibility seed — same seed for compress/decompress.
        """
        assert bits >= 2, "TurboQuant needs at least 2 bits (1 for PQ, 1 for QJL)"

        self.dim = dim
        self.bits = bits
        self.seed = seed

        # Stage 1: TurboQuantMSE gets (bits - 1) of the budget
        self.turboquant_mse = TurboQuantMSE(dim=dim, bits=bits - 1, seed=seed)

        # Stage 2: QJL gets 1 bit, with n_proj = dim // 4 projections
        self.qjl = QJL(dim=dim, n_proj=max(dim // 4, 8), seed=seed)

    def compress(self, x: np.ndarray) -> TurboQuantState:
        """
        Full TurboQuant compression: x → (X_Base, X_residual).

        Args:
            x: KV vectors, shape (seq_len, dim) or (dim,)
               Should be float32 or float16.
        Returns:
            TurboQuantState — everything needed for reconstruction
        """
        original_dtype = x.dtype
        x = x.astype(np.float32)
        original_shape = x.shape

        if x.ndim == 1:
            x = x[np.newaxis, :]

        # ── Stage 1: TurboQuantMSE ──────────────────────────────────────
        pq_state = self.turboquant_mse.quantize(x)

        # Reconstruct X_Base to compute residual
        x_base = self.turboquant_mse.dequantize(pq_state)   # (N, dim)

        # ── Compute residual on unit sphere (normalised space) ───────
        # QJL works best on normalised residuals — scale-invariant
        norms = pq_state.norms if pq_state.norms is not None else np.ones((x.shape[0], 1), dtype=np.float32)
        x_unit  = x / np.where(norms < 1e-8, 1.0, norms)
        xb_unit = x_base / np.where(norms < 1e-8, 1.0, norms)
        residual = x_unit - xb_unit   # residual on unit sphere

        # ── Stage 2: QJL on the residual ────────────────────────────
        qjl_state = self.qjl.compress(residual)

        return TurboQuantState(
            pq_state=pq_state,
            qjl_state=qjl_state,
            bits=self.bits,
            original_shape=original_shape,
            dtype=original_dtype,
        )

    def decompress(self, state: TurboQuantState) -> np.ndarray:
        """
        Full reconstruction: x̃ = X_Base + X_residual

        Note: For attention computation, use attention_score() instead —
        it's faster and more accurate (no reconstruction needed).

        Args:
            state: TurboQuantState from compress()
        Returns:
            Reconstructed vectors, shape matching original input
        """
        # Reconstruct X_Base from TurboQuantMSE
        x_base = self.turboquant_mse.dequantize(state.pq_state)

        # Reconstruct approximate residual from QJL
        x_residual = self.qjl.reconstruct_residual(state.qjl_state)

        # x̃ = X_Base + X_residual
        reconstructed = x_base + x_residual
        return reconstructed.reshape(state.original_shape).astype(state.dtype)

    def attention_score(
        self,
        query: np.ndarray,
        state: TurboQuantState,
    ) -> np.ndarray:
        """
        Compute attention scores WITHOUT full decompression.

        ⟨q, x̃⟩ = ⟨q, X_Base⟩ + ⟨q, X_residual⟩

        The QJL term provides an unbiased estimate of ⟨q, residual⟩,
        making the total estimator unbiased for the full inner product.

        Args:
            query: Query vector(s), shape (dim,) or (n_heads, dim)
            state: TurboQuantState from compress()
        Returns:
            Attention scores, shape (n_queries, seq_len)
        """
        if query.ndim == 1:
            query = query[np.newaxis, :]

        # Term 1: ⟨q, X_Base⟩ using reconstructed TurboQuantMSE vectors
        x_base = self.turboquant_mse.dequantize(state.pq_state)   # (N_kv, dim)
        score_base = query @ x_base.T   # (N_q, N_kv)

        # Term 2: unbiased QJL estimate of ⟨q, residual⟩
        score_residual = self.qjl.estimate_inner_product(query, state.qjl_state)

        return score_base + score_residual

    def compress_tensor(self, x: torch.Tensor) -> TurboQuantState:
        """Compress a torch tensor (handles device transfer internally)."""
        x_np = x.detach().cpu().float().numpy()
        return self.compress(x_np)

    def decompress_tensor(
        self,
        state: TurboQuantState,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Decompress to a torch tensor."""
        reconstructed_np = self.decompress(state)
        t = torch.from_numpy(reconstructed_np.astype(np.float32))
        if device is not None:
            t = t.to(device)
        return t.to(dtype)

    def compression_stats(
        self,
        original: np.ndarray,
        state: TurboQuantState,
    ) -> dict:
        """
        Report memory savings and reconstruction quality.

        Returns dict with:
          - original_mb:    Memory before compression (MB)
          - compressed_mb:  Memory after compression (MB)
          - reduction_ratio: How many times smaller
          - mse:            Mean squared reconstruction error
          - cosine_sim:     Average cosine similarity (1.0 = perfect)
        """
        if original.ndim == 1:
            original = original[np.newaxis, :]

        n, d = original.shape

        # Memory calculation
        fp16_bytes = n * d * 2   # 2 bytes per FP16 value
        pq_bytes   = n * d * (self.bits - 1) / 8
        qjl_bytes  = n * self.qjl.n_proj / 8   # 1 bit per projection
        total_compressed = pq_bytes + qjl_bytes

        # Reconstruction quality
        reconstructed = self.decompress(state)
        mse = float(np.mean((original - reconstructed) ** 2))

        # Cosine similarity
        norms_orig = np.linalg.norm(original, axis=1, keepdims=True) + 1e-8
        norms_rec  = np.linalg.norm(reconstructed, axis=1, keepdims=True) + 1e-8
        cos_sim = float(np.mean(
            np.sum((original / norms_orig) * (reconstructed / norms_rec), axis=1)
        ))

        return {
            "original_mb":     fp16_bytes / 1e6,
            "compressed_mb":   total_compressed / 1e6,
            "reduction_ratio": fp16_bytes / total_compressed,
            "mse":             mse,
            "cosine_similarity": cos_sim,
            "bits":            self.bits,
            "n_vectors":       n,
            "dim":             d,
        }

    def __repr__(self) -> str:
        return (
            f"TurboQuantizer(\n"
            f"  dim={self.dim}, bits={self.bits},\n"
            f"  Stage1={self.turboquant_mse},\n"
            f"  Stage2={self.qjl}\n"
            f")"
        )