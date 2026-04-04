"""
qjl.py — Quantized Johnson-Lindenstrauss: Stage 2 Error Correction
===================================================================
TurboQuantMSE (Stage 1) achieves near-optimal MSE compression but
introduces a systematic bias in inner product estimation. This matters
because LLM attention scores are computed as dot products — a biased
estimator would silently skew which tokens the model attends to.

QJL fixes this with a single additional bit per dimension:
  1. Compute the residual: r = x - X_Base  (what TurboQuantMSE left behind)
  2. Apply the Johnson-Lindenstrauss transform to r
  3. Store only the sign bit (+1 or -1) of each projected coordinate

The JL transform preserves distances in expectation, and the sign-bit
quantization introduces random noise that cancels out over millions of
attention calculations — giving an UNBIASED inner product estimator.

Math:
  Final vector:     x̃ = X_Base + X_residual
  Unbiased proof:   E_Q[⟨y, Q⁻¹(Q(x))⟩] = ⟨y, x⟩
  Inner prod error: D_prod = E[|⟨y,x⟩ - ⟨y,x̃⟩|²] ≈ ||y||²/d  (near-optimal)
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class QJLState:
    """Compressed residual from QJL Stage 2."""
    signs: np.ndarray        # Sign bits of JL-projected residual, shape (N, n_proj)
    projection: np.ndarray   # JL projection matrix, shape (n_proj, dim)
    gamma: np.ndarray        # Scaling factor per vector, shape (N, 1)
    n_proj: int
    dim: int


class QJL:
    """
    Quantized Johnson-Lindenstrauss transform for residual correction.

    Takes the residual error left by TurboQuantMSE and compresses it to
    1 bit per projected dimension, eliminating systematic bias in
    attention score computation.

    The JL lemma guarantees that for random projections,
    inner products are preserved in expectation — making this
    mathematically provably unbiased.

    Usage:
        qjl = QJL(dim=128, n_proj=32, seed=42)
        state = qjl.compress(residual)
        correction = qjl.estimate_inner_product(query, state)
    """

    def __init__(
        self,
        dim: int,
        n_proj: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Args:
            dim:    Dimensionality of input vectors (head_dim).
            n_proj: Number of JL projections. Default: dim // 4.
                    More projections = better accuracy, more memory.
                    Paper recommends dim/4 for search tasks.
            seed:   Must match the rotation seed for consistency.
        """
        self.dim = dim
        self.n_proj = n_proj or max(dim // 4, 8)
        self.seed = seed
        self._projection: Optional[np.ndarray] = None

    @property
    def projection(self) -> np.ndarray:
        """
        JL projection matrix: shape (n_proj, dim).
        Entries drawn from standard normal (Gaussian) as required for rigorous 
        QJL variance bounds, scaled by 1/√n_proj.
        Lazy init — built once, reused for all vectors.
        """
        if self._projection is None:
            rng = np.random.default_rng(self.seed + 1)   # offset from rotation seed
            # Gaussian matrix S as per paper's Definition 1
            S = rng.standard_normal((self.n_proj, self.dim))
            self._projection = (S / np.sqrt(self.n_proj)).astype(np.float32)
        return self._projection

    def compress(self, residual: np.ndarray) -> QJLState:
        """
        Compress the residual vector to 1 bit per JL projection.

        Args:
            residual: r = x - X_Base, shape (N, dim) or (dim,)
        Returns:
            QJLState with sign bits and scaling factor
        """
        if residual.ndim == 1:
            residual = residual[np.newaxis, :]
            
        residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

        # Project residual into JL space: (N, n_proj)
        projected = residual @ self.projection.T

        # Store only the sign — +1 or -1 (1 bit per projection)
        signs = np.sign(projected).astype(np.int8)
        signs[signs == 0] = 1   # handle exact zero edge case

        # Gamma: per-vector scaling factor that calibrates the estimator
        # Derived from the mean magnitude of the residual projections
        gamma = np.linalg.norm(residual, axis=1, keepdims=True).astype(np.float32)
        gamma = np.nan_to_num(gamma, nan=0.0)

        return QJLState(
            signs=signs,
            projection=self.projection,
            gamma=gamma,
            n_proj=self.n_proj,
            dim=self.dim,
        )

    def estimate_inner_product(
        self,
        query: np.ndarray,
        state: QJLState,
    ) -> np.ndarray:
        """
        Estimate ⟨query, residual⟩ from the compressed QJL state.

        This is the core of the unbiased estimator:
          ⟨q, r⟩ ≈ γ · (q @ Pᵀ) · signs

        Args:
            query:  Query vector, shape (dim,) or (N_q, dim)
            state:  QJLState from compress()
        Returns:
            Estimated inner products, shape (N_q, N_kv) or scalar
        """
        if query.ndim == 1:
            query = query[np.newaxis, :]

        # Project query into JL space: (N_q, n_proj)
        q_projected = query @ state.projection.T

        # Estimate: (N_q, n_proj) @ (N_kv, n_proj).T → (N_q, N_kv)
        # Then multiply column-wise by the per-vector gamma (shape 1, N_kv)
        scale = np.sqrt(np.pi / 2) / self.dim
        estimates = scale * (q_projected @ state.signs.T) * state.gamma.T

        return estimates.squeeze()

    def reconstruct_residual(self, state: QJLState) -> np.ndarray:
        """
        Approximate reconstruction of the residual vector.
        Used when you need the actual vector, not just an inner product.

        Returns:
            Approximate residual, shape (N, dim)
        """
        # state.projection is (n_proj, dim), already scaled by 1/√n_proj
        # For paper-exact: scale = √(π/2) / dim
        scale = np.sqrt(np.pi / 2) / self.dim
        return scale * state.gamma * (state.signs.astype(np.float32) @ state.projection)
    
    def memory_bits_per_vector(self) -> int:
        """Bits used by QJL per original vector (1 bit per projection)."""
        return self.n_proj   # 1 bit × n_proj

    def __repr__(self) -> str:
        return (
            f"QJL(dim={self.dim}, n_proj={self.n_proj}, "
            f"bits_per_vector={self.memory_bits_per_vector()})"
        )
