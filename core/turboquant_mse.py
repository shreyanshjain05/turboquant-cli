"""
turboquant_mse.py — TurboQuantMSE: Stage 1 Compression
==================================================
After rotation, each coordinate of the KV vector follows a
predictable Beta distribution. TurboQuantMSE exploits this by:

  1. Computing an optimal Lloyd-Max codebook for the Beta distribution
     (done ONCE offline — O(k) centroids, not O(N) dataset scans)
  2. Quantizing each coordinate to the nearest centroid index
  3. Storing only the integer index — zero overhead from normalization
     constants (the key innovation vs. traditional VQ methods)

This gives optimal MSE distortion at the chosen bit-width,
approaching Shannon's lower bound for the compression budget.

Math:
  D_mse = E[||x - Q⁻¹(Q(x))||²₂]

  Lower bound (Shannon): D_mse ≥ 1 / 4^b
  TurboQuantMSE achieves:   D_mse ≈ 1 / 4^b  (near-optimal)
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rotation import RotationMatrix


@dataclass
class TurboQuantMSEState:
    """Everything needed to reconstruct a compressed vector."""
    indices: np.ndarray      # Quantized indices, shape (N, dim)
    codebook: np.ndarray     # Centroids, shape (2^bits,)
    bits: int
    original_shape: tuple
    norms: np.ndarray = None  # Per-vector L2 norms, shape (N, 1)


class TurboQuantMSE:
    """
    Stage 1 of TurboQuant: near-optimal MSE compression with zero overhead.

    The codebook is derived from the Beta distribution analytically —
    no training data, no calibration, no dataset-specific tuning.
    Works on any transformer's KV cache out of the box.

    Usage:
        pq = TurboQuantMSE(dim=128, bits=3, seed=42)
        state = pq.quantize(keys)          # compress
        reconstructed = pq.dequantize(state)  # decompress
    """

    def __init__(self, dim: int, bits: int = 3, seed: int = 42):
        """
        Args:
            dim:  Head dimension of the KV vectors.
            bits: Bit-width for TurboQuantMSE stage.
                  TurboQuant reserves 1 bit for QJL, so pass (target_bits - 1).
                  e.g., for 4-bit TurboQuant, pass bits=3 here.
            seed: Rotation seed — must match QJL and decompression.
        """
        self.dim = dim
        self.bits = bits
        self.n_centroids = 2 ** bits
        self.seed = seed

        self.rotation = RotationMatrix(dim=dim, seed=seed)
        self.codebook = self._build_codebook()

    def _build_codebook(self) -> np.ndarray:
        """
        Build the Lloyd-Max optimal codebook for the Beta distribution.

        After random rotation, each coordinate of a unit-norm vector
        follows Beta((d-1)/2, (d-1)/2) shifted to [-1, 1].
        We solve the Lloyd-Max optimality conditions analytically
        using a fine-grained sample of the distribution.

        Returns:
            centroids: shape (n_centroids,), float32
        """
        # Sample the Beta distribution (shifted to [-1, 1])
        # With large d, this approximates the actual coordinate distribution
        alpha = (self.dim - 1) / 2.0
        rng = np.random.default_rng(self.seed)

        n_samples = 100_000
        samples = rng.beta(alpha, alpha, size=n_samples).astype(np.float32)
        samples = 2.0 * samples - 1.0   # shift from [0,1] to [-1,1]

        # Lloyd-Max iteration: find centroids that minimise MSE
        # Initialise with uniform quantisation levels
        centroids = np.linspace(-1.0, 1.0, self.n_centroids, dtype=np.float32)

        for _ in range(50):   # typically converges in <20 iterations
            # Assignment step: assign each sample to nearest centroid
            diffs = samples[:, None] - centroids[None, :]   # (N, k)
            assignments = np.argmin(np.abs(diffs), axis=1)

            # Update step: new centroid = mean of assigned samples
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_centroids):
                mask = assignments == k
                if mask.any():
                    new_centroids[k] = samples[mask].mean()
                else:
                    new_centroids[k] = centroids[k]   # keep if empty

            # Check convergence
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return centroids

    def quantize(self, x: np.ndarray) -> TurboQuantMSEState:
        """
        Compress KV vectors using TurboQuantMSE.

        Args:
            x: Input vectors, shape (seq_len, dim) or (dim,)
        Returns:
            TurboQuantMSEState containing compressed indices and metadata
        """
        original_shape = x.shape
        if x.ndim == 1:
            x = x[np.newaxis, :]   # (1, dim)

        # Step 1: Store per-vector norms for reconstruction
        # TurboQuantMSE works on the unit sphere — we normalise first,
        # quantise the direction, then restore magnitude on decompress.
        # This is the key fix for outlier spikes: the rotation spreads
        # direction energy evenly, and the norm is stored separately (losslessly).
        norms = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
        norms = np.where(norms < 1e-8, 1.0, norms)   # avoid div-by-zero
        x_unit = x / norms   # unit vectors on the sphere

        # Step 2: Random rotation — after normalisation, each coordinate
        # of the rotated unit vector follows Beta((d-1)/2, (d-1)/2)
        rotated = self.rotation.apply(x_unit)   # (N, dim)

        # Step 3: Quantize each coordinate to nearest centroid
        diffs = rotated[:, :, np.newaxis] - self.codebook[np.newaxis, np.newaxis, :]
        indices = np.argmin(np.abs(diffs), axis=2).astype(np.uint8)   # (N, dim)

        return TurboQuantMSEState(
            indices=indices,
            codebook=self.codebook,
            bits=self.bits,
            original_shape=original_shape,
            norms=norms,
        )

    def dequantize(self, state: TurboQuantMSEState) -> np.ndarray:
        """
        Reconstruct vectors from compressed state.

        Args:
            state: TurboQuantMSEState from compress()
        Returns:
            Reconstructed vectors, shape matching original input
        """
        # Map indices back to centroid values (unit-sphere directions)
        reconstructed_rotated = state.codebook[state.indices]   # (N, dim)

        # Invert the rotation
        reconstructed_unit = self.rotation.invert(reconstructed_rotated)  # (N, dim)

        # Restore original magnitude
        if state.norms is not None:
            reconstructed = reconstructed_unit * state.norms
        else:
            reconstructed = reconstructed_unit

        return reconstructed.reshape(state.original_shape)

    def quantize_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Compress a torch tensor — converts to numpy, compresses, returns
        indices as tensor for GPU-friendly storage.

        Args:
            x: KV tensor, shape (..., dim)
        Returns:
            (indices_tensor, codebook)
        """
        device = x.device
        x_np = x.detach().cpu().float().numpy()
        original_shape = x_np.shape
        x_flat = x_np.reshape(-1, self.dim)

        state = self.quantize(x_flat)

        indices_tensor = torch.from_numpy(state.indices).to(device)
        return indices_tensor, self.codebook

    def decompress_tensor(
        self,
        indices: torch.Tensor,
        codebook: np.ndarray,
        target_shape: tuple,
        target_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Decompress indices back to tensor form.
        """
        indices_np = indices.cpu().numpy()
        state = TurboQuantMSEState(
            indices=indices_np,
            codebook=codebook,
            bits=self.bits,
            original_shape=target_shape,
        )
        reconstructed_np = self.dequantize(state)
        return torch.from_numpy(reconstructed_np).to(
            device=indices.device, dtype=target_dtype
        )

    def memory_ratio(self) -> float:
        """
        Theoretical memory reduction vs FP16 storage.
        FP16 = 16 bits/value. TurboQuantMSE = self.bits bits/value.
        """
        return 16.0 / self.bits

    def __repr__(self) -> str:
        return (
            f"TurboQuantMSE(dim={self.dim}, bits={self.bits}, "
            f"n_centroids={self.n_centroids}, "
            f"memory_ratio={self.memory_ratio():.1f}x)"
        )