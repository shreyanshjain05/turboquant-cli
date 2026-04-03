"""
rotation.py — Random Orthogonal Rotation Matrix
================================================
TurboQuant begins by multiplying every KV vector by a random
rotation matrix Π. This spreads energy evenly across all dimensions
(eliminating outlier spikes) without changing the vector's magnitude.

Key property: ||Π·x||₂ = ||x||₂  (rotation preserves L2 norm)

The matrix is Haar-distributed (uniformly random orthogonal),
generated once via QR decomposition and reused for all vectors.
Storing it once costs O(d²) — computed offline, applied online.
"""

import numpy as np
import torch
from typing import Optional


class RotationMatrix:
    """
    Haar-distributed random orthogonal matrix via QR decomposition.

    Usage:
        rot = RotationMatrix(dim=128, seed=42)
        rotated = rot.apply(x)          # numpy array
        rotated = rot.apply_tensor(x)   # torch tensor
    """

    def __init__(self, dim: int, seed: int = 42):
        """
        Args:
            dim:  Dimensionality of the vectors to rotate (head_dim).
            seed: Random seed for reproducibility. Keep this fixed —
                  the same seed must be used for compression and
                  decompression, or reconstruction will be nonsense.
        """
        self.dim = dim
        self.seed = seed
        self._matrix: Optional[np.ndarray] = None
        self._matrix_tensor: Optional[torch.Tensor] = None

    def _build(self) -> np.ndarray:
        """Build the rotation matrix (called lazily on first use)."""
        rng = np.random.default_rng(self.seed)
        # QR decomposition of a random Gaussian matrix gives a
        # Haar-distributed orthogonal matrix
        G = rng.standard_normal((self.dim, self.dim))
        Q, _ = np.linalg.qr(G)
        return Q.astype(np.float32)

    @property
    def matrix(self) -> np.ndarray:
        """Rotation matrix as numpy array (lazy init)."""
        if self._matrix is None:
            self._matrix = self._build()
        return self._matrix

    @property
    def matrix_tensor(self) -> torch.Tensor:
        """Rotation matrix as torch tensor (lazy init)."""
        if self._matrix_tensor is None:
            self._matrix_tensor = torch.from_numpy(self.matrix)
        return self._matrix_tensor

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply rotation: y = Π · x

        Args:
            x: Input vector(s), shape (dim,) or (N, dim)
        Returns:
            Rotated vector(s), same shape as input
        """
        if x.ndim == 1:
            return self.matrix @ x
        return x @ self.matrix.T   # (N, dim) @ (dim, dim).T

    def apply_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to a torch tensor.

        Args:
            x: Shape (..., dim) — works on any leading batch dims
        Returns:
            Rotated tensor, same shape
        """
        Pi = self.matrix_tensor.to(x.device, dtype=x.dtype)
        return x @ Pi.T

    def invert(self, y: np.ndarray) -> np.ndarray:
        """
        Invert rotation: x = Πᵀ · y
        (For orthogonal matrices, inverse = transpose)
        """
        if y.ndim == 1:
            return self.matrix.T @ y
        return y @ self.matrix

    def invert_tensor(self, y: torch.Tensor) -> torch.Tensor:
        """Invert rotation on a torch tensor."""
        Pi = self.matrix_tensor.to(y.device, dtype=y.dtype)
        return y @ Pi

    def __repr__(self) -> str:
        return f"RotationMatrix(dim={self.dim}, seed={self.seed})"


# ── Utility: Hadamard-based fast rotation (optional, faster for large d) ──

def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform as an alternative to full QR rotation.
    O(d log d) vs O(d²) — useful when head_dim is large (>= 512).
    Requires dim to be a power of 2.
    """
    d = x.shape[-1]
    assert d & (d - 1) == 0, "Hadamard transform requires dim to be power of 2"

    h = x.clone()
    step = 1
    while step < d:
        for i in range(0, d, step * 2):
            left  = h[..., i:i + step]
            right = h[..., i + step:i + step * 2]
            h[..., i:i + step]            = left + right
            h[..., i + step:i + step * 2] = left - right
        step *= 2

    return h / (d ** 0.5)   # normalise to preserve L2 norm
