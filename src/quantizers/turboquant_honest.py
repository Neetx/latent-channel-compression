"""Variant B — Haar random rotation + Lloyd-Max-for-Gaussian scalar quantizer.

The "honest" TurboQuant minimal: matches the algorithmic core of
arxiv 2504.19874 (Zandieh et al., ICLR 2026), Algorithm 1 (MSE-optimal).

Differences from Variant A (`hadamard_uniform.py`):
- Rotation is **Haar uniform on the orthogonal group** (QR of a Gaussian
  with sign correction), not randomized Hadamard. Cost is O(d²) per forward
  and O(d²) memory; acceptable for d ≤ 4096 at inference.
- Scalar quantizer uses the **Lloyd-Max codebook** for N(0, 1/d), not
  uniform symmetric. The codebook is precomputed once per (d, bits) and
  cached.
- Inputs are L2-normalized to the unit sphere before rotation and rescaled
  after the inverse rotation. The Lloyd-Max marginal guarantee only holds
  on the sphere; the norm is preserved exactly outside the quantized pipe.
- No per-channel mode. The whole point of the Haar rotation is to make all
  coordinates i.i.d. so a single codebook is optimal for every channel.

This module is the screening-vs-honest counterpart to Variant A. Failure of
Variant B at a given bit-rate is a *meaningful* signal that the RecursiveLink
channel is incompressible; failure of Variant A at the same bit-rate is not.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.utils.lloyd_max import lloyd_max_gaussian_torch


class HaarRotation(nn.Module):
    """Random orthogonal matrix uniformly sampled from the Haar measure.

    Construction: QR of a Gaussian matrix, then sign-correct the columns so
    the resulting Q has determinant +1 with the proper Haar measure. Matches
    ``external/turboquant_ref/turboquant/main/rotation.py``.

    Conventions: ``forward(x)`` returns ``x @ R.T`` (row vectors), matching
    the reference's ``rotated = x @ R.T``. ``inverse(y)`` returns ``y @ R``
    since R is orthogonal.
    """

    def __init__(self, d: int, seed: int = 42, dtype: torch.dtype = torch.float32):
        super().__init__()
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        rng = np.random.default_rng(int(seed))
        A = rng.standard_normal((d, d))
        Q, R = np.linalg.qr(A)
        signs = np.sign(np.diag(R))
        # Avoid the (measure-zero) edge case where any sign is 0.
        signs = np.where(signs == 0, 1.0, signs)
        Q = Q * signs[np.newaxis, :]
        self.register_buffer("R", torch.from_numpy(Q).to(dtype))
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.R.T.to(x.dtype)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.R.to(y.dtype)


class TurboQuantHonest(nn.Module):
    """Variant B: HaarRotation → Lloyd-Max quantize → inverse rotation.

    Parameters
    ----------
    d : int
        Input vector dimension.
    bits : int
        Number of bits per coordinate. Codebook has ``2**bits`` levels.
    normalize : bool, default True
        If True, L2-normalize the input to the unit sphere before rotation
        and rescale by the stored per-vector norm after the inverse. Set to
        False only when you know inputs are already unit-norm.
    seed : int, default 42
        Seed for the Haar rotation.
    lloyd_n_iter : int, default 100
        Lloyd-Max iterations.
    """

    def __init__(
        self,
        d: int,
        bits: int,
        *,
        normalize: bool = True,
        seed: int = 42,
        lloyd_n_iter: int = 100,
    ):
        super().__init__()
        self.rot = HaarRotation(d, seed=seed)
        codebook = lloyd_max_gaussian_torch(d, bits)
        midpoints = 0.5 * (codebook[:-1] + codebook[1:])
        self.register_buffer("codebook", codebook)
        self.register_buffer("midpoints", midpoints)
        self.d = d
        self.bits = bits
        self.normalize = normalize

    def _nearest_neighbor(self, y: torch.Tensor) -> torch.Tensor:
        """Vectorized nearest-neighbor on the 1-D sorted codebook.

        Uses ``torch.bucketize`` for O(N · d · log K) time and O(N · d) memory,
        avoiding the naïve ``[N, d, K]`` distance tensor.
        """
        idx = torch.bucketize(y.contiguous(), self.midpoints.to(y.dtype))
        return self.codebook.to(y.dtype)[idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            u = x / norm
        else:
            u = x
        y = self.rot(u)
        y_q = self._nearest_neighbor(y)
        u_q = self.rot.inverse(y_q)
        if self.normalize:
            return u_q * norm
        return u_q

    def extra_repr(self) -> str:
        return f"d={self.d}, bits={self.bits}, normalize={self.normalize}"


__all__ = ["HaarRotation", "TurboQuantHonest"]
