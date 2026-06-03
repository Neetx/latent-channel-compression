"""Lloyd-Max optimal scalar quantizer codebook for TurboQuant-style quantization.

The TurboQuant rotation makes each coordinate (approximately) i.i.d. with marginal
N(0, 1/d) in the high-d limit of the exact Beta(1/2, (d-1)/2) marginal of a
uniform-on-sphere coordinate. The optimal scalar quantizer for that distribution
is the Lloyd-Max codebook for N(0, 1/d).

This implementation mirrors `external/turboquant_ref/turboquant/main/lloyd_max.py`
analytically (scipy.integrate.quad on the Gaussian PDF). It is one-time
precomputation, cached by (d, bits).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from scipy import integrate


def _gaussian_pdf(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def lloyd_max_gaussian(
    d: int,
    bits: int,
    *,
    n_iter: int = 100,
    tol: float = 1e-6,
    range_sigmas: float = 5.0,
) -> np.ndarray:
    """Compute the Lloyd-Max codebook for the marginal N(0, 1/d) distribution.

    Returns ``2**bits`` sorted centroids as a numpy array.
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    n_levels = 2 ** bits
    sigma = 1.0 / np.sqrt(d)
    lo, hi = -range_sigmas * sigma, range_sigmas * sigma

    centroids = np.linspace(lo, hi, n_levels)
    for _ in range(n_iter):
        midpoints = 0.5 * (centroids[:-1] + centroids[1:])
        bounds = np.concatenate([[lo], midpoints, [hi]])
        new_c = np.empty_like(centroids)
        for i in range(n_levels):
            a, b = bounds[i], bounds[i + 1]
            num, _ = integrate.quad(lambda x: x * _gaussian_pdf(x, sigma), a, b)
            den, _ = integrate.quad(lambda x: _gaussian_pdf(x, sigma), a, b)
            new_c[i] = num / den if den > 1e-12 else centroids[i]
        if np.max(np.abs(new_c - centroids)) < tol:
            centroids = new_c
            break
        centroids = new_c
    return np.sort(centroids)


@lru_cache(maxsize=128)
def lloyd_max_gaussian_torch(d: int, bits: int) -> torch.Tensor:
    """Cached torch float32 view of :func:`lloyd_max_gaussian`."""
    arr = lloyd_max_gaussian(d, bits)
    return torch.from_numpy(arr).float()


__all__ = ["lloyd_max_gaussian", "lloyd_max_gaussian_torch"]
