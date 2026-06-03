"""Variant A — Randomized Hadamard rotation + uniform symmetric scalar quantizer.

This is the *screening* quantizer, NOT a faithful TurboQuant implementation.
TurboQuant proper uses Haar-random rotation and Lloyd-Max-for-Beta quantization
(implemented separately as Variant B). The purpose of this module is to give a
cheap positive screen: if Variant A passes at a given bit-rate, the honest
TurboQuant (Variant B) will pass at least as well.

See RESEARCH.md §5.1 for the rationale.

Computational profile
---------------------
- Rotation: O(d log d) per vector via FWHT (in-place butterfly).
- Quantizer: per-tensor (default) or per-channel uniform symmetric, signed.
- No packed storage. No custom kernels. Pure PyTorch, autograd-compatible.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Fast Walsh-Hadamard Transform along the last axis.

    Requires ``x.shape[-1]`` to be a power of two. Returns a tensor of the same
    shape, with a 1/sqrt(d) normalization so the transform is its own inverse:
    ``fwht(fwht(x)) == x`` (up to floating-point error).

    Cost: O(N * d * log d) where N = prod(x.shape[:-1]).
    Memory: O(N * d) — does not mutate the input.
    """
    d = x.shape[-1]
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"FWHT requires last-dim to be a power of two, got d={d}")
    orig_shape = x.shape
    y = x.reshape(-1, d).clone()
    h = 1
    while h < d:
        # Group consecutive coordinates into butterfly pairs of size h.
        y = y.view(-1, d // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        # torch.stack avoids in-place aliasing issues that plague a naive
        # implementation; the temporary is unavoidable for an autograd-safe FWHT.
        y = torch.stack([a + b, a - b], dim=-2)
        y = y.reshape(-1, d)
        h *= 2
    y = y / math.sqrt(d)
    return y.reshape(orig_shape)


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class RandomizedHadamard(nn.Module):
    """Rotation H · diag(s) where s ∈ {±1}^d_pad is fixed once at construction.

    For input dims not a power of two, the input is zero-padded to the next
    power of two before rotation and unpadded after the inverse. The padding
    coordinates are quantized along with the real ones — they receive zero
    information, but they don't break the inverse because diag(s) and H are
    both orthonormal on the padded space.

    The rotation is deterministic given ``seed``. Shape: input ``[..., d]``,
    output (forward / inverse) same shape.
    """

    def __init__(self, d: int, seed: int = 0):
        super().__init__()
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        d_pad = _next_pow2(d)
        g = torch.Generator().manual_seed(int(seed))
        signs = (torch.randint(0, 2, (d_pad,), generator=g, dtype=torch.float32) * 2.0 - 1.0)
        self.register_buffer("signs", signs)
        self.d = d
        self.d_pad = d_pad

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.d == self.d_pad:
            return x
        pad_shape = list(x.shape)
        pad_shape[-1] = self.d_pad - self.d
        pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply D then H: y = H · D · x  (note: not H·D·x in matrix sense, but
        # element-wise scaling by signs followed by FWHT, which equals HDx).
        y = self._pad(x) * self.signs.to(x.dtype)
        return fwht(y)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # (HD)^{-1} = D^{-1} H^{-1} = D · H  (both H and D are self-inverse).
        x = fwht(y) * self.signs.to(y.dtype)
        return x[..., : self.d]


class UniformSymmetricQuantizer(nn.Module):
    """Signed symmetric uniform scalar quantizer (fake-quant; float in, float out).

    For ``bits`` ≥ 2:
        levels ∈ {-Qmax, ..., -1, 0, 1, ..., Qmax}  where  Qmax = 2^(bits-1) - 1
        That is 2·Qmax + 1 distinct levels (so 3, 7, 15, 255 for bits=2,3,4,8).
        Scale = max(|x|) / Qmax.

    No packed storage. Use this only to measure distortion, not to claim VRAM
    savings.
    """

    def __init__(self, bits: int, per_channel: bool = False):
        super().__init__()
        if bits < 2:
            raise ValueError(f"bits must be >= 2 (use a sign-only quantizer for 1-bit), got {bits}")
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.per_channel = per_channel

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            # Reduce over all dims except the last (channel).
            reduce_dims = tuple(range(y.dim() - 1))
            if reduce_dims:
                scale = y.abs().amax(dim=reduce_dims, keepdim=True)
            else:
                scale = y.abs()
        else:
            scale = y.abs().amax()
        scale = scale.clamp(min=1e-8) / self.qmax
        q = torch.round(y / scale).clamp(-self.qmax, self.qmax)
        return q * scale


class HadamardUniformQuantizer(nn.Module):
    """Variant A: RandomizedHadamard rotate → uniform quantize → inverse rotate.

    Designed to drop into ``CrossModelAdapter.forward`` (or ``Adapter.forward``)
    right after the final LayerNorm; see RESEARCH.md §5.2.
    """

    def __init__(
        self,
        d: int,
        bits: int,
        *,
        per_channel: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.rot = RandomizedHadamard(d, seed=seed)
        self.quant = UniformSymmetricQuantizer(bits, per_channel=per_channel)
        self.d = d
        self.bits = bits
        self.per_channel = per_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.rot(x)
        y_q = self.quant(y)
        return self.rot.inverse(y_q)

    def extra_repr(self) -> str:
        return f"d={self.d}, bits={self.bits}, per_channel={self.per_channel}"


__all__ = [
    "fwht",
    "RandomizedHadamard",
    "UniformSymmetricQuantizer",
    "HadamardUniformQuantizer",
]
