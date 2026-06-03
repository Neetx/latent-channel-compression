"""Distortion metrics for quantized hidden states.

All functions operate on the last axis of their input and broadcast over the
leading dimensions. Returned tensors have shape ``x.shape[:-1]``.

Conventions:
- ``x`` is the reference (un-quantized) tensor.
- ``x_q`` is the quantized reconstruction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-12


def relative_mse(x: torch.Tensor, x_q: torch.Tensor) -> torch.Tensor:
    """Per-vector ‖x − x_q‖² / ‖x‖²."""
    num = (x - x_q).pow(2).sum(dim=-1)
    den = x.pow(2).sum(dim=-1).clamp(min=_EPS)
    return num / den


def cosine(x: torch.Tensor, x_q: torch.Tensor) -> torch.Tensor:
    """Per-vector cosine similarity."""
    return F.cosine_similarity(x, x_q, dim=-1)


def norm_ratio(x: torch.Tensor, x_q: torch.Tensor) -> torch.Tensor:
    """Per-vector ‖x_q‖ / ‖x‖. Should be ≈ 1 if the quantizer preserves scale."""
    return x_q.norm(dim=-1) / x.norm(dim=-1).clamp(min=_EPS)


def inner_product_error(
    x: torch.Tensor, y: torch.Tensor, x_q: torch.Tensor, y_q: torch.Tensor
) -> torch.Tensor:
    """|⟨x, y⟩ − ⟨x_q, y_q⟩| / (‖x‖ · ‖y‖).

    Predicts whether the quantizer will hurt downstream attention-like consumers.
    """
    ip_true = (x * y).sum(dim=-1)
    ip_q = (x_q * y_q).sum(dim=-1)
    denom = (x.norm(dim=-1) * y.norm(dim=-1)).clamp(min=_EPS)
    return (ip_true - ip_q).abs() / denom


__all__ = ["relative_mse", "cosine", "norm_ratio", "inner_product_error"]
