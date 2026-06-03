"""Logit-level fidelity metrics at egress (paired REF vs INT4).

Operates on per-decode-position logit vectors produced by the Solver's
``model.generate(..., output_scores=True)``.

Conventions
-----------
- Inputs are 2-D tensors ``(T, V)`` — T decoded positions, V vocab size.
- One pair per problem; metrics are averaged over positions, then over problems
  in the calling code.
- Computations use log-softmax for numerical stability and add ``eps`` to avoid
  log(0). Reported probabilities are at temperature 1.0 regardless of the
  generation-time temperature (the metric is on the underlying distribution).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

_EPS = 1e-12


def per_position_mse(
    logits_ref: torch.Tensor, logits_int: torch.Tensor
) -> torch.Tensor:
    """Per-position mean squared error over the vocab dimension.

    Returns shape ``(T,)``.
    """
    return (logits_ref - logits_int).pow(2).mean(dim=-1)


def per_position_kl(
    logits_ref: torch.Tensor, logits_int: torch.Tensor
) -> torch.Tensor:
    """KL(p_REF || p_INT4) per decode position, at temperature 1.0.

    Uses log-softmax for stability. Returns shape ``(T,)``.
    """
    log_p = F.log_softmax(logits_ref.to(torch.float32), dim=-1)
    log_q = F.log_softmax(logits_int.to(torch.float32), dim=-1)
    p = log_p.exp()
    # KL = sum p (log p - log q); guarded by stable log-softmax
    return (p * (log_p - log_q)).sum(dim=-1)


def per_position_js(
    logits_ref: torch.Tensor, logits_int: torch.Tensor
) -> torch.Tensor:
    """Symmetric Jensen-Shannon divergence per position, at temperature 1.0.

    JS(p,q) = 0.5 * (KL(p||m) + KL(q||m)) with m = 0.5 * (p+q).
    Returns shape ``(T,)``.
    """
    log_p = F.log_softmax(logits_ref.to(torch.float32), dim=-1)
    log_q = F.log_softmax(logits_int.to(torch.float32), dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = (m + _EPS).log()
    return 0.5 * ((p * (log_p - log_m)).sum(dim=-1) + (q * (log_q - log_m)).sum(dim=-1))


@dataclass
class PairedLogitStats:
    """Aggregate stats for one (problem, T_value) pair.

    Position-level stats are summarized into per-problem means; the calling code
    then averages over problems and computes bootstrap CIs.
    """

    mse_mean: float
    mse_max: float
    kl_mean: float
    kl_max: float
    js_mean: float
    n_positions: int
    n_positions_clamped: int


def summarize_pair(
    logits_ref: torch.Tensor,
    logits_int: torch.Tensor,
    *,
    max_positions: int = 0,
) -> PairedLogitStats:
    """Compute per-problem-mean MSE / KL / JS for one paired (REF, INT4) pair.

    If ``max_positions > 0``, truncate both sequences to the first N positions
    before measuring (use when REF and INT4 generated different sequence lengths).
    """
    if logits_ref.shape[-1] != logits_int.shape[-1]:
        raise ValueError(
            f"vocab mismatch: REF V={logits_ref.shape[-1]} INT4 V={logits_int.shape[-1]}"
        )
    T = min(logits_ref.shape[0], logits_int.shape[0])
    if max_positions > 0:
        T = min(T, max_positions)
    n_clamped = max(logits_ref.shape[0], logits_int.shape[0]) - T
    lr = logits_ref[:T]
    li = logits_int[:T]

    mse = per_position_mse(lr, li)
    kl = per_position_kl(lr, li)
    js = per_position_js(lr, li)

    return PairedLogitStats(
        mse_mean=float(mse.mean()),
        mse_max=float(mse.max()) if T > 0 else 0.0,
        kl_mean=float(kl.mean()),
        kl_max=float(kl.max()) if T > 0 else 0.0,
        js_mean=float(js.mean()),
        n_positions=T,
        n_positions_clamped=n_clamped,
    )


__all__ = [
    "per_position_mse",
    "per_position_kl",
    "per_position_js",
    "PairedLogitStats",
    "summarize_pair",
]
