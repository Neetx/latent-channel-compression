"""Channel fidelity metrics for the RecursiveMAS Outer-Link channel.

Operates on the post-LN vector that B (the downstream agent) actually receives
after the (Variant B) quantize/de-quantize round-trip.

All metrics are computed in fp32 for numerical safety, regardless of the input
dtype of the channel.

Conventions
-----------
- ``x_ref`` is the vector B would receive in the unquantized channel (BF16/FP32).
- ``x_recv`` is the vector B actually receives after de-quantization.
- Last axis is the hidden dimension ``d``. Leading dims aggregate (batch, seq).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F

from .distortion import cosine, relative_mse  # re-use the existing primitives

_EPS = 1e-12


def relative_l2(x_ref: torch.Tensor, x_recv: torch.Tensor) -> torch.Tensor:
    """Per-vector ||Δ||_2 / ||x_ref||_2 (square root of relative_mse).

    Matches the "relative L2" the user asked for at the injection point.
    """
    return relative_mse(x_ref, x_recv).clamp(min=0.0).sqrt()


def effective_rank(x: torch.Tensor) -> float:
    """Participation ratio of the singular-value spectrum, on the *channel* vectors.

    Defined as ``(Σ σ_i)² / Σ σ_i²`` — i.e. exp(H(p)) where p_i = σ_i² / Σ σ_j².
    Equals d when all singular values are equal (channel uses its full dimension),
    1 when the channel is rank-1. Reported on the centered matrix; the calling code
    should pass a 2-D matrix shaped ``(N_calls, d)`` (one channel vector per row).
    """
    if x.ndim != 2:
        raise ValueError(f"effective_rank expects 2-D (N, d), got {tuple(x.shape)}")
    if x.shape[0] < 2:
        return 1.0
    x32 = x.detach().to(torch.float32)
    x32 = x32 - x32.mean(dim=0, keepdim=True)
    # economical SVD; only singular values needed
    s = torch.linalg.svdvals(x32)
    s2 = s.pow(2)
    denom = s2.pow(2).sum()
    # All-identical-rows degenerate case: centred → 0 matrix → all σ=0.
    # Semantically this is a rank-1 channel (one point in feature space).
    if denom < _EPS:
        return 1.0
    return float(s2.sum().pow(2) / denom)


def codebook_extreme_rate(
    y_rotated: torch.Tensor, codebook: torch.Tensor, *, tail_levels: int = 1
) -> float:
    """Fraction of post-rotation coordinates that hit the outermost ``tail_levels``
    codebook levels on either side.

    This is the Variant-B analog of "INT4 outlier-clipping rate". A non-zero value
    means the marginal post-rotation distribution has heavier tails than the
    Lloyd-Max codebook accommodated, so the quantizer is saturating instead of
    interpolating. Compute by index lookup at quantize time and pass here.

    Parameters
    ----------
    y_rotated : (..., d) tensor of values AFTER Haar rotation, BEFORE bucketize.
    codebook : 1-D sorted tensor of codebook levels (length 2**bits).
    tail_levels : how many levels on each side count as "extreme" (default 1).
    """
    if codebook.ndim != 1 or codebook.numel() < 2 * tail_levels + 1:
        raise ValueError("codebook must be 1-D and longer than 2*tail_levels")
    midpoints = 0.5 * (codebook[:-1] + codebook[1:])
    idx = torch.bucketize(y_rotated.contiguous().to(midpoints.dtype), midpoints)
    K = codebook.numel()
    extreme_mask = (idx < tail_levels) | (idx >= K - tail_levels)
    return float(extreme_mask.float().mean())


# ---------------------------------------------------------------------------
# Aggregator over many calls (one per CrossModelAdapter forward).
# ---------------------------------------------------------------------------


@dataclass
class FidelityRun:
    """Accumulates per-call channel-fidelity stats for one paired REF/INT4 run.

    Each ``record(...)`` call corresponds to one CrossModelAdapter forward and
    appends per-call means of (cos, rel_l2). Channel vectors themselves can also
    be stashed for later effective-rank / spectral analysis.
    """

    label: str = ""
    cosines: List[float] = field(default_factory=list)
    rel_l2s: List[float] = field(default_factory=list)
    n_tokens: List[int] = field(default_factory=list)
    channel_samples: List[torch.Tensor] = field(default_factory=list)
    max_samples_per_call: int = 0  # 0 = don't stash, >0 = stash at most N rows

    def record(self, x_ref: torch.Tensor, x_recv: torch.Tensor) -> None:
        with torch.no_grad():
            x = x_ref.detach().to(torch.float32).reshape(-1, x_ref.shape[-1])
            xr = x_recv.detach().to(torch.float32).reshape(-1, x_recv.shape[-1])
            n_t = x.shape[0]
            self.cosines.append(float(cosine(x, xr).mean()))
            self.rel_l2s.append(float(relative_l2(x, xr).mean()))
            self.n_tokens.append(n_t)
            if self.max_samples_per_call > 0:
                k = min(self.max_samples_per_call, n_t)
                idx = torch.randperm(n_t)[:k]
                self.channel_samples.append(x[idx].cpu())

    def summary(self) -> dict:
        import statistics

        def _agg(xs):
            if not xs:
                return {"n": 0}
            return {
                "n": len(xs),
                "mean": statistics.mean(xs),
                "median": statistics.median(xs),
                "stdev": statistics.stdev(xs) if len(xs) >= 2 else 0.0,
                "min": min(xs),
                "max": max(xs),
            }

        out = {
            "label": self.label,
            "n_calls": len(self.cosines),
            "n_tokens_total": int(sum(self.n_tokens)),
            "cosine": _agg(self.cosines),
            "rel_l2": _agg(self.rel_l2s),
        }
        if self.channel_samples:
            mat = torch.cat(self.channel_samples, dim=0)
            out["effective_rank"] = effective_rank(mat)
            out["channel_matrix_shape"] = list(mat.shape)
        return out


__all__ = [
    "relative_l2",
    "effective_rank",
    "codebook_extreme_rate",
    "FidelityRun",
]
