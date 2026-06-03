"""Standalone Kaggle script: Variant A distortion sweep on synthetic data.

Self-contained — no external imports beyond torch/numpy/json.
Outputs `/kaggle/working/variant_a_sweep.json` with the full grid of metrics.

This is the script body that gets pushed via the Kaggle MCP. Keep it pure stdlib
+ torch + numpy. Do NOT add imports that aren't in the default Kaggle image.
"""
import json
import math
import time
from typing import List

import torch


# ----- Variant A implementation (inlined for portability) -----

def fwht(x):
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"d={d} must be power of 2"
    orig_shape = x.shape
    y = x.reshape(-1, d).clone()
    h = 1
    while h < d:
        y = y.view(-1, d // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack([a + b, a - b], dim=-2)
        y = y.reshape(-1, d)
        h *= 2
    return (y / math.sqrt(d)).reshape(orig_shape)


def next_pow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class RandomizedHadamard(torch.nn.Module):
    def __init__(self, d, seed=0):
        super().__init__()
        d_pad = next_pow2(d)
        g = torch.Generator().manual_seed(int(seed))
        signs = (torch.randint(0, 2, (d_pad,), generator=g, dtype=torch.float32) * 2 - 1)
        self.register_buffer("signs", signs)
        self.d, self.d_pad = d, d_pad

    def _pad(self, x):
        if self.d == self.d_pad:
            return x
        pad_shape = list(x.shape); pad_shape[-1] = self.d_pad - self.d
        return torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=-1)

    def forward(self, x):
        return fwht(self._pad(x) * self.signs.to(x.dtype))

    def inverse(self, y):
        return (fwht(y) * self.signs.to(y.dtype))[..., : self.d]


class UniformSymmetricQuantizer(torch.nn.Module):
    def __init__(self, bits, per_channel=False):
        super().__init__()
        assert bits >= 2
        self.bits, self.qmax, self.per_channel = bits, (1 << (bits - 1)) - 1, per_channel

    def forward(self, y):
        if self.per_channel:
            reduce_dims = tuple(range(y.dim() - 1))
            scale = y.abs().amax(dim=reduce_dims, keepdim=True) if reduce_dims else y.abs()
        else:
            scale = y.abs().amax()
        scale = scale.clamp(min=1e-8) / self.qmax
        return (torch.round(y / scale).clamp(-self.qmax, self.qmax)) * scale


class HadamardUniformQuantizer(torch.nn.Module):
    def __init__(self, d, bits, per_channel=False, seed=0):
        super().__init__()
        self.rot = RandomizedHadamard(d, seed=seed)
        self.quant = UniformSymmetricQuantizer(bits, per_channel=per_channel)
        self.d, self.bits, self.per_channel = d, bits, per_channel

    def forward(self, x):
        y = self.rot(x)
        return self.rot.inverse(self.quant(y))


# ----- Metrics -----

EPS = 1e-12

def relative_mse(x, xq):
    return (x - xq).pow(2).sum(-1) / x.pow(2).sum(-1).clamp(min=EPS)

def cosine(x, xq):
    return torch.nn.functional.cosine_similarity(x, xq, dim=-1)

def norm_ratio(x, xq):
    return xq.norm(dim=-1) / x.norm(dim=-1).clamp(min=EPS)

def inner_product_error(x, y, xq, yq):
    ip_true = (x * y).sum(-1); ip_q = (xq * yq).sum(-1)
    denom = (x.norm(dim=-1) * y.norm(dim=-1)).clamp(min=EPS)
    return (ip_true - ip_q).abs() / denom


# ----- Sweep -----

def bootstrap_ci95(values: torch.Tensor, n_boot: int = 1000, seed: int = 0):
    """Return (mean, low95, high95)."""
    g = torch.Generator().manual_seed(seed)
    n = values.numel()
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    boots = values[idx].mean(dim=1)
    return values.mean().item(), torch.quantile(boots, 0.025).item(), torch.quantile(boots, 0.975).item()


def run_sweep():
    # Representative of RecursiveMAS inner-link hidden dims (Sequential-Light).
    # 1536 = Qwen2.5-Math-1.5B, 2048 = Qwen3-1.7B / Llama3.2-1B.
    # 256 included as a sanity dim where coordinates aren't extremely concentrated.
    dims = [256, 1536, 2048]
    bit_widths = [16, 8, 4, 3, 2]
    seeds = [0, 1, 2, 3, 4]   # 5 seeds for CI
    N = 1000                  # samples per seed
    per_channel_modes = [False, True]

    results: List[dict] = []

    t0 = time.time()
    for d in dims:
        for per_channel in per_channel_modes:
            for bits in bit_widths:
                # Aggregate metrics across seeds
                rmse_acc, cos_acc, nr_acc, ip_acc = [], [], [], []
                for seed in seeds:
                    g = torch.Generator().manual_seed(seed)
                    x_raw = torch.randn(N, d, generator=g)
                    x_sphere = x_raw / x_raw.norm(dim=-1, keepdim=True).clamp(min=EPS)
                    y_raw = torch.randn(N, d, generator=g)
                    y_sphere = y_raw / y_raw.norm(dim=-1, keepdim=True).clamp(min=EPS)

                    q = HadamardUniformQuantizer(d=d, bits=bits, per_channel=per_channel, seed=seed)
                    xq = q(x_sphere); yq = q(y_sphere)

                    rmse_acc.append(relative_mse(x_sphere, xq))
                    cos_acc.append(cosine(x_sphere, xq))
                    nr_acc.append(norm_ratio(x_sphere, xq))
                    ip_acc.append(inner_product_error(x_sphere, y_sphere, xq, yq))

                rmse_all = torch.cat(rmse_acc); cos_all = torch.cat(cos_acc)
                nr_all = torch.cat(nr_acc);     ip_all = torch.cat(ip_acc)

                rmse_m, rmse_lo, rmse_hi = bootstrap_ci95(rmse_all)
                cos_m, cos_lo, cos_hi    = bootstrap_ci95(cos_all)
                nr_m,  nr_lo,  nr_hi     = bootstrap_ci95(nr_all)
                ip_m,  ip_lo,  ip_hi     = bootstrap_ci95(ip_all)

                results.append({
                    "d": d, "bits": bits, "per_channel": per_channel,
                    "n_total": int(rmse_all.numel()),
                    "rMSE":       {"mean": rmse_m, "ci95_lo": rmse_lo, "ci95_hi": rmse_hi},
                    "cosine":     {"mean": cos_m,  "ci95_lo": cos_lo,  "ci95_hi": cos_hi},
                    "norm_ratio": {"mean": nr_m,   "ci95_lo": nr_lo,   "ci95_hi": nr_hi},
                    "ip_error":   {"mean": ip_m,   "ci95_lo": ip_lo,   "ci95_hi": ip_hi},
                })
                print(f"d={d:5d} bits={bits:2d} per_channel={per_channel!s:5s}  "
                      f"rMSE={rmse_m:.4f}  cos={cos_m:.4f}  nr={nr_m:.4f}  ip={ip_m:.4f}")

    elapsed = time.time() - t0

    out = {
        "variant": "A (Hadamard + uniform)",
        "config": {
            "dims": dims, "bits": bit_widths, "seeds": seeds,
            "samples_per_seed": N, "per_channel_modes": per_channel_modes,
        },
        "results": results,
        "elapsed_seconds": elapsed,
        "torch_version": torch.__version__,
    }
    return out


if __name__ == "__main__":
    print(f"torch {torch.__version__}  CUDA={torch.cuda.is_available()}")
    out = run_sweep()
    out_path = "/kaggle/working/variant_a_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total runtime: {out['elapsed_seconds']:.1f}s, {len(out['results'])} configurations")
