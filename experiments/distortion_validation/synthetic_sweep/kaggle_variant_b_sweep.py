"""Standalone Kaggle script: Variant B distortion sweep on synthetic data.

Self-contained: numpy + scipy + torch only. Outputs
``/kaggle/working/variant_b_sweep.json`` with the full grid + 95% bootstrap CIs.

Same grid as Variant A so head-to-head is direct.
"""
import json
import time

import numpy as np
import torch
from scipy import integrate


# ----- Lloyd-Max codebook (analytical, ported from turboquant_ref) -----

def _gaussian_pdf(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def lloyd_max_gaussian(d, bits, n_iter=100, tol=1e-6, range_sigmas=5.0):
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


# ----- Haar rotation -----

class HaarRotation(torch.nn.Module):
    def __init__(self, d, seed=42):
        super().__init__()
        rng = np.random.default_rng(int(seed))
        A = rng.standard_normal((d, d))
        Q, R = np.linalg.qr(A)
        signs = np.sign(np.diag(R))
        signs = np.where(signs == 0, 1.0, signs)
        Q = Q * signs[np.newaxis, :]
        self.register_buffer("R", torch.from_numpy(Q).float())
        self.d = d

    def forward(self, x):
        return x @ self.R.T

    def inverse(self, y):
        return y @ self.R


class TurboQuantHonest(torch.nn.Module):
    def __init__(self, d, bits, normalize=True, seed=42):
        super().__init__()
        self.rot = HaarRotation(d, seed=seed)
        cb = torch.from_numpy(lloyd_max_gaussian(d, bits)).float()
        self.register_buffer("codebook", cb)
        self.register_buffer("midpoints", 0.5 * (cb[:-1] + cb[1:]))
        self.d, self.bits, self.normalize = d, bits, normalize

    def _nn(self, y):
        idx = torch.bucketize(y.contiguous(), self.midpoints)
        return self.codebook[idx]

    def forward(self, x):
        if self.normalize:
            norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            u = x / norm
        else:
            u = x
        y = self.rot(u)
        y_q = self._nn(y)
        u_q = self.rot.inverse(y_q)
        return (u_q * norm) if self.normalize else u_q


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

def bootstrap_ci95(values, n_boot=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = values.numel()
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    boots = values[idx].mean(dim=1)
    return values.mean().item(), torch.quantile(boots, 0.025).item(), torch.quantile(boots, 0.975).item()


def run_sweep():
    dims = [256, 1536, 2048]
    bit_widths = [16, 8, 4, 3, 2]
    seeds = [0, 1, 2, 3, 4]
    N = 1000

    results = []
    t0 = time.time()

    # Build one Haar rotation per (d, seed) — codebooks cached per (d, bits) implicitly.
    print(f"Precomputing Haar rotations for dims={dims}, seeds={seeds}...")
    rotations = {}
    for d in dims:
        for seed in seeds:
            rotations[(d, seed)] = HaarRotation(d, seed=seed)

    for d in dims:
        for bits in bit_widths:
            rmse_acc, cos_acc, nr_acc, ip_acc = [], [], [], []
            for seed in seeds:
                g = torch.Generator().manual_seed(seed)
                x_raw = torch.randn(N, d, generator=g)
                x_sphere = x_raw / x_raw.norm(dim=-1, keepdim=True).clamp(min=EPS)
                y_raw = torch.randn(N, d, generator=g)
                y_sphere = y_raw / y_raw.norm(dim=-1, keepdim=True).clamp(min=EPS)

                # Build quantizer with the precomputed rotation
                q = TurboQuantHonest(d=d, bits=bits, normalize=True, seed=seed)
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
                "d": d, "bits": bits, "normalize": True,
                "n_total": int(rmse_all.numel()),
                "rMSE":       {"mean": rmse_m, "ci95_lo": rmse_lo, "ci95_hi": rmse_hi},
                "cosine":     {"mean": cos_m,  "ci95_lo": cos_lo,  "ci95_hi": cos_hi},
                "norm_ratio": {"mean": nr_m,   "ci95_lo": nr_lo,   "ci95_hi": nr_hi},
                "ip_error":   {"mean": ip_m,   "ci95_lo": ip_lo,   "ci95_hi": ip_hi},
            })
            print(f"d={d:5d} bits={bits:2d}  rMSE={rmse_m:.4f}  cos={cos_m:.4f}  nr={nr_m:.4f}  ip={ip_m:.4f}")

    elapsed = time.time() - t0
    return {
        "variant": "B (Haar + Lloyd-Max-Gaussian)",
        "config": {
            "dims": dims, "bits": bit_widths, "seeds": seeds,
            "samples_per_seed": N, "normalize": True,
            "lloyd_max_n_iter": 100, "lloyd_max_range_sigmas": 5.0,
        },
        "results": results,
        "elapsed_seconds": elapsed,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


if __name__ == "__main__":
    print(f"torch {torch.__version__}  CUDA={torch.cuda.is_available()}")
    out = run_sweep()
    out_path = "/kaggle/working/variant_b_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total runtime: {out['elapsed_seconds']:.1f}s, {len(out['results'])} configurations")
