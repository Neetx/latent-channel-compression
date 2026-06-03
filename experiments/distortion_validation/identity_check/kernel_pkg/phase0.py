"""Phase 0 — RecursiveLink quantization on REAL inner-adapter output.

Loads the smallest released Sequential-Light agent (Solver Qwen2.5-Math-1.5B)
and its trained inner adapter ("ln_res_adapter" type with proj1/proj2/pre_ln/
post_ln), captures the adapter output on a small batch of math prompts, and
runs:

  Gate 0 — identity wrapper sanity (bits=16 must be ~identity).
  Variant B — Haar + Lloyd-Max-Gaussian distortion sweep at bits ∈ {8,4,3,2}
              on REAL data, head-to-head against synthetic baseline numbers.

Outputs `/kaggle/working/phase0_results.json`.
Requires `enable_internet=true` (downloads from HuggingFace).
GPU recommended but not required: fp16 + 1.5B params runs OK on T4.

Self-contained: only PyPI packages assumed to exist on the Kaggle image
(torch, transformers, huggingface_hub, numpy, scipy).
"""
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from scipy import integrate
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# Section 1 — Adapter class (matches RecursiveMAS/modeling.py Adapter exactly)
# ============================================================================

class InnerAdapter(nn.Module):
    """Mirrors `Adapter(adapter_type='ln_res_adapter')` from RecursiveMAS."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_size, hidden_size)
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        h = self.pre_ln(x)
        out = self.proj2(self.act(self.proj1(h)))
        out = x + out
        return self.post_ln(out)


# ============================================================================
# Section 2 — Lloyd-Max codebook (analytical, matches turboquant_ref)
# ============================================================================

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


# ============================================================================
# Section 3 — Variant B (Haar + Lloyd-Max), normalize-on-sphere
# ============================================================================

class HaarRotation(nn.Module):
    def __init__(self, d, seed=42, dtype=torch.float32):
        super().__init__()
        rng = np.random.default_rng(int(seed))
        A = rng.standard_normal((d, d))
        Q, R = np.linalg.qr(A)
        signs = np.sign(np.diag(R))
        signs = np.where(signs == 0, 1.0, signs)
        Q = Q * signs[np.newaxis, :]
        self.register_buffer("R", torch.from_numpy(Q).to(dtype))
        self.d = d

    def forward(self, x):
        return x @ self.R.T.to(x.dtype)

    def inverse(self, y):
        return y @ self.R.to(y.dtype)


class TurboQuantHonest(nn.Module):
    def __init__(self, d, bits, normalize=True, seed=42):
        super().__init__()
        self.rot = HaarRotation(d, seed=seed)
        cb = torch.from_numpy(lloyd_max_gaussian(d, bits)).float()
        self.register_buffer("codebook", cb)
        self.register_buffer("midpoints", 0.5 * (cb[:-1] + cb[1:]))
        self.d, self.bits, self.normalize = d, bits, normalize

    def _nn(self, y):
        idx = torch.bucketize(y.contiguous(), self.midpoints.to(y.dtype))
        return self.codebook.to(y.dtype)[idx]

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


# ============================================================================
# Section 4 — Metrics
# ============================================================================

EPS = 1e-12

def rel_mse(x, xq):
    return ((x - xq).pow(2).sum(-1) / x.pow(2).sum(-1).clamp(min=EPS))

def cosine(x, xq):
    return F.cosine_similarity(x, xq, dim=-1)

def norm_ratio(x, xq):
    return xq.norm(dim=-1) / x.norm(dim=-1).clamp(min=EPS)

def ip_error(x, y, xq, yq):
    ip_t = (x * y).sum(-1); ip_q = (xq * yq).sum(-1)
    denom = (x.norm(dim=-1) * y.norm(dim=-1)).clamp(min=EPS)
    return (ip_t - ip_q).abs() / denom

def bootstrap_ci95(values, n_boot=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = values.detach().cpu().float().flatten()
    n = v.numel()
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    boots = v[idx].mean(dim=1)
    return float(v.mean()), float(torch.quantile(boots, 0.025)), float(torch.quantile(boots, 0.975))


# ============================================================================
# Section 5 — Model + adapter loading
# ============================================================================

SOLVER_REPO = "RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B"


def find_adapter_file(repo_dir: Path) -> Path:
    """Mirror RecursiveMAS/hf_resolver.resolve_inner_adapter for task='math'."""
    manifest = repo_dir / "innerlink_config.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        tasks = data.get("tasks", {})
        if "math" in tasks:
            name = tasks["math"].get("adapter.pt", "adapter.pt")
            cand = repo_dir / name
            if cand.is_file():
                return cand
    for name in ["adapter(math).pt", "adapter.pt"]:
        cand = repo_dir / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No adapter found under {repo_dir}; files={list(repo_dir.iterdir())}")


def load_model_and_adapter(device, dtype):
    print(f"[load] snapshot_download {SOLVER_REPO} ...")
    t0 = time.time()
    repo_dir = Path(snapshot_download(SOLVER_REPO))
    print(f"[load]   -> {repo_dir}  ({time.time()-t0:.1f}s)")
    print(f"[load]   files: {sorted(p.name for p in repo_dir.iterdir())[:20]}")

    # Build a model-only view by symlinking out adapter junk that newer
    # transformers might mistake for a PEFT adapter.
    view = repo_dir / "_plain_model_view"
    if not view.is_dir() and (repo_dir / "adapter_config.json").is_file():
        view.mkdir()
        excluded_names = {"adapter_config.json", "innerlink_config.json", "README.md"}
        for item in repo_dir.iterdir():
            if item == view or item.name in excluded_names or item.suffix == ".pt":
                continue
            if item.name.startswith("adapter("):
                continue
            (view / item.name).symlink_to(item.resolve())
        model_dir = view
    else:
        model_dir = repo_dir

    print(f"[load] AutoModelForCausalLM.from_pretrained({model_dir.name})")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=dtype, trust_remote_code=True,
    ).to(device).eval()
    print(f"[load]   model loaded  ({time.time()-t0:.1f}s)  hidden_size={model.config.hidden_size}")

    adapter_file = find_adapter_file(repo_dir)
    print(f"[load] adapter file: {adapter_file.name}")
    state = torch.load(adapter_file, map_location="cpu", weights_only=True)
    adapter = InnerAdapter(model.config.hidden_size).to(device).to(dtype).eval()
    missing, unexpected = adapter.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"adapter state mismatch: missing={missing} unexpected={unexpected}")
    print(f"[load]   adapter loaded into InnerAdapter(d={model.config.hidden_size})")
    return model, tokenizer, adapter


# ============================================================================
# Section 6 — Prompts + hidden-state capture
# ============================================================================

PROMPTS = [
    "Compute the integral of x^2 from 0 to 3.",
    "Solve for x: 2x + 5 = 17.",
    "What is the derivative of sin(x) * exp(x)?",
    "Find the roots of x^2 - 4x - 5 = 0.",
    "How many ways can 5 books be arranged on a shelf?",
    "Convert 3/8 to a decimal.",
    "What is the area of a circle with radius 4?",
    "Sum of the first 100 natural numbers?",
    "Solve the system: x + y = 7, x - y = 1.",
    "What is the limit of (sin x)/x as x -> 0?",
    "Factor x^3 - 8.",
    "Find the inverse of the matrix [[1,2],[3,4]].",
    "Evaluate the determinant of [[2,0,1],[1,1,0],[0,1,2]].",
    "What is the value of e to four decimal places?",
    "Solve the differential equation dy/dx = y.",
]


@torch.no_grad()
def capture_adapter_output(model, tokenizer, adapter, device, dtype, max_len=128):
    """For each prompt, run the LM, take last-layer hidden states, push them
    through the inner adapter. Returns a flat tensor [N_tokens, hidden_size]
    plus per-prompt token counts.
    """
    all_hidden = []
    all_adapted = []
    tok_counts = []
    t0 = time.time()
    for i, p in enumerate(PROMPTS):
        ids = tokenizer(p, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        out = model(**ids, output_hidden_states=True, return_dict=True)
        last_h = out.hidden_states[-1]  # [1, T, H]
        adapted = adapter(last_h)        # [1, T, H]
        all_hidden.append(last_h.squeeze(0).cpu().float())
        all_adapted.append(adapted.squeeze(0).cpu().float())
        tok_counts.append(int(ids["input_ids"].shape[1]))
    print(f"[capture] {len(PROMPTS)} prompts, total tokens={sum(tok_counts)}, "
          f"hidden_size={all_hidden[0].shape[-1]}, elapsed={time.time()-t0:.1f}s")
    return torch.cat(all_hidden, dim=0), torch.cat(all_adapted, dim=0), tok_counts


# ============================================================================
# Section 7 — Distortion stats on REAL output
# ============================================================================

def describe_distribution(z: torch.Tensor) -> dict:
    """Sanity stats on adapter output: per-coord mean/std, sphere distance, etc."""
    z = z.float()
    norms = z.norm(dim=-1)
    return {
        "n_tokens": int(z.shape[0]),
        "hidden_dim": int(z.shape[-1]),
        "abs_mean": float(z.abs().mean()),
        "abs_max": float(z.abs().max()),
        "coord_std_min": float(z.std(dim=0).min()),
        "coord_std_max": float(z.std(dim=0).max()),
        "coord_std_mean": float(z.std(dim=0).mean()),
        "vector_norm_mean": float(norms.mean()),
        "vector_norm_std": float(norms.std()),
        "vector_norm_min": float(norms.min()),
        "vector_norm_max": float(norms.max()),
    }


def run_quantizer_sweep(z: torch.Tensor, dims: int, bits_list, seed=0) -> list:
    """For each bit-rate run Variant B and report metrics."""
    rows = []
    # Pair vectors for inner-product error: use a shifted version of z so we have
    # something to inner-product with. Real downstream uses keys/queries from a
    # different sequence, but on Phase 0.A we don't have that — shifted pairs is
    # a reasonable proxy.
    N = z.shape[0]
    if N < 2:
        raise RuntimeError("need >= 2 tokens for inner-product metric")
    z_pair = torch.cat([z[1:], z[:1]], dim=0)

    for bits in bits_list:
        q = TurboQuantHonest(d=dims, bits=bits, normalize=True, seed=seed)
        zq = q(z)
        zqp = q(z_pair)
        rmse_m, rmse_lo, rmse_hi = bootstrap_ci95(rel_mse(z, zq))
        cos_m,  cos_lo,  cos_hi  = bootstrap_ci95(cosine(z, zq))
        nr_m,   nr_lo,   nr_hi   = bootstrap_ci95(norm_ratio(z, zq))
        ip_m,   ip_lo,   ip_hi   = bootstrap_ci95(ip_error(z, z_pair, zq, zqp))
        rows.append({
            "bits": bits,
            "rMSE":       {"mean": rmse_m, "ci95_lo": rmse_lo, "ci95_hi": rmse_hi},
            "cosine":     {"mean": cos_m,  "ci95_lo": cos_lo,  "ci95_hi": cos_hi},
            "norm_ratio": {"mean": nr_m,   "ci95_lo": nr_lo,   "ci95_hi": nr_hi},
            "ip_error":   {"mean": ip_m,   "ci95_lo": ip_lo,   "ci95_hi": ip_hi},
        })
        print(f"  bits={bits:2d}  rMSE={rmse_m:.4f}  cos={cos_m:.4f}  "
              f"nr={nr_m:.4f}  ip={ip_m:.4f}")
    return rows


# ============================================================================
# Section 8 — Gate 0 identity check
# ============================================================================

def gate0_identity(z: torch.Tensor, dims: int, seed=0) -> dict:
    """At bits=16, Variant B with normalize=True should reproduce z to within
    fp16-ish tolerance. This validates the patching pipeline (rotate -> trivial
    quantize -> inverse -> rescale) introduces no surprise drift."""
    q = TurboQuantHonest(d=dims, bits=16, normalize=True, seed=seed)
    zq = q(z)
    rmse = rel_mse(z, zq).mean().item()
    cos  = cosine(z, zq).mean().item()
    nr   = norm_ratio(z, zq).mean().item()
    passed = (rmse < 1e-3) and (cos > 0.9999) and abs(nr - 1.0) < 1e-2
    print(f"[gate0] rMSE={rmse:.6f}  cos={cos:.6f}  nr={nr:.6f}  passed={passed}")
    return {"rMSE": rmse, "cosine": cos, "norm_ratio": nr, "passed": bool(passed)}


# ============================================================================
# Main
# ============================================================================

def pick_device():
    """Pick CUDA only if PyTorch supports the assigned GPU's compute capability.
    Kaggle assigns P100 (sm_60) by default; current Kaggle PyTorch images are
    often built without sm_60 support and would crash with
    `cudaErrorNoKernelImageForDevice`. Fall back to CPU gracefully.
    """
    if not torch.cuda.is_available():
        return "cpu", torch.float32
    try:
        major, minor = torch.cuda.get_device_capability(0)
        my_arch = f"sm_{major}{minor}"
        supported = torch.cuda.get_arch_list()  # e.g. ['sm_70', 'sm_75', ...]
        gpu_name = torch.cuda.get_device_name(0)
        if my_arch in supported:
            print(f"[env] using GPU: {gpu_name} ({my_arch} in {supported})")
            return "cuda", torch.float16
        print(f"[env] GPU {gpu_name} has {my_arch}, PyTorch supports {supported}. "
              f"Falling back to CPU (no crash but slower).")
        return "cpu", torch.float32
    except Exception as e:
        print(f"[env] CUDA probe failed ({e}); falling back to CPU")
        return "cpu", torch.float32


def main():
    device, dtype = pick_device()
    print(f"[env] torch={torch.__version__}  device={device}  dtype={dtype}")

    model, tokenizer, adapter = load_model_and_adapter(device, dtype)
    hidden_size = model.config.hidden_size

    hidden_raw, hidden_adapted, tok_counts = capture_adapter_output(
        model, tokenizer, adapter, device, dtype
    )
    # Free model memory; from now on we only work with captured tensors.
    del model, adapter, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print(f"\n[dist] raw last-hidden distribution:")
    raw_stats = describe_distribution(hidden_raw)
    for k, v in raw_stats.items():
        print(f"   {k:>18}: {v}")

    print(f"\n[dist] adapted output (the actual RecursiveLink signal):")
    adapted_stats = describe_distribution(hidden_adapted)
    for k, v in adapted_stats.items():
        print(f"   {k:>18}: {v}")

    print("\n[gate0] identity check on adapted output (bits=16):")
    gate0 = gate0_identity(hidden_adapted, hidden_size, seed=0)

    print("\n[sweep] Variant B on REAL adapter output:")
    sweep = run_quantizer_sweep(hidden_adapted, hidden_size, [8, 4, 3, 2], seed=0)

    print("\n[sweep:control] Variant B on RAW last-hidden (pre-adapter), for comparison:")
    sweep_raw = run_quantizer_sweep(hidden_raw, hidden_size, [8, 4, 3, 2], seed=0)

    out = {
        "phase": "0.A",
        "model_repo": SOLVER_REPO,
        "hidden_size": hidden_size,
        "n_prompts": len(PROMPTS),
        "n_tokens_total": int(sum(tok_counts)),
        "token_counts_per_prompt": tok_counts,
        "raw_hidden_stats": raw_stats,
        "adapted_output_stats": adapted_stats,
        "gate0_identity_on_adapted": gate0,
        "variant_b_sweep_on_adapted": sweep,
        "variant_b_sweep_on_raw_hidden": sweep_raw,
        "env": {
            "torch": torch.__version__,
            "device": device,
            "dtype": str(dtype),
            "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        },
    }

    out_path = "/kaggle/working/phase0_results.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
