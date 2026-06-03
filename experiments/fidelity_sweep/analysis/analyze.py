"""Post-hoc analysis of fidelity_sweep kernel outputs.

Consumes the per-kernel JSON outputs (fidelity_vbN_TM_n50_b4_float32.json) and
their associated per-call stats dumps, computes paired statistics across T
values, emits:

  results.md      — per-T table (acc_REF, acc_INT4, Δacc + CI, TOST verdict,
                    mean cosine, mean rel_l2, n_calls)
  figures/*.png   — matplotlib plots (one per metric vs T)
  raw.npz         — concatenated per-call arrays for further analysis

Usage:
  .venv/bin/python experiments/fidelity_sweep/analysis/analyze.py \
      --inputs path/to/fid_kernels_dir \
      --out experiments/fidelity_sweep/analysis/results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

# Make `src` importable when this script is run directly as
# `python experiments/fidelity_sweep/analysis/analyze.py` (Python only puts the
# script's own dir on sys.path, not the project root).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "lscr_mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from src.metrics.bootstrap import (
    paired_bootstrap_delta,
    paired_tost_binary,
    bootstrap_ci_mean,
)
# Logit pairing is done in numpy probability-space (see dist_over_union); the
# torch per_position_* helpers in src.metrics.logit_metrics remain available for
# full-vector use but are not needed here.


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_runs(input_dir: Path, *, allow_overwrite: bool = False) -> Dict[Tuple[int, int], dict]:
    """Returns dict keyed by (bits, T) → kernel-output JSON.

    Searches ``input_dir`` recursively for ``fidelity_vbN_TM_*.json`` so the
    recommended download layout (one ``vb{bits}_T{T}/`` subdir per kernel, which
    keeps the same-named ``fidelity_logits.npz`` from colliding) works directly.

    By default, duplicate (bits, T) runs are a hard error. This prevents
    accidental mixing of n=50, n=250, inner/outer, or REF-vs-REF control outputs
    in a single analysis directory.
    """
    runs: Dict[Tuple[int, int], dict] = {}
    for jf in sorted(input_dir.rglob("fidelity_vb*_T*.json")):
        try:
            data = json.loads(jf.read_text())
        except Exception as exc:
            print(f"  skip (parse error): {jf.name} — {exc}")
            continue
        cfg = data.get("config", {})
        bits = int(cfg.get("variant_b_bits", 0))
        T = int(cfg.get("num_recursive_rounds", 3))
        key = (bits, T)
        if key in runs and not allow_overwrite:
            prev = runs[key].get("_json_path", runs[key].get("_dir", "unknown"))
            raise ValueError(
                "duplicate fidelity run for "
                f"bits={bits}, T={T}: {prev} and {jf}. "
                "Analyze one comparison at a time, or pass "
                "--allow-duplicate-overwrite if you intentionally want the "
                "last sorted file to win."
            )
        data["_dir"] = str(jf.parent)  # so the NPZ can be found next to the JSON
        data["_json_path"] = str(jf)
        runs[key] = data
        print(f"  loaded: bits={bits} T={T} acc={data.get('final_accuracy')}%"
              f"  n_call_records={len((data.get('fidelity_summary') or {}).get('per_call', []))}")
    return runs


def verify_npz(path: Path) -> None:
    """Raise a clear error if an NPZ/ZIP artifact is corrupt."""
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"corrupt NPZ artifact: {path} ({exc})") from exc
    if bad is not None:
        raise RuntimeError(f"corrupt NPZ artifact: {path} (first bad member: {bad})")


def _find_logits_npz(run: dict, logit_dir: "Path | None", bits: int, T: int) -> "Path | None":
    """Locate a run's fidelity_logits.npz: prefer the dir next to its JSON, then
    fall back to the --logit-dir ``vb{bits}_T{T}/`` convention. Returns None if
    neither exists.
    """
    candidates: List[Path] = []
    if run.get("_dir"):
        candidates.append(Path(run["_dir"]) / "fidelity_logits.npz")
    if logit_dir is not None:
        candidates.append(logit_dir / f"vb{bits}_T{T}/fidelity_logits.npz")
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Tier 2 — paired per-problem accuracy via JSON's per_problem list
# ---------------------------------------------------------------------------


def _correct_flag(rec: dict):
    """Boolean correctness from a per-problem record, robust to the two
    upstream JSONL schemas (flat ``correct`` vs nested ``rollouts[0].correct``).
    Returns the bool or None when it cannot be determined.
    """
    c = rec.get("correct")
    if isinstance(c, bool):
        return c
    rollouts = rec.get("rollouts")
    if isinstance(rollouts, list) and rollouts:
        cc = rollouts[0].get("correct")
        if isinstance(cc, bool):
            return cc
    pak = rec.get("pass_at_k_correct")
    if isinstance(pak, bool):
        return pak
    return None


def paired_correctness(ref: dict, intq: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Align per-problem `correct` flags by sample_idx. Returns matched arrays
    suitable for paired_bootstrap_delta + paired_tost_binary.

    Problems where either side has an undetermined correctness flag are DROPPED
    (not coerced to 0). This is a safety property: a JSONL-parsing failure then
    yields an EMPTY paired array -> aggregate_per_T reports NO_PAIRED_DATA,
    instead of silently producing all-zeros that masquerade as an EQUIVALENT
    verdict.
    """
    by_idx_ref = {r["sample_idx"]: r for r in ref.get("per_problem", []) if r.get("sample_idx") is not None}
    by_idx_int = {r["sample_idx"]: r for r in intq.get("per_problem", []) if r.get("sample_idx") is not None}
    common = sorted(set(by_idx_ref) & set(by_idx_int))
    ref_list: List[int] = []
    int_list: List[int] = []
    n_dropped = 0
    for i in common:
        cr = _correct_flag(by_idx_ref[i])
        ci = _correct_flag(by_idx_int[i])
        if cr is None or ci is None:
            n_dropped += 1
            continue
        ref_list.append(1 if cr else 0)
        int_list.append(1 if ci else 0)
    if n_dropped:
        print(f"  [paired_correctness] dropped {n_dropped}/{len(common)} problems "
              f"with undetermined correctness")
    return np.array(ref_list, dtype=np.int32), np.array(int_list, dtype=np.int32)


# ---------------------------------------------------------------------------
# Tier 2 — paired top-K logit metrics with union-support approximation
# ---------------------------------------------------------------------------


def dist_over_union(
    vals: np.ndarray, idxs: np.ndarray, full_lse: float, union: np.ndarray
) -> np.ndarray:
    """Proper probability vector over ``union`` tokens + 1 residual-tail bucket.

    Works in PROBABILITY space (not logit space) using ``full_lse`` — the
    log-sum-exp over the FULL vocab, which the kernel captured and which is always
    finite — for correct normalization: ``p_token = exp(logit_token - full_lse)``.

    A token in ``union`` that is OUTSIDE this run's top-K is, by definition, no
    larger than the run's smallest kept logit; we approximate its prob by that
    boundary prob ``exp(min_kept - full_lse)`` (a tight upper bound, not the −inf
    that produced NaNs / blow-ups in the earlier logit-space construction). The
    final bucket holds the residual mass ``1 − Σ top-K probs``. The vector is
    renormalized to sum to 1.
    """
    full_lse = float(full_lse)
    vals = np.asarray(vals, dtype=np.float64)
    logit = {int(i): float(v) for i, v in zip(idxs, vals)}
    boundary = float(vals.min()) if vals.size else full_lse
    probs = np.empty(union.size + 1, dtype=np.float64)
    for j, u in enumerate(union):
        lv = logit.get(int(u), boundary)
        probs[j] = np.exp(lv - full_lse)
    topk_mass = float(np.exp(vals - full_lse).sum())
    probs[-1] = max(0.0, 1.0 - topk_mass)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs[:] = 1.0 / probs.size  # degenerate position -> uniform (KL=0 vs uniform)
    else:
        probs /= s
    return probs


def kl_probs(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p ‖ q) on probability vectors (nats). Skips p==0 terms; floors q."""
    p = np.clip(p, 0.0, None)
    q = np.clip(q, eps, None)
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))


def js_probs(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (nats, symmetric, ≤ ln 2) on prob vectors."""
    m = 0.5 * (p + q)
    return 0.5 * kl_probs(p, m, eps) + 0.5 * kl_probs(q, m, eps)


def compute_logit_metrics_pair(
    ref_npz: Path, int_npz: Path, *, max_positions_per_batch: int = 256
) -> dict:
    """Load both kernels' top-K logit dumps, align per (batch, step, item),
    compute MSE/KL/JS over the union-support approximation, return aggregated
    per-problem means.

    Memory: union size is ≤ 2K = 1024, processed in fp64 → ~kB per position.
    """
    verify_npz(ref_npz)
    verify_npz(int_npz)
    ref_z = np.load(ref_npz, allow_pickle=True)
    int_z = np.load(int_npz, allow_pickle=True)
    n_ref = int(ref_z["n_batches"]) if "n_batches" in ref_z else 0
    n_int = int(int_z["n_batches"]) if "n_batches" in int_z else 0
    n_batches = min(n_ref, n_int)
    print(f"  pairing {n_batches} logit batches (REF={n_ref}, INT={n_int})")

    per_problem_mse: List[float] = []
    per_problem_kl: List[float] = []
    per_problem_js: List[float] = []
    matched_lens: List[int] = []   # aligned positions before the greedy paths diverge
    n_items = 0
    n_diverged = 0

    for b in range(n_batches):
        try:
            r_vals = ref_z[f"batch{b}_vals"]
            r_idxs = ref_z[f"batch{b}_idxs"]
            r_fll = ref_z[f"batch{b}_full_lse"]
            r_tl = ref_z[f"batch{b}_tail_log"]
            i_vals = int_z[f"batch{b}_vals"]
            i_idxs = int_z[f"batch{b}_idxs"]
            i_fll = int_z[f"batch{b}_full_lse"]
            i_tl = int_z[f"batch{b}_tail_log"]
        except KeyError as e:
            print(f"    skip batch {b}: missing field {e}")
            continue
        # Shapes (T, B, K). Truncate to shortest T and shared B.
        T = min(r_vals.shape[0], i_vals.shape[0], max_positions_per_batch)
        B = min(r_vals.shape[1], i_vals.shape[1])
        for b_i in range(B):
            n_items += 1
            mse_pos: List[float] = []
            kl_pos: List[float] = []
            js_pos: List[float] = []
            diverged = False
            for t in range(T):
                union = np.union1d(r_idxs[t, b_i], i_idxs[t, b_i])
                pr = dist_over_union(r_vals[t, b_i], r_idxs[t, b_i], r_fll[t, b_i], union)
                pi = dist_over_union(i_vals[t, b_i], i_idxs[t, b_i], i_fll[t, b_i], union)
                kl = kl_probs(pr, pi)
                js = js_probs(pr, pi)
                mse = float(np.mean((pr - pi) ** 2))  # MSE on PROBABILITIES (bounded)
                if np.isfinite(kl) and np.isfinite(js) and np.isfinite(mse):
                    kl_pos.append(kl)
                    js_pos.append(js)
                    mse_pos.append(mse)
                # Greedy token = top-1. Once REF and INT4 pick different tokens, the
                # generated prefixes (and thus contexts) diverge and any later
                # positional comparison is apples-to-oranges. This position itself is
                # still a valid same-context comparison, so count it, then stop.
                if int(r_idxs[t, b_i][0]) != int(i_idxs[t, b_i][0]):
                    diverged = True
                    break
            if mse_pos:
                per_problem_mse.append(float(np.mean(mse_pos)))
                per_problem_kl.append(float(np.mean(kl_pos)))
                per_problem_js.append(float(np.mean(js_pos)))
                matched_lens.append(len(mse_pos))
            if diverged:
                n_diverged += 1
    return {
        "per_problem_mse": np.array(per_problem_mse, dtype=np.float64),
        "per_problem_kl": np.array(per_problem_kl, dtype=np.float64),
        "per_problem_js": np.array(per_problem_js, dtype=np.float64),
        "mean_matched_len": float(np.mean(matched_lens)) if matched_lens else 0.0,
        "divergence_rate": (n_diverged / n_items) if n_items else 0.0,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_per_T(
    runs: Dict[Tuple[int, int], dict],
    *,
    logit_dir: Path | None = None,
    eps_pp: float = 2.0,
) -> List[dict]:
    """One row per T value. Computes paired Δaccuracy + per-T continuous
    metric means with bootstrap CIs.

    Returns sorted by T.
    """
    Ts = sorted({T for (_, T) in runs.keys()})
    rows = []
    for T in Ts:
        ref = runs.get((0, T))
        # Pick the highest-bits run as the INT4 analog (typically 4).
        int_candidates = [(b, T) for (b, t) in runs.keys() if t == T and b > 0]
        if not int_candidates:
            print(f"  T={T}: no INT4 run, skipping")
            continue
        int_bits = max(b for (b, _) in int_candidates)
        intq = runs[(int_bits, T)]

        if ref is None:
            print(f"  T={T}: no REF run, skipping")
            continue

        # === Paired per-problem accuracy + bootstrap + TOST (Tier 2) ===
        ref_arr, int_arr = paired_correctness(ref, intq)
        if ref_arr.size > 0:
            d_obs, ci_lo, ci_hi, _ = paired_bootstrap_delta(
                ref_arr, int_arr, n_resamples=10_000, seed=42
            )
            tost = paired_tost_binary(ref_arr, int_arr, eps=eps_pp)
            tost_verdict = tost.verdict
            tost_p = tost.p_tost
        else:
            d_obs = ci_lo = ci_hi = None
            tost_verdict = "NO_PAIRED_DATA"
            tost_p = None

        acc_ref = ref.get("final_accuracy")
        acc_int = intq.get("final_accuracy")

        # === Per-call continuous metrics from QuantStats ===
        # The fidelity_summary["per_call"] list has one entry per patched
        # adapter; each entry contains a list of per-call means (cosine_means,
        # rmse_means, norm_ratio_means).
        cos_means: List[float] = []
        rmse_means: List[float] = []
        n_calls = 0
        n_tokens_total = 0
        for ad in (intq.get("fidelity_summary") or {}).get("per_call", []):
            cos_means.extend(ad.get("cosine_means", []))
            rmse_means.extend(ad.get("rmse_means", []))
            n_calls += int(ad.get("n_calls", 0))
            n_tokens_total += int(ad.get("n_tokens_total", 0))

        cos_arr = np.array(cos_means, dtype=np.float64)
        rmse_arr = np.array(rmse_means, dtype=np.float64)
        rel_l2_arr = np.sqrt(np.clip(rmse_arr, 0.0, None))

        if cos_arr.size > 0:
            mean_cos, cos_lo, cos_hi = bootstrap_ci_mean(cos_arr, n_resamples=10_000, seed=42)
            mean_rel_l2, rl_lo, rl_hi = bootstrap_ci_mean(rel_l2_arr, n_resamples=10_000, seed=42)
            mean_rmse, _, _ = bootstrap_ci_mean(rmse_arr, n_resamples=10_000, seed=42)
        else:
            mean_cos = cos_lo = cos_hi = float("nan")
            mean_rel_l2 = rl_lo = rl_hi = float("nan")
            mean_rmse = float("nan")

        # === Tier 2: logit-level KL/JS/MSE if NPZ dumps are available ===
        logit_metrics = None
        ref_npz = _find_logits_npz(ref, logit_dir, 0, T)
        int_npz = _find_logits_npz(intq, logit_dir, int_bits, T)
        if ref_npz is not None and int_npz is not None:
            print(f"  T={T}: computing paired logit metrics from {ref_npz.parent.name}/ + {int_npz.parent.name}/")
            lm = compute_logit_metrics_pair(ref_npz, int_npz)
            if lm["per_problem_kl"].size > 0:
                m_kl, lo_kl, hi_kl = bootstrap_ci_mean(lm["per_problem_kl"], n_resamples=10_000, seed=42)
                m_js, lo_js, hi_js = bootstrap_ci_mean(lm["per_problem_js"], n_resamples=10_000, seed=42)
                m_mse, lo_mse, hi_mse = bootstrap_ci_mean(lm["per_problem_mse"], n_resamples=10_000, seed=42)
                logit_metrics = {
                    "kl_mean": m_kl, "kl_ci": (lo_kl, hi_kl),
                    "js_mean": m_js, "js_ci": (lo_js, hi_js),
                    "mse_mean": m_mse, "mse_ci": (lo_mse, hi_mse),
                    "n_problems": int(lm["per_problem_kl"].size),
                    "mean_matched_len": lm["mean_matched_len"],
                    "divergence_rate": lm["divergence_rate"],
                }

        rows.append({
            "T": T,
            "bits_int": int_bits,
            "acc_ref": acc_ref,
            "acc_int": acc_int,
            "delta_acc_obs": d_obs,
            "delta_acc_ci": (ci_lo, ci_hi),
            "tost_verdict": tost_verdict,
            "tost_p": tost_p,
            "tost_eps": eps_pp,
            "n_paired_problems": int(ref_arr.size),
            "n_calls": n_calls,
            "n_tokens_total": n_tokens_total,
            "mean_cosine": mean_cos,
            "cosine_ci": (cos_lo, cos_hi),
            "mean_rel_l2": mean_rel_l2,
            "rel_l2_ci": (rl_lo, rl_hi),
            "mean_rmse": mean_rmse,
            "logit_metrics": logit_metrics,
        })
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_results_md(rows: List[dict], out_path: Path) -> None:
    lines = []
    lines.append("# Fidelity sweep — results")
    lines.append("")
    lines.append("Paired REF (bits=0) vs INT4 (bits=4) runs at varying channel-")
    lines.append("traversal counts T = num_recursive_rounds. Greedy decoding, fp32.")
    lines.append("All CIs are 95% bootstrap (≥10k resamples).")
    lines.append("")
    lines.append("## Table 1 — Accuracy with paired bootstrap + TOST")
    lines.append("")
    lines.append("Per-problem outcomes paired by `sample_idx`. ε=2pp pre-specified.")
    lines.append("TOST `NOT_EQUIVALENT` means equivalence was not established at this margin; it is not evidence of harm by itself.")
    lines.append("")
    lines.append("| T | acc_REF | acc_INT | Δacc (95% CI) | n_paired | TOST verdict (p) |")
    lines.append("|---|---:|---:|---:|---:|:---:|")
    for r in rows:
        if r["delta_acc_obs"] is not None:
            d_ci = r["delta_acc_ci"]
            delta_cell = f"{r['delta_acc_obs']:+.2f}pp [{d_ci[0]:+.2f}, {d_ci[1]:+.2f}]"
        else:
            delta_cell = "n/a"
        tost_cell = f"{r['tost_verdict']}"
        if r["tost_p"] is not None:
            tost_cell += f" (p={r['tost_p']:.3f})"
        lines.append(
            f"| {r['T']} | {r['acc_ref']}% | {r['acc_int']}% | {delta_cell} | "
            f"{r['n_paired_problems']} | {tost_cell} |"
        )
    lines.append("")
    lines.append("## Table 2 — Channel fidelity (per-adapter-call, INT4 round-trip)")
    lines.append("")
    lines.append("Recorded at every Adapter / CrossModelAdapter forward in the INT4 run.")
    lines.append("Cos ≈ 1 + rel L2 ≈ 0 means the quantizer preserved the post-LN output bit-for-bit AT THAT CALL.")
    lines.append("")
    lines.append("| T | n_calls | mean cos | cos 95% CI | mean rel L2 | rel L2 95% CI |")
    lines.append("|---|---:|---:|:---:|---:|:---:|")
    for r in rows:
        cos_ci = r["cosine_ci"]; rl_ci = r["rel_l2_ci"]
        lines.append(
            f"| {r['T']} | {r['n_calls']} | "
            f"{r['mean_cosine']:.4f} | "
            f"[{cos_ci[0]:.4f}, {cos_ci[1]:.4f}] | "
            f"{r['mean_rel_l2']:.4f} | "
            f"[{rl_ci[0]:.4f}, {rl_ci[1]:.4f}] |"
        )
    lines.append("")
    lines.append("## Table 3 — Egress distributional fidelity (matched-prefix, per-step)")
    lines.append("")
    lines.append("KL(p_REF ‖ p_INT) / JS / MSE on the next-token **probability** distributions")
    lines.append("(normalized via the captured full-vocab log-sum-exp), over the union of the")
    lines.append("two top-K supports. Under greedy free generation the two runs' sequences can")
    lines.append("diverge; once they pick different tokens the contexts differ and a positional")
    lines.append("comparison is meaningless. So the metric is computed **only over the matched")
    lines.append("prefix** (positions up to and including the first token mismatch). `div_rate`")
    lines.append("is the fraction of sequences whose greedy path diverged within the window;")
    lines.append("`match_len` is the mean number of aligned positions measured.")
    lines.append("")
    lines.append("| T | mean KL (nats) | KL 95% CI | mean JS | JS 95% CI | prob-MSE | div_rate | match_len |")
    lines.append("|---|---:|:---:|---:|:---:|---:|---:|---:|")
    for r in rows:
        lm = r.get("logit_metrics")
        if lm is None:
            lines.append(f"| {r['T']} | n/a | — | n/a | — | n/a | — | — |")
            continue
        lines.append(
            f"| {r['T']} | "
            f"{lm['kl_mean']:.4f} | [{lm['kl_ci'][0]:.4f}, {lm['kl_ci'][1]:.4f}] | "
            f"{lm['js_mean']:.4f} | [{lm['js_ci'][0]:.4f}, {lm['js_ci'][1]:.4f}] | "
            f"{lm['mse_mean']:.2e} | "
            f"{lm.get('divergence_rate', float('nan')):.2f} | "
            f"{lm.get('mean_matched_len', float('nan')):.1f} |"
        )
    lines.append("")
    lines.append("## Reading the verdict")
    lines.append("")
    lines.append("- **Table 2 (channel) cos≈1, rel L2≈const across T** → the 4-bit round-trip")
    lines.append("  preserves the inter-agent vector geometry, and the per-call distortion does")
    lines.append("  NOT grow with channel-traversal depth.")
    lines.append("- **Table 3 matched-prefix KL is small** → where REF and INT4 share context the")
    lines.append("  quantizer barely perturbs the next-token distribution (near-lossless per step).")
    lines.append("- **KL does not explode with T** → no catastrophic depth-amplification of the")
    lines.append("  per-step drift; `div_rate` quantifies how often a tiny perturbation flips a")
    lines.append("  greedy token (a trajectory effect, separate from per-step fidelity).")
    lines.append("- **Table 1 TOST** needs adequate n to return EQUIVALENT within ±2pp; at small n")
    lines.append("  it will read INCONCLUSIVE/NOT_EQUIVALENT (wide CI), which is *underpowered*, not")
    lines.append("  evidence of harm. Use n≈250 for the formal equivalence claim.")
    out_path.write_text("\n".join(lines))


def plot_metric_vs_T(rows: List[dict], key: str, ylabel: str, out_path: Path) -> None:
    Ts = [r["T"] for r in rows]
    vals = [r[key] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ts, vals, marker="o", linewidth=2, color="#3498db")
    ax.set_xticks(Ts)
    ax.set_xlabel("T = num_recursive_rounds (channel-traversal count)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs T — Variant B in-loop fidelity")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_acc_vs_T(rows: List[dict], out_path: Path) -> None:
    Ts = [r["T"] for r in rows]
    accs_ref = [r["acc_ref"] for r in rows]
    accs_int = [r["acc_int"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ts, accs_ref, marker="o", label="REF (bits=0)", color="#2c3e50", linewidth=2)
    ax.plot(Ts, accs_int, marker="s", label="INT4 (Variant B)", color="#3498db", linewidth=2)
    ax.set_xticks(Ts)
    ax.set_xlabel("T = num_recursive_rounds")
    ax.set_ylabel("math500 accuracy (%)")
    ax.set_title("Accuracy vs T — REF vs Variant B INT4")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_raw_npz(rows: List[dict], out_path: Path) -> None:
    arrs = {
        "T": np.array([r["T"] for r in rows]),
        "bits_int": np.array([r["bits_int"] for r in rows]),
        "acc_ref": np.array([r["acc_ref"] for r in rows], dtype=np.float64),
        "acc_int": np.array([r["acc_int"] for r in rows], dtype=np.float64),
        "mean_cosine": np.array([r["mean_cosine"] for r in rows], dtype=np.float64),
        "mean_rel_l2": np.array([r["mean_rel_l2"] for r in rows], dtype=np.float64),
        "mean_rmse": np.array([r["mean_rmse"] for r in rows], dtype=np.float64),
        "n_calls": np.array([r["n_calls"] for r in rows]),
        "n_tokens_total": np.array([r["n_tokens_total"] for r in rows]),
    }
    np.savez(out_path, **arrs)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def plot_logit_metric_vs_T(rows: List[dict], key: str, ylabel: str, out_path: Path) -> None:
    have = [r for r in rows if r.get("logit_metrics") is not None]
    if not have:
        return
    Ts = [r["T"] for r in have]
    vals = [r["logit_metrics"][f"{key}_mean"] for r in have]
    ci_lo = [r["logit_metrics"][f"{key}_ci"][0] for r in have]
    ci_hi = [r["logit_metrics"][f"{key}_ci"][1] for r in have]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ts, vals, marker="o", linewidth=2, color="#e74c3c")
    ax.fill_between(Ts, ci_lo, ci_hi, color="#e74c3c", alpha=0.2, label="95% bootstrap CI")
    ax.set_xticks(Ts)
    ax.set_xlabel("T = num_recursive_rounds")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs T — Tier 2 distributional drift")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True,
                   help="directory containing fidelity_vbN_TM_*.json kernel outputs")
    p.add_argument("--logit-dir", type=Path, default=None,
                   help="directory containing per-kernel subdirs with fidelity_logits.npz "
                        "(naming: vb{bits}_T{T}/fidelity_logits.npz). "
                        "If omitted, Tier 2 logit metrics are skipped.")
    p.add_argument("--eps-pp", type=float, default=2.0,
                   help="TOST equivalence margin in percentage points (default 2)")
    p.add_argument("--out", type=Path,
                   default=Path("experiments/fidelity_sweep/analysis/results"))
    p.add_argument("--allow-duplicate-overwrite", action="store_true",
                   help="allow later sorted JSON files to overwrite earlier runs "
                        "with the same (bits, T). Default is to fail loudly.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)

    print(f"Loading runs from {args.inputs} ...")
    runs = load_runs(args.inputs, allow_overwrite=args.allow_duplicate_overwrite)
    if not runs:
        print("✗ no runs found")
        return 1
    print(f"  → {len(runs)} runs loaded")

    print(f"\nAggregating per T (eps={args.eps_pp}pp, logit_dir={args.logit_dir}) ...")
    rows = aggregate_per_T(runs, logit_dir=args.logit_dir, eps_pp=args.eps_pp)
    print(f"  → {len(rows)} (T, REF, INT4) triples assembled")

    print(f"\nWriting {args.out}/results.md ...")
    write_results_md(rows, args.out / "results.md")

    print(f"Writing plots ...")
    plot_metric_vs_T(rows, "mean_cosine", "Mean cosine (channel fidelity)",
                     args.out / "figures" / "fidelity_cosine_vs_T.png")
    plot_metric_vs_T(rows, "mean_rel_l2", "Mean relative L2 (channel error)",
                     args.out / "figures" / "fidelity_rel_l2_vs_T.png")
    plot_acc_vs_T(rows, args.out / "figures" / "fidelity_accuracy_vs_T.png")
    # Tier 2 plots
    plot_logit_metric_vs_T(rows, "kl", "Mean KL(p_REF ‖ p_INT)", args.out / "figures" / "fidelity_kl_vs_T.png")
    plot_logit_metric_vs_T(rows, "js", "Mean Jensen-Shannon", args.out / "figures" / "fidelity_js_vs_T.png")
    plot_logit_metric_vs_T(rows, "mse", "Mean logit MSE", args.out / "figures" / "fidelity_mse_vs_T.png")

    print(f"Writing raw npz ...")
    save_raw_npz(rows, args.out / "raw.npz")

    print(f"\n✓ done. results in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
