#!/usr/bin/env python3
"""Tier-2 logit-fidelity + trajectory analysis across the local cells (no GPU).

Reuses the tested `compute_logit_metrics_pair` from the cloud analyzer
(`experiments/fidelity_sweep/analysis/analyze.py`) on the captured top-K logit NPZs
(`fidelity_logits.npz`) of each greedy REF/INT4 pair. For every cell it reports:
  - trajectory divergence rate within the captured window (fraction of primary
    solver sequences whose top-1 token differs REF vs INT4),
  - mean common-prefix length (positions strictly before the first mismatch),
  - matched-prefix per-step KL / JS (nats) between the REF and INT4 next-token
    distributions, over the union top-K support with a tail bucket,
  - per-call channel cosine (Tier 1) from the INT4 run's fidelity_call_stats.json.

Only the fixed primary solver batches are paired. Conditional answer-retry calls
are excluded because REF and INT4 may retry different problems. This remains the
matched-prefix variant (right-censored at the capture window and confounded by
divergence itself); a clean teacher-forced version needs a fresh GPU capture.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
ANALYZE = REPO / "experiments" / "fidelity_sweep" / "analysis" / "analyze.py"
FID = Path.home() / "lcc" / "fid_out"


def _run_dir(current: str, legacy: str | None = None) -> Path:
    """Return the current output directory, falling back to a legacy tag."""
    current_path = FID / current
    if current_path.exists() or legacy is None:
        return current_path
    return FID / legacy

spec = importlib.util.spec_from_file_location("analyze_cloud", str(ANALYZE))
az = importlib.util.module_from_spec(spec)
spec.loader.exec_module(az)

CELLS = [
    ("math500 / light",   _run_dir("math500_vb0_T3_n250_b2_auto", "vb0_T3_n250_b2_auto"),
                           _run_dir("math500_vb4_T3_n250_b2_auto", "vb4_T3_n250_b2_auto"), 2),
    ("mbppplus / light",  FID / "mbppplus_vb0_T3_n250_b2_auto",  FID / "mbppplus_vb4_T3_n250_b2_auto", 2),
    ("mbppplus / scaled", FID / "mbppplus_vb0_T3_n250_b1_auto",  FID / "mbppplus_vb4_T3_n250_b1_auto", 1),
    ("medqa / light",     FID / "medqa_vb0_T3_n250_b2_auto",     FID / "medqa_vb4_T3_n250_b2_auto", 2),
]


def channel_cosine(callstats: Path):
    try:
        d = json.loads(callstats.read_text())
        cos = [s["cosine"]["mean"] for s in d.get("per_adapter", [])
               if isinstance(s.get("cosine"), dict) and "mean" in s["cosine"]]
        return float(np.mean(cos)) if cos else None
    except Exception:
        return None


def ci95_mean(x, boot=10000, seed=42):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bm = x[rng.integers(0, x.size, size=(boot, x.size))].mean(axis=1)
    return (float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)))


print(f"{'cell':18} {'div.rate':>8} {'prefix':>8} {'KL nats (95% CI)':>22} {'JS':>7} {'chan_cos':>9} {'n':>4} {'window':>6}")
print("-" * 102)
for label, refd, intd, batch_size in CELLS:
    rnpz, inpz = refd / "fidelity_logits.npz", intd / "fidelity_logits.npz"
    if not (rnpz.is_file() and inpz.is_file()):
        print(f"{label:18}  (NPZ missing: {rnpz.is_file()=}, {inpz.is_file()=})")
        continue
    n_samples = 250
    primary_batches = (n_samples + batch_size - 1) // batch_size
    m = az.compute_logit_metrics_pair(rnpz, inpz, max_batches=primary_batches)
    kl = m["per_problem_kl"]
    klm = float(kl.mean()) if kl.size else float("nan")
    lo, hi = ci95_mean(kl)
    jsm = float(m["per_problem_js"].mean()) if m["per_problem_js"].size else float("nan")
    cos = channel_cosine(intd / "fidelity_call_stats.json")
    cos_s = f"{cos:.4f}" if cos is not None else "n/a"
    print(f"{label:18} {m['divergence_rate']*100:7.1f}% {m['mean_prefix_len']:8.1f} "
          f"{klm:7.3f} [{lo:.3f},{hi:.3f}] {jsm:7.3f} {cos_s:>9} "
          f"{m['n_items']:4d} {m['max_positions_seen']:6d}")
    print(
        f"  primary batches={m['n_batches_paired']}; excluded retries "
        f"REF={m['n_retry_batches_ref_excluded']} INT={m['n_retry_batches_int_excluded']}"
    )
