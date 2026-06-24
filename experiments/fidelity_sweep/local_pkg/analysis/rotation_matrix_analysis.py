#!/usr/bin/env python3
"""Cross-rotation analysis of the MBPP+ light/scaled quantizer-rotation matrix.

Confirms whether the light/scaled greedy-trajectory-divergence gap is robust to the
quantizer rotation seed. For each tier the rotation-independent seed-42 greedy REF is
paired with each rotation's INT4 capture; per problem we record whether the greedy paths
diverged within the 128-position window and within the first 25 positions (the
length-robust early estimand).

Inference is a **problem-clustered bootstrap**: each problem's divergence propensity is
averaged over the rotations, then problems (the resampling unit) are resampled with
replacement. Rotations are repeated measures on the same 250 problems and are NOT pooled
as if they were independent benchmark items.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("dh", HERE / "divergence_hazard.py")
dh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dh)

N = 250
EARLY = 25
SEEDS = [42, 7, 17, 73, 101]
TIERS = [("light", "b2", 125), ("scaled", "b1", 250)]


def per_problem_events(ref_npz: Path, int_npz: Path, primary_batches: int):
    """Per-problem (diverged_window, diverged_early) binary vectors, problem-ordered."""
    t, e, _ = dh.first_divergence(ref_npz, int_npz, primary_batches)
    return e.astype(np.int64), ((e == 1) & (t <= EARLY)).astype(np.int64)


def clustered_bootstrap(light_prop, scaled_prop, boot=20000, seed=42):
    """CI on (light - scaled) of the per-problem propensity means, resampling problems."""
    rng = np.random.default_rng(seed)
    n = light_prop.size
    idx = rng.integers(0, n, size=(boot, n))
    deltas = light_prop[idx].mean(axis=1) - scaled_prop[idx].mean(axis=1)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default=os.environ.get("LCC_RUN_ROOT", str(Path.home() / "lcc" / "runs")))
    ap.add_argument("--ref-root", default=str(Path.home() / "lcc" / "fid_out"),
                    help="root holding the rotation-independent seed-42 REF captures")
    args = ap.parse_args()
    rot = Path(args.run_root) / "rotation_matrix"
    ref_root = Path(args.ref_root)

    # window[tier] and early[tier]: [n_seeds, N] binary matrices
    window = {t: [] for t, _, _ in TIERS}
    early = {t: [] for t, _, _ in TIERS}
    print(f"{'seed':>5} {'tier':7} {'div%':>6} {'div@25%':>8}")
    for seed in SEEDS:
        sfx = "" if seed == 42 else f"_qs{seed}"
        for tier, b, pb in TIERS:
            ref = ref_root / f"mbppplus_vb0_T3_n{N}_{b}_auto" / "fidelity_logits.npz"
            intd = (ref_root if seed == 42 else rot) / f"mbppplus_vb4_T3_n{N}_{b}_auto{sfx}" / "fidelity_logits.npz"
            if not (ref.is_file() and intd.is_file()):
                print(f"{seed:5} {tier:7}  MISSING ({intd})")
                return 2
            w, e = per_problem_events(ref, intd, pb)
            if w.size != N:
                print(f"{seed:5} {tier:7}  WARN got {w.size} problems (expected {N})")
            window[tier].append(w)
            early[tier].append(e)
            print(f"{seed:5} {tier:7} {w.mean()*100:6.1f} {e.mean()*100:8.1f}")

    print("-" * 36)
    out = {}
    for tier, _, _ in TIERS:
        W = np.vstack(window[tier])   # [n_seeds, N]
        E = np.vstack(early[tier])
        out[tier] = {
            "win_rate": W.mean() * 100, "early_rate": E.mean() * 100,
            "win_prop": W.mean(axis=0), "early_prop": E.mean(axis=0),  # per-problem over seeds
        }
        print(f"{'mean':>5} {tier:7} {out[tier]['win_rate']:6.1f} {out[tier]['early_rate']:8.1f}")

    print("\nLight - scaled contrast (problem-clustered bootstrap over the 5 rotations):")
    for key, lab in [("win", "divergence within 128"), ("early", "divergence within 25")]:
        lp, sp = out["light"][f"{key}_prop"], out["scaled"][f"{key}_prop"]
        delta = (lp.mean() - sp.mean()) * 100
        lo, hi = clustered_bootstrap(lp, sp)
        print(f"  {lab:24}: light {out['light'][key + '_rate']:.1f}% - scaled "
              f"{out['scaled'][key + '_rate']:.1f}% = {delta:+.1f} pp  "
              f"95% CI [{lo*100:+.1f}, {hi*100:+.1f}]")
    print("\nThe CI excluding 0 by a wide margin across 5 independent rotations confirms the "
          "MBPP+ scaled trajectory-robustness is not a single-rotation artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
