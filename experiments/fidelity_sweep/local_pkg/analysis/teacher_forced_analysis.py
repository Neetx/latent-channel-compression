#!/usr/bin/env python3
"""Teacher-forced (aligned) mechanism analysis for MBPP+.

For each tier it pairs the full-precision REF capture (the teacher-forced REF, identical to the
free-running REF by gate G0) with the INT4 teacher-forced capture and, at every aligned position,
computes: the arg-max flip (REF top-1 != INT4 top-1), the REF top1-top2 logit margin, the
magnitude of the per-position perturbation (|top-1 logit diff|), and the rank of the REF token
under INT4. It then reports the flip-vs-margin curve (the margin-tipping signature) and an
Oaxaca-style decomposition of the light-minus-scaled flip-rate gap into a margin-distribution
component (scaled decides with larger margins) and a per-margin/sensitivity component (the
channel perturbs the larger model's logits less), with a problem-clustered bootstrap.

Captures are b=1 (gate G0 holds only at b=1), so each NPZ batch == one problem == the bootstrap
cluster. Run: python teacher_forced_analysis.py [--tf-root DIR] [--dataset DS]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

LOCAL_PKG = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(LOCAL_PKG))
from fidelity_local import build_config_tag  # noqa: E402

TIERS = {"light": "sequential_light", "scaled": "sequential_scaled"}
MARGIN_EDGES = [0, 1, 2, 4, 8, 16, 1e9]
MARGIN_LABELS = ["<1", "1-2", "2-4", "4-8", "8-16", ">16"]


def resolve_ref_dir(roots, style, dataset, n):
    tag = build_config_tag(dataset, 0, 3, n, 1, "auto")
    for root in roots:
        for d in (root / f"{style}_{dataset}" / tag, root / tag):
            if (d / "fidelity_logits.npz").is_file():
                return d
    return None


def gather(ref_npz, int_npz):
    """Per-position arrays + the problem (cluster) id for each position."""
    r = np.load(ref_npz); t = np.load(int_npz)
    nb = min(int(r["n_batches"]), int(t["n_batches"]))
    prob, margin, flip, ldiff, rank = [], [], [], [], []
    for b in range(nb):
        ri, ii = r[f"batch{b}_idxs"], t[f"batch{b}_idxs"]
        rv, iv = r[f"batch{b}_vals"], t[f"batch{b}_vals"]
        rl, il = r[f"batch{b}_full_lse"], t[f"batch{b}_full_lse"]
        T = min(ri.shape[0], ii.shape[0]); B = min(ri.shape[1], ii.shape[1])
        K = ii.shape[2]
        for e in range(B):
            for p in range(T):
                if rl[p, e] == 0 or il[p, e] == 0:
                    continue
                prob.append(b * B + e)
                margin.append(float(rv[p, e, 0] - rv[p, e, 1]))
                rt = int(ri[p, e, 0])
                flip.append(0 if rt == int(ii[p, e, 0]) else 1)
                ldiff.append(abs(float(rv[p, e, 0]) - float(iv[p, e, 0])))
                w = np.where(ii[p, e] == rt)[0]
                rank.append(int(w[0]) + 1 if w.size else K + 1)
    return (np.array(prob), np.array(margin), np.array(flip, float),
            np.array(ldiff), np.array(rank))


def bin_idx(margin):
    return np.digitize(margin, MARGIN_EDGES[1:-1])  # 0..5


def curve(margin, flip):
    bi = bin_idx(margin)
    return [(MARGIN_LABELS[k], float(flip[bi == k].mean()) if (bi == k).any() else float("nan"),
             int((bi == k).sum())) for k in range(len(MARGIN_LABELS))]


def decompose(mL, fL, mS, fS):
    """Oaxaca split of (light flip-rate - scaled flip-rate), light as reference weights."""
    biL, biS = bin_idx(mL), bin_idx(mS)
    K = len(MARGIN_LABELS)
    wL = np.array([(biL == k).mean() for k in range(K)])
    rateL = np.array([fL[biL == k].mean() if (biL == k).any() else 0.0 for k in range(K)])
    wS = np.array([(biS == k).mean() for k in range(K)])
    rateS = np.array([fS[biS == k].mean() if (biS == k).any() else 0.0 for k in range(K)])
    margin_comp = float(((wL - wS) * rateS).sum())        # different margin mix
    sens_comp = float((wL * (rateL - rateS)).sum())       # different per-margin flip (attenuation)
    return margin_comp, sens_comp


def boot_gap(pL, fL, pS, fS, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    uL, uS = np.unique(pL), np.unique(pS)
    # precompute per-problem (sum, count) so a resample is a fast gather
    def perprob(p, f, u):
        s = np.array([f[p == k].sum() for k in u]); c = np.array([(p == k).sum() for k in u])
        return s, c
    sL, cL = perprob(pL, fL, uL); sS, cS = perprob(pS, fS, uS)
    out = np.empty(n)
    for i in range(n):
        iL = rng.integers(0, len(uL), len(uL)); iS = rng.integers(0, len(uS), len(uS))
        out[i] = sL[iL].sum() / cL[iL].sum() - sS[iS].sum() / cS[iS].sum()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tf-root", default=None)
    ap.add_argument("--dataset", default="mbppplus")
    ap.add_argument("--n", type=int, default=250)
    args = ap.parse_args()
    run_root = Path(os.environ.get("LCC_RUN_ROOT", Path.home() / "lcc" / "runs"))
    tf_root = Path(args.tf_root) if args.tf_root else (run_root / "teacher_forced")
    ref_roots = [run_root, Path.home() / "lcc" / "fid_out"]
    tf_tag = build_config_tag(args.dataset, 4, 3, args.n, 1, "auto", teacher_forced=True)

    data = {}
    for tier, style in TIERS.items():
        int_npz = tf_root / tier / tf_tag / "fidelity_logits.npz"
        ref_dir = resolve_ref_dir([tf_root / tier] + ref_roots, style, args.dataset, args.n)
        if not int_npz.is_file() or ref_dir is None:
            print(f"[skip] {tier}: missing capture (int={int_npz.is_file()}, ref={ref_dir})")
            continue
        data[tier] = gather(ref_dir / "fidelity_logits.npz", int_npz)

    print(f"=== teacher-forced mechanism — {args.dataset} (b=1, greedy, INT4) ===\n")
    for tier, (prob, margin, flip, ldiff, rank) in data.items():
        print(f"[{tier}]  positions={flip.size}  problems={np.unique(prob).size}")
        print(f"  flip-rate={100*flip.mean():.1f}%   median REF margin={np.median(margin):.2f}   "
              f"median |logit diff|={np.median(ldiff):.3f}   REF top-1 rank-1 under INT4="
              f"{100*np.mean(rank==1):.1f}%")
        for lab, fr, nn in curve(margin, flip):
            print(f"    margin {lab:>5}: flip {100*fr:5.1f}%  (n={nn})")
        print()

    if "light" in data and "scaled" in data:
        pL, mL, fL = data["light"][0], data["light"][1], data["light"][2]
        pS, mS, fS = data["scaled"][0], data["scaled"][1], data["scaled"][2]
        raw = 100 * (fL.mean() - fS.mean())
        mc, sc = decompose(mL, fL, mS, fS)
        lo, hi = boot_gap(pL, fL, pS, fS)
        print("=== light - scaled flip-rate gap decomposition ===")
        print(f"  raw gap           : {raw:+.2f} pp   (95% CI problem-clustered [{100*lo:+.2f}, {100*hi:+.2f}])")
        print(f"  margin-distribution component : {100*mc:+.2f} pp  (scaled decides with larger margins)")
        print(f"  per-margin / sensitivity comp : {100*sc:+.2f} pp  (channel perturbs scaled's logits less)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
