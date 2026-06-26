#!/usr/bin/env python3
"""Is the greedy-trajectory divergence functionally inert?

A high trajectory-divergence rate (51--93%) can read as alarming, but a different token path
need not change the task outcome: divergent MBPP+ code can still pass the same unit tests, and a
divergent Math500 trace can still reach the same boxed answer. For each non-confounded cell this
cross-tabulates per-problem *trajectory divergence within 128* (from the paired REF/INT4 captures)
against per-problem *outcome change* (correctness flip, from the committed per-problem JSONLs), and
reports the fraction of diverged trajectories that nonetheless preserve the final outcome.

Divergence needs the local NPZ captures (not committed, by size); correctness is read from the
committed JSONLs. Run: python functional_equivalence.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

LOCAL_PKG = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(LOCAL_PKG / "analysis"))
import divergence_hazard as dh  # resolve_npz, _valid_len, HORIZON, PRIMARY_N

RESULTS = LOCAL_PKG / "results"
# (label, style, dataset, batch, results-subdir, jsonl tag)
CELLS = [
    ("math500 / light",  "light",  "math500",  2, "step0",                 "vb0_T3_n250_b2_auto",          "vb4_T3_n250_b2_auto"),
    ("mbppplus / light", "light",  "mbppplus", 2, "step1_mbppplus",        "mbppplus_vb0_T3_n250_b2_auto", "mbppplus_vb4_T3_n250_b2_auto"),
    ("mbppplus / scaled","scaled", "mbppplus", 1, "step2_scaled_mbppplus", "mbppplus_vb0_T3_n250_b1_auto", "mbppplus_vb4_T3_n250_b1_auto"),
    ("math500 / scaled", "scaled", "math500",  1, "step4_scaled_math500",  "math500_vb0_T3_n250_b1_auto",  "math500_vb4_T3_n250_b1_auto"),
]


def per_problem_diverged(ref_npz, int_npz, primary):
    """problem_idx -> diverged-within-128 (bool), None where neither side has a valid capture."""
    r, t = np.load(ref_npz), np.load(int_npz)
    n = min(int(r["n_batches"]), int(t["n_batches"]), primary)
    out = {}
    for b in range(n):
        ri, ii = r[f"batch{b}_idxs"], t[f"batch{b}_idxs"]
        rl, il = r[f"batch{b}_full_lse"], t[f"batch{b}_full_lse"]
        B = min(ri.shape[1], ii.shape[1])
        for bi in range(B):
            pidx = b * B + bi
            L = min(dh._valid_len(rl[:, bi]), dh._valid_len(il[:, bi]), dh.HORIZON)
            if L == 0:
                continue
            out[pidx] = bool(np.any(ri[:L, bi, 0] != ii[:L, bi, 0]))
    return out


def correctness(subdir, tag):
    p = RESULTS / subdir / "fidelity" / f"per_problem_{tag}.jsonl"
    return {json.loads(l)["sample_idx"]: bool(json.loads(l)["correct"]) for l in p.open()}


def main() -> int:
    roots = [Path(os.environ.get("LCC_RUN_ROOT", Path.home() / "lcc" / "runs")),
             Path.home() / "lcc" / "fid_out"]
    hdr = (f"{'cell':18} {'n':>4} {'diverged':>9} {'churn':>6} "
           f"{'P(flip|div)':>11} {'outcome-preserved|diverged':>27}")
    print(hdr); print("-" * len(hdr))
    agg_div, agg_pres = [], []
    for label, style, dataset, batch, subdir, reftag, inttag in CELLS:
        ref = dh.resolve_npz(roots, style, dataset, 0, batch)
        intd = dh.resolve_npz(roots, style, dataset, 4, batch)
        div = per_problem_diverged(ref, intd, (dh.PRIMARY_N + batch - 1) // batch)
        cref, cint = correctness(subdir, reftag), correctness(subdir, inttag)
        idx = sorted(set(div) & set(cref) & set(cint))
        d = np.array([div[i] for i in idx])
        flip = np.array([cref[i] != cint[i] for i in idx])
        n = len(idx)
        n_div = int(d.sum())
        churn = 100 * flip.mean()
        p_flip_div = 100 * flip[d].mean() if n_div else 0.0
        preserved = 100 * (1 - flip[d].mean()) if n_div else float("nan")
        print(f"{label:18} {n:4d} {100*d.mean():8.1f}% {churn:5.1f}% "
              f"{p_flip_div:10.1f}% {preserved:26.1f}%")
        agg_div.append(100 * d.mean()); agg_pres.append(preserved)
    print("-" * len(hdr))
    print(f"Across cells: divergence {min(agg_div):.0f}--{max(agg_div):.0f}% of trajectories, "
          f"but {min(agg_pres):.0f}--{max(agg_pres):.0f}% of those diverged trajectories preserve "
          f"the final outcome (pass/fail).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
