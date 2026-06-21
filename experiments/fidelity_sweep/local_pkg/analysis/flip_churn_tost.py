#!/usr/bin/env python3
"""Flip-churn + paired TOST equivalence analysis for a REF-vs-INT4 greedy run.

Reproducible: by default it reads the per-problem JSONLs committed under
``../results/step0/fidelity/`` (the local bf16 math500 Step 0), so it runs from a
fresh checkout with no access to the original scratch/ext4 paths. Override with
``--ref`` / ``--int4`` to analyse any other paired greedy run.

Outputs the 2x2 contingency, flip-churn (how many answers flip despite the net),
a paired bootstrap CI on Δacc, the TOST verdict at ±2 pp, and exact McNemar.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

LOCAL_PKG = Path(__file__).resolve().parent.parent
DEF_REF = LOCAL_PKG / "results/step0/fidelity/per_problem_vb0_T3_n250_b2_auto.jsonl"
DEF_INT4 = LOCAL_PKG / "results/step0/fidelity/per_problem_vb4_T3_n250_b2_auto.jsonl"


def load(p: Path) -> dict:
    recs = {}
    for line in Path(p).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not isinstance(r, dict) or r.get("type") == "summary":
            continue
        si, c = r.get("sample_idx"), r.get("correct")
        if si is not None and isinstance(c, bool):
            recs[si] = c
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default=str(DEF_REF), help="REF (bits=0) per-problem JSONL")
    ap.add_argument("--int4", default=str(DEF_INT4), help="INT4 (bits=4) per-problem JSONL")
    ap.add_argument("--eps-pp", type=float, default=2.0, help="TOST equivalence margin (pp)")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ref, int4 = load(Path(args.ref)), load(Path(args.int4))
    keys = sorted(set(ref) & set(int4))
    n = len(keys)
    if n == 0:
        raise SystemExit("no aligned correctness records found; check --ref/--int4 paths")
    r = np.array([ref[k] for k in keys], dtype=bool)
    q = np.array([int4[k] for k in keys], dtype=bool)

    both = int((r & q).sum())
    loss = int((r & ~q).sum())
    gain = int((~r & q).sum())
    neither = int((~r & ~q).sum())

    print(f"REF  = {Path(args.ref).name}")
    print(f"INT4 = {Path(args.int4).name}")
    print(f"paired n={n}   REF acc={r.mean()*100:.2f}%   INT4 acc={q.mean()*100:.2f}%   "
          f"Δ={(q.mean()-r.mean())*100:+.2f} pp\n")
    print("2x2 contingency:")
    print(f"  both correct={both}  REF✓/INT4✗(loss)={loss}  REF✗/INT4✓(gain)={gain}  both wrong={neither}\n")
    print("FLIP-CHURN:")
    print(f"  flipped={loss+gain} ({(loss+gain)/n*100:.1f}% of problems)  "
          f"losses={loss} gains={gain} net={gain-loss:+d}")
    if gain != loss:
        print(f"  churn ({loss+gain}) = {(loss+gain)/abs(gain-loss):.1f}x the net change ({abs(gain-loss)})\n")

    d = q.astype(int) - r.astype(int)
    rng = np.random.default_rng(args.seed)
    boot = d[rng.integers(0, n, size=(args.boot, n))].mean(axis=1) * 100
    ci95 = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
    ci90 = (np.percentile(boot, 5), np.percentile(boot, 95))
    eps = args.eps_pp
    within90 = ci90[0] > -eps and ci90[1] < eps
    sig = not (ci95[0] <= 0 <= ci95[1])
    verdict = ("EQUIVALENT" if within90 else
               "DIFFERENT" if sig else "INCONCLUSIVE")
    print(f"Paired bootstrap (B={args.boot}, seed={args.seed}):")
    print(f"  Δ={d.mean()*100:+.2f} pp  95% CI [{ci95[0]:+.2f},{ci95[1]:+.2f}]  90% CI [{ci90[0]:+.2f},{ci90[1]:+.2f}]")
    print(f"  TOST @±{eps:g}pp: {verdict}\n")

    nd = loss + gain
    k = min(loss, gain)
    p = min(1.0, 2 * sum(comb(nd, i) for i in range(k + 1)) / (2 ** nd)) if nd else 1.0
    print(f"McNemar exact: discordant={nd} (b={loss}, c={gain})  two-sided p={p:.3f}")


if __name__ == "__main__":
    main()
