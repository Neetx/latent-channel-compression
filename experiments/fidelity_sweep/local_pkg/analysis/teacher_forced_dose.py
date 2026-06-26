#!/usr/bin/env python3
"""Teacher-forced bit-rate dose-response (MBPP+): flip-rate and perturbation vs bit-width.

For each tier and each INT bit-width, this compares the teacher-forced INT capture against the
shared full-precision REF (the same b=1 REF the headline mechanism uses) at every aligned
position, reporting the arg-max flip rate and the median per-position top-1 logit perturbation.
It tests whether the trajectory drift tracks the channel bit-rate monotonically, and whether the
flip-rate is governed by the perturbation magnitude or (as the margin-tipping mechanism predicts)
buffered by the REF margin distribution.

Captures are b=1 teacher-forced (gate G0). Run: python teacher_forced_dose.py [--bits ...]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

TIERS = ["light", "scaled"]


def ref_dir(tf_root, tier, n):
    """The b=1 full-precision REF for this tier (TF-REF == REF by gate G0)."""
    tag = f"mbppplus_vb0_T3_n{n}_b1_auto"
    for cand in (tf_root / tier / tag,
                 Path.home() / "lcc" / "fid_out" / tag,
                 Path(os.environ.get("LCC_RUN_ROOT", Path.home() / "lcc" / "runs"))
                 / f"sequential_{tier}_mbppplus" / tag):
        if (cand / "fidelity_logits.npz").is_file():
            return cand
    return None


def curve(ref_npz, int_npz):
    r, t = np.load(ref_npz), np.load(int_npz)
    nb = min(int(r["n_batches"]), int(t["n_batches"]))
    flips, ldiff = [], []
    for b in range(nb):
        ri, ii = r[f"batch{b}_idxs"], t[f"batch{b}_idxs"]
        rv, iv = r[f"batch{b}_vals"], t[f"batch{b}_vals"]
        rl, il = r[f"batch{b}_full_lse"], t[f"batch{b}_full_lse"]
        T = min(ri.shape[0], ii.shape[0])
        for p in range(T):
            for e in range(min(ri.shape[1], ii.shape[1])):
                if rl[p, e] == 0 or il[p, e] == 0:
                    continue
                flips.append(int(ri[p, e, 0]) != int(ii[p, e, 0]))
                ldiff.append(abs(float(rv[p, e, 0]) - float(iv[p, e, 0])))
    f = np.array(flips)
    return 100 * f.mean(), float(np.median(ldiff)), f.size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4, 6, 8])
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--tf-root", default=None)
    args = ap.parse_args()
    run_root = Path(os.environ.get("LCC_RUN_ROOT", Path.home() / "lcc" / "runs"))
    tf_root = Path(args.tf_root) if args.tf_root else (run_root / "teacher_forced")

    print(f"MBPP+ teacher-forced bit-rate dose-response (b=1, REF reused), bits={args.bits}\n")
    for tier in TIERS:
        rd = ref_dir(tf_root, tier, args.n)
        if rd is None:
            print(f"[{tier}] REF missing"); continue
        print(f"[{tier}]  bits   flip-rate   median|logit diff|")
        for bit in args.bits:
            tag = f"mbppplus_vb{bit}_T3_n{args.n}_b1_auto_tf"
            p = tf_root / tier / tag / "fidelity_logits.npz"
            if not p.is_file():
                print(f"        {bit:>4}   (pending)"); continue
            fr, md, n = curve(rd / "fidelity_logits.npz", p)
            print(f"        {bit:>4}     {fr:5.1f}%        {md:.3f}   (n={n})")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
