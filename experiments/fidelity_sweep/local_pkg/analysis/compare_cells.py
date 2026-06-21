#!/usr/bin/env python3
"""Rigorous cross-cell comparison of the greedy paired REF-vs-INT4 runs.

For each cell it reports the paired accuracy delta, the discordant pairs (losses b,
gains c), the exact McNemar test, AND a 95% CI on the loss-fraction b/(b+c) — which is
what tells us whether an apparent loss/gain *asymmetry* is real or just noise. With the
small discordant counts here (single seed, n=250) the honest expectation is that every
CI straddles 0.5 (no significant asymmetry) and every Δ CI straddles 0.

Reads the committed per-problem JSONLs for the two light cells and (by default) the
scaled cell from the working cache. Override paths with --scaled-ref/--scaled-int4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

LOCAL_PKG = Path(__file__).resolve().parent.parent
FID = Path.home() / "lcc" / "fid_out"


def load(p: Path) -> dict:
    d = {}
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
            d[si] = c
    return d


def analyse(label, refp, intp, boot=20000, seed=42):
    ref, intq = load(refp), load(intp)
    keys = sorted(set(ref) & set(intq))
    n = len(keys)
    r = np.array([ref[k] for k in keys], dtype=bool)
    q = np.array([intq[k] for k in keys], dtype=bool)
    loss = int((r & ~q).sum())   # REF correct, INT4 wrong
    gain = int((~r & q).sum())   # REF wrong, INT4 correct
    nd = loss + gain
    delta = (q.mean() - r.mean()) * 100
    # paired bootstrap CI on delta
    d = q.astype(int) - r.astype(int)
    rng = np.random.default_rng(seed)
    bd = d[rng.integers(0, n, size=(boot, n))].mean(axis=1) * 100
    dci = (np.percentile(bd, 2.5), np.percentile(bd, 97.5))
    # McNemar exact + asymmetry CI (loss fraction among discordants)
    mc = binomtest(min(loss, gain), nd, 0.5).pvalue if nd else 1.0
    if nd:
        bt = binomtest(loss, nd, 0.5)
        lo, hi = bt.proportion_ci(0.95)
    else:
        lo = hi = float("nan")
    return dict(label=label, n=n, refacc=r.mean()*100, intacc=q.mean()*100, delta=delta,
                dci=dci, loss=loss, gain=gain, nd=nd, churn=(loss+gain)/n*100,
                mcnemar=mc, lossfrac=(loss/nd if nd else float("nan")), lf_ci=(lo, hi))


def main():
    ap = argparse.ArgumentParser()
    s0 = LOCAL_PKG / "results/step0/fidelity"
    s1 = LOCAL_PKG / "results/step1_mbppplus/fidelity"
    s3 = LOCAL_PKG / "results/step3_light_medqa/fidelity"
    ap.add_argument("--scaled-ref", default=str(FID / "mbppplus_vb0_T3_n250_b1_auto/per_problem_mbppplus_vb0_T3_n250_b1_auto.jsonl"))
    ap.add_argument("--scaled-int4", default=str(FID / "mbppplus_vb4_T3_n250_b1_auto/per_problem_mbppplus_vb4_T3_n250_b1_auto.jsonl"))
    args = ap.parse_args()

    cells = [
        ("math500  / light", s0/"per_problem_vb0_T3_n250_b2_auto.jsonl",          s0/"per_problem_vb4_T3_n250_b2_auto.jsonl"),
        ("mbppplus / light", s1/"per_problem_mbppplus_vb0_T3_n250_b2_auto.jsonl",  s1/"per_problem_mbppplus_vb4_T3_n250_b2_auto.jsonl"),
        ("mbppplus / scaled", Path(args.scaled_ref),                              Path(args.scaled_int4)),
        ("medqa    / light", s3/"per_problem_medqa_vb0_T3_n250_b2_auto.jsonl",    s3/"per_problem_medqa_vb4_T3_n250_b2_auto.jsonl"),
    ]

    print(f"{'cell':18} {'REF':>6} {'INT4':>6} {'Δpp':>6} {'Δ 95%CI':>16} "
          f"{'b/c':>7} {'churn%':>7} {'McNemar':>8} {'loss-frac (95% CI)':>22}")
    print("-" * 110)
    rows = []
    for lbl, rp, ip in cells:
        x = analyse(lbl, rp, ip)
        rows.append(x)
        print(f"{x['label']:18} {x['refacc']:5.1f}% {x['intacc']:5.1f}% {x['delta']:+5.1f} "
              f"[{x['dci'][0]:+4.1f},{x['dci'][1]:+4.1f}]  {x['loss']:>2}/{x['gain']:<2}  "
              f"{x['churn']:5.1f}   p={x['mcnemar']:.2f}   "
              f"{x['lossfrac']:.2f} [{x['lf_ci'][0]:.2f},{x['lf_ci'][1]:.2f}]")

    print("\nReading:")
    print("- math500 / mbppplus (light + scaled): every Δ 95% CI straddles 0 and every")
    print("  McNemar is non-significant — no significant accuracy effect, no significant")
    print("  loss/gain asymmetry. Single seed, 11-25 discordants: the per-cell directions")
    print("  (+2 / 0 / -2) are within noise. The faint scaled -2pp needs power to resolve.")
    print("- medqa is the LONE significant cell (Δ=+15pp, p<0.001) but is CONFOUNDED: under")
    print("  greedy the unquantized REF develops a strong 'A'-bias (picks A ~46% vs ~25%")
    print("  expected) and scores 21.2%, while the quantizer's dither breaks the bias (36.4%).")
    print("  The +15pp is a greedy-decoding pathology of the REF, NOT a channel-fidelity")
    print("  result. medqa must be read via its (flat) sampled ladder; its greedy paired")
    print("  comparison is not a clean answer-preservation measure. See")
    print("  results/step3_light_medqa/SUMMARY.md.")


if __name__ == "__main__":
    main()
