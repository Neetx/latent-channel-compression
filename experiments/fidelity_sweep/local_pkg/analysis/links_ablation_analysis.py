#!/usr/bin/env python3
"""Inner/outer link-compression ablation analysis (MBPP+).

Localizes the INT4 trajectory drift: for each tier it compares the greedy REF against three
INT4 captures — all links (the headline), inner-only (``_li``), and outer-only (``_lo``) —
using the same windowed ``divergence within 128`` first-top1-mismatch estimand as
``divergence_hazard.py``. Reuses the shared first_divergence / kaplan_meier so the numbers
are directly comparable to the committed tier-effect and rotation-matrix results.

Run: python links_ablation_analysis.py
     [--roots <capture roots>]  (default: $LCC_RUN_ROOT, ~/lcc/runs/links_ablation, ~/lcc/fid_out)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import divergence_hazard as dh  # first_divergence, kaplan_meier, HORIZON, PROBES, PRIMARY_N

TIERS = [("light", 2), ("scaled", 1)]
LINKSETS = [("all", ""), ("inner", "_li"), ("outer", "_lo")]


def find(roots, tag):
    for root in roots:
        p = root / tag / "fidelity_logits.npz"
        if p.is_file():
            return p
    return None


def find_json(roots, tag):
    for root in roots:
        p = root / tag / f"fidelity_{tag}.json"
        if p.is_file():
            return p
    return None


def accuracy(roots, tag):
    p = find_json(roots, tag)
    if not p:
        return None
    try:
        return json.load(open(p)).get("final_accuracy")
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", default=None)
    args = ap.parse_args()
    if args.roots:
        roots = [Path(r) for r in args.roots]
    else:
        run_root = Path(os.environ.get("LCC_RUN_ROOT", Path.home() / "lcc" / "runs"))
        roots = [run_root / "links_ablation", run_root, Path.home() / "lcc" / "fid_out"]

    n = dh.PRIMARY_N
    print(f"MBPP+ inner/outer link ablation — divergence within {dh.HORIZON} "
          f"(REF vs INT4), n={n}, greedy, quantizer seed 42\n")
    hdr = f"{'tier':7} {'links':6} {'acc':>6} {'within128':>10} {'med.len':>8}  cum 1-S at {dh.PROBES}"
    print(hdr); print("-" * len(hdr))
    missing = 0
    table = {}
    for tier, batch in TIERS:
        ref = find(roots, f"mbppplus_vb0_T3_n{n}_b{batch}_auto")
        ref_acc = accuracy(roots, f"mbppplus_vb0_T3_n{n}_b{batch}_auto")
        print(f"{tier:7} {'REF':6} {str(ref_acc):>6} {'--':>10}")
        if not ref:
            print(f"{tier:7} REF MISSING"); missing += 1; continue
        primary = (n + batch - 1) // batch
        for label, sfx in LINKSETS:
            tag = f"mbppplus_vb4_T3_n{n}_b{batch}_auto{sfx}"
            intd = find(roots, tag)
            if not intd:
                print(f"{tier:7} {label:6} {'--':>6} {'MISSING':>10}"); missing += 1; continue
            t, e, L = dh.first_divergence(ref, intd, primary)
            S, _, _ = dh.kaplan_meier(t, e)
            evt = float(np.mean(e)) * 100
            acc = accuracy(roots, tag)
            probes = "  ".join(f"{1.0 - S[p]:.3f}" for p in dh.PROBES)
            print(f"{tier:7} {label:6} {str(acc):>6} {evt:9.1f}% {int(np.median(L)):8d}  {probes}")
            table[(tier, label)] = evt
        # localization deltas vs the all-links headline
        if (tier, "all") in table:
            a = table[(tier, "all")]
            parts = [f"{lk}-vs-all={table[(tier, lk)]-a:+.1f}pp"
                     for lk in ("inner", "outer") if (tier, lk) in table]
            if parts:
                print(f"        -> {tier}: " + "   ".join(parts))
        print()
    if missing:
        print(f"{missing} capture(s) missing; run run_links_ablation.py or pass --roots.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
