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


def resolve_dir(roots, style, dataset, tag):
    """Containing dir of a condition's captures, across all three capture layouts: the
    run_cell nested layout (``sequential_{style}_{dataset}/tag``), the flat ``~/lcc/fid_out``
    tag, and the run_links_ablation flat tag. Mirrors divergence_hazard.resolve_npz so the
    analysis works no matter how REF/all/inner/outer were regenerated."""
    flat = tag[len(f"{dataset}_"):] if tag.startswith(f"{dataset}_") else tag
    for root in roots:
        for cand in (root / f"sequential_{style}_{dataset}" / tag, root / tag, root / flat):
            if (cand / "fidelity_logits.npz").is_file():
                return cand
    return None


def accuracy(d, tag):
    p = d / f"fidelity_{tag}.json"
    if not p.is_file():
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
        ref_tag = f"mbppplus_vb0_T3_n{n}_b{batch}_auto"
        ref_dir = resolve_dir(roots, tier, "mbppplus", ref_tag)
        if not ref_dir:
            print(f"{tier:7} REF MISSING"); missing += 1; continue
        ref = ref_dir / "fidelity_logits.npz"
        print(f"{tier:7} {'REF':6} {str(accuracy(ref_dir, ref_tag)):>6} {'--':>10}")
        primary = (n + batch - 1) // batch
        for label, sfx in LINKSETS:
            tag = f"mbppplus_vb4_T3_n{n}_b{batch}_auto{sfx}"
            d = resolve_dir(roots, tier, "mbppplus", tag)
            if not d:
                print(f"{tier:7} {label:6} {'--':>6} {'MISSING':>10}"); missing += 1; continue
            intd = d / "fidelity_logits.npz"
            t, e, L = dh.first_divergence(ref, intd, primary)
            S, _, _ = dh.kaplan_meier(t, e)
            evt = float(np.mean(e)) * 100
            acc = accuracy(d, tag)
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
