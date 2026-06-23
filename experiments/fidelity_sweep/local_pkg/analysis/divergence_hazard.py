#!/usr/bin/env python3
"""Length-censored first-divergence survival/hazard for the greedy REF-vs-INT4 captures.

The Tier-2 ``divergence within 128`` metric is windowed AND padding-inclusive: each
capture stores a fixed 128 positions, padded with zeros once a sequence has finished
(detected here as ``full_lse == 0``). Comparing padded positions conflates "the two
greedy sequences generated identical text and one simply stopped sooner" with "the
sequences diverged", and it makes a cell's apparent robustness depend on how long its
generations happen to be.

This analysis instead treats first-divergence as a right-censored survival process:

* observe each problem only while BOTH sequences are still generating real tokens, i.e.
  up to ``L = min(len_ref, len_int, 128)`` valid positions;
* event = first position where the greedy top-1 tokens differ (``time = that position``);
* otherwise censor at ``L``.

The per-position hazard ``h(t)`` and survival ``S(t)`` then have a length-independent
early regime (almost every problem is still at risk at small ``t``), so ``1 - S(t)`` at a
fixed early position is a fair cross-task / cross-tier comparison. If the MBPP+
light/scaled gap survives there but the Math500 one stays small, the task-specific tier
contrast is real and not a generation-length artifact.

Only the fixed primary solver batches are used (``ceil(n/batch_size)``); conditional
answer-retry batches are excluded, matching ``tier2_logit_fidelity.py``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

HORIZON = 128
PRIMARY_N = 250
PROBES = (10, 25, 50, 100)


def kaplan_meier(time: np.ndarray, event: np.ndarray, horizon: int = HORIZON):
    """Discrete-time KM. ``time`` is the 1-indexed position of the event (or of right
    censoring); ``event`` is 1 for a divergence, 0 for a censored (identical) sequence.
    Returns ``(S, h, at_risk)`` arrays indexed by position 1..horizon (index 0 unused)."""
    time = np.asarray(time, dtype=np.int64)
    event = np.asarray(event, dtype=np.int64)
    S = np.ones(horizon + 1, dtype=np.float64)
    h = np.zeros(horizon + 1, dtype=np.float64)
    at_risk = np.zeros(horizon + 1, dtype=np.int64)
    surv = 1.0
    for t in range(1, horizon + 1):
        n_risk = int(np.sum(time >= t))
        at_risk[t] = n_risk
        d = int(np.sum((time == t) & (event == 1)))
        ht = (d / n_risk) if n_risk > 0 else 0.0
        h[t] = ht
        surv *= (1.0 - ht)
        S[t] = surv
    return S, h, at_risk


def _valid_len(full_lse_col: np.ndarray) -> int:
    """Number of leading real positions: a finished sequence is zero-padded, and a real
    next-token full-vocab log-sum-exp is never exactly 0."""
    return int(np.sum(full_lse_col != 0.0))


def first_divergence(ref_npz: Path, int_npz: Path, primary_batches: int):
    """Per-problem (time, event, observed_length) with min-length right-censoring."""
    ref_z = np.load(ref_npz, allow_pickle=True)
    int_z = np.load(int_npz, allow_pickle=True)
    n = min(int(ref_z["n_batches"]), int(int_z["n_batches"]), primary_batches)
    times, events, lengths = [], [], []
    for b in range(n):
        r_idx = ref_z[f"batch{b}_idxs"]      # (T, B, K)
        i_idx = int_z[f"batch{b}_idxs"]
        r_lse = ref_z[f"batch{b}_full_lse"]  # (T, B)
        i_lse = int_z[f"batch{b}_full_lse"]
        B = min(r_idx.shape[1], i_idx.shape[1])
        for bi in range(B):
            L = min(_valid_len(r_lse[:, bi]), _valid_len(i_lse[:, bi]), HORIZON)
            if L == 0:
                continue
            r_top1 = r_idx[:L, bi, 0]
            i_top1 = i_idx[:L, bi, 0]
            mism = np.nonzero(r_top1 != i_top1)[0]
            if mism.size:
                times.append(int(mism[0]) + 1)   # 1-indexed divergence position
                events.append(1)
            else:
                times.append(L)                  # censored: identical for all L positions
                events.append(0)
            lengths.append(L)
    return (np.array(times, dtype=np.int64),
            np.array(events, dtype=np.int64),
            np.array(lengths, dtype=np.int64))


def resolve_npz(roots, style, dataset, bits, batch):
    """Find a cell's fidelity_logits.npz across the canonical run layout and the legacy
    ~/lcc/fid_out flat tags (math500/light has no dataset prefix)."""
    tag = f"{dataset}_vb{bits}_T3_n{PRIMARY_N}_b{batch}_auto"
    legacy_math = f"vb{bits}_T3_n{PRIMARY_N}_b{batch}_auto"
    for root in roots:
        for cand in (root / f"sequential_{style}_{dataset}" / tag / "fidelity_logits.npz",
                     root / tag / "fidelity_logits.npz",
                     root / legacy_math / "fidelity_logits.npz"):
            if cand.is_file():
                return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", default=None,
                    help="capture roots to search (default: $LCC_RUN_ROOT and ~/lcc/fid_out)")
    args = ap.parse_args()
    if args.roots:
        roots = [Path(r) for r in args.roots]
    else:
        roots = [Path(os.environ.get("LCC_RUN_ROOT", Path.home() / "lcc" / "runs")),
                 Path.home() / "lcc" / "fid_out"]

    cells = [
        ("math500 / light", "light", "math500", 2),
        ("math500 / scaled", "scaled", "math500", 1),
        ("mbppplus / light", "light", "mbppplus", 2),
        ("mbppplus / scaled", "scaled", "mbppplus", 1),
        ("medqa / light", "light", "medqa", 2),
    ]

    hdr = (f"{'cell':18} {'n':>4} {'med.len':>7} {'%len<128':>8} "
           + " ".join(f"1-S({t}){'':>1}" for t in PROBES)
           + f" {'evt%':>6} {'eh(1-25)':>9}")
    print(hdr)
    print("-" * len(hdr))
    missing = 0
    for label, style, dataset, batch in cells:
        ref = resolve_npz(roots, style, dataset, 0, batch)
        intd = resolve_npz(roots, style, dataset, 4, batch)
        if not (ref and intd):
            missing += 1
            print(f"{label:18} MISSING")
            continue
        primary = (PRIMARY_N + batch - 1) // batch
        time, event, length = first_divergence(ref, intd, primary)
        S, h, _ = kaplan_meier(time, event)
        med_len = int(np.median(length))
        frac_short = float(np.mean(length < HORIZON))
        early_hazard = float(np.mean(h[1:26]))   # mean per-position hazard, positions 1-25
        evt_rate = float(np.mean(event))
        probes = "  ".join(f"{1.0 - S[t]:6.3f}" for t in PROBES)
        print(f"{label:18} {time.size:4d} {med_len:7d} {frac_short:8.2f} "
              f"{probes} {evt_rate*100:5.1f}% {early_hazard:9.4f}")
    if missing:
        print(f"\n{missing} cell(s) missing NPZ; regenerate captures or pass --roots.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
