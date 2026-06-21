#!/usr/bin/env python3
"""Step 2 — sequential_scaled x mbppplus orchestrator (pure Python).

The high-baseline test of the guiding hypothesis: does compression break on code once
the system is actually good at it (removing Step 1's floor-effect confound)?

It loops the bit-rate ladder + the greedy paired fidelity, running EACH condition as a
separate `fidelity_local.py` subprocess. Separate processes (not one long process) are
deliberate: every run gets a fresh CUDA context (no memory fragmentation carryover), is
crash-isolated (one OOM does not abort the rest), and the driver's own git-restore keeps
the upstream pristine per run. The scientific logic lives entirely in the driver and the
analysis scripts — this file is just a repeatable, portable run recipe.

Batches are CLI args because a 4B agent's KV cache at max_new_tokens=4000 is large
(set them from setup/probe_scaled_feasibility.py).

  python run_step2_scaled_mbppplus.py --ladder-batch 4 --cap-batch 1 --n 250
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

LOCAL_PKG = Path(__file__).resolve().parent
REPO_ROOT = LOCAL_PKG.parents[2]                 # latent-channel-compression
DRIVER = LOCAL_PKG / "fidelity_local.py"
STYLE, DATASET = "sequential_scaled", "mbppplus"


def acc_from(logpath: Path):
    try:
        m = re.findall(r"accuracy=([0-9.]+)%", logpath.read_text())
        return float(m[-1]) if m else None
    except Exception:
        return None


def run_one(bits: int, n: int, batch: int, capture: bool, logpath: Path, py: str):
    cmd = [py, str(DRIVER), "--style", STYLE, "--dataset", DATASET, "--bits", str(bits),
           "--t", "3", "--n-samples", str(n), "--batch-size", str(batch)]
    if not capture:
        cmd.append("--no-capture")
    t0 = time.time()
    with logpath.open("w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT)).returncode
    return rc, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--ladder-batch", type=int, default=4, help="sampled ladder batch (from VRAM probe)")
    ap.add_argument("--cap-batch", type=int, default=1, help="greedy capture batch (tightest case)")
    ap.add_argument("--out", default=str(Path.home() / "lcc" / "step2_scaled_mbppplus"))
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== STEP2 {STYLE}/{DATASET}  n={args.n}  ladder_b={args.ladder_batch}  "
          f"cap_b={args.cap_batch}  {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)

    print("### PHASE 1: sampled ladder ###", flush=True)
    for b in (0, 2, 4, 8):
        lp = out / f"ladder_b{b}_n{args.n}.log"
        print(f"[ladder bits={b}] start {time.strftime('%H:%M:%S')}", flush=True)
        rc, dt = run_one(b, args.n, args.ladder_batch, capture=False, logpath=lp, py=args.python)
        print(f"[ladder bits={b}] rc={rc}  {dt/60:.1f} min  acc={acc_from(lp)}", flush=True)

    print("### PHASE 2: greedy paired fidelity ###", flush=True)
    for b in (0, 4):
        lp = out / f"fidelity_b{b}_n{args.n}.log"
        print(f"[fidelity bits={b}] start {time.strftime('%H:%M:%S')}", flush=True)
        rc, dt = run_one(b, args.n, args.cap_batch, capture=True, logpath=lp, py=args.python)
        print(f"[fidelity bits={b}] rc={rc}  {dt/60:.1f} min  acc={acc_from(lp)}", flush=True)

    print(f"=== DONE {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    for b in (0, 2, 4, 8):
        print(f"  ladder   bits={b}: {acc_from(out / f'ladder_b{b}_n{args.n}.log')}")
    for b in (0, 4):
        print(f"  fidelity bits={b}: {acc_from(out / f'fidelity_b{b}_n{args.n}.log')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
