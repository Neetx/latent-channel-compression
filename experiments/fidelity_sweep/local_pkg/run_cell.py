#!/usr/bin/env python3
"""Generic cell orchestrator: run the bit-rate ladder + greedy paired fidelity for any
(style, dataset) cell of the RecursiveMAS × Variant B matrix.

Each condition runs as a separate `fidelity_local.py` subprocess (fresh CUDA context,
crash isolation, per-run git-restore). The scientific logic lives in the driver; this is
a portable, repeatable run recipe. Generalises run_step2_scaled_mbppplus.py.

  python run_cell.py --style sequential_light --dataset medqa --ladder-batch 16 --cap-batch 2 --n 250
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

LOCAL_PKG = Path(__file__).resolve().parent
REPO_ROOT = LOCAL_PKG.parents[2]
DRIVER = LOCAL_PKG / "fidelity_local.py"


def acc_from(logpath: Path):
    try:
        m = re.findall(r"accuracy=([0-9.]+)%", logpath.read_text())
        return float(m[-1]) if m else None
    except Exception:
        return None


def run_one(style, dataset, bits, n, batch, capture, logpath, py):
    cmd = [py, str(DRIVER), "--style", style, "--dataset", dataset, "--bits", str(bits),
           "--t", "3", "--n-samples", str(n), "--batch-size", str(batch)]
    if not capture:
        cmd.append("--no-capture")
    t0 = time.time()
    with logpath.open("w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT)).returncode
    return rc, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--style", required=True, choices=["sequential_light", "sequential_scaled"])
    ap.add_argument("--dataset", required=True, choices=["math500", "medqa", "gpqa", "mbppplus"])
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--ladder-batch", type=int, default=16)
    ap.add_argument("--cap-batch", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    tag = f"{args.style}_{args.dataset}"
    out = Path(args.out) if args.out else (Path.home() / "lcc" / f"cell_{tag}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== CELL {tag}  n={args.n}  ladder_b={args.ladder_batch}  cap_b={args.cap_batch}  "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)

    print("### PHASE 1: sampled ladder ###", flush=True)
    for b in (0, 2, 4, 8):
        lp = out / f"ladder_b{b}_n{args.n}.log"
        print(f"[ladder bits={b}] start {time.strftime('%H:%M:%S')}", flush=True)
        rc, dt = run_one(args.style, args.dataset, b, args.n, args.ladder_batch, False, lp, args.python)
        print(f"[ladder bits={b}] rc={rc}  {dt/60:.1f} min  acc={acc_from(lp)}", flush=True)

    print("### PHASE 2: greedy paired fidelity ###", flush=True)
    for b in (0, 4):
        lp = out / f"fidelity_b{b}_n{args.n}.log"
        print(f"[fidelity bits={b}] start {time.strftime('%H:%M:%S')}", flush=True)
        rc, dt = run_one(args.style, args.dataset, b, args.n, args.cap_batch, True, lp, args.python)
        print(f"[fidelity bits={b}] rc={rc}  {dt/60:.1f} min  acc={acc_from(lp)}", flush=True)

    print(f"=== DONE {tag} {time.strftime('%H:%M:%S')} ===", flush=True)
    for b in (0, 2, 4, 8):
        print(f"  ladder   bits={b}: {acc_from(out / f'ladder_b{b}_n{args.n}.log')}")
    for b in (0, 4):
        print(f"  fidelity bits={b}: {acc_from(out / f'fidelity_b{b}_n{args.n}.log')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
