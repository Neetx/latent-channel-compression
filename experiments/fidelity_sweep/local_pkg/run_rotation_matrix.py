#!/usr/bin/env python3
"""Resumable orchestrator for the MBPP+ light/scaled quantizer-rotation matrix.

Runs only the INT4 (bits=4) greedy capture for each (tier, quantizer_seed). The greedy
REF (bits=0) is rotation-independent, so the existing seed-42 REF captures pair with every
rotation and are NOT re-run here. Any (tier, seed) whose valid result JSON already exists
is skipped, so the matrix survives interruptions/reboots (re-launch to continue). A
``MATRIX_DONE`` marker is written on completion (a lightweight observer can watch for it).

Each condition is an isolated ``fidelity_local.py`` subprocess. Launch detached for a
multi-day run, e.g. (scaled needs the expandable allocator):

  setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 \\
    .venv/bin/python experiments/fidelity_sweep/local_pkg/run_rotation_matrix.py \\
    > $LCC_RUN_ROOT/rotation_matrix/orchestrator.out 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LOCAL_PKG = Path(__file__).resolve().parent
DRIVER = LOCAL_PKG / "fidelity_local.py"
REPO_ROOT = LOCAL_PKG.parents[2]

TIERS = {"light": ("sequential_light", 2), "scaled": ("sequential_scaled", 1)}


def cond_tag(dataset: str, batch: int, seed: int, n: int) -> str:
    return f"{dataset}_vb4_T3_n{n}_b{batch}_auto" + ("" if seed == 42 else f"_qs{seed}")


def valid_result(out: Path, dataset: str, batch: int, seed: int, n: int) -> bool:
    tag = cond_tag(dataset, batch, seed, n)
    p = out / tag / f"fidelity_{tag}.json"
    if not p.is_file():
        return False
    try:
        r = json.loads(p.read_text())
    except Exception:
        return False
    return (r.get("return_code") == 0 and r.get("final_accuracy") is not None
            and r.get("n_per_problem") == n and r.get("n_logit_batches", 0) > 0
            and bool(r.get("call_stats_present")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="mbppplus")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 73, 101])
    ap.add_argument("--tiers", nargs="+", default=["light", "scaled"], choices=list(TIERS))
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--out", default=None)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    run_root = Path(os.environ.get("LCC_RUN_ROOT", Path.home() / "lcc" / "runs"))
    out = Path(args.out) if args.out else (run_root / "rotation_matrix")
    out.mkdir(parents=True, exist_ok=True)
    # seeds outer / tiers inner: completes each rotation's light+scaled contrast first.
    plan = [(t, s) for s in args.seeds for t in args.tiers]

    print(f"=== ROTATION MATRIX {args.dataset} INT4  seeds={args.seeds} tiers={args.tiers}  "
          f"{len(plan)} runs  out={out}  {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    summary = []
    for tier, seed in plan:
        style, batch = TIERS[tier]
        label = f"{tier} qs{seed} b{batch}"
        if valid_result(out, args.dataset, batch, seed, args.n):
            print(f"[skip] {label} — valid result exists", flush=True)
            summary.append({"cond": label, "status": "reused"})
            continue
        log = out / f"int4_{tier}_qs{seed}.log"
        cmd = [args.python, str(DRIVER), "--style", style, "--dataset", args.dataset,
               "--bits", "4", "--quantizer-seed", str(seed), "--n-samples", str(args.n),
               "--batch-size", str(batch), "--t", "3", "--out", str(out)]
        print(f"[run] {label} start {time.strftime('%H:%M:%S')}", flush=True)
        t0 = time.time()
        with log.open("w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(REPO_ROOT)).returncode
        dt = (time.time() - t0) / 60.0
        print(f"[run] {label} rc={rc}  {dt:.1f} min", flush=True)
        if rc != 0:
            print(f"[fatal] {label} failed; inspect {log}", file=sys.stderr)
            return rc
        if not valid_result(out, args.dataset, batch, seed, args.n):
            print(f"[fatal] {label} produced an invalid/incomplete result", file=sys.stderr)
            return 3
        summary.append({"cond": label, "status": f"{dt:.1f}min"})

    (out / "MATRIX_DONE").write_text(json.dumps(
        {"dataset": args.dataset, "seeds": args.seeds, "tiers": args.tiers,
         "completed": summary, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print(f"=== MATRIX DONE {time.strftime('%H:%M:%S')} -> {out / 'MATRIX_DONE'} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
