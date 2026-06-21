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
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

LOCAL_PKG = Path(__file__).resolve().parent
REPO_ROOT = LOCAL_PKG.parents[2]
DRIVER = LOCAL_PKG / "fidelity_local.py"


def environment_metadata() -> dict:
    """Capture enough runtime identity to audit a cell without exposing secrets."""
    metadata = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        metadata["repository_commit"] = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        metadata["repository_commit"] = None
    try:
        import torch
        metadata.update({
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        })
        if torch.cuda.is_available():
            metadata["gpu"] = torch.cuda.get_device_name(0)
            metadata["compute_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:
        metadata["torch_probe_error"] = type(exc).__name__
    return metadata


def acc_from(logpath: Path):
    try:
        m = re.findall(r"accuracy=([0-9.]+)%", logpath.read_text())
        return float(m[-1]) if m else None
    except Exception:
        return None


def validate_result(out: Path, dataset: str, bits: int, n: int, batch: int, capture: bool):
    """Validate the machine-readable contract emitted by one completed condition."""
    tag = f"{dataset}_vb{bits}_T3_n{n}_b{batch}_auto"
    path = out / tag / f"fidelity_{tag}.json"
    if not path.is_file():
        return False, f"missing result JSON: {path}"
    try:
        result = json.loads(path.read_text())
    except Exception as exc:
        return False, f"invalid result JSON {path}: {exc}"
    if result.get("return_code") != 0 or result.get("final_accuracy") is None:
        return False, f"unsuccessful result contract: {path}"
    if capture:
        if result.get("n_per_problem") != n:
            return False, f"expected {n} paired records, got {result.get('n_per_problem')}"
        if result.get("n_logit_batches", 0) <= 0:
            return False, f"capture produced no logit batches: {path}"
        if bits > 0 and not result.get("call_stats_present"):
            return False, f"INT{bits} capture produced no channel statistics: {path}"
    return True, str(path)


def run_one(style, dataset, bits, n, batch, capture, logpath, py, out):
    cmd = [py, str(DRIVER), "--style", style, "--dataset", dataset, "--bits", str(bits),
           "--t", "3", "--n-samples", str(n), "--batch-size", str(batch),
           "--out", str(out)]
    if not capture:
        cmd.append("--no-capture")
    t0 = time.time()
    with logpath.open("w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT)).returncode
    return rc, time.time() - t0, cmd


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
    run_root = Path(os.environ.get("LCC_RUN_ROOT", Path.home() / "lcc" / "runs"))
    out = Path(args.out) if args.out else (run_root / tag)
    out.mkdir(parents=True, exist_ok=True)
    runs = []

    print(f"=== CELL {tag}  n={args.n}  ladder_b={args.ladder_batch}  cap_b={args.cap_batch}  "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)

    print("### PHASE 1: sampled ladder ###", flush=True)
    for b in (0, 8, 4, 2):
        lp = out / f"ladder_b{b}_n{args.n}.log"
        print(f"[ladder bits={b}] start {time.strftime('%H:%M:%S')}", flush=True)
        rc, dt, cmd = run_one(
            args.style, args.dataset, b, args.n, args.ladder_batch, False, lp,
            args.python, out,
        )
        runs.append({"phase": "ladder", "bits": b, "return_code": rc,
                     "seconds": round(dt, 1), "command": cmd, "log": str(lp)})
        print(f"[ladder bits={b}] rc={rc}  {dt/60:.1f} min  acc={acc_from(lp)}", flush=True)
        if rc != 0:
            (out / "cell_manifest.json").write_text(json.dumps(runs, indent=2))
            print(f"[fatal] stopping after failed condition; inspect {lp}", file=sys.stderr)
            return rc
        valid, detail = validate_result(
            out, args.dataset, b, args.n, args.ladder_batch, False
        )
        if not valid:
            print(f"[fatal] {detail}", file=sys.stderr)
            return 3

    print("### PHASE 2: greedy paired fidelity ###", flush=True)
    for b in (0, 4):
        lp = out / f"fidelity_b{b}_n{args.n}.log"
        print(f"[fidelity bits={b}] start {time.strftime('%H:%M:%S')}", flush=True)
        rc, dt, cmd = run_one(
            args.style, args.dataset, b, args.n, args.cap_batch, True, lp,
            args.python, out,
        )
        runs.append({"phase": "fidelity", "bits": b, "return_code": rc,
                     "seconds": round(dt, 1), "command": cmd, "log": str(lp)})
        print(f"[fidelity bits={b}] rc={rc}  {dt/60:.1f} min  acc={acc_from(lp)}", flush=True)
        if rc != 0:
            (out / "cell_manifest.json").write_text(json.dumps(runs, indent=2))
            print(f"[fatal] stopping after failed condition; inspect {lp}", file=sys.stderr)
            return rc
        valid, detail = validate_result(
            out, args.dataset, b, args.n, args.cap_batch, True
        )
        if not valid:
            print(f"[fatal] {detail}", file=sys.stderr)
            return 3

    manifest = {
        "style": args.style, "dataset": args.dataset, "n": args.n,
        "ladder_batch": args.ladder_batch, "capture_batch": args.cap_batch,
        "python_executable": args.python, "output_directory": str(out),
        "environment": environment_metadata(), "runs": runs,
    }
    (out / "cell_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"=== DONE {tag} {time.strftime('%H:%M:%S')} ===", flush=True)
    for b in (0, 8, 4, 2):
        print(f"  ladder   bits={b}: {acc_from(out / f'ladder_b{b}_n{args.n}.log')}")
    for b in (0, 4):
        print(f"  fidelity bits={b}: {acc_from(out / f'fidelity_b{b}_n{args.n}.log')}")
    print(f"  manifest: {out / 'cell_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
