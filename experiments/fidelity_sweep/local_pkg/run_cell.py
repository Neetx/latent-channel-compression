#!/usr/bin/env python3
"""Generic cell orchestrator: run the bit-rate ladder + greedy paired fidelity for any
(style, dataset) cell of the RecursiveMAS × Variant B matrix.

Each condition runs as a separate `fidelity_local.py` subprocess (fresh CUDA context,
crash isolation; the driver instruments a disposable upstream copy, never the read-only
clone). The scientific logic lives in the driver; this is a portable, repeatable run
recipe. Generalises run_step2_scaled_mbppplus.py.

  python run_cell.py --style sequential_light --dataset medqa --ladder-batch 16 --cap-batch 2 --n 250

`--resume` skips any condition that already has a valid result JSON, so an interrupted
multi-hour cell (e.g. after a reboot) continues from the first incomplete condition
instead of recomputing finished ones. Skipping is safe because every condition is
deterministic (seed 42, fixed decoding); a reused condition is byte-identical to a rerun.
"""
from __future__ import annotations

import argparse
import atexit
import errno
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


def qs_suffix(quantizer_seed: int) -> str:
    """Tag/log suffix for a non-default quantizer rotation. Seed 42 stays unsuffixed so
    the original results and analyzers keep resolving unchanged."""
    return "" if quantizer_seed == 42 else f"_qs{quantizer_seed}"


def validate_result(out: Path, dataset: str, bits: int, n: int, batch: int, capture: bool,
                    quantizer_seed: int = 42):
    """Validate the machine-readable contract emitted by one completed condition."""
    tag = f"{dataset}_vb{bits}_T3_n{n}_b{batch}_auto{qs_suffix(quantizer_seed)}"
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


def _pid_alive(pid: int) -> bool:
    """True if a process with this PID currently exists (POSIX/WSL)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


def claim_cell_lock(out: Path) -> Path | None:
    """Atomically claim a cell's output dir so two orchestrators cannot run it at once.
    A double-run writes the SAME shared captures (NPZ/JSONL/call_stats) and contends for
    VRAM, silently corrupting results — this guard prevents that. A stale lock whose
    recorded PID is gone (e.g. after a reboot or kill) is reclaimed. Returns the lock
    path on success, or None if a live instance already holds it."""
    lock = out / ".run_cell.lock"
    for _ in range(2):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return lock
        except FileExistsError:
            try:
                holder = int(lock.read_text().strip() or "0")
            except Exception:
                holder = 0
            if holder and _pid_alive(holder):
                return None
            try:
                lock.unlink()  # stale: the recorded PID is gone, reclaim it
            except FileNotFoundError:
                pass
    return None


def release_cell_lock(lock: Path) -> None:
    """Remove the lock iff we still own it (best-effort; stale locks self-heal anyway)."""
    try:
        if int(lock.read_text().strip() or "0") == os.getpid():
            lock.unlink()
    except Exception:
        pass


def run_one(style, dataset, bits, n, batch, capture, logpath, py, out, quantizer_seed=42):
    cmd = [py, str(DRIVER), "--style", style, "--dataset", dataset, "--bits", str(bits),
           "--t", "3", "--n-samples", str(n), "--batch-size", str(batch),
           "--quantizer-seed", str(quantizer_seed), "--out", str(out)]
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
    ap.add_argument("--resume", action="store_true",
                    help="skip conditions whose valid result JSON already exists "
                         "(continue an interrupted cell without recomputing finished conditions)")
    ap.add_argument("--quantizer-seed", type=int, default=42,
                    help="quantizer rotation seed for the whole cell (distinct from the "
                         "generation seed); 42 keeps the original tags and log names")
    args = ap.parse_args()

    sfx = qs_suffix(args.quantizer_seed)
    tag = f"{args.style}_{args.dataset}"
    run_root = Path(os.environ.get("LCC_RUN_ROOT", Path.home() / "lcc" / "runs"))
    out = Path(args.out) if args.out else (run_root / tag)
    out.mkdir(parents=True, exist_ok=True)
    lock = claim_cell_lock(out)
    if lock is None:
        print(f"[abort] another run_cell instance is already active for {out} "
              f"(lock {out / '.run_cell.lock'}); refusing to double-run, which would "
              f"corrupt the shared captures and contend for VRAM.", file=sys.stderr)
        return 4
    atexit.register(release_cell_lock, lock)
    runs = []

    print(f"=== CELL {tag}  n={args.n}  ladder_b={args.ladder_batch}  cap_b={args.cap_batch}  "
          f"qseed={args.quantizer_seed}  {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)

    print("### PHASE 1: sampled ladder ###", flush=True)
    for b in (0, 8, 4, 2):
        lp = out / f"ladder_b{b}_n{args.n}{sfx}.log"
        if args.resume:
            valid, detail = validate_result(
                out, args.dataset, b, args.n, args.ladder_batch, False, args.quantizer_seed
            )
            if valid:
                runs.append({"phase": "ladder", "bits": b, "return_code": 0,
                             "reused": True, "log": str(lp)})
                print(f"[ladder bits={b}] reuse valid result, skip ({detail})", flush=True)
                continue
        print(f"[ladder bits={b}] start {time.strftime('%H:%M:%S')}", flush=True)
        rc, dt, cmd = run_one(
            args.style, args.dataset, b, args.n, args.ladder_batch, False, lp,
            args.python, out, args.quantizer_seed,
        )
        runs.append({"phase": "ladder", "bits": b, "return_code": rc,
                     "seconds": round(dt, 1), "command": cmd, "log": str(lp)})
        print(f"[ladder bits={b}] rc={rc}  {dt/60:.1f} min  acc={acc_from(lp)}", flush=True)
        if rc != 0:
            (out / "cell_manifest.json").write_text(json.dumps(runs, indent=2))
            print(f"[fatal] stopping after failed condition; inspect {lp}", file=sys.stderr)
            return rc
        valid, detail = validate_result(
            out, args.dataset, b, args.n, args.ladder_batch, False, args.quantizer_seed
        )
        if not valid:
            print(f"[fatal] {detail}", file=sys.stderr)
            return 3

    print("### PHASE 2: greedy paired fidelity ###", flush=True)
    for b in (0, 4):
        lp = out / f"fidelity_b{b}_n{args.n}{sfx}.log"
        if args.resume:
            valid, detail = validate_result(
                out, args.dataset, b, args.n, args.cap_batch, True, args.quantizer_seed
            )
            if valid:
                runs.append({"phase": "fidelity", "bits": b, "return_code": 0,
                             "reused": True, "log": str(lp)})
                print(f"[fidelity bits={b}] reuse valid result, skip ({detail})", flush=True)
                continue
        print(f"[fidelity bits={b}] start {time.strftime('%H:%M:%S')}", flush=True)
        rc, dt, cmd = run_one(
            args.style, args.dataset, b, args.n, args.cap_batch, True, lp,
            args.python, out, args.quantizer_seed,
        )
        runs.append({"phase": "fidelity", "bits": b, "return_code": rc,
                     "seconds": round(dt, 1), "command": cmd, "log": str(lp)})
        print(f"[fidelity bits={b}] rc={rc}  {dt/60:.1f} min  acc={acc_from(lp)}", flush=True)
        if rc != 0:
            (out / "cell_manifest.json").write_text(json.dumps(runs, indent=2))
            print(f"[fatal] stopping after failed condition; inspect {lp}", file=sys.stderr)
            return rc
        valid, detail = validate_result(
            out, args.dataset, b, args.n, args.cap_batch, True, args.quantizer_seed
        )
        if not valid:
            print(f"[fatal] {detail}", file=sys.stderr)
            return 3

    manifest = {
        "style": args.style, "dataset": args.dataset, "n": args.n,
        "ladder_batch": args.ladder_batch, "capture_batch": args.cap_batch,
        "quantizer_seed": args.quantizer_seed, "generation_seed": 42,
        "python_executable": args.python, "output_directory": str(out),
        "environment": environment_metadata(), "runs": runs,
    }
    (out / "cell_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"=== DONE {tag} {time.strftime('%H:%M:%S')} ===", flush=True)
    for b in (0, 8, 4, 2):
        print(f"  ladder   bits={b}: {acc_from(out / f'ladder_b{b}_n{args.n}{sfx}.log')}")
    for b in (0, 4):
        print(f"  fidelity bits={b}: {acc_from(out / f'fidelity_b{b}_n{args.n}{sfx}.log')}")
    print(f"  manifest: {out / 'cell_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
