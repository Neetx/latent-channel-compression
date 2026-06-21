#!/usr/bin/env python3
"""Local single-GPU backend for the fidelity_sweep experiment.

Primary execution backend. It runs on a local CUDA GPU where the upstream
RecursiveMAS source is cloned at ``external/RecursiveMAS`` and pinned to the tested
commit. The upstream clone is treated as read-only: this driver copies it into the
run directory and patches only that disposable working copy.

It reuses the unit-tested ``patch_run_py`` / ``patch_inference_mas`` from
``kernel_pkg/fidelity_kernel.py`` (loaded by path, so no package import is
needed), so the injected Variant B quantizer + Tier-2 logit capture are byte-for
-byte the same patches the cloud backends apply. Tests covering those functions:
``tests/test_fidelity_kernel.py``.

Flow:
  1. Verify the upstream commit and that tracked source files are clean.
  2. Copy the upstream source into the run directory (excluding .git/ and caches).
  3. Patch the copied run.py (sample cap, dtype, force-greedy, T default, --result_jsonl)
     and inference_mas.py (batch_size pin + Variant B head + adapter wrapping).
  4. Subprocess the copied ``run.py`` with VARIANT_B_BITS / CAPTURE_MODE / FIDELITY_SRC_ROOT
     / FIDELITY_WORK_DIR propagated into the child env.
  5. Collect artifacts and delete the disposable source copy.

Blackwell note: this machine is sm_120 with NATIVE bf16, so ``--dtype auto`` is
safe and is the default here (the cloud drivers force float32 only because the T4
is pre-Ampere — see docs REPORT_05). float32 sequential_light would be ~17 GB and
will NOT fit the 16 GB card; keep dtype at auto/bfloat16 locally.

Example (tiny dry-run, exercises the full path incl. the quantizer):
  python fidelity_local.py --bits 4 --t 3 --n-samples 4 --batch-size 2 \
      --topk 256 --maxpos 128
A paired comparison is two runs: --bits 0 (REF) then --bits 4 (INT4), identical
seed + greedy, so they pair per problem for analysis/analyze.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve()
SWEEP_DIR = THIS.parent.parent                 # experiments/fidelity_sweep
REPO_ROOT = SWEEP_DIR.parent.parent            # repo root (contains src/)
UPSTREAM_SOURCE = REPO_ROOT / "external" / "RecursiveMAS"
UPSTREAM_FILES = ["run.py", "inference_utils/inference_mas.py"]
EXPECTED_UPSTREAM_COMMIT = "f95d512017fb713e9ac519248fbfd3d270dafd68"


def _load_kernel():
    """Load the tested patch functions from kernel_pkg by file path."""
    path = SWEEP_DIR / "kernel_pkg" / "fidelity_kernel.py"
    spec = importlib.util.spec_from_file_location("fidelity_kernel", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_upstream_source() -> str:
    """Verify the external upstream without modifying it."""
    if not UPSTREAM_SOURCE.is_dir():
        raise RuntimeError(f"upstream not found: {UPSTREAM_SOURCE}")
    commit = subprocess.check_output(
        ["git", "-C", str(UPSTREAM_SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError(
            f"RecursiveMAS commit mismatch: got {commit}, expected {EXPECTED_UPSTREAM_COMMIT}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(UPSTREAM_SOURCE), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            "RecursiveMAS has tracked modifications; use a clean pinned clone. "
            "The local backend will not overwrite upstream files."
        )
    return commit


def copy_upstream_source(destination: Path) -> None:
    """Create a disposable source copy while leaving the upstream clone untouched."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        UPSTREAM_SOURCE,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo"),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local fidelity_sweep backend (single CUDA GPU).")
    p.add_argument("--bits", type=int, default=0, help="0 = REF (no quant), N>0 = Variant B N-bit")
    p.add_argument("--t", type=int, default=3, help="channel-traversal count = num_recursive_rounds")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--dataset", default="math500", choices=["math500", "medqa", "gpqa", "mbppplus"])
    p.add_argument("--style", default="sequential_light", choices=["sequential_light", "sequential_scaled"])
    p.add_argument("--dtype", default="auto", help="auto (Blackwell bf16) | bfloat16 | float32")
    p.add_argument("--topk", type=int, default=256, help="top-K logits captured per decode position")
    p.add_argument("--maxpos", type=int, default=128, help="cap on decode positions stored per generate()")
    p.add_argument("--links", default="all", choices=["all", "inner", "outer"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--out", default=str(Path.home() / "lcc" / "fid_out"))
    p.add_argument("--no-capture", action="store_true",
                   help="disable Tier-2 logit capture + greedy (sampled, lighter memory)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    capture = not args.no_capture

    if not (REPO_ROOT / "src").is_dir():
        print(f"[fatal] src/ not found under repo root: {REPO_ROOT}", file=sys.stderr)
        return 2

    try:
        upstream_commit = verify_upstream_source()
    except Exception as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        return 2

    k = _load_kernel()

    config_tag = f"{args.dataset}_vb{args.bits}_T{args.t}_n{args.n_samples}_b{args.batch_size}_{args.dtype}"
    work_dir = Path(args.out) / config_tag
    work_dir.mkdir(parents=True, exist_ok=True)
    upstream_work = work_dir / "_RecursiveMAS_work"
    jsonl_path = work_dir / f"per_problem_{config_tag}.jsonl"

    print(f"=== fidelity_local — {config_tag} ===")
    print(f"  repo_root      = {REPO_ROOT}")
    print(f"  upstream(src)  = {UPSTREAM_SOURCE} @ {upstream_commit[:8]} (read-only)")
    print(f"  upstream(work) = {upstream_work}")
    print(f"  work_dir       = {work_dir}")
    print(f"  HF_HOME        = {os.environ.get('HF_HOME', '(unset)')}")
    print(f"  bits={args.bits} T={args.t} n={args.n_samples} batch={args.batch_size} "
          f"dtype={args.dtype} capture={capture} links={args.links}")

    t0 = time.time()
    final_acc = None
    rc = -1
    try:
        # ---- [1/3] disposable source copy, then patch only the copy ----
        copy_upstream_source(upstream_work)
        run_py = upstream_work / "run.py"
        src, rc_counts = k.patch_run_py(
            run_py.read_text(encoding="utf-8"),
            n_samples=args.n_samples,
            dtype=args.dtype,
            num_recursive_rounds=args.t,
            capture_mode=capture,
            jsonl_path=str(jsonl_path),
        )
        run_py.write_text(src, encoding="utf-8")

        infer_mas = upstream_work / "inference_utils" / "inference_mas.py"
        isrc, im_counts = k.patch_inference_mas(
            infer_mas.read_text(encoding="utf-8"),
            batch_size=args.batch_size,
            dataset=args.dataset,
            style=args.style,
        )
        infer_mas.write_text(isrc, encoding="utf-8")
        print(f"  run.py patches: {rc_counts}")
        print(f"  inference_mas patches: {im_counts}")

        # ---- [2/3] child env (the injected head reads these in the subprocess) ----
        env = dict(os.environ)
        env.update({
            "MAS_FORCE_DISABLE_TORCHVISION": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "VARIANT_B_BITS": str(args.bits),
            "CAPTURE_MODE": "1" if capture else "0",
            "TOPK_LOGITS": str(args.topk),
            "MAX_LOGIT_POSITIONS": str(args.maxpos),
            "FIDELITY_WORK_DIR": str(work_dir),
            "FIDELITY_SRC_ROOT": str(REPO_ROOT),
            "VB_LINKS": args.links,
            "PYTHONDONTWRITEBYTECODE": "1",
        })

        cmd = [
            sys.executable, "run.py",
            "--style", args.style,
            "--dataset", args.dataset,
            "--seed", "42",
            "--num_recursive_rounds", str(args.t),
            "--trust_remote_code", "1",
            "--device", args.device,
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
        ]
        print(f"\n[run] cwd={upstream_work}")
        print(f"[run] {' '.join(cmd)}")
        print(f"[run] child: VARIANT_B_BITS={env['VARIANT_B_BITS']} CAPTURE_MODE={env['CAPTURE_MODE']} "
              f"FIDELITY_SRC_ROOT={env['FIDELITY_SRC_ROOT']}\n", flush=True)

        # ---- [3/3] run upstream, stream output, parse accuracy ----
        t_run = time.time()
        proc = subprocess.Popen(
            cmd, cwd=str(upstream_work),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, bufsize=1, text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            m = re.search(r"accuracy=([0-9.]+)%", line)
            if m:
                final_acc = float(m.group(1))
        proc.wait()
        rc = proc.returncode
        run_secs = time.time() - t_run
    finally:
        # Delete only our disposable copy. The external upstream is never written.
        if upstream_work.exists():
            shutil.rmtree(upstream_work, ignore_errors=True)

    # ---- collect child-dumped artifacts ----
    call_stats = None
    p = work_dir / "fidelity_call_stats.json"
    if p.is_file():
        call_stats = json.loads(p.read_text())

    per_problem = []
    if capture and jsonl_path.is_file():
        per_problem = k.parse_per_problem_jsonl(str(jsonl_path))
    n_correct = sum(1 for r in per_problem if isinstance(r.get("correct"), bool))

    n_logit_batches = 0
    lp = work_dir / "fidelity_logits.npz"
    if lp.is_file():
        try:
            import numpy as np
            with np.load(lp) as z:
                n_logit_batches = int(z["n_batches"]) if "n_batches" in z else 0
        except Exception:
            pass

    result = {
        "config": {
            "bits": args.bits, "t": args.t, "n_samples": args.n_samples,
            "batch_size": args.batch_size, "dataset": args.dataset, "dtype": args.dtype,
            "capture": capture, "links": args.links, "config_tag": config_tag,
            "seed": 42, "decoding": "greedy" if capture else "sampled",
            "upstream_commit": upstream_commit,
        },
        "final_accuracy": final_acc,
        "return_code": rc,
        "run_seconds": round(time.time() - t0, 1),
        "n_per_problem": len(per_problem),
        "n_per_problem_with_correct": n_correct,
        "n_logit_batches": n_logit_batches,
        "call_stats_present": call_stats is not None,
    }
    out_json = work_dir / f"fidelity_{config_tag}.json"
    out_json.write_text(json.dumps(result, indent=2))

    print(f"\n[result] rc={rc} final_acc={final_acc} "
          f"per_problem={len(per_problem)}({n_correct} w/correct) "
          f"logit_batches={n_logit_batches} call_stats={'yes' if call_stats else 'no'}")
    print(f"[done] wrote {out_json}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
