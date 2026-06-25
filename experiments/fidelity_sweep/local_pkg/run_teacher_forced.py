#!/usr/bin/env python3
"""Resumable orchestrator for the teacher-forced (aligned) mechanism capture on MBPP+.

For each tier it produces ONE INT4 teacher-forced capture (bits=4, --teacher-forced), forcing
decoding along the paired full-precision REF tokens while the channel stays quantized, so the
per-position REF-vs-INT4 distributions are aligned (no post-divergence confound). The REF that
supplies the teacher tokens (and is itself the TF-REF reference, identical to the capture by
gate G0) is reused when a b=1 REF already exists (scaled's committed b=1 REF), else generated.

Teacher-forced captures run at **batch_size=1**: gate G0 reproduces the free-running REF exactly
at b=1, whereas batched (b>1) generation diverges in its post-EOS padding dynamics under forcing.
Each tier writes to its own sub-dir (light/, scaled/) because the b=1 config tag is identical
across tiers. Any (tier) whose valid TF result exists is skipped (resumable). A
``TEACHER_FORCED_DONE`` marker is written on completion. Launch detached for a multi-hour run:

  setsid env PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 \\
    .venv/bin/python experiments/fidelity_sweep/local_pkg/run_teacher_forced.py \\
    > $LCC_RUN_ROOT/teacher_forced/orchestrator.out 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LOCAL_PKG = Path(__file__).resolve().parent
DRIVER = LOCAL_PKG / "fidelity_local.py"
REPO_ROOT = LOCAL_PKG.parents[2]
sys.path.insert(0, str(LOCAL_PKG))
from run_cell import claim_cell_lock, release_cell_lock  # noqa: E402

TIERS = {"light": "sequential_light", "scaled": "sequential_scaled"}
BATCH = 1  # see module docstring: G0 holds only at b=1


def _valid(path: Path, n: int, want_call_stats: bool) -> bool:
    if not path.is_file():
        return False
    try:
        r = json.loads(path.read_text())
    except Exception:
        return False
    ok = (r.get("return_code") == 0 and r.get("final_accuracy") is not None
          and r.get("n_per_problem") == n and r.get("n_logit_batches", 0) > 0)
    return ok and (r.get("call_stats_present") if want_call_stats else True)


def _run(python, style, dataset, n, bits, out_dir, log, extra):
    cmd = [python, str(DRIVER), "--style", style, "--dataset", dataset, "--bits", str(bits),
           "--t", "3", "--n-samples", str(n), "--batch-size", str(BATCH),
           "--out", str(out_dir)] + extra
    print(f"[run] {' '.join(cmd[3:])}\n      start {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    with log.open("w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT)).returncode
    print(f"[run] rc={rc}  {(time.time()-t0)/60:.1f} min", flush=True)
    return rc


def resolve_ref_npz(tier, out_tier, n, committed):
    """b=1 full-precision REF npz that supplies the teacher tokens. Reuse a committed b=1 REF
    for this tier (scaled), else a fresh one already in out_tier; the caller generates it if None."""
    cr = committed.get(tier)
    if cr and Path(cr).is_file():
        return Path(cr), "committed"
    local = out_tier / f"mbppplus_vb0_T3_n{n}_b{BATCH}_auto"
    if _valid(local / f"fidelity_mbppplus_vb0_T3_n{n}_b{BATCH}_auto.json", n, False) and \
            (local / "fidelity_logits.npz").is_file():
        return local / "fidelity_logits.npz", "reused-local"
    return None, "missing"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="mbppplus")
    ap.add_argument("--tiers", nargs="+", default=["light", "scaled"], choices=list(TIERS))
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--out", default=None)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    run_root = Path(os.environ.get("LCC_RUN_ROOT", Path.home() / "lcc" / "runs"))
    out = Path(args.out) if args.out else (run_root / "teacher_forced")
    out.mkdir(parents=True, exist_ok=True)
    lock = claim_cell_lock(out)
    if lock is None:
        print(f"[abort] another teacher-forced orchestrator is active for {out}", file=sys.stderr)
        return 4
    atexit.register(release_cell_lock, lock)

    # committed b=1 REFs that can be reused as teacher sources (verified per tier by construction)
    committed = {"scaled": Path.home() / "lcc" / "fid_out"
                 / f"mbppplus_vb0_T3_n{args.n}_b{BATCH}_auto" / "fidelity_logits.npz"}

    print(f"=== TEACHER-FORCED {args.dataset} INT4 b={BATCH}  tiers={args.tiers}  out={out}  "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    summary = []
    for tier in args.tiers:
        style = TIERS[tier]
        out_tier = out / tier
        out_tier.mkdir(parents=True, exist_ok=True)
        tf_json = (out_tier / f"mbppplus_vb4_T3_n{args.n}_b{BATCH}_auto_tf"
                   / f"fidelity_mbppplus_vb4_T3_n{args.n}_b{BATCH}_auto_tf.json")
        if _valid(tf_json, args.n, True):
            print(f"[skip] {tier}: valid TF result exists", flush=True)
            summary.append({"tier": tier, "status": "reused"}); continue

        ref_npz, how = resolve_ref_npz(tier, out_tier, args.n, committed)
        if ref_npz is None:
            print(f"[ref] {tier}: generating fresh b=1 REF", flush=True)
            if _run(args.python, style, args.dataset, args.n, 0, out_tier,
                    out_tier / "ref_b1.log", []) != 0:
                print(f"[fatal] {tier} REF failed", file=sys.stderr); return 3
            ref_npz, how = resolve_ref_npz(tier, out_tier, args.n, committed)
            if ref_npz is None:
                print(f"[fatal] {tier} REF produced no valid npz", file=sys.stderr); return 3
        print(f"[ref] {tier}: teacher tokens from {ref_npz} ({how})", flush=True)

        if _run(args.python, style, args.dataset, args.n, 4, out_tier,
                out_tier / "tf_int4.log",
                ["--teacher-forced", "--tf-ref-npz", str(ref_npz)]) != 0:
            print(f"[fatal] {tier} TF-INT4 failed", file=sys.stderr); return 3
        if not _valid(tf_json, args.n, True):
            print(f"[fatal] {tier} TF-INT4 invalid result", file=sys.stderr); return 3
        summary.append({"tier": tier, "status": "done", "ref": how})

    (out / "TEACHER_FORCED_DONE").write_text(json.dumps(
        {"dataset": args.dataset, "tiers": args.tiers, "batch": BATCH,
         "completed": summary, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print(f"=== TEACHER-FORCED DONE {time.strftime('%H:%M:%S')} -> {out/'TEACHER_FORCED_DONE'} ===",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
