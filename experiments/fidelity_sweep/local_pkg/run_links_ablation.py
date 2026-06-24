#!/usr/bin/env python3
"""Resumable orchestrator for the MBPP+ inner/outer link-compression ablation.

The headline runs quantize *all* inter-agent links together. This orchestrator localizes
the trajectory effect by quantizing exactly one link type at a time: it runs only the INT4
(bits=4) greedy capture for each (tier, links) with links in {inner, outer}, on MBPP+
light and scaled. The greedy REF (bits=0) is quantizer-independent — no quantizer runs at
full precision — so the existing seed-42 ``all``-links REF captures pair with both the
inner-only and outer-only INT4 captures and are NOT re-run here. Any (tier, links) whose
valid result JSON already exists is skipped, so the ablation survives interruptions/reboots
(re-launch to continue). A ``LINKS_ABLATION_DONE`` marker is written on completion (a
lightweight observer can watch for it).

Each condition is an isolated ``fidelity_local.py`` subprocess (fresh CUDA context). A
single-instance lock on the output dir refuses a second concurrent orchestrator, which
would contend for VRAM and corrupt shared captures. Launch detached for a multi-hour run:

  setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 \\
    .venv/bin/python experiments/fidelity_sweep/local_pkg/run_links_ablation.py \\
    > $LCC_RUN_ROOT/links_ablation/orchestrator.out 2>&1 < /dev/null &
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
from run_cell import claim_cell_lock, release_cell_lock  # noqa: E402  (battle-tested lock)
from fidelity_local import build_config_tag  # noqa: E402  (shared single source of truth for tags)

TIERS = {"light": ("sequential_light", 2), "scaled": ("sequential_scaled", 1)}


def cond_tag(dataset: str, batch: int, links: str, n: int) -> str:
    """Tag for a seed-42 INT4 single-link capture, via the driver's own builder so the
    orchestrator and fidelity_local can never disagree on the output directory."""
    return build_config_tag(dataset, 4, 3, n, batch, "auto", quantizer_seed=42, links=links)


def valid_result(out: Path, dataset: str, batch: int, links: str, n: int) -> bool:
    tag = cond_tag(dataset, batch, links, n)
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
    ap.add_argument("--links", nargs="+", default=["inner", "outer"],
                    choices=["inner", "outer"])
    ap.add_argument("--tiers", nargs="+", default=["light", "scaled"], choices=list(TIERS))
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--out", default=None)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    run_root = Path(os.environ.get("LCC_RUN_ROOT", Path.home() / "lcc" / "runs"))
    out = Path(args.out) if args.out else (run_root / "links_ablation")
    out.mkdir(parents=True, exist_ok=True)
    lock = claim_cell_lock(out)
    if lock is None:
        print(f"[abort] another links-ablation orchestrator is already active for {out} "
              f"(lock {out / '.run_cell.lock'}); refusing to double-run.", file=sys.stderr)
        return 4
    atexit.register(release_cell_lock, lock)

    # tiers outer / links inner: finishes the cheap light tier (inner+outer) before scaled.
    plan = [(t, l) for t in args.tiers for l in args.links]
    print(f"=== LINKS ABLATION {args.dataset} INT4  tiers={args.tiers} links={args.links}  "
          f"{len(plan)} runs  out={out}  {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    summary = []
    for tier, links in plan:
        style, batch = TIERS[tier]
        label = f"{tier} {links} b{batch}"
        if valid_result(out, args.dataset, batch, links, args.n):
            print(f"[skip] {label} — valid result exists", flush=True)
            summary.append({"cond": label, "status": "reused"})
            continue
        log = out / f"int4_{tier}_{links}.log"
        cmd = [args.python, str(DRIVER), "--style", style, "--dataset", args.dataset,
               "--bits", "4", "--links", links, "--n-samples", str(args.n),
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
        if not valid_result(out, args.dataset, batch, links, args.n):
            print(f"[fatal] {label} produced an invalid/incomplete result", file=sys.stderr)
            return 3
        summary.append({"cond": label, "status": f"{dt:.1f}min"})

    (out / "LINKS_ABLATION_DONE").write_text(json.dumps(
        {"dataset": args.dataset, "tiers": args.tiers, "links": args.links,
         "completed": summary, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print(f"=== LINKS ABLATION DONE {time.strftime('%H:%M:%S')} -> "
          f"{out / 'LINKS_ABLATION_DONE'} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
