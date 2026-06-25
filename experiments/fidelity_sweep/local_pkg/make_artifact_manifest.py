#!/usr/bin/env python3
"""Checksum + environment manifest for the raw capture artifacts backing the paper.

The raw NPZ captures are too large to commit, so this records a SHA256 of every
``fidelity_logits.npz``, ``fidelity_call_stats.json``, and result JSON under the capture roots,
together with each condition's provenance (config tag, links, quantizer seed, teacher-forced flag)
and the runtime environment. The committed manifest lets a reproduction verify that regenerated
artifacts match byte-for-byte (or flag where they diverge). No secrets or personal paths are
recorded: paths are stored relative to their capture root, keyed by the root's basename.

Run: python make_artifact_manifest.py [--roots DIR ...] [--out results/artifact_manifest.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

LOCAL_PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_PKG))
from run_cell import environment_metadata  # reuse the audited env capture


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def provenance(p: Path) -> dict:
    """Pull a few non-secret provenance fields from a result JSON, if present."""
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    c = d.get("config", {})
    return {k: c.get(k) for k in ("config_tag", "links", "quantizer_seed", "teacher_forced",
                                  "upstream_commit") if k in c} | {
        "final_accuracy": d.get("final_accuracy"),
        "n_logit_batches": d.get("n_logit_batches"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    run_root = Path(os.environ.get("LCC_RUN_ROOT", Path.home() / "lcc" / "runs"))
    ap.add_argument("--roots", nargs="+", default=[str(run_root), str(Path.home() / "lcc" / "fid_out")])
    ap.add_argument("--out", default=str(LOCAL_PKG / "results" / "artifact_manifest.json"))
    args = ap.parse_args()

    artifacts = []
    for root_s in args.roots:
        root = Path(root_s)
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            name = p.name
            is_npz = name == "fidelity_logits.npz"
            is_json = name.startswith("fidelity_") and name.endswith(".json")
            if not (is_npz or is_json):
                continue
            entry = {
                "root": root.name,
                "path": str(p.relative_to(root)).replace(os.sep, "/"),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
            if is_json and name != "fidelity_call_stats.json":
                entry["provenance"] = provenance(p)
            artifacts.append(entry)

    manifest = {
        "description": "SHA256 + provenance of raw capture artifacts for the paper (not committed; regenerable).",
        "n_artifacts": len(artifacts),
        "n_npz": sum(a["path"].endswith(".npz") for a in artifacts),
        "environment": environment_metadata(),
        "artifacts": artifacts,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    print(f"wrote {out}  ({len(artifacts)} artifacts, {manifest['n_npz']} NPZ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
