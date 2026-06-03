#!/usr/bin/env python3
"""Verify downloaded experiment artifacts and optionally write SHA256 manifest.

Checks performed:
- every JSON file parses;
- every NPZ file is a valid ZIP archive and passes CRC for every member;
- SHA256 is computed for every file under the requested roots.

Example:
    .venv/bin/python bin/verify_artifacts.py /tmp/fid_outputs \
        --manifest /tmp/fid_outputs/SHA256SUMS.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactStatus:
    path: str
    size_bytes: int
    sha256: str
    kind: str
    ok: bool
    error: str | None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_json(path: Path) -> str | None:
    try:
        json.loads(path.read_text())
    except Exception as exc:
        return str(exc)
    return None


def check_npz(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
    except Exception as exc:
        return str(exc)
    if bad is not None:
        return f"bad zip member: {bad}"
    return None


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".npz":
        return "npz"
    return "file"


def verify_file(path: Path) -> ArtifactStatus:
    kind = classify(path)
    error = None
    if kind == "json":
        error = check_json(path)
    elif kind == "npz":
        error = check_npz(path)
    return ArtifactStatus(
        path=str(path),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        kind=kind,
        ok=error is None,
        error=error,
    )


def iter_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(p for p in root.rglob("*") if p.is_file())
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path,
                        help="artifact files or directories to verify")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="write JSON manifest to this path")
    args = parser.parse_args()

    statuses = [verify_file(path) for path in iter_files(args.paths)]
    for status in statuses:
        label = "OK" if status.ok else f"BAD: {status.error}"
        print(f"{label}\t{status.kind}\t{status.size_bytes}\t{status.sha256}\t{status.path}")

    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps([asdict(s) for s in statuses], indent=2))
        print(f"wrote manifest: {args.manifest}")

    return 0 if all(s.ok for s in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
