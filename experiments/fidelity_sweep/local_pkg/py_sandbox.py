#!/usr/bin/env python3
"""Run untrusted (model-generated) Python in a locked-down ephemeral Docker container.

The RecursiveMAS deliberation toolcaller executes ``<python>`` blocks the model writes. Upstream
runs them with a bare ``subprocess`` + 10 s timeout and the full host environment/permissions --
fine for benign benchmark code, but not a security boundary, and on our machine the host env holds
API keys. This drop-in replacement runs each block in an ephemeral container with: no network,
read-only rootfs (writes only to a small tmpfs ``/tmp``), NO host environment (so secrets cannot
leak), unprivileged ``nobody`` user, all capabilities dropped, ``no-new-privileges``, memory/cpu/
pids limits, and a wall-clock timeout that force-removes a lingering container. It keeps upstream's
prelude (math/itertools/functools/fractions/statistics/sympy) and output formatting.

Build the image once:  docker build -t lcc-pysandbox experiments/fidelity_sweep/local_pkg/sandbox
Self-test:             python py_sandbox.py
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import uuid

IMAGE = os.environ.get("LCC_PYSANDBOX_IMAGE", "lcc-pysandbox")
PRELUDE = (
    "import math\nimport itertools\nimport functools\nimport fractions\nimport statistics\n"
    "try:\n    import sympy as sp\nexcept Exception:\n    sp = None\n\n"
)


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "\n[...truncated...]"


def run_python_sandboxed(code: str, timeout: float = 10.0, max_chars: int = 4000) -> str:
    workdir = tempfile.mkdtemp(prefix="lcc_pybox_")
    os.chmod(workdir, 0o755)                        # so the container's 'nobody' can traverse in
    script = os.path.join(workdir, "script.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(PRELUDE); f.write(code); f.write("\n")
    os.chmod(script, 0o644)
    name = f"lcc_py_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    cmd = [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=64m",
        "--memory", "512m", "--memory-swap", "512m",
        "--cpus", "1.0", "--pids-limit", "64",
        "--user", "65534:65534",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--env", "HOME=/tmp",                       # only this; NO host env is inherited
        "-v", f"{workdir}:/work:ro", "--workdir", "/work",
        IMAGE, "python", "/work/script.py",
    ]
    try:
        out = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        pieces = []
        if out.stdout:
            pieces.append(out.stdout.rstrip())
        if out.stderr:
            pieces.append("[stderr]\n" + out.stderr.rstrip())
        if out.returncode != 0:
            pieces.append(f"[exit_code] {out.returncode}")
        return truncate("\n".join(pieces).strip() or "[no output]", max_chars)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        return f"[timeout] Python tool exceeded {timeout} seconds."
    finally:
        try:
            os.unlink(script); os.rmdir(workdir)
        except OSError:
            pass


if __name__ == "__main__":
    if not docker_available():
        raise SystemExit("docker daemon not available; start it and build lcc-pysandbox first.")
    os.environ["FAKE_SECRET"] = "tvly-SHOULD-NOT-LEAK"   # probe: must not reach the container
    probes = [
        ("benign arithmetic", "print(2+2)"),
        ("sympy available", "print(sp.simplify('x + x'))"),
        ("env scrubbed (must be NONE)",
         "import os; print('SECRET=', os.environ.get('FAKE_SECRET','NONE'))"),
        ("network blocked (must fail)",
         "import urllib.request; print(urllib.request.urlopen('http://example.com', timeout=4).status)"),
        ("write outside /tmp (must fail)", "open('/work/x','w').write('hi'); print('WROTE')"),
        ("write /tmp (allowed)", "open('/tmp/x','w').write('hi'); print('wrote /tmp ok')"),
        ("timeout (must time out)", "while True:\n    pass"),
    ]
    for label, code in probes:
        print(f"\n### {label}\n{run_python_sandboxed(code, timeout=6.0)}")
