"""Security properties of the deliberation Python sandbox. Skipped where Docker (and the
lcc-pysandbox image) are unavailable, e.g. CI; run locally after building the image.

Run: .venv/bin/python -m pytest tests/test_py_sandbox.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PKG = ROOT / "experiments" / "fidelity_sweep" / "local_pkg"
sys.path.insert(0, str(LOCAL_PKG))
import py_sandbox  # noqa: E402


def _image_ready() -> bool:
    if not py_sandbox.docker_available():
        return False
    out = subprocess.run(["docker", "image", "inspect", py_sandbox.IMAGE],
                         capture_output=True)
    return out.returncode == 0


pytestmark = pytest.mark.skipif(not _image_ready(),
                                reason="docker daemon or lcc-pysandbox image not available")


def test_benign_runs():
    assert py_sandbox.run_python_sandboxed("print(2 + 2)", timeout=6).strip() == "4"


def test_host_env_not_leaked():
    os.environ["FAKE_SECRET_PROBE"] = "tvly-SHOULD-NOT-LEAK"
    out = py_sandbox.run_python_sandboxed(
        "import os; print(os.environ.get('FAKE_SECRET_PROBE', 'NONE'))", timeout=6)
    assert "SHOULD-NOT-LEAK" not in out and "NONE" in out


def test_network_blocked():
    out = py_sandbox.run_python_sandboxed(
        "import urllib.request; urllib.request.urlopen('http://example.com', timeout=4)", timeout=6)
    assert "[exit_code]" in out  # name resolution / connection fails with no network


def test_rootfs_read_only():
    out = py_sandbox.run_python_sandboxed("open('/work/x', 'w')", timeout=6)
    assert "Read-only file system" in out


def test_timeout_enforced():
    out = py_sandbox.run_python_sandboxed("while True:\n    pass", timeout=4)
    assert "[timeout]" in out
