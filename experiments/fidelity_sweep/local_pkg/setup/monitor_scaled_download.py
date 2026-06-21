#!/usr/bin/env python3
"""Print a completion percentage for the sequential_scaled checkpoint download.

Computes % = (bytes on disk for the 4 scaled repos) / (tier total ~24.1 GB), every
30 s with an ASCII bar, until ~complete or the download process exits. Read-only.
"""
import subprocess
import time
from pathlib import Path

HUB = Path.home() / "lcc" / "hf-cache" / "hub"
DIRS = [
    HUB / "models--RecursiveMAS--Sequential-Scaled-Planner-Gemma3-4B",
    HUB / "models--RecursiveMAS--Sequential-Scaled-Critic-Llama3.2-3B",
    HUB / "models--RecursiveMAS--Sequential-Scaled-Solver-Qwen3.5-4B",
    HUB / "models--RecursiveMAS--Sequential-Scaled-Outerlinks",
]
TOTAL_GB = 24.14  # planner 8.69 + critic 6.52 + solver 8.48 + outer 0.45


def scaled_gb() -> float:
    tot = 0
    for d in DIRS:
        try:
            r = subprocess.run(["du", "-sb", str(d)], capture_output=True, text=True)
            if r.returncode == 0:
                tot += int(r.stdout.split()[0])
        except Exception:
            pass
    return tot / 1e9


def dl_running() -> bool:
    return subprocess.run(["pgrep", "-f", "[d]ownload_scaled.py"], capture_output=True).returncode == 0


while True:
    gb = scaled_gb()
    pct = min(100.0, gb / TOTAL_GB * 100)
    filled = int(pct / 5)
    bar = "#" * filled + "-" * (20 - filled)
    running = dl_running()
    print(f"[{time.strftime('%H:%M:%S')}] scaled [{bar}] {pct:5.1f}%  "
          f"({gb:4.1f}/{TOTAL_GB} GB)  {'downloading' if running else 'process ended'}", flush=True)
    if pct >= 99.0 or not running:
        state = "COMPLETE" if pct >= 99.0 else f"download process ended at {pct:.1f}%"
        print(f"[{time.strftime('%H:%M:%S')}] === {state} ===", flush=True)
        break
    time.sleep(30)
