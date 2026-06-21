#!/usr/bin/env python3
"""Download the sequential_scaled checkpoints into HF_HOME — self-healing.

Unauthenticated HF downloads throttle and stall on big shards. This wrapper sets a
short per-request timeout (so a hung connection RAISES instead of hanging forever) and
retries each repo with resume until it completes. Set HF_TOKEN for fast, stall-free
downloads (then this just succeeds on the first try). Total tier ~24 GB.
"""
import os

# Must be set before importing huggingface_hub (read at import time).
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "20"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # standard downloader: honours timeout+resume on flaky links

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4] / "external" / "RecursiveMAS"
sys.path.insert(0, str(REPO))
from huggingface_hub import snapshot_download
from hf_resolver import resolve_inner_adapter, resolve_outer_paths

REPOS = {
    "critic": "RecursiveMAS/Sequential-Scaled-Critic-Llama3.2-3B",
    "solver": "RecursiveMAS/Sequential-Scaled-Solver-Qwen3.5-4B",
    "outer": "RecursiveMAS/Sequential-Scaled-Outerlinks",
    "planner": "RecursiveMAS/Sequential-Scaled-Planner-Gemma3-4B",
}


def fetch(repo: str, attempts: int = 60) -> Path:
    for i in range(1, attempts + 1):
        try:
            return Path(snapshot_download(repo_id=repo, repo_type="model", max_workers=2)).resolve()
        except Exception as e:  # timeout / connection reset / throttle → resume on retry
            print(f"  [retry {i}/{attempts}] {type(e).__name__}: {str(e)[:140]} — resume in 5s", flush=True)
            time.sleep(5)
    raise RuntimeError(f"gave up on {repo} after {attempts} attempts")


paths = {}
for key, repo in REPOS.items():
    t0 = time.time()
    print(f"[download] {key}: {repo}", flush=True)
    paths[key] = fetch(repo)
    print(f"[done]     {key} in {time.time() - t0:.1f}s", flush=True)

for task in ("math", "code"):
    print(f"\n[validate task={task}]")
    for key in ("planner", "critic", "solver"):
        try:
            a = resolve_inner_adapter(paths[key], task)
            print(f"  inner {key:7s} -> exists={a.is_file()}")
        except Exception as e:
            print(f"  inner {key:7s} -> FAIL {type(e).__name__}")
    try:
        o = resolve_outer_paths(paths["outer"], task=task)
        print(f"  outer -> {{{', '.join(f'{k}:{v.is_file()}' for k, v in o.items())}}}")
    except Exception as e:
        print(f"  outer -> FAIL {type(e).__name__}: {e}")

print("\nALL_SCALED_DOWNLOADED")
