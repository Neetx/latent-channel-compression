import os
import sys
import time
from pathlib import Path

# Make upstream resolvers importable
REPO = Path(__file__).resolve().parents[4] / "external" / "RecursiveMAS"
sys.path.insert(0, str(REPO))

from huggingface_hub import snapshot_download
from hf_resolver import resolve_inner_adapter, resolve_outer_paths, task_for_inner_repo

print("HF_HOME =", os.environ.get("HF_HOME"), flush=True)

REPOS = {
    "planner": "RecursiveMAS/Sequential-Light-Planner-Qwen3-1.7B",
    "critic": "RecursiveMAS/Sequential-Light-Critic-Llama3.2-1B",
    "solver": "RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B",
    "outer": "RecursiveMAS/Sequential-Light-Outerlinks",
}

paths = {}
for key, repo in REPOS.items():
    t0 = time.time()
    print(f"[download] {key}: {repo} ...", flush=True)
    resolved = snapshot_download(repo_id=repo, repo_type="model")
    paths[key] = Path(resolved).resolve()
    print(f"[done]     {key} in {time.time()-t0:.1f}s -> {paths[key]}", flush=True)

# Validate task-scoped adapter/outer resolution for math500 (task='math')
task = task_for_inner_repo("math500")
print(f"\n[validate] task={task}", flush=True)
for key in ["planner", "critic", "solver"]:
    a = resolve_inner_adapter(paths[key], task)
    print(f"  inner {key:8s} adapter -> {a.name}  exists={a.is_file()}")
outer = resolve_outer_paths(paths["outer"], task=task)
for k, v in outer.items():
    print(f"  outer {k:8s} -> {v.name}  exists={v.is_file()}")

print("\nALL_OK", flush=True)
