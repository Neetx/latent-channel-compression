import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[4] / "external" / "RecursiveMAS"
sys.path.insert(0, str(REPO))
from huggingface_hub import snapshot_download
from hf_resolver import resolve_inner_adapter, resolve_outer_paths, task_for_inner_repo

print("task(mbppplus) =", task_for_inner_repo("mbppplus"))
inner = {
    "planner": "RecursiveMAS/Sequential-Light-Planner-Qwen3-1.7B",
    "critic": "RecursiveMAS/Sequential-Light-Critic-Llama3.2-1B",
    "solver": "RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B",
}
for k, r in inner.items():
    d = Path(snapshot_download(r))
    man = d / "innerlink_config.json"
    tasks = json.loads(man.read_text()).get("tasks", {}) if man.is_file() else {}
    print(f"{k}: manifest tasks = {list(tasks.keys())}")
    for t in ("math", "code"):
        try:
            a = resolve_inner_adapter(d, t)
            print(f"   {t}: exists={a.is_file()}  ({a.name[:24]}...)")
        except Exception as e:
            print(f"   {t}: RESOLVE FAIL {type(e).__name__}: {e}")

d = Path(snapshot_download("RecursiveMAS/Sequential-Light-Outerlinks"))
data = json.loads((d / "outerlink_config.json").read_text())
print("outer manifest top-keys:", list(data.keys()))
if "tasks" in data:
    print("outer tasks:", list(data["tasks"].keys()))
for t in ("math", "code"):
    try:
        o = resolve_outer_paths(d, task=t)
        print(f"outer {t}: {{{', '.join(f'{kk}:{v.is_file()}' for kk, v in o.items())}}}")
    except Exception as e:
        print(f"outer {t}: RESOLVE FAIL {type(e).__name__}: {e}")
