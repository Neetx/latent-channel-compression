"""Phase 0.C diagnostic v4 — Solver ONLY on math500 (no RecursiveMAS pipeline).

v3 fixed the torchvision crash but introduced a NEW crash:
`AttributeError: 'list' object has no attribute 'keys'` at
`transformers.tokenization_utils_base._set_model_specific_special_tokens`.
Root cause: we pinned `transformers==4.50.0` which has a bug with the
Solver checkpoint's `tokenizer_config.json` (where `extra_special_tokens`
is a list, not a dict).

v4 fix: replicate our `baseline.py` install pattern verbatim — install
torch==2.4.1+cu121, then install upstream's full `requirements.txt`
*minus* the torch line (so we pick up `transformers==5.3.0`,
`datasets==4.4.2`, `huggingface-hub==1.7.1`, etc. exactly as the paper
specifies). This is the install pattern that produced the suspect 30%
baseline — so we know it works to the point of returning numbers; the
question is what number it returns when we strip the multi-agent dance.

Still applies from v3:
1. MAS_FORCE_DISABLE_TORCHVISION=1 + importlib.find_spec patch
2. Upstream compare_answers (4-strategy match: intpart, latex_text,
   nospace, digits)
3. Upstream SYSTEM_PROMPT
"""
import json
import os
import re
import subprocess
import sys
import time

T0 = time.time()
N_SAMPLES = int(os.environ.get("N_SAMPLES", "100"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2000"))
DO_SAMPLE = os.environ.get("DO_SAMPLE", "0") == "1"  # default greedy for cleanest signal
RECURSIVEMAS_COMMIT = os.environ.get(
    "RECURSIVEMAS_COMMIT",
    "f95d512017fb713e9ac519248fbfd3d270dafd68",
)

print(f"=== Single-agent Solver diagnostic v4 ===")
print(f"  N_SAMPLES={N_SAMPLES}  MAX_NEW_TOKENS={MAX_NEW_TOKENS}  DO_SAMPLE={DO_SAMPLE}")


def run(cmd):
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    return subprocess.check_call(cmd)


# ------- PATCH 1: install ENV + install -------
#
# Set the upstream env var BEFORE any transformers/torch import. This is
# checked at inference_mas.py:25 to gate the importlib monkey-patch.
os.environ["MAS_FORCE_DISABLE_TORCHVISION"] = "1"

print("\n[1a/5] install torch 2.4.1+cu121")
t = time.time()
run([sys.executable, "-m", "pip", "install", "-q",
     "torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cu121"])
print(f"  done in {time.time() - t:.1f}s")

# Clone upstream repo first so we can use its exact requirements.txt
# (pins transformers==5.3.0, datasets==4.4.2, etc.) AND import its
# answer_utils + SYSTEM_PROMPT in step [3/5].
UPSTREAM_DIR = "/kaggle/working/RecursiveMAS"
print(f"\n[1b/5] clone upstream RecursiveMAS")
t = time.time()
if not os.path.isdir(UPSTREAM_DIR):
    run(["git", "clone", "https://github.com/RecursiveMAS/RecursiveMAS.git", UPSTREAM_DIR])
run(["git", "-C", UPSTREAM_DIR, "checkout", RECURSIVEMAS_COMMIT])
print(f"  done in {time.time() - t:.1f}s")

print(f"\n[1c/5] install upstream requirements.txt (skip torch line)")
t = time.time()
req_in = f"{UPSTREAM_DIR}/requirements.txt"
req_out = "/kaggle/working/_requirements_no_torch.txt"
with open(req_in) as f, open(req_out, "w") as g:
    for line in f:
        if line.strip().lower().startswith("torch"):
            continue
        g.write(line)
run([sys.executable, "-m", "pip", "install", "-q", "-r", req_out])
print(f"  done in {time.time() - t:.1f}s")


# ------- PATCH 1b: importlib monkey-patch -------
#
# Even with MAS_FORCE_DISABLE_TORCHVISION=1 set, we need to actively make
# find_spec("torchvision") return None — otherwise transformers 4.50 will
# try to import torchvision and fail at the C-level `torchvision::nms`
# registration step (because we just pinned a torch that doesn't match the
# preinstalled torchvision in the Kaggle base image).
import importlib.util  # noqa: E402
_ORIG_FIND_SPEC = importlib.util.find_spec
def _patched_find_spec(name, *args, **kwargs):
    if name == "torchvision" or name.startswith("torchvision."):
        return None
    return _ORIG_FIND_SPEC(name, *args, **kwargs)
importlib.util.find_spec = _patched_find_spec  # type: ignore[assignment]


# Imports after install + patch.
import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from datasets import load_dataset  # noqa: E402

print(f"\n[2/5] env: torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

# Defensive GPU check (P100 sm_60 vs newer arch).
if torch.cuda.is_available():
    try:
        major, minor = torch.cuda.get_device_capability(0)
        if f"sm_{major}{minor}" not in torch.cuda.get_arch_list():
            print(f"  GPU sm_{major}{minor} not in supported {torch.cuda.get_arch_list()}; falling to CPU")
            device, dtype = "cpu", torch.float32
        else:
            device, dtype = "cuda", torch.float16
    except Exception:
        device, dtype = "cpu", torch.float32
else:
    device, dtype = "cpu", torch.float32
print(f"  using device={device}  dtype={dtype}")


# ------- PATCH 2 + 3: import upstream SYSTEM_PROMPT + compare_answers -------

sys.path.insert(0, UPSTREAM_DIR)

# Upstream's exact comparator (4 normalization strategies).
from inference_utils.answer_utils import (  # noqa: E402
    compare_answers,
    extract_pred_answer,
)
# Upstream's exact system prompt (turns out to be "You are a helpful assistant.").
from prompts import SYSTEM_PROMPT  # noqa: E402
print(f"\n[3/5] upstream loaded. SYSTEM_PROMPT={SYSTEM_PROMPT!r}")


# ------- Load Solver -------

SOLVER_REPO = "RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B"

print(f"\n[4/5] load model {SOLVER_REPO}")
t = time.time()
repo_dir = snapshot_download(SOLVER_REPO)
from pathlib import Path  # noqa: E402
view = Path(repo_dir) / "_plain_view"
if not view.is_dir() and (Path(repo_dir) / "adapter_config.json").is_file():
    view.mkdir()
    for item in Path(repo_dir).iterdir():
        if item == view or item.name in ("adapter_config.json", "innerlink_config.json", "README.md"):
            continue
        if item.suffix == ".pt" or item.name.startswith("adapter("):
            continue
        (view / item.name).symlink_to(item.resolve())
    model_dir = str(view)
else:
    model_dir = repo_dir
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
print(f"  model + tokenizer loaded in {time.time() - t:.1f}s (hidden={model.config.hidden_size})")


# ------- Load math500 -------

ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
problems = list(ds)[:N_SAMPLES]
print(f"\n[5/5] math500: using first {len(problems)} problems")


# ------- Inference loop -------

@torch.no_grad()
def generate_answer(prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
    if DO_SAMPLE:
        gen_kwargs.update(do_sample=True, temperature=0.6, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)
    out = model.generate(**inputs, **gen_kwargs)
    gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen_text


correct = 0
no_pred = 0       # extracted nothing
wrong = 0         # extracted but mismatch
results = []
t = time.time()
DATASET_NAME = "math500"  # matches _is_math500_dataset

for i, p in enumerate(problems):
    question = p["problem"]
    gold = p["answer"]
    gen = generate_answer(question)

    # Upstream comparator: 4-strategy normalization (intpart, latex_text, nospace, digits)
    gold_norm_str, pred_norm_str, ok, gold_dbg, pred_dbg = compare_answers(gold, gen, DATASET_NAME)
    pred_raw = extract_pred_answer(gen)  # for logging

    if ok:
        correct += 1
    elif pred_raw is None:
        no_pred += 1
    else:
        wrong += 1

    results.append({
        "i": i,
        "gold": gold,
        "pred_raw": pred_raw,
        "ok": ok,
        "gold_norm": gold_dbg,
        "pred_norm": pred_dbg,
        "gen_len": len(gen),
    })
    if (i + 1) % 10 == 0:
        elapsed = time.time() - t
        print(f"  [{i+1}/{len(problems)}] correct={correct} no_pred={no_pred} wrong={wrong} "
              f"acc={correct/(i+1)*100:.1f}%  elapsed={elapsed:.0f}s")

acc = correct / len(problems) * 100
print(f"\n=== FINAL: {correct}/{len(problems)} = {acc:.1f}% accuracy ===")
print(f"  breakdown:  correct={correct}  wrong_pred={wrong}  no_pred_extracted={no_pred}")
print(f"  inference time: {time.time() - t:.0f}s")

out = {
    "config": {
        "n_samples": N_SAMPLES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "model_repo": SOLVER_REPO,
        "system_prompt": SYSTEM_PROMPT,
        "comparator": "upstream.compare_answers(math500)",
        "recursive_mas_commit": RECURSIVEMAS_COMMIT,
    },
    "accuracy": acc,
    "correct": correct,
    "wrong_with_pred": wrong,
    "no_pred_extracted": no_pred,
    "total": len(problems),
    "inference_seconds": time.time() - t,
    "total_seconds": time.time() - T0,
    "env": {
        "torch": torch.__version__,
        "device": device,
        "dtype": str(dtype),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    },
    "samples": results[:5],
    "all_results": results,
}
with open("/kaggle/working/solver_only.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[done] wrote /kaggle/working/solver_only.json")
