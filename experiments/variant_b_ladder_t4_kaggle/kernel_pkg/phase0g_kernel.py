"""Phase 0.G — Kaggle T4 (Turing sm_75, Tensor Cores) free-tier runner.

Goal: validate that T4 (Turing, has TC, fp16-accumulated-in-fp32) reproduces
the A100 baseline ~85% AND supports the Variant B ladder — without Modal cost.

This kernel does both jobs depending on env var VARIANT_B_BITS:
  VARIANT_B_BITS=0    → baseline pristine (sanity that T4 matches Modal ~85%)
  VARIANT_B_BITS=4    → Variant B 4-bit in-loop (replicates Phase 0.F result)
  VARIANT_B_BITS=N    → arbitrary bit-rate for ladder

Configurable also via env:
  N_SAMPLES    (default 50)
  BATCH_SIZE   (default 8)
  TEMPERATURE  (default 0.6)
  TOP_P        (default 0.95)

Uses FILE-BASED diagnostic logging (/kaggle/working/vb_patches.log) to
avoid Modal-style stdout multiplexing ambiguity. After the run, read the log
to count patch activations (expect ~6+ per recursive round if bits>0).

Our src/ is bundled via a Kaggle Dataset (<YOUR_KAGGLE_USERNAME>/lqc-src) mounted at
/kaggle/input/lqc-src/src/. The kernel-metadata.json declares the dependency.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

T0 = time.time()

# ---- Env-configurable knobs ----
N_SAMPLES = int(os.environ.get("N_SAMPLES", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
VARIANT_B_BITS = int(os.environ.get("VARIANT_B_BITS", "0"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
DTYPE_OVERRIDE = os.environ.get("DTYPE_OVERRIDE", "float32")  # "" = upstream default (auto → bf16)
RECURSIVEMAS_COMMIT = os.environ.get(
    "RECURSIVEMAS_COMMIT",
    "f95d512017fb713e9ac519248fbfd3d270dafd68",
)

print(f"=== Phase 0.G Kaggle T4 — VB_BITS={VARIANT_B_BITS} n={N_SAMPLES} b={BATCH_SIZE} dtype={DTYPE_OVERRIDE or 'auto'} ===")


def run(cmd, check=True, env=None):
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    if check:
        return subprocess.check_call(cmd, env=env)
    return subprocess.call(cmd, env=env)


# Upstream torchvision-disable env var (same as Phase 0.D / 0.E / 0.F)
os.environ["MAS_FORCE_DISABLE_TORCHVISION"] = "1"


# ---- [1/5] install torch 2.4.1+cu121 (works on T4 sm_75 + P100 sm_60 fallback) ----
print("\n[1/5] install torch 2.4.1+cu121")
t = time.time()
run([sys.executable, "-m", "pip", "install", "-q",
     "torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cu121"])
print(f"  done in {time.time() - t:.1f}s")


# ---- [2/5] clone upstream + install requirements ----
UPSTREAM_DIR = "/kaggle/working/RecursiveMAS"
print(f"\n[2/5] clone upstream RecursiveMAS")
t = time.time()
if not os.path.isdir(UPSTREAM_DIR):
    run(["git", "clone", "https://github.com/RecursiveMAS/RecursiveMAS.git", UPSTREAM_DIR])
run(["git", "-C", UPSTREAM_DIR, "checkout", RECURSIVEMAS_COMMIT])
print(f"  done in {time.time() - t:.1f}s")

print(f"\n[3/5] install upstream requirements.txt (skip torch line)")
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


# ---- GPU sanity ----
import torch  # noqa: E402

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"\n[gpu] {name} sm_{major}{minor}")
    print(f"  Tensor Cores: {'YES' if major >= 7 else 'NO'}")
    print(f"  Native bf16 HW: {'YES' if major >= 8 else 'NO (sm_80+ required)'}")
    print(f"  torch arch list: {torch.cuda.get_arch_list()}")
    print(f"  torch version: {torch.__version__}")
    print(f"  cuda version (torch): {torch.version.cuda}")
    # SDPA backend availability
    try:
        print(f"  SDPA flash_sdp_enabled:        {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"  SDPA mem_efficient_sdp_enabled:{torch.backends.cuda.mem_efficient_sdp_enabled()}")
        print(f"  SDPA math_sdp_enabled:         {torch.backends.cuda.math_sdp_enabled()}")
    except Exception as e:
        print(f"  SDPA backend probe failed: {e}")
    # matmul precision flags
    print(f"  allow_tf32 (cuda.matmul):   {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  allow_tf32 (cudnn):         {torch.backends.cudnn.allow_tf32}")
    try:
        print(f"  allow_fp16_reduced_precision_reduction: "
              f"{torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}")
        print(f"  allow_bf16_reduced_precision_reduction: "
              f"{torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}")
    except Exception:
        pass
    if major < 8:
        print("  ⚠️  WARNING: sm < 80 → no native bf16 → expect collapse to ~30% if --dtype auto/bfloat16 (REPORT_05 §15.5)")
else:
    print("\n[gpu] NO CUDA — fatal")
    sys.exit(1)


# ---- [4/5] patch upstream code ----

print(f"\n[4/5] patches: num_samples={N_SAMPLES} batch_size={BATCH_SIZE} VB_bits={VARIANT_B_BITS}")

# Patch 1: num_samples in run.py + optional dtype override
run_py = f"{UPSTREAM_DIR}/run.py"
src = open(run_py).read()
src, n1 = re.subn(r'("--num_samples",\s*)"-1"', rf'\1"{N_SAMPLES}"', src, count=1)
if n1 == 0:
    raise RuntimeError("Failed to MATCH --num_samples in run.py")
if DTYPE_OVERRIDE:
    # Replace BOTH hardcoded "auto" args for --dtype and --outer_dtype
    # to force a specific dtype throughout the pipeline.
    src, nd = re.subn(
        r'("--dtype",\s*)"auto"',
        rf'\1"{DTYPE_OVERRIDE}"',
        src, count=1,
    )
    src, nod = re.subn(
        r'("--outer_dtype",\s*)"auto"',
        rf'\1"{DTYPE_OVERRIDE}"',
        src, count=1,
    )
    if nd + nod < 2:
        raise RuntimeError(f"Failed to override dtype (nd={nd} nod={nod})")
    print(f"  dtype override applied: {DTYPE_OVERRIDE}")
open(run_py, "w").write(src)

# Patch 2 + 3: batch_size + Variant B injection in inference_mas.py
infer_mas = f"{UPSTREAM_DIR}/inference_utils/inference_mas.py"
isrc = open(infer_mas).read()

# 2) batch_size pin
isrc, n2 = re.subn(
    r'\("sequential_light",\s*"math500"\):\s*\{[^}]*"batch_size":\s*\d+',
    f'("sequential_light", "math500"): {{"seed": 42, "batch_size": {BATCH_SIZE}',
    isrc,
    count=1,
)
if n2 == 0:
    raise RuntimeError("Failed to MATCH batch_size")

# 3) Variant B head injection (only matters if bits>0)
# Dataset structure: /kaggle/input/lqc-src-bundle/src/{adapters,quantizers,utils}/...
# (auto-extracted by Kaggle from lqc_src.tar.gz, src/ wrapper preserved).
VARIANT_B_HEAD = '''
# ===== Variant B injection (Phase 0.G) =====
import os as _vb_os, sys as _vb_sys
_VB_BITS = int(_vb_os.environ.get("VARIANT_B_BITS", "0"))
_VB_LOG = "/kaggle/working/vb_patches.log"
# Deep diagnostic: walk /kaggle/input/ to find any patch.py
print("[VB-DIAG] /kaggle/input/ tree:", flush=True)
for _r, _d, _f in _vb_os.walk("/kaggle/input"):
    _depth = _r.count("/") - 2
    if _depth > 4: continue
    print(f"  {' ' * _depth}{_r}/  ({len(_d)} dirs, {len(_f)} files)", flush=True)
with open(_VB_LOG, "w") as _vbf:
    print(f"=== Variant B injector loaded, bits={_VB_BITS} ===", file=_vbf)
if _VB_BITS > 0:
    # Auto-detect dataset mount: look for any path containing src/adapters/patch.py
    _vb_src_root = None
    for _r, _d, _f in _vb_os.walk("/kaggle/input"):
        if _r.endswith("/src") and "adapters" in _d and "quantizers" in _d:
            _vb_src_root = _vb_os.path.dirname(_r)  # parent of src/
            break
    if _vb_src_root is None:
        raise RuntimeError(f"src/ wrapper not found anywhere under /kaggle/input")
    print(f"[VB-DIAG] found src/ at: {_vb_src_root}/src", flush=True)
    _vb_sys.path.insert(0, _vb_src_root)
    from src.adapters.patch import patch_adapter as _vb_patch
    from src.quantizers.turboquant_honest import TurboQuantHonest as _VB_Quant
    def _vb_quant_factory(d):
        return _VB_Quant(d=d, bits=_VB_BITS, seed=42)
# ============================================
'''

head_match = re.search(r"^from modeling import \([^)]*\)\n", isrc, re.MULTILINE)
if not head_match:
    raise RuntimeError("Failed to find `from modeling import (...)` block")
isrc = isrc[:head_match.end()] + VARIANT_B_HEAD + isrc[head_match.end():]

# 4) Patch BOTH `return adapter` lines with VB activation
VARIANT_B_RETURN = '''    if _VB_BITS > 0:
        _vb_label = "outer" if "CrossModel" in type(adapter).__name__ else "inner"
        _vb_patch(adapter, _vb_quant_factory, label=_vb_label)
        with open(_VB_LOG, "a") as _vbf:
            print(f"loader_call bits={_VB_BITS} class={type(adapter).__name__} label={_vb_label}", file=_vbf)
    return adapter'''

isrc, n3 = re.subn(
    r"^    return adapter\n",
    VARIANT_B_RETURN + "\n",
    isrc,
    flags=re.MULTILINE,
)
if n3 != 2:
    raise RuntimeError(f"Expected 2 return adapter patches, got {n3}")

open(infer_mas, "w").write(isrc)
print(f"  patches applied: num_samples ✓ batch_size ✓ head_injection ✓ "
      f"loader_returns ✓ (2 sites)")


# ---- [5/5] subprocess upstream run.py ----

print(f"\n[5/5] invoke upstream run.py")
t_run = time.time()
env = dict(os.environ)
env["MAS_FORCE_DISABLE_TORCHVISION"] = "1"
env["TOKENIZERS_PARALLELISM"] = "false"
env["PYTHONUNBUFFERED"] = "1"
env["VARIANT_B_BITS"] = str(VARIANT_B_BITS)

cmd = [
    sys.executable, "run.py",
    "--style", "sequential_light",
    "--dataset", "math500",
    "--seed", "42",
    "--trust_remote_code", "1",
    "--device", "cuda",
    "--temperature", str(TEMPERATURE),
    "--top_p", str(TOP_P),
]
# Optional dtype override (ablation §15.5)
if DTYPE_OVERRIDE:
    # `run.py` passes --dtype/--outer_dtype to inference_mas.py via build_common_cli.
    # We can't add flags via run.py directly (it constructs them internally).
    # Workaround: patch inference_mas.py's resolve_dtype to override.
    # Simpler: pre-set the value in run.py:200 → "--dtype", "auto", → "--dtype", DTYPE_OVERRIDE.
    pass  # handled by an additional regex patch below if DTYPE_OVERRIDE set
print(f"  cwd={UPSTREAM_DIR}  cmd={' '.join(cmd)}")
proc = subprocess.Popen(
    cmd, cwd=UPSTREAM_DIR,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    env=env, bufsize=1, text=True,
)

all_out = []
final_acc = None
for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
    all_out.append(line)
    m = re.search(r"accuracy=([0-9.]+)%", line)
    if m:
        final_acc = float(m.group(1))

proc.wait()
rc = proc.returncode
run_secs = time.time() - t_run

# ---- Diagnostic: count patches from file log ----
patches_log = "/kaggle/working/vb_patches.log"
n_patches = 0
if os.path.exists(patches_log):
    with open(patches_log) as f:
        for line in f:
            if line.startswith("loader_call"):
                n_patches += 1
    print(f"\n[diagnostic] /kaggle/working/vb_patches.log contains {n_patches} loader_call entries")
else:
    print(f"\n[diagnostic] log file not created (head injection never ran?)")

print(f"\n[result] rc={rc}  elapsed={run_secs:.0f}s ({run_secs/60:.1f} min)  "
      f"final_acc={final_acc}  patches_logged={n_patches}")


# ---- Persist JSON ----
result = {
    "config": {
        "n_samples": N_SAMPLES,
        "batch_size": BATCH_SIZE,
        "variant_b_bits": VARIANT_B_BITS,
        "gpu": torch.cuda.get_device_name(0),
        "sm": f"sm_{major}{minor}",
        "has_tensor_cores": major >= 7,
        "style": "sequential_light",
        "dataset": "math500",
        "seed": 42,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "recursive_mas_commit": RECURSIVEMAS_COMMIT,
    },
    "final_accuracy": final_acc,
    "n_patches_logged_from_file": n_patches,
    "return_code": rc,
    "run_seconds": run_secs,
    "total_seconds": time.time() - T0,
    "tail_log_lines": all_out[-200:],
}
out_path = f"/kaggle/working/phase0g_vb{VARIANT_B_BITS}_n{N_SAMPLES}_b{BATCH_SIZE}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n[done] wrote {out_path}")
