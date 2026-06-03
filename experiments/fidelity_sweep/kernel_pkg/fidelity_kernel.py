"""Fidelity-instrumentation kernel for the RecursiveMAS Outer-Link channel.

Runs RecursiveMAS Sequential-Light with optional Variant B Lloyd-Max-Gaussian
quantization injected at every Adapter / CrossModelAdapter forward, and emits
per-call channel-fidelity stats (cosine, relative L2, n_tokens) plus per-decode
top-K logits for every quantized call. Designed to be run twice — once at
VARIANT_B_BITS=0 (REF) and once at VARIANT_B_BITS>0 (INT4 analog) — with identical
seed + greedy decoding, so the two runs are exactly paired per problem.

Env vars (the push helper bakes the per-run values into the module-level
constants below; the injected head re-reads the *propagated* env in the upstream
subprocess — see `main()` where they are explicitly written into the child env):
  CAPTURE_MODE           — "1" enables instrumentation + greedy decoding
  VARIANT_B_BITS         — 0 = REF (no quantization), N>0 = Variant B N-bit
  NUM_RECURSIVE_ROUNDS   — channel-traversal count T (upstream default 3)
  N_SAMPLES              — how many math500 problems
  BATCH_SIZE             — kept small for fp32 (T4: 4; b=2 if output_scores OOMs)
  DTYPE_OVERRIDE         — forced dtype string (e.g. "float32" on T4)
  TOPK_LOGITS            — top-K logits captured per decode position (default 512)
  MAX_LOGIT_POSITIONS    — cap on decode positions stored per generate() call
  TEMPERATURE / TOP_P    — only used when CAPTURE_MODE=0

Outputs in /kaggle/working/:
  fidelity_{config}.json       summary (accuracy, FidelityRun summaries, env, per-problem)
  fidelity_call_stats.json     per-adapter channel-fidelity stats (INT4 runs)
  fidelity_logits.npz          raw top-K logits per decode position (both REF/INT4)
  per_problem_vb{N}_T{T}.jsonl upstream per-problem correctness dump

The patch logic and the JSONL parser are module-level pure functions so they can
be unit-tested against the real upstream source WITHOUT running the kernel — see
tests/test_fidelity_kernel.py. `main()` holds all the side effects (pip install,
clone, subprocess) and only runs under `__main__`.
"""
import json
import os
import re
import subprocess
import sys
import time

# ---- Env-configurable knobs (push helper sed-patches these defaults) ----
N_SAMPLES = int(os.environ.get("N_SAMPLES", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
VARIANT_B_BITS = int(os.environ.get("VARIANT_B_BITS", "0"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
DTYPE_OVERRIDE = os.environ.get("DTYPE_OVERRIDE", "float32")
NUM_RECURSIVE_ROUNDS = int(os.environ.get("NUM_RECURSIVE_ROUNDS", "3"))
CAPTURE_MODE = os.environ.get("CAPTURE_MODE", "1") == "1"
TOPK_LOGITS = int(os.environ.get("TOPK_LOGITS", "512"))
MAX_LOGIT_POSITIONS = int(os.environ.get("MAX_LOGIT_POSITIONS", "256"))
RECURSIVEMAS_COMMIT = os.environ.get(
    "RECURSIVEMAS_COMMIT",
    "f95d512017fb713e9ac519248fbfd3d270dafd68",
)
# Output dir: /kaggle/working on Kaggle (default). The Modal driver sets its own.
WORK_DIR = os.environ.get("FIDELITY_WORK_DIR", "/kaggle/working")

CONFIG_TAG = (
    f"vb{VARIANT_B_BITS}_T{NUM_RECURSIVE_ROUNDS}"
    f"_n{N_SAMPLES}_b{BATCH_SIZE}_{DTYPE_OVERRIDE}"
)

UPSTREAM_DIR = f"{WORK_DIR}/RecursiveMAS"


# ===========================================================================
# Injected source (runs INSIDE the upstream subprocess after import).
# Kept as module-level strings so patch_inference_mas() can be tested.
# ===========================================================================

VARIANT_B_HEAD = '''
# ===== Variant B + Tier 2 fidelity capture (fidelity_kernel) =====
import os as _vb_os, sys as _vb_sys
import atexit as _vb_atexit
import json as _vb_json
_VB_BITS = int(_vb_os.environ.get("VARIANT_B_BITS", "0"))
_VB_CAPTURE = _vb_os.environ.get("CAPTURE_MODE", "0") == "1"
_VB_TOPK = int(_vb_os.environ.get("TOPK_LOGITS", "512"))
_VB_MAXPOS = int(_vb_os.environ.get("MAX_LOGIT_POSITIONS", "256"))
# Output dir + src location are env-configurable so the SAME head runs on Kaggle
# (defaults below) and Modal (FIDELITY_WORK_DIR=/work, FIDELITY_SRC_ROOT=/root).
_VB_WORK = _vb_os.environ.get("FIDELITY_WORK_DIR", "/kaggle/working")
_VB_SRC_ROOT_ENV = _vb_os.environ.get("FIDELITY_SRC_ROOT", "")
_VB_LOG = _vb_os.path.join(_VB_WORK, "vb_patches.log")
with open(_VB_LOG, "w") as _vbf:
    print(f"=== fidelity injector — bits={_VB_BITS} capture={_VB_CAPTURE} topk={_VB_TOPK} maxpos={_VB_MAXPOS} work={_VB_WORK} ===", file=_vbf)
_vb_stats_registry = []
if _VB_BITS > 0:
    if _VB_SRC_ROOT_ENV:
        _vb_src_root = _VB_SRC_ROOT_ENV
    else:
        # Auto-detect mounted src/ wrapper anywhere under /kaggle/input/
        _vb_src_root = None
        for _r, _d, _f in _vb_os.walk("/kaggle/input"):
            if _r.endswith("/src") and "adapters" in _d and "quantizers" in _d:
                _vb_src_root = _vb_os.path.dirname(_r); break
        if _vb_src_root is None:
            raise RuntimeError("src/ wrapper not found under /kaggle/input")
    print(f"[fidelity] src/ root: {_vb_src_root}", flush=True)
    _vb_sys.path.insert(0, _vb_src_root)
    from src.adapters.patch import patch_adapter as _vb_patch, QuantStats as _VBQuantStats
    from src.quantizers.turboquant_honest import TurboQuantHonest as _VB_Quant

    def _vb_quant_factory(d):
        return _VB_Quant(d=d, bits=_VB_BITS, seed=42)

# ---------- Tier 2: per-problem top-K logit capture ------------------------
#
# Strategy: monkey-patch transformers.GenerationMixin.generate to force
# output_scores=True on every call site (all 3 in upstream inference_mas.py),
# capture top-K logits + indices per generated step, but RETURN the raw
# sequences tensor (backward compatible with all call sites that don't
# request return_dict_in_generate themselves).
#
# Memory: positions are capped at _VB_MAXPOS per call. At K=512, MAXPOS=256,
# B=4: 256*4*512*4 bytes * 2 (vals+idxs) ~= 4 MB per generate() call (CPU).
_vb_logit_buffer = []  # list of dicts, one per generated batch
def _vb_install_generate_hook():
    if not _VB_CAPTURE:
        return
    try:
        import torch as _t
        from transformers.generation.utils import GenerationMixin as _GM
    except Exception as _e:
        print(f"[fidelity] generate-hook install skipped: {_e}", flush=True)
        return
    _orig_generate = _GM.generate

    def _patched_generate(self, *args, **kwargs):
        user_wants_dict = kwargs.get("return_dict_in_generate", False)
        kwargs["return_dict_in_generate"] = True
        kwargs["output_scores"] = True
        out = _orig_generate(self, *args, **kwargs)
        # out.scores: tuple of (B, V) tensors, one per generated step
        if _VB_CAPTURE and getattr(out, "scores", None) is not None:
            try:
                with _t.no_grad():
                    scores = out.scores
                    if _VB_MAXPOS > 0:
                        scores = scores[:_VB_MAXPOS]
                    if len(scores) == 0:
                        raise RuntimeError("empty scores")
                    # Stack to (T, B, V); top-K over V
                    stacked = _t.stack([s.detach().float().cpu() for s in scores], dim=0)
                    T_steps, B, V = stacked.shape
                    k = min(_VB_TOPK, V)
                    vals, idxs = _t.topk(stacked, k=k, dim=-1)
                    # Tail mass: log-sum-exp of the non-top-K logits, for KL tail correction
                    full_lse = _t.logsumexp(stacked, dim=-1, keepdim=True)  # (T, B, 1)
                    topk_lse = _t.logsumexp(vals, dim=-1, keepdim=True)     # (T, B, 1)
                    # tail_mass_log = log(exp(full_lse) - exp(topk_lse))
                    diff = full_lse - topk_lse
                    # log1p(-exp(-diff)) numerically stable for diff>0
                    tail_log = full_lse + _t.log1p(-_t.exp(-diff).clamp(min=1e-12))
                    _vb_logit_buffer.append({
                        "vals": vals.numpy(),                       # (T, B, K)
                        "idxs": idxs.numpy().astype("int32"),
                        "full_lse": full_lse.squeeze(-1).numpy(),   # (T, B)
                        "tail_log": tail_log.squeeze(-1).numpy(),   # (T, B)
                    })
            except Exception as _e:
                print(f"[fidelity] top-K capture failed: {_e}", flush=True)
        if user_wants_dict:
            return out
        return out.sequences

    _GM.generate = _patched_generate
    print(f"[fidelity] GenerationMixin.generate patched (top-K={_VB_TOPK}, maxpos={_VB_MAXPOS})", flush=True)

# Install at import time
_vb_install_generate_hook()

# ---------- atexit dumps ---------------------------------------------------

def _vb_dump_stats():
    if not _VB_CAPTURE:
        return
    out = {
        "n_adapters": len(_vb_stats_registry),
        "bits": _VB_BITS,
        "per_adapter": [s.summary() for s in _vb_stats_registry],
        "per_call": [
            {
                "adapter_idx": i,
                "label": s.label,
                "rmse_means": s.rmse_means,
                "cosine_means": s.cosine_means,
                "norm_ratio_means": s.norm_ratio_means,
                "n_tokens_total": s.n_tokens_total,
                "n_calls": s.n_calls,
            }
            for i, s in enumerate(_vb_stats_registry)
        ],
    }
    with open(_vb_os.path.join(_VB_WORK, "fidelity_call_stats.json"), "w") as _f:
        _vb_json.dump(out, _f, indent=2)
_vb_atexit.register(_vb_dump_stats)

def _vb_dump_logits():
    if not _VB_CAPTURE or not _vb_logit_buffer:
        return
    try:
        import numpy as _np
        # Concatenate along batch axis (T may vary per batch -> store as list)
        _np.savez_compressed(
            _vb_os.path.join(_VB_WORK, "fidelity_logits.npz"),
            **{f"batch{i}_{k}": v for i, b in enumerate(_vb_logit_buffer) for k, v in b.items()},
            n_batches=_np.array(len(_vb_logit_buffer)),
            bits=_np.array(_VB_BITS),
            topk=_np.array(_VB_TOPK),
        )
        print(f"[fidelity] dumped {len(_vb_logit_buffer)} logit batches to fidelity_logits.npz", flush=True)
    except Exception as _e:
        print(f"[fidelity] logit dump failed: {_e}", flush=True)
_vb_atexit.register(_vb_dump_logits)
# =====================================================================
'''

# Replaces every `    return adapter` in the two loader functions. When
# CAPTURE_MODE=1 we ALWAYS create stats (even at bits=0) so the two runs are
# structurally symmetric; at bits=0 the `if _VB_BITS > 0` guard skips patching.
VARIANT_B_RETURN = '''    if _VB_BITS > 0:
        _vb_label = "outer" if "CrossModel" in type(adapter).__name__ else "inner"
        _vb_links = _vb_os.environ.get("VB_LINKS", "all")
        if _vb_links == "all" or _vb_links == _vb_label:
            _vb_stats = _VBQuantStats(label=_vb_label)
            _vb_patch(adapter, _vb_quant_factory, label=_vb_label,
                      stats=_vb_stats, record=_VB_CAPTURE)
            _vb_stats_registry.append(_vb_stats)
            with open(_VB_LOG, "a") as _vbf:
                print(f"loader_call bits={_VB_BITS} links={_vb_links} class={type(adapter).__name__} label={_vb_label}", file=_vbf)
    return adapter'''


# ===========================================================================
# Pure, testable functions
# ===========================================================================


def patch_run_py(
    src: str,
    *,
    n_samples: int,
    dtype: str,
    num_recursive_rounds: int,
    capture_mode: bool,
    jsonl_path: str,
) -> "tuple[str, dict]":
    """Apply the run.py surgical patches. Returns (patched_src, counts).

    Raises RuntimeError if any required anchor is missing (so a silent
    upstream drift fails loudly here rather than after 95 min on Kaggle).
    """
    counts: dict = {}

    # (a) num_samples cap
    src, n1 = re.subn(r'("--num_samples",\s*)"-1"', rf'\1"{n_samples}"', src, count=1)
    if n1 != 1:
        raise RuntimeError(f"run.py: --num_samples anchor not found (got {n1})")
    counts["num_samples"] = n1

    # (b) dtype override (both --dtype and --outer_dtype)
    src, nd = re.subn(r'("--dtype",\s*)"auto"', rf'\1"{dtype}"', src, count=1)
    src, nod = re.subn(r'("--outer_dtype",\s*)"auto"', rf'\1"{dtype}"', src, count=1)
    if nd != 1 or nod != 1:
        raise RuntimeError(f"run.py: dtype anchors not found (dtype={nd} outer={nod})")
    counts["dtype"] = nd
    counts["outer_dtype"] = nod

    # (c) FORCE GREEDY in capture mode: drop the unconditional do_sample append
    if capture_mode:
        src, ng = re.subn(
            r'^(\s+)out\.append\("--do_sample"\)\s*\n',
            r"\1# CAPTURE_MODE: do_sample removed for paired greedy\n",
            src,
            count=1,
            flags=re.MULTILINE,
        )
        if ng != 1:
            raise RuntimeError(f"run.py: do_sample anchor not found (got {ng})")
        counts["greedy"] = ng

    # (d) num_recursive_rounds argparse default
    src, nrr = re.subn(
        r'(--num_recursive_rounds",\s*type=int,\s*default=)\d+',
        rf'\g<1>{num_recursive_rounds}',
        src,
        count=1,
    )
    if nrr != 1:
        raise RuntimeError(f"run.py: num_recursive_rounds default not found (got {nrr})")
    counts["num_recursive_rounds"] = nrr

    # (e) inject --result_jsonl so the per-problem correctness dump is produced
    if capture_mode:
        src, nrj = re.subn(
            r'("--ans_max_new_tokens",\s*"-1",)',
            rf'\1\n        "--result_jsonl", "{jsonl_path}",',
            src,
            count=1,
        )
        if nrj != 1:
            raise RuntimeError(f"run.py: ans_max_new_tokens anchor not found (got {nrj})")
        counts["result_jsonl"] = nrj

    return src, counts


def patch_inference_mas(
    src: str,
    *,
    batch_size: int,
    head_code: str = VARIANT_B_HEAD,
    return_code: str = VARIANT_B_RETURN,
) -> "tuple[str, dict]":
    """Apply the inference_mas.py patches: batch_size pin + head injection +
    return-adapter replacement. Returns (patched_src, counts).
    """
    counts: dict = {}

    # batch_size pin in the recommended-settings dict
    src, n2 = re.subn(
        r'\("sequential_light",\s*"math500"\):\s*\{[^}]*"batch_size":\s*\d+',
        f'("sequential_light", "math500"): {{"seed": 42, "batch_size": {batch_size}',
        src,
        count=1,
    )
    if n2 != 1:
        raise RuntimeError(f"inference_mas.py: batch_size anchor not found (got {n2})")
    counts["batch_size"] = n2

    # head injection right after the `from modeling import (...)` block
    head_match = re.search(r"^from modeling import \([^)]*\)\n", src, re.MULTILINE)
    if not head_match:
        raise RuntimeError("inference_mas.py: `from modeling import (...)` block not found")
    src = src[:head_match.end()] + head_code + src[head_match.end():]
    counts["head_injection"] = 1

    # replace both `    return adapter` lines (the two adapter loaders)
    src, rcount = re.subn(
        r"^    return adapter\n",
        return_code + "\n",
        src,
        flags=re.MULTILINE,
    )
    if rcount != 2:
        raise RuntimeError(f"inference_mas.py: expected 2 return-adapter patches, got {rcount}")
    counts["return_adapter"] = rcount

    return src, counts


def _record_correct(rec: dict):
    """Extract a boolean correctness flag from one upstream JSONL record.

    Upstream writes TWO schemas depending on --num_rollouts:
      * num_rollouts == 1  -> FLAT record with "correct" at the top level.
      * num_rollouts  > 1  -> NESTED record with "rollouts": [{"correct": ...}].
    Returns the bool, or None if it cannot be determined.
    """
    c = rec.get("correct")
    if isinstance(c, bool):
        return c
    rollouts = rec.get("rollouts")
    if isinstance(rollouts, list) and rollouts:
        cc = rollouts[0].get("correct")
        if isinstance(cc, bool):
            return cc
    pak = rec.get("pass_at_k_correct")
    if isinstance(pak, bool):
        return pak
    return None


def parse_per_problem_jsonl(path: str) -> "list[dict]":
    """Parse the upstream --result_jsonl dump into per-problem records.

    Handles both upstream schemas, skips the trailing ``type=="summary"`` row,
    and drops rows without a sample_idx. Each output record is
    ``{"sample_idx", "correct" (bool|None), "gold"}``.
    """
    per_problem: list = []
    if not os.path.exists(path):
        return per_problem
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("type") == "summary":
                continue
            sample_idx = rec.get("sample_idx")
            if sample_idx is None:
                continue
            per_problem.append({
                "sample_idx": sample_idx,
                "correct": _record_correct(rec),
                "gold": rec.get("gold_answer_raw"),
            })
    return per_problem


# ===========================================================================
# Side-effectful driver (Kaggle only)
# ===========================================================================


def _run(cmd, check=True, env=None):
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    if check:
        return subprocess.check_call(cmd, env=env)
    return subprocess.call(cmd, env=env)


def main() -> None:
    t0 = time.time()
    print(f"=== Fidelity kernel — {CONFIG_TAG} ===")
    print(f"  CAPTURE_MODE = {CAPTURE_MODE}  TOPK={TOPK_LOGITS}  MAXPOS={MAX_LOGIT_POSITIONS}")

    os.environ["MAS_FORCE_DISABLE_TORCHVISION"] = "1"

    # ---- [1/5] install torch ----
    print("\n[1/5] install torch 2.4.1+cu121")
    t = time.time()
    _run([sys.executable, "-m", "pip", "install", "-q",
          "torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cu121"])
    print(f"  done in {time.time() - t:.1f}s")

    # ---- [2/5] clone upstream ----
    print("\n[2/5] clone upstream RecursiveMAS")
    t = time.time()
    if not os.path.isdir(UPSTREAM_DIR):
        _run(["git", "clone", "https://github.com/RecursiveMAS/RecursiveMAS.git", UPSTREAM_DIR])
    _run(["git", "-C", UPSTREAM_DIR, "checkout", RECURSIVEMAS_COMMIT])
    print(f"  done in {time.time() - t:.1f}s")

    # ---- [3/5] upstream requirements (skip torch line) ----
    print("\n[3/5] install upstream requirements.txt (skip torch line)")
    t = time.time()
    req_in = f"{UPSTREAM_DIR}/requirements.txt"
    req_out = "/kaggle/working/_requirements_no_torch.txt"
    with open(req_in) as f, open(req_out, "w") as g:
        for line in f:
            if line.strip().lower().startswith("torch"):
                continue
            g.write(line)
    _run([sys.executable, "-m", "pip", "install", "-q", "-r", req_out])
    print(f"  done in {time.time() - t:.1f}s")

    import torch  # noqa: E402
    if not torch.cuda.is_available():
        print("[gpu] NO CUDA — fatal")
        sys.exit(1)
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"\n[gpu] {name} sm_{major}{minor}  bf16-native={major >= 8}")

    # ---- [4/5] patch upstream ----
    print(f"\n[4/5] patches: N_SAMPLES={N_SAMPLES} BATCH_SIZE={BATCH_SIZE} "
          f"VB_BITS={VARIANT_B_BITS} T={NUM_RECURSIVE_ROUNDS} CAPTURE={CAPTURE_MODE}")

    jsonl_path = f"{WORK_DIR}/per_problem_vb{VARIANT_B_BITS}_T{NUM_RECURSIVE_ROUNDS}.jsonl"

    run_py = f"{UPSTREAM_DIR}/run.py"
    src, rc_counts = patch_run_py(
        open(run_py).read(),
        n_samples=N_SAMPLES,
        dtype=DTYPE_OVERRIDE,
        num_recursive_rounds=NUM_RECURSIVE_ROUNDS,
        capture_mode=CAPTURE_MODE,
        jsonl_path=jsonl_path,
    )
    open(run_py, "w").write(src)

    infer_mas = f"{UPSTREAM_DIR}/inference_utils/inference_mas.py"
    isrc, im_counts = patch_inference_mas(open(infer_mas).read(), batch_size=BATCH_SIZE)
    open(infer_mas, "w").write(isrc)
    print(f"  run.py patches: {rc_counts}")
    print(f"  inference_mas patches: {im_counts}")

    # ---- [5/5] run upstream (with env propagated to the child!) ----
    print(f"\n[5/5] invoke upstream run.py with T={NUM_RECURSIVE_ROUNDS}")
    t_run = time.time()
    env = dict(os.environ)
    env["MAS_FORCE_DISABLE_TORCHVISION"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    # CRITICAL: the injected head reads these from os.environ in the child.
    env["VARIANT_B_BITS"] = str(VARIANT_B_BITS)
    env["CAPTURE_MODE"] = "1" if CAPTURE_MODE else "0"
    env["TOPK_LOGITS"] = str(TOPK_LOGITS)
    env["MAX_LOGIT_POSITIONS"] = str(MAX_LOGIT_POSITIONS)
    env["FIDELITY_WORK_DIR"] = WORK_DIR
    env["VB_LINKS"] = os.environ.get("VB_LINKS", "all")  # all | inner | outer

    cmd = [
        sys.executable, "run.py",
        "--style", "sequential_light",
        "--dataset", "math500",
        "--seed", "42",
        "--num_recursive_rounds", str(NUM_RECURSIVE_ROUNDS),
        "--trust_remote_code", "1",
        "--device", "cuda",
        "--temperature", str(TEMPERATURE),
        "--top_p", str(TOP_P),
    ]
    print(f"  cwd={UPSTREAM_DIR}")
    print(f"  cmd={' '.join(cmd)}")
    print(f"  child env: VARIANT_B_BITS={env['VARIANT_B_BITS']} CAPTURE_MODE={env['CAPTURE_MODE']} "
          f"TOPK_LOGITS={env['TOPK_LOGITS']} MAX_LOGIT_POSITIONS={env['MAX_LOGIT_POSITIONS']}\n")
    proc = subprocess.Popen(
        cmd, cwd=UPSTREAM_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, bufsize=1, text=True,
    )
    all_out, final_acc = [], None
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

    # ---- collect channel-fidelity stats (dumped by the child's atexit) ----
    stats_dump_path = f"{WORK_DIR}/fidelity_call_stats.json"
    fidelity_summary = None
    if os.path.exists(stats_dump_path):
        fidelity_summary = json.loads(open(stats_dump_path).read())
        print(f"\n[fidelity] loaded {len(fidelity_summary.get('per_call', []))} call records "
              f"across {fidelity_summary.get('n_adapters', '?')} patched adapters")
    else:
        print("\n[fidelity] no stats dump found (REF run, or no patches fired)")

    # ---- diagnostic: count loader calls from log ----
    patches_log = f"{WORK_DIR}/vb_patches.log"
    n_patches = 0
    if os.path.exists(patches_log):
        with open(patches_log) as f:
            for line in f:
                if line.startswith("loader_call"):
                    n_patches += 1

    # ---- per-problem correctness (robust to both upstream schemas) ----
    per_problem = []
    if CAPTURE_MODE:
        per_problem = parse_per_problem_jsonl(jsonl_path)
        n_with_correct = sum(1 for p in per_problem if isinstance(p["correct"], bool))
        print(f"[fidelity] parsed {len(per_problem)} per-problem records "
              f"({n_with_correct} with a correctness flag) from {jsonl_path}")
        if per_problem and n_with_correct == 0:
            print("[fidelity][WARN] per-problem records present but NONE had a correctness "
                  "flag — JSONL schema may have changed; paired TOST will have no data.")

    # ---- logit dump diagnostics ----
    logits_dump_path = f"{WORK_DIR}/fidelity_logits.npz"
    n_logit_batches = 0
    if os.path.exists(logits_dump_path):
        import numpy as _np
        with _np.load(logits_dump_path) as _z:
            n_logit_batches = int(_z["n_batches"]) if "n_batches" in _z else 0
        print(f"[fidelity] {n_logit_batches} top-K logit batches dumped")

    print(f"\n[result] rc={rc}  elapsed={run_secs:.0f}s ({run_secs/60:.1f} min)  "
          f"final_acc={final_acc}  patches_logged={n_patches} "
          f"per_problem={len(per_problem)} logit_batches={n_logit_batches}")

    result = {
        "config": {
            "n_samples": N_SAMPLES,
            "batch_size": BATCH_SIZE,
            "variant_b_bits": VARIANT_B_BITS,
            "num_recursive_rounds": NUM_RECURSIVE_ROUNDS,
            "capture_mode": CAPTURE_MODE,
            "dtype_override": DTYPE_OVERRIDE,
            "topk_logits": TOPK_LOGITS,
            "max_logit_positions": MAX_LOGIT_POSITIONS,
            "gpu": name,
            "sm": f"sm_{major}{minor}",
            "config_tag": CONFIG_TAG,
            "seed": 42,
            "decoding": "greedy" if CAPTURE_MODE else "sampled",
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "recursive_mas_commit": RECURSIVEMAS_COMMIT,
        },
        "final_accuracy": final_acc,
        "n_patches_logged": n_patches,
        "n_per_problem_records": len(per_problem),
        "n_logit_batches": n_logit_batches,
        "return_code": rc,
        "run_seconds": run_secs,
        "total_seconds": time.time() - t0,
        "fidelity_summary": fidelity_summary,
        "per_problem": per_problem,
        "logits_npz_path": logits_dump_path if n_logit_batches > 0 else None,
        "tail_log_lines": all_out[-200:],
    }
    out_path = f"{WORK_DIR}/fidelity_{CONFIG_TAG}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
