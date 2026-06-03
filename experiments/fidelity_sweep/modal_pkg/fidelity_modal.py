"""Modal A100 driver for the paired REF vs INT4 fidelity sweep.

This is the Modal counterpart of the Kaggle ``fidelity_kernel.py``. It reuses
that kernel's **already-tested** pure functions (`patch_run_py`,
`patch_inference_mas`, `parse_per_problem_jsonl`, and the portable injected head)
so there is a single source of truth for the surgical patches and the JSONL
parsing — see tests/test_fidelity_kernel.py.

Why Modal A100 fp32 (not bf16):
  Forcing ``--dtype float32`` on A100 avoids BOTH failure modes documented in
  REPORT 05/06: the bf16-fallback collapse on non-Ampere HW (irrelevant on A100
  but kept for consistency) AND the bf16<->fp32 boundary-cast artifact that
  produced the spurious Phase 0.F -19pp. A100 runs fp32 natively, ~15x faster
  than the Kaggle T4, with 40GB so ``output_scores=True`` never OOMs.

Cost control ($1/day budget):
  A100-40GB is ~$2/h, so ~28 min/day. A single n=50 fp32 capture run is ~12-16
  min (~$0.45-0.55). Validate first with a tiny dry-run, then run ONE paired
  comparison (REF+INT4 at a chosen T) per day. Runs are deterministic
  (seed=42, greedy), so a REF run today and its INT4 partner tomorrow pair
  exactly by sample_idx.

IMPORTANT — always launch long runs with `--detach`. A plain `modal run` ties
the app to the local client; if the session disconnects, Modal stops the
containers. `--detach` keeps the run alive server-side. NOTE: detached mode only
guarantees the *last triggered* function survives, so for a long paired run
launch REF and INT4 as TWO separate single-function `::main` runs (each triggers
exactly one function) rather than the `::sweep` fan-out. The function commits its
outputs to the `rmas-fidelity-out` volume at the end, so detached runs are
recovered by reading the volume — no need to stay attached.

Usage:
  # tiny validation (~$0.20): INT4 exercises the full path incl. the quantizer
  modal run experiments/fidelity_sweep/modal_pkg/fidelity_modal.py::main \
      --bits 4 --t 3 --n-samples 8 --batch-size 4 --maxpos 128 --topk 256

  # real paired comparison at depth T=3 — DETACHED, one function per run
  modal run --detach .../fidelity_modal.py::main --bits 0 --t 3 --n-samples 250 --batch-size 8  # REF
  modal run --detach .../fidelity_modal.py::main --bits 4 --t 3 --n-samples 250 --batch-size 8  # INT4

  # the ::sweep entrypoint fans out all (T x {REF,INT4}) at once; convenient when
  # you will stay connected, but NOT disconnect-safe under --detach (see note).
  modal run .../fidelity_modal.py::sweep --t-values 1,2,3,4 --n-samples 50 --batch-size 8

  # recover outputs from the volume (works even if the client disconnected) + analyze
  modal volume get rmas-fidelity-out vb0_T3_n250 /tmp/fid_outputs/
  modal volume get rmas-fidelity-out vb4_T3_n250 /tmp/fid_outputs/
  .venv/bin/python experiments/fidelity_sweep/analysis/analyze.py \
      --inputs /tmp/fid_outputs --out experiments/fidelity_sweep/analysis/results
"""
import pathlib

import modal

RECURSIVEMAS_COMMIT = "f95d512017fb713e9ac519248fbfd3d270dafd68"

app = modal.App("rmas-fidelity-sweep")

_base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # Match upstream RecursiveMAS requirements.txt (minus torch).
        "transformers==5.3.0",
        "datasets==4.4.2",
        "huggingface-hub==1.7.1",
        "accelerate==1.12.0",
        "safetensors==0.7.0",
        "tokenizers==0.22.2",
        "tqdm",
        "numpy",
        "scipy",  # src/utils/lloyd_max.py needs scipy.integrate (Kaggle's base image had it; debian_slim does not)
        "sentencepiece",
        "python-dotenv",
    )
    .run_commands(
        "cd /opt && git clone https://github.com/RecursiveMAS/RecursiveMAS.git && "
        f"cd /opt/RecursiveMAS && git checkout {RECURSIVEMAS_COMMIT}"
    )
    .env({
        "MAS_FORCE_DISABLE_TORCHVISION": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/cache/hf",
    })
)

# Attach our source tree + the tested kernel-patch module. This is LOCAL-ONLY:
# the path computation and mounts must not run when Modal re-imports this module
# inside the container (where __file__=/root/fidelity_modal.py has no parents[3]
# and the local paths don't exist). The mounted files are already present at
# /root/src and /root/fidelity_kernel.py inside the container.
if modal.is_local():
    _repo = pathlib.Path(__file__).resolve().parents[3]
    image = (
        _base_image
        .add_local_dir(str(_repo / "src"), "/root/src")
        .add_local_file(
            str(_repo / "experiments" / "fidelity_sweep" / "kernel_pkg" / "fidelity_kernel.py"),
            "/root/fidelity_kernel.py",
        )
    )
else:
    image = _base_image

# Reuse the populated 9 GB checkpoint cache from the baseline runs (no download).
hf_cache = modal.Volume.from_name("rmas-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("rmas-fidelity-out", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=2 * 60 * 60,
    volumes={"/cache/hf": hf_cache, "/out": out_vol},
)
def run_fidelity(
    bits: int = 0,
    T: int = 3,
    n_samples: int = 50,
    batch_size: int = 4,
    topk: int = 512,
    maxpos: int = 256,
    dtype: str = "float32",
    tag: str = "",
    links: str = "all",
) -> dict:
    import importlib.util
    import json
    import os
    import re
    import shutil
    import subprocess
    import sys
    import time

    t0 = time.time()
    WORK = "/work"
    os.makedirs(WORK, exist_ok=True)

    # Env consumed by the injected head IN THE CHILD process (propagated via
    # env=dict(os.environ) on Popen). FIDELITY_SRC_ROOT lets the head import
    # `src` from the baked-in /root/src without the Kaggle /kaggle/input walk.
    os.environ["FIDELITY_WORK_DIR"] = WORK
    os.environ["FIDELITY_SRC_ROOT"] = "/root"
    os.environ["VARIANT_B_BITS"] = str(bits)
    os.environ["CAPTURE_MODE"] = "1"
    os.environ["TOPK_LOGITS"] = str(topk)
    os.environ["MAX_LOGIT_POSITIONS"] = str(maxpos)
    os.environ["VB_LINKS"] = links  # all | inner | outer (selective quantization)

    gpu_name = os.popen(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
    ).read().strip()
    config_tag = f"vb{bits}_T{T}_n{n_samples}_b{batch_size}_{dtype}"
    print(f"=== Modal fidelity — {config_tag} on {gpu_name} ===")

    # Load the tested kernel-patch functions (single source of truth).
    spec = importlib.util.spec_from_file_location("fk", "/root/fidelity_kernel.py")
    fk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fk)

    # Copy upstream to a writable dir and apply the surgical patches.
    upstream = "/tmp/RecursiveMAS"
    if os.path.isdir(upstream):
        shutil.rmtree(upstream)
    shutil.copytree("/opt/RecursiveMAS", upstream)

    jsonl_path = f"{WORK}/per_problem_vb{bits}_T{T}.jsonl"
    run_py = f"{upstream}/run.py"
    rsrc, rc_counts = fk.patch_run_py(
        open(run_py).read(),
        n_samples=n_samples, dtype=dtype, num_recursive_rounds=T,
        capture_mode=True, jsonl_path=jsonl_path,
    )
    open(run_py, "w").write(rsrc)

    infer = f"{upstream}/inference_utils/inference_mas.py"
    isrc, im_counts = fk.patch_inference_mas(open(infer).read(), batch_size=batch_size)
    open(infer, "w").write(isrc)
    print(f"  run.py patches: {rc_counts}")
    print(f"  inference_mas patches: {im_counts}")

    # Run upstream. os.environ already carries the head's knobs.
    cmd = [
        sys.executable, "run.py",
        "--style", "sequential_light",
        "--dataset", "math500",
        "--seed", "42",
        "--num_recursive_rounds", str(T),
        "--trust_remote_code", "1",
        "--device", "cuda",
        "--temperature", "0.6",
        "--top_p", "0.95",
    ]
    print(f"  cmd={' '.join(cmd)}\n", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=upstream,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=dict(os.environ), bufsize=1, text=True,
    )
    all_out, final_acc = [], None
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        all_out.append(line)
        m = re.search(r"accuracy=([0-9.]+)%", line)
        if m:
            final_acc = float(m.group(1))
    proc.wait()
    rc = proc.returncode
    run_secs = time.time() - t0

    # ---- collect artifacts ----
    per_problem = fk.parse_per_problem_jsonl(jsonl_path)
    n_with_correct = sum(1 for p in per_problem if isinstance(p["correct"], bool))

    call_stats = None
    csp = f"{WORK}/fidelity_call_stats.json"
    if os.path.exists(csp):
        call_stats = json.loads(open(csp).read())

    n_patches = 0
    plog = f"{WORK}/vb_patches.log"
    if os.path.exists(plog):
        with open(plog) as f:
            n_patches = sum(1 for line in f if line.startswith("loader_call"))

    import numpy as np
    lpath = f"{WORK}/fidelity_logits.npz"
    n_logit_batches = 0
    if os.path.exists(lpath):
        with np.load(lpath) as z:
            n_logit_batches = int(z["n_batches"]) if "n_batches" in z else 0

    print(f"\n[result] rc={rc} elapsed={run_secs/60:.1f}min acc={final_acc} "
          f"patches={n_patches} per_problem={len(per_problem)} "
          f"(with_correct={n_with_correct}) logit_batches={n_logit_batches}")
    if per_problem and n_with_correct == 0:
        print("[WARN] per-problem records present but NONE had a correctness flag.")

    result = {
        "config": {
            "n_samples": n_samples,
            "batch_size": batch_size,
            "variant_b_bits": bits,
            "num_recursive_rounds": T,
            "capture_mode": True,
            "dtype_override": dtype,
            "topk_logits": topk,
            "max_logit_positions": maxpos,
            "gpu": gpu_name,
            "config_tag": config_tag,
            "seed": 42,
            "decoding": "greedy",
            "platform": "modal-a100",
            "recursive_mas_commit": "f95d512017fb713e9ac519248fbfd3d270dafd68",
        },
        "final_accuracy": final_acc,
        "n_patches_logged": n_patches,
        "n_per_problem_records": len(per_problem),
        "n_logit_batches": n_logit_batches,
        "return_code": rc,
        "run_seconds": run_secs,
        "fidelity_summary": call_stats,
        "per_problem": per_problem,
        "tail_log_lines": all_out[-150:],
    }

    # Persist to the output volume under a per-config subdir (so the same-named
    # fidelity_logits.npz from different configs/n never collide). analyze.py
    # finds the NPZ next to each run's JSON.
    outdir = (f"/out/vb{bits}_T{T}_n{n_samples}"
              + (f"_{links}" if links != "all" else "")
              + (f"_{tag}" if tag else ""))
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/fidelity_{config_tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    if os.path.exists(lpath):
        shutil.copy(lpath, f"{outdir}/fidelity_logits.npz")
    if call_stats is not None:
        shutil.copy(csp, f"{outdir}/fidelity_call_stats.json")
    out_vol.commit()
    print(f"[done] persisted to volume rmas-fidelity-out:/vb{bits}_T{T}/")

    # Return the small dict (drop the big logits + verbose log for transport).
    light = dict(result)
    light.pop("tail_log_lines", None)
    return light


@app.local_entrypoint()
def sweep(
    t_values: str = "1,2,3,4",
    n_samples: int = 50,
    batch_size: int = 8,
    topk: int = 512,
    maxpos: int = 256,
    dtype: str = "float32",
):
    """Fan out the full paired T-sweep concurrently (one container per config).

    Usage:
      modal run experiments/fidelity_sweep/modal_pkg/fidelity_modal.py::sweep \
          --t-values 1,2,3,4 --n-samples 50 --batch-size 8
    """
    Ts = [int(x) for x in t_values.split(",") if x.strip()]
    configs = [
        (bits, T, n_samples, batch_size, topk, maxpos, dtype)
        for T in Ts for bits in (0, 4)
    ]
    print(f"Sweep: {len(configs)} runs — T={Ts} x {{REF(0), INT4(4)}} "
          f"at n={n_samples} b={batch_size} fp32 (concurrent).")
    rows = []
    for r in run_fidelity.starmap(configs):
        rows.append(r)
        c = r["config"]
        print(f"  done {c['config_tag']}: acc={r['final_accuracy']}% "
              f"per_problem={r['n_per_problem_records']} logit_batches={r['n_logit_batches']} rc={r['return_code']}")
    print("\n=== sweep complete ===")
    for r in sorted(rows, key=lambda x: (x['config']['num_recursive_rounds'], x['config']['variant_b_bits'])):
        c = r["config"]
        print(f"  T={c['num_recursive_rounds']} bits={c['variant_b_bits']:>1} "
              f"acc={r['final_accuracy']}% logit_batches={r['n_logit_batches']}")
    print("\nDownload all: modal volume get rmas-fidelity-out . /tmp/fid_outputs/ --force")


@app.local_entrypoint()
def main(
    bits: int = 0,
    t: int = 3,
    n_samples: int = 50,
    batch_size: int = 4,
    topk: int = 512,
    maxpos: int = 256,
    dtype: str = "float32",
    tag: str = "",
    links: str = "all",
):
    cond = "REF" if bits == 0 else f"INT{bits}"
    print(f"Launching Modal A100 fidelity run: {cond} T={t} n={n_samples} "
          f"b={batch_size} topk={topk} maxpos={maxpos} dtype={dtype} links={links} tag='{tag}' ...")
    r = run_fidelity.remote(
        bits=bits, T=t, n_samples=n_samples, batch_size=batch_size,
        topk=topk, maxpos=maxpos, dtype=dtype, tag=tag, links=links,
    )
    print("\n" + "=" * 64)
    print("=== LOCAL: result returned from Modal ===")
    print("=" * 64)
    print(f"  config_tag:        {r['config']['config_tag']}")
    print(f"  final_accuracy:    {r['final_accuracy']}%")
    print(f"  patches_logged:    {r['n_patches_logged']}")
    print(f"  per_problem recs:  {r['n_per_problem_records']}")
    print(f"  logit batches:     {r['n_logit_batches']}")
    print(f"  return_code:       {r['return_code']}")
    print(f"  run_seconds:       {r['run_seconds']:.0f}  ({r['run_seconds']/60:.1f} min)")
    print()
    _suffix = (f"_{links}" if links != "all" else "") + (f"_{tag}" if tag else "")
    print("Download artifacts for analysis:")
    print(f"  modal volume get rmas-fidelity-out vb{bits}_T{t}_n{n_samples}{_suffix} /tmp/fid_outputs/")
