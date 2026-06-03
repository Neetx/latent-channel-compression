"""Phase 0.E — Modal A100 40GB: upstream run.py with paper's batch_size=32.

The definitive test. Phase 0.D got 35% on P100 b=8 (forced by 16GB VRAM).
Paper claims 75.8% on math500 r=1 with b=32 on A100/H100. This run answers:
"does forcing b=8 alone account for the gap, or is the demo code itself
the bottleneck?"

Outcome interpretation:
  - acc ≈ 70-78%  → batch_size was the bug. Paper reproducible on demo code.
  - acc ≈ 50-65%  → batch_size matters but isn't the whole story.
  - acc ≈ 35-45%  → batch_size irrelevant. Demo code is incomplete (matches
                    Issue #20 conclusion). Pivot publication framing.

Cost estimate: ~$1.50-3.00 (A100 40GB @ $1.42/h × ~1-2h).

Usage:
  modal run phase0e_modal.py                    # default n=100, b=32
  modal run phase0e_modal.py --n-samples 200    # bigger eval
  modal run phase0e_modal.py --batch-size 16    # if curious about b=16
"""
import modal

RECURSIVEMAS_COMMIT = "f95d512017fb713e9ac519248fbfd3d270dafd68"

app = modal.App("rmas-phase0e-upstream-b32")

# Pre-install everything at image build time → faster cold start, cached.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # Match upstream RecursiveMAS requirements.txt exactly (minus torch).
        "transformers==5.3.0",
        "datasets==4.4.2",
        "huggingface-hub==1.7.1",
        "accelerate==1.12.0",
        "safetensors==0.7.0",
        "tokenizers==0.22.2",
        "tqdm",
        "numpy",
        "sentencepiece",
        "python-dotenv",
    )
    # Clone upstream once at image build time. /opt is read-only; we'll
    # copytree to /tmp at runtime to apply our 2 patches.
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

# Persistent volume for HF model cache — first run downloads 9 GB,
# subsequent runs reuse instantly.
hf_cache = modal.Volume.from_name("rmas-hf-cache", create_if_missing=True)
# Persistent volume for output JSON.
out_vol = modal.Volume.from_name("rmas-phase0e-out", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3 * 60 * 60,  # 3h hard cap (Modal kills past this)
    volumes={"/cache/hf": hf_cache, "/out": out_vol},
)
def run_phase0e(n_samples: int = 100, batch_size: int = 32) -> dict:
    import json
    import os
    import re
    import shutil
    import subprocess
    import sys
    import time

    T0 = time.time()
    print(f"=== Phase 0.E Modal A100-40GB — n={n_samples} b={batch_size} ===")
    print(f"  GPU: {os.popen('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader').read().strip()}")

    # Copy upstream into writable /tmp so we can patch in-place.
    upstream = "/tmp/RecursiveMAS"
    if os.path.isdir(upstream):
        shutil.rmtree(upstream)
    shutil.copytree("/opt/RecursiveMAS", upstream)

    # ---- PATCH 1: cap num_samples (run.py:185 hardcodes -1 = full 500) ----
    # Use re.subn so we can check that the pattern actually MATCHED (count>0),
    # rather than that the output differs (which fails if new value == old).
    run_py = f"{upstream}/run.py"
    src = open(run_py).read()
    patched, n_subs = re.subn(
        r'("--num_samples",\s*)"-1"', rf'\1"{n_samples}"', src, count=1
    )
    if n_subs == 0:
        raise RuntimeError("Failed to MATCH --num_samples line in run.py")
    open(run_py, "w").write(patched)

    # ---- PATCH 2: pin batch_size in RELEASE_RECOMMENDED_SETTINGS ----
    infer_mas = f"{upstream}/inference_utils/inference_mas.py"
    isrc = open(infer_mas).read()
    ipatched, ni_subs = re.subn(
        r'\("sequential_light",\s*"math500"\):\s*\{[^}]*"batch_size":\s*\d+',
        f'("sequential_light", "math500"): {{"seed": 42, "batch_size": {batch_size}',
        isrc,
        count=1,
    )
    if ni_subs == 0:
        raise RuntimeError("Failed to MATCH batch_size in RELEASE_RECOMMENDED_SETTINGS")
    open(infer_mas, "w").write(ipatched)

    # Verify
    print("  patched lines:")
    for line in patched.splitlines():
        if "--num_samples" in line:
            print(f"    run.py:           {line.strip()}")
    for line in ipatched.splitlines():
        if "sequential_light" in line and "math500" in line and "batch_size" in line:
            print(f"    inference_mas.py: {line.strip()}")

    # ---- RUN upstream run.py ----
    print(f"\n[invoke] upstream run.py")
    t = time.time()
    cmd = [
        sys.executable, "run.py",
        "--style", "sequential_light",
        "--dataset", "math500",
        "--seed", "42",
        "--trust_remote_code", "1",
        "--device", "cuda",
        "--temperature", "0.6",
        "--top_p", "0.95",
    ]
    print(f"  cwd={upstream}  cmd={' '.join(cmd)}\n")
    proc = subprocess.Popen(
        cmd, cwd=upstream,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )

    all_out: list[str] = []
    final_acc: float | None = None
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
    run_secs = time.time() - t
    print(f"\n[result] rc={rc}  elapsed={run_secs:.0f}s ({run_secs/60:.1f} min)  "
          f"final_acc={final_acc}")

    result = {
        "config": {
            "n_samples": n_samples,
            "batch_size": batch_size,
            "gpu": "A100-40GB",
            "style": "sequential_light",
            "dataset": "math500",
            "seed": 42,
            "temperature": 0.6,
            "top_p": 0.95,
            "recursive_mas_commit": "f95d512017fb713e9ac519248fbfd3d270dafd68",
        },
        "final_accuracy": final_acc,
        "return_code": rc,
        "run_seconds": run_secs,
        "total_seconds": time.time() - T0,
        "tail_log_lines": all_out[-200:],
    }
    out_path = f"/out/phase0e_n{n_samples}_b{batch_size}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    out_vol.commit()  # persist for future modal volume get
    print(f"\n[done] wrote {out_path}")
    return result


@app.local_entrypoint()
def main(n_samples: int = 100, batch_size: int = 32):
    """Local CLI: `modal run phase0e_modal.py --n-samples 100 --batch-size 32`."""
    print(f"Launching Modal A100-40GB run: n={n_samples}, b={batch_size} ...")
    result = run_phase0e.remote(n_samples=n_samples, batch_size=batch_size)
    print("\n" + "=" * 60)
    print("=== LOCAL: result returned from Modal ===")
    print("=" * 60)
    print(f"  final_accuracy:  {result['final_accuracy']}%")
    print(f"  return_code:     {result['return_code']}")
    print(f"  run_seconds:     {result['run_seconds']:.0f}  ({result['run_seconds']/60:.1f} min)")
    print(f"  total_seconds:   {result['total_seconds']:.0f}")
    print()
    print(f"JSON saved to Modal volume 'rmas-phase0e-out':")
    print(f"  modal volume get rmas-phase0e-out phase0e_n{n_samples}_b{batch_size}.json")
