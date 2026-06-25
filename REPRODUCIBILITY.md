# Reproducing the study locally on an RTX 5070 Ti

This is the primary reproduction path for the paper. It starts from a clean clone,
runs the four published local cells on one 16 GB GPU, and regenerates the answer and
trajectory tables. Kaggle and Modal are historical independent checks and are
documented only in the appendix below.

## What this reproduces

The local experiment is RecursiveMAS at commit `f95d512`, seed 42, three recursive
rounds, native bf16, and $n=250$ per condition:

| cell | sampled ladder | paired greedy capture | measured time |
|---|---|---|---:|
| `sequential_light x math500` | REF/8/4/2 bit | REF/INT4 | 6.3 h |
| `sequential_light x mbppplus` | REF/8/4/2 bit | REF/INT4 | 9.7 h |
| `sequential_scaled x mbppplus` | REF/8/4/2 bit | REF/INT4 | 9.4 h |
| `sequential_light x medqa` | REF/8/4/2 bit | REF/INT4 | 11.0 h |

The measured total is about 36 GPU-hours when run sequentially. Times are from the
published RTX 5070 Ti run and will vary. The light and scaled checkpoints require
roughly 34 GB of cache; reserve at least 50 GB for checkpoints, outputs, and temporary
files. Raw NPZ capture size also depends on generated length.

## 1. Required platform

The published configuration was:

- NVIDIA RTX 5070 Ti, 16 GB, compute capability 12.0;
- Windows 10 with WSL2 Ubuntu 20.04;
- NVIDIA driver visible inside WSL;
- Python 3.12;
- PyTorch 2.9.0+cu128 and the versions pinned in `requirements.txt`.

Another Ampere-or-newer CUDA GPU with native bf16 can be used, but that is a new
hardware replication rather than an exact reproduction. Pre-Ampere GPUs must use
fp32 and are not the primary path.

## 2. Clone this repository and the read-only upstream

```bash
git clone https://github.com/Neetx/latent-channel-compression.git
cd latent-channel-compression

git clone https://github.com/RecursiveMAS/RecursiveMAS.git external/RecursiveMAS
git -C external/RecursiveMAS checkout f95d512017fb713e9ac519248fbfd3d270dafd68
git -C external/RecursiveMAS status --porcelain --untracked-files=no
```

The last command must print nothing. RecursiveMAS is third-party upstream code. The
local driver verifies its commit, copies the source into the run directory, patches
only that disposable copy, and deletes the copy afterward. It never edits or restores
files in `external/RecursiveMAS`.

## 3. Create the tested Python/CUDA environment

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip

# Exact CUDA build used by the RTX 5070 Ti run.
.venv/bin/python -m pip install torch==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128

.venv/bin/python -m pip install -r requirements.txt
```

Choose persistent locations outside the git checkout:

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export LCC_RUN_ROOT="${LCC_RUN_ROOT:-$HOME/lcc/runs}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HF_HOME" "$LCC_RUN_ROOT"
```

Run the complete CPU test suite and GPU sanity check:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python experiments/fidelity_sweep/local_pkg/setup/verify_gpu.py
```

The GPU check must report CUDA available, capability `(12, 0)` on the exact target,
and a successful bf16 matrix multiplication.

## 4. Download and validate the released checkpoints

```bash
# Sequential-Light (~10 GB) and its adapter manifests.
.venv/bin/python experiments/fidelity_sweep/local_pkg/setup/download_checkpoints.py

# Sequential-Scaled (~24 GB), with resumable retries.
.venv/bin/python experiments/fidelity_sweep/local_pkg/setup/download_scaled.py

# Confirm math/code adapter and outer-link resolution.
.venv/bin/python experiments/fidelity_sweep/local_pkg/setup/preflight_checks.py
```

Set `HF_TOKEN` only if Hugging Face throttles anonymous downloads. Never store it in
the repository. All checkpoint repositories are listed explicitly in the setup
scripts and are resolved through the pinned RecursiveMAS manifest logic.

## 5. Run a cheap end-to-end smoke test first

This exercises the source-copy isolation, patch injection, GPU model load, INT4
quantizer, greedy capture, JSONL output, call statistics, and NPZ output:

```bash
.venv/bin/python experiments/fidelity_sweep/local_pkg/fidelity_local.py \
  --style sequential_light --dataset math500 --bits 4 --t 3 \
  --n-samples 4 --batch-size 2 --topk 256 --maxpos 128 \
  --out "$LCC_RUN_ROOT/smoke"
```

Success requires return code 0 and a final line reporting four per-problem records,
at least one logit batch, and channel statistics. After the run:

```bash
git -C external/RecursiveMAS status --porcelain --untracked-files=no
```

must still print nothing.

## 6. Reproduce the four local cells

`run_cell.py` is the canonical orchestrator. Each condition runs in a fresh Python
process. It fails fast on a nonzero child process and validates the result JSON,
paired-record count, NPZ presence, and INT4 call statistics before continuing.

```bash
.venv/bin/python experiments/fidelity_sweep/local_pkg/run_cell.py \
  --style sequential_light --dataset math500 --n 250 \
  --ladder-batch 16 --cap-batch 2

.venv/bin/python experiments/fidelity_sweep/local_pkg/run_cell.py \
  --style sequential_light --dataset mbppplus --n 250 \
  --ladder-batch 16 --cap-batch 2

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python experiments/fidelity_sweep/local_pkg/run_cell.py \
  --style sequential_scaled --dataset mbppplus --n 250 \
  --ladder-batch 4 --cap-batch 1

.venv/bin/python experiments/fidelity_sweep/local_pkg/run_cell.py \
  --style sequential_light --dataset medqa --n 250 \
  --ladder-batch 16 --cap-batch 2
```

The output layout is deterministic:

```text
$LCC_RUN_ROOT/
  sequential_light_math500/
  sequential_light_mbppplus/
  sequential_scaled_mbppplus/
  sequential_light_medqa/
```

Each cell contains six condition directories, six logs, and
`cell_manifest.json`. The condition directories contain machine-readable result JSON;
greedy conditions additionally contain per-problem JSONL and `fidelity_logits.npz`;
INT4 greedy conditions contain `fidelity_call_stats.json`.

## 7. Regenerate the published analyses

### Answer-level table from a fresh run

```bash
.venv/bin/python \
  experiments/fidelity_sweep/local_pkg/analysis/compare_cells.py \
  --run-root "$LCC_RUN_ROOT"
```

With no `--run-root`, the same command analyzes the compact committed JSONLs and
reproduces the published paired answer table without a GPU:

```bash
.venv/bin/python experiments/fidelity_sweep/local_pkg/analysis/compare_cells.py
```

For a detailed single-cell contingency and TOST, pass the REF and INT4 JSONLs to
`analysis/flip_churn_tost.py --ref ... --int4 ...`.

### Corrected Tier-2 table from raw local captures

```bash
.venv/bin/python \
  experiments/fidelity_sweep/local_pkg/analysis/tier2_logit_fidelity.py \
  --run-root "$LCC_RUN_ROOT"
```

The analyzer pairs only the fixed primary batches and excludes conditional answer
retries. Local capture is top-K=256 and right-censored at 128 positions. The expected
table is recorded in REPORT_08 and
`experiments/fidelity_sweep/local_pkg/results/tier2_logit_fidelity_SUMMARY.md`.
Raw NPZs are not committed because of their size; exact Tier-2 reproduction therefore
requires running the capture conditions and verifying them against the committed checksum
manifest (below).

### Additional confirmatory analyses (rotation matrix, link ablation, mechanism)

The same capture driver produces the three follow-up analyses. Each orchestrator is resumable and
reuses the rotation-independent greedy REF where possible.

```bash
# Quantizer-rotation matrix (MBPP+): five INT4 rotations per tier, REF reused.
.venv/bin/python experiments/fidelity_sweep/local_pkg/run_rotation_matrix.py
.venv/bin/python experiments/fidelity_sweep/local_pkg/analysis/rotation_matrix_analysis.py

# Inner/outer link ablation (MBPP+): one INT4 capture per link type per tier (_li/_lo tag).
.venv/bin/python experiments/fidelity_sweep/local_pkg/run_links_ablation.py
.venv/bin/python experiments/fidelity_sweep/local_pkg/analysis/links_ablation_analysis.py

# Teacher-forced mechanism (MBPP+ and a Math500 control). Forced-decoding capture; runs at
# batch size 1 only -- gate G0 (teacher-forcing the full-precision REF reproduces the
# free-running REF exactly) holds at b=1 but not at b>1 (batched post-EOS padding diverges).
.venv/bin/python experiments/fidelity_sweep/local_pkg/run_teacher_forced.py
.venv/bin/python experiments/fidelity_sweep/local_pkg/run_teacher_forced.py \
  --dataset math500 --out "$LCC_RUN_ROOT/teacher_forced_math500"
.venv/bin/python experiments/fidelity_sweep/local_pkg/analysis/teacher_forced_analysis.py
.venv/bin/python experiments/fidelity_sweep/local_pkg/analysis/teacher_forced_analysis.py \
  --dataset math500 --tf-root "$LCC_RUN_ROOT/teacher_forced_math500"
```

Expected numbers are in `experiments/fidelity_sweep/local_pkg/results/` as
`rotation_matrix_SUMMARY.md`, `links_ablation_SUMMARY.md`, and `teacher_forced_SUMMARY.md`.

### Verify raw artifacts against the checksum manifest

The raw NPZ captures are too large to commit, so a SHA256 + provenance manifest of every capture
is committed at
`experiments/fidelity_sweep/local_pkg/results/artifact_manifest.json`. It records, per artifact,
the capture-root-relative path, byte size, SHA256, and provenance (config tag, links, quantizer
seed, teacher-forced flag, upstream commit), plus the runtime environment (Python/PyTorch/CUDA/GPU
and repository commit) -- no secrets or absolute paths. Re-verify a regenerated set:

```bash
.venv/bin/python experiments/fidelity_sweep/local_pkg/make_artifact_manifest.py \
  --out /tmp/regenerated_manifest.json
# compare the artifact lists (requires jq); identical SHA256s mean a byte-exact reproduction.
diff <(jq -S .artifacts experiments/fidelity_sweep/local_pkg/results/artifact_manifest.json) \
     <(jq -S .artifacts /tmp/regenerated_manifest.json)
```

## 8. Rebuild the paper

```bash
.venv/bin/python writeup/figures/make_figures.py
cd writeup
tectonic main.tex
```

The paper PDF is committed as `writeup/main.pdf`.

## 9. Reproducibility boundaries

- The experiment uses fake quantization: it measures the reconstructed vector and
  task behavior, not serialized network traffic or codec latency.
- A single seed does not establish formal equivalence or a causal capacity law.
- MedQA greedy REF is confounded by a first-option bias; use its sampled ladder for
  task-level interpretation.
- Matched-prefix KL is approximate and selection-conditioned; the committed teacher-forced
  per-position analysis is its aligned complement, but is itself MBPP+ and a Math500 control at
  a single generation seed.
- Exact stochastic sampled accuracies require the pinned software, checkpoint cache,
  sample order, and hardware/dtype path above.

## Appendix: historical cloud replications

Kaggle T4 fp32 produced the original Math500 bit-rate ladder (REPORT_06). Modal A100
fp32 produced the original greedy controls and depth sweep (REPORT_07). They are
valuable independent hardware checks and document the dtype failures that led to the
safe local configuration, but they are no longer the primary reproduction path.

Cloud commands and artifact retrieval remain documented in
`experiments/variant_b_ladder_t4_kaggle/` and
`experiments/fidelity_sweep/{kernel_pkg,modal_pkg}/`. REPORT_07's historical KL and
divergence values used the legacy all-call/tail estimator; reanalyze its raw NPZs with
the current analyzer before comparing them numerically with the corrected local table.
