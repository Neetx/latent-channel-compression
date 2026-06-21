# Primary local RTX 5070 Ti backend

This directory contains the primary execution path for the four-cell study in
REPORT_08. The exact clean-clone environment and full commands are in the root
[`REPRODUCIBILITY.md`](../../../REPRODUCIBILITY.md); this file documents the backend
contract for maintainers.

## Scientific configuration

| item | published value |
|---|---|
| GPU | RTX 5070 Ti 16 GB, Blackwell sm_120 |
| runtime | WSL2, Python 3.12, PyTorch 2.9.0+cu128 |
| dtype | native bf16 (`--dtype auto`) |
| upstream | RecursiveMAS `f95d512017fb713e9ac519248fbfd3d270dafd68` |
| seed / rounds / n | 42 / 3 / 250 |
| capture | top-K=256, maximum 128 positions |
| intervention | Variant B at every inner and outer link |

Completed cells are light x {Math500, MBPP+, MedQA} and scaled x MBPP+.

## Upstream isolation

RecursiveMAS is third-party upstream code and is never modified. `fidelity_local.py`:

1. verifies the exact upstream commit and a clean tracked worktree;
2. copies the source, excluding `.git` and Python caches, into the condition output;
3. applies instrumentation only to the disposable copy;
4. runs that copy against checkpoints in `HF_HOME`;
5. deletes the disposable source copy in `finally`.

The runner fails rather than checking out, restoring, or overwriting upstream files.

## Canonical orchestrator

Use `run_cell.py`, not the historical `run_step*.sh` wrappers:

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export LCC_RUN_ROOT="${LCC_RUN_ROOT:-$HOME/lcc/runs}"

.venv/bin/python experiments/fidelity_sweep/local_pkg/run_cell.py \
  --style sequential_light --dataset math500 --n 250 \
  --ladder-batch 16 --cap-batch 2
```

For each cell it runs the sampled ladder in order REF/8/4/2 and then paired greedy
REF/INT4. Each condition has a fresh Python/CUDA process. The orchestrator stops on
the first failure and validates its machine-readable result contract before moving
on. `cell_manifest.json` records commands, durations, return codes, and paths.

Default output:

```text
${LCC_RUN_ROOT}/sequential_<tier>_<dataset>/
  cell_manifest.json
  ladder_b*_n250.log
  fidelity_b*_n250.log
  <dataset>_vb<bit>_T3_n250_b<batch>_auto/
    fidelity_<tag>.json
    per_problem_<tag>.jsonl       # greedy only
    fidelity_logits.npz           # greedy only
    fidelity_call_stats.json      # quantized greedy only
```

## Setup and smoke test

```bash
.venv/bin/python experiments/fidelity_sweep/local_pkg/setup/verify_gpu.py
.venv/bin/python experiments/fidelity_sweep/local_pkg/setup/download_checkpoints.py
.venv/bin/python experiments/fidelity_sweep/local_pkg/setup/download_scaled.py
.venv/bin/python experiments/fidelity_sweep/local_pkg/setup/preflight_checks.py

.venv/bin/python experiments/fidelity_sweep/local_pkg/fidelity_local.py \
  --style sequential_light --dataset math500 --bits 4 --t 3 \
  --n-samples 4 --batch-size 2 --out "$LCC_RUN_ROOT/smoke"
```

Light uses batch 16 for sampled runs and 2 for capture. Scaled MBPP+ uses batch 4
and 1 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Analysis

```bash
# Recompute the published answer table from committed compact artifacts.
.venv/bin/python experiments/fidelity_sweep/local_pkg/analysis/compare_cells.py

# Analyze a newly completed four-cell run.
.venv/bin/python experiments/fidelity_sweep/local_pkg/analysis/compare_cells.py \
  --run-root "$LCC_RUN_ROOT"
.venv/bin/python experiments/fidelity_sweep/local_pkg/analysis/tier2_logit_fidelity.py \
  --run-root "$LCC_RUN_ROOT"
```

Tier-2 pairs only fixed primary calls and excludes conditional answer retries. Its
divergence is windowed and its KL is a top-K, matched-prefix approximation.

## Artifact policy

The repository keeps compact configuration/result JSONs, minimized correctness
JSONLs, summaries, and analysis code. It excludes NPZs, verbose generations, raw
prompts, logs, checkpoints, caches, and personal paths. Public results are in
[`results/`](results/); raw local captures must be regenerated for exact Tier-2
reproduction.
