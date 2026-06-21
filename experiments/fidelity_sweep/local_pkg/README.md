# Local single-GPU fidelity backend

This is the third execution backend for `fidelity_sweep`, alongside Modal A100 and
Kaggle T4. It injects the same unit-tested Variant B and capture patches into a pinned
RecursiveMAS checkout, but runs locally on one CUDA GPU.

## Completed study

The backend produced four n=250 cells on an RTX 5070 Ti 16 GB under WSL2, with
native bf16, seed 42, and three recursive rounds:

- `sequential_light x math500`;
- `sequential_light x mbppplus`;
- `sequential_scaled x mbppplus`;
- `sequential_light x medqa`.

Each cell contains a sampled bit-rate ladder and paired greedy REF/INT4 runs. The
canonical cross-cell interpretation is
[`docs/reports/08_local_cross_cell_generalization.md`](../../../docs/reports/08_local_cross_cell_generalization.md).
Compact public JSONL files contain only fields required to reproduce paired answer
statistics; raw prompts, traces, logs, and NPZ captures are intentionally excluded.

## Environment

| item | value |
|---|---|
| GPU | NVIDIA RTX 5070 Ti, 16 GB, Blackwell sm_120 |
| OS | Windows 10 + WSL2 Ubuntu 20.04 |
| Python | CPython 3.12, PyTorch 2.9.0+cu128 |
| dtype | native bf16 (`--dtype auto`) |
| upstream | `external/RecursiveMAS` at `f95d512` |
| seed | 42 |

The scripts derive the repository root from their own location. Set optional
`LCC_RUN_ROOT` and `HF_HOME` environment variables to choose scratch and cache paths;
no personal absolute path is required.

## Reproduce

From this directory inside WSL:

```bash
python setup/verify_gpu.py
python setup/download_checkpoints.py
python setup/preflight_checks.py

# One sampled condition (no logit capture)
python fidelity_local.py --style sequential_light --dataset math500 --bits 4 \
  --t 3 --n-samples 250 --batch-size 16 --no-capture

# One paired-greedy capture condition
python fidelity_local.py --style sequential_light --dataset math500 --bits 4 \
  --t 3 --n-samples 250 --batch-size 2
```

The orchestration scripts reproduce the completed cells:

```bash
bash run_step0.sh
bash run_step1_mbppplus.sh
bash run_step2_scaled_mbppplus.sh
python run_cell.py --style sequential_light --dataset medqa \
  --ladder-batch 16 --cap-batch 2 --n 250
```

For scaled runs set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Batch 4 is
safe for the sampled ladder and batch 1 for greedy capture; batch 8 thrashes the
16 GB allocator.

## Analyze

```bash
python analysis/flip_churn_tost.py
python analysis/compare_cells.py
python analysis/tier2_logit_fidelity.py
```

Tier-2 pairs only the fixed primary generate calls implied by
`ceil(n_samples / batch_size)`. Conditional answer-retry calls are excluded because
REF and INT4 need not invoke them on the same samples. Local captures use top-K=256
and at most 128 positions; reported divergence is therefore windowed, and the KL is
an approximate matched-prefix metric.

## Artifact policy

The driver writes to `${LCC_RUN_ROOT:-$HOME/lcc/runs}`. Commit only small summaries,
configuration JSON, call statistics, and minimized per-problem correctness records.
Do not commit `fidelity_logits.npz`, full prompts, execution traces, cache paths, or
machine-specific logs. NPZ captures are regenerated from the commands above.

## Hazards

- Capture retains generation scores and needs a smaller batch than sampled runs.
- Pre-Ampere GPUs should use fp32; this Blackwell run safely uses bf16.
- MedQA greedy REF has a first-option bias, so its paired greedy accuracy is a
  diagnostic failure case. Use the sampled ladder for the task-level conclusion.
- Equal channel cosine does not imply equal semantic perturbations or prove that model
  size causes downstream robustness.
