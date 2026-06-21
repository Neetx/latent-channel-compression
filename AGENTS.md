# AGENTS.md

Project-root instructions for AI coding agents. Read this before changing code or
interpreting results.

## Project and evidence hierarchy

This repository instruments the latent communication channel of the third-party
RecursiveMAS system with TurboQuant Variant B (Haar rotation plus Lloyd--Max scalar
quantization) and measures channel distortion, answer behavior, and greedy
trajectory drift.

The **primary result** is the four-cell local RTX 5070 Ti study in
[`docs/reports/08_local_cross_cell_generalization.md`](docs/reports/08_local_cross_cell_generalization.md).
Kaggle/Modal reports 06--07 are independent historical cloud checks, not the primary
reproduction path.

Current local findings, n=250, seed 42, T=3:

- sampled REF/8/4/2-bit ladders show no detected monotonic degradation in four cells;
- clean greedy deltas are +2 pp (Math500/light), 0 pp (MBPP+/light), and -2 pp
  (MBPP+/scaled), all with 95% intervals spanning zero;
- answer churn is 4.4--10% in those clean cells;
- corrected divergence within 128 positions is 86.4%, 92.8%, 51.2%, and 96.4%;
- MBPP+/scaled is more trajectory-robust than MBPP+/light, an exploratory tier
  association rather than a causal capacity law;
- MedQA greedy is confounded by a pathological REF first-option bias.

Never call the intervention lossless, formally equivalent, trajectory preserving, or
a deployed 4x--16x bandwidth saving. It is fake quantization with nominal packed
payload ratios.

## Ownership boundary: RecursiveMAS is read-only upstream

`external/RecursiveMAS` is a gitignored clone of the code released with the
RecursiveMAS paper. It is not our repository.

- Never edit, patch in place, restore, commit, or push files in that clone.
- Pin it to `f95d512017fb713e9ac519248fbfd3d270dafd68`.
- The local driver verifies the commit and tracked cleanliness, copies the source to
  the run directory, and instruments only the disposable copy.
- Do not vendor upstream code or model checkpoints into this repository.

## Primary local reproduction

Follow [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). The tested environment is Python
3.12, PyTorch 2.9.0+cu128, native bf16 on RTX 5070 Ti 16 GB under WSL2.

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export LCC_RUN_ROOT="${LCC_RUN_ROOT:-$HOME/lcc/runs}"
export PYTHONDONTWRITEBYTECODE=1

.venv/bin/python -m pytest tests/ -q
.venv/bin/python experiments/fidelity_sweep/local_pkg/setup/verify_gpu.py

.venv/bin/python experiments/fidelity_sweep/local_pkg/run_cell.py \
  --style sequential_light --dataset math500 --n 250 \
  --ladder-batch 16 --cap-batch 2
```

`run_cell.py` is canonical. Historical `run_step*.sh` wrappers are compatibility
helpers, not the documented entrypoint. A full four-cell reproduction takes about
36 sequential GPU-hours and roughly 34 GB of checkpoint cache.

Post-hoc checks:

```bash
# Works from committed compact correctness records, no GPU/raw NPZ required.
.venv/bin/python experiments/fidelity_sweep/local_pkg/analysis/compare_cells.py

# Fresh local run; raw NPZ required for Tier-2.
.venv/bin/python experiments/fidelity_sweep/local_pkg/analysis/compare_cells.py \
  --run-root "$LCC_RUN_ROOT"
.venv/bin/python experiments/fidelity_sweep/local_pkg/analysis/tier2_logit_fidelity.py \
  --run-root "$LCC_RUN_ROOT"
```

## Repository map

| path | role |
|---|---|
| `src/quantizers/turboquant_honest.py` | Variant B implementation; do not change while adding measurement |
| `src/adapters/patch.py` | reversible adapter instrumentation |
| `src/metrics/` | deterministic fidelity/bootstrap metrics |
| `experiments/fidelity_sweep/local_pkg/` | primary local runner, setup, analyses, compact results |
| `experiments/fidelity_sweep/kernel_pkg/` | shared patch functions and historical Kaggle backend |
| `experiments/fidelity_sweep/modal_pkg/` | historical Modal backend |
| `docs/reports/08_*` | canonical current result |
| `docs/reports/05_*` | hardware/dtype failure analysis |
| `writeup/main.tex`, `main.pdf` | current paper and compiled PDF |

## Scientific invariants

1. **Instrumentation only.** Do not alter model weights, prompts, quantizer math, or
   evaluation semantics while adding measurement.
2. **Determinism.** Statistical helpers take an explicit seed and use
   `numpy.random.default_rng(seed)`.
3. **Primary-call pairing.** Tier-2 pairs only fixed primary generate batches;
   condition-dependent answer retries are excluded.
4. **Correct tail accounting.** Missing union-token estimates are deducted from the
   residual tail; do not double count them.
5. **Censoring.** Local divergence means divergence within at most 128 captured
   positions, not full-generation divergence.
6. **Matched-prefix limitation.** KL/JS are top-K and selection-conditioned; do not
   infer teacher-forced full-trajectory fidelity or a reliable depth trend.
7. **MedQA exclusion.** Do not use the greedy MedQA delta as a clean channel effect.

## Hardware safety

- Ampere or newer: `--dtype auto` may use native bf16; verify with `verify_gpu.py`.
- Pre-Ampere: force fp32. Auto/bf16 silently collapsed earlier RecursiveMAS runs.
- Local light batches: 16 sampled, 2 capture.
- Local scaled MBPP+ batches: 4 sampled, 1 capture, with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Do not extrapolate exact sampled accuracies across hardware/dtype configurations.

## Public-repository hygiene

- No secrets, API tokens, account usernames, author email, or personal absolute paths.
- The approved publication author name is `Antonio Pastorelli`.
- Do not commit `external/`, `.venv/`, checkpoints, model weights, NPZs, raw prompts,
  generations, caches, or machine logs.
- Compact correctness JSONLs may be committed after removing prompts, dataset paths,
  traces, and error payloads.
- Raw artifacts should be archived separately with SHA256 manifests if they are to be
  cited as a reproducibility bundle.

## Tests and documentation discipline

Run before committing:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m py_compile \
  experiments/fidelity_sweep/local_pkg/*.py \
  experiments/fidelity_sweep/local_pkg/analysis/*.py \
  experiments/fidelity_sweep/local_pkg/setup/*.py
git diff --check
```

When results or methods change, update together:

- `README.md`, `REPRODUCIBILITY.md`, `ROADMAP.md`, and this file;
- `docs/RESEARCH.md` and REPORT_08;
- `writeup/main.tex` and the compiled `writeup/main.pdf`;
- relevant experiment README/result summaries and tests.

## Highest-value open work

1. multi-seed MBPP+ light/scaled replication and scaled Math500;
2. teacher-forced position-aligned KL plus token-margin mediation;
3. rate-distortion sweep with scalar/no-rotation baseline and QJL residual;
4. deliberation/tool-calling topology and a strong non-math benchmark;
5. real packed bytes, codec latency, VRAM, and end-to-end throughput;
6. public raw-artifact archive with manifests.
