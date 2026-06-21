# AGENTS.md

Project-root instructions for AI coding agents working on this repository.
Read this first. For deeper context follow the links to `docs/`.

## What this project is

Research codebase that applies **TurboQuant Variant B** (Haar rotation +
Lloyd-Max-Gaussian per-coordinate quantizer) to RecursiveMAS latent channels and
measures channel distortion, generated-trajectory drift, and downstream accuracy
across Sequential-Light/Scaled and math, code, and medical-QA tasks.

**Canonical cloud finding (n=250):** Variant B compresses the
Sequential-Light/math500 channel **4× to 16× with no detected accuracy change under
sampled decoding**. Under greedy decoding, a stricter paired ±2 pp equivalence test
at 4-bit is **INCONCLUSIVE** (Δ=−2.0 pp, not significant), while the trajectory
changes in most captured sequences. See reports [06](docs/reports/06_variant_b_in_loop_HEADLINE.md)
and [07](docs/reports/07_fidelity_sweep_modal.md).

**Latest local extension (2026-06-21):** four cells are complete on a 16 GB Blackwell
GPU: light×{math500, MBPP+, MedQA} and scaled×MBPP+. The three clean math/code cells
show small, non-significant aggregate accuracy deltas with 4.4--10% correctness churn.
On MBPP+, corrected primary-solver-only Tier-2 analysis finds divergence within the
first 128 positions of **92.8% (light) vs 51.2% (scaled)** at the same mean channel
cosine (0.9953). Treat this as a strong tier association, **not yet a causal model-
capacity law**. MedQA greedy is confounded by a pathological first-option bias in REF.
See [report 08](docs/reports/08_local_cross_cell_generalization.md).

## Repository layout

```
.
├── AGENTS.md                ← this file
├── README.md                ← public overview (TL;DR + repro)
├── .gitignore               ← excludes secrets, model artifacts, captured outputs
├── bin/
│   ├── kaggle               ← uvx wrapper for Kaggle CLI 2.x (use this, not `pip install kaggle`)
│   ├── push_kaggle_vb_kernel.sh    ← portable push for the headline experiment
│   └── push_fidelity_kernel.sh     ← portable push for the fidelity sweep
├── docs/                    ← ALL documentation lives here (no .md sprinkled around)
│   ├── README.md            ← documentation index
│   ├── RESEARCH.md          ← master research design — read after headline
│   ├── reports/             ← 8 numbered reports (06 headline, 07 cloud fidelity, 08 local generalization)
│   ├── design/              ← architectural plans
│   ├── operations/          ← experiments log + external reproducibility audit
│   └── figures/             ← matplotlib PNG + the script that regenerates them
├── src/                     ← Variant B + patcher + new fidelity metrics
│   ├── quantizers/turboquant_honest.py  ← THE quantizer (don't touch logic, measure only)
│   ├── adapters/patch.py    ← monkey-patches CrossModelAdapter.forward
│   ├── metrics/
│   │   ├── distortion.py          ← rMSE, cosine, norm_ratio, inner_product_error
│   │   ├── channel_fidelity.py    ← FidelityRun, effective_rank, codebook_extreme_rate
│   │   ├── logit_metrics.py       ← MSE, KL, JS at egress
│   │   └── bootstrap.py           ← paired bootstrap + TOST equivalence
│   └── utils/lloyd_max.py
├── tests/                   ← unit/integration tests; run pytest for the current count
└── experiments/             ← only 4 active folders — historical clutter archived
    ├── distortion_validation/        ← per-link rMSE validation (write-up §4.1)
    ├── solver_diagnostic/            ← Solver-alone math500 sanity (83%)
    ├── baseline_a100_modal/          ← Modal A100 baseline reproduction
    ├── variant_b_ladder_t4_kaggle/   ← main accuracy experiment — n=250 bit-rate ladder
    └── fidelity_sweep/               ← paired fidelity; Kaggle, Modal, and local single-GPU backends share tested patch functions
```

## Build / test / run

```bash
# Run all tests (must stay green; the TurboQuant reference module may skip if absent)
.venv/bin/python -m pytest tests/

# Validate the fidelity sweep's risky logic WITHOUT a GPU: the kernel's regex
# patches are checked against the real cloned upstream, and the JSONL parser +
# analysis glue are exercised on synthetic data. Run before spending Kaggle quota.
.venv/bin/python -m pytest tests/test_fidelity_metrics.py \
    tests/test_fidelity_kernel.py tests/test_fidelity_analyze.py -v

# Regenerate publication figures (matplotlib PNGs into docs/figures/)
.venv/bin/python docs/figures/_generate_figures.py

# Re-clone upstream (NOT in tree; required for any Kaggle/Modal experiment)
git clone https://github.com/RecursiveMAS/RecursiveMAS.git external/RecursiveMAS

# Headline experiment: push one Kaggle T4 kernel
./bin/push_kaggle_vb_kernel.sh <bits> <n_samples> <batch_size>
# e.g.  ./bin/push_kaggle_vb_kernel.sh 4 250 4

# Fidelity sweep (Tier 2) — Modal A100 fp32 (PRIMARY since Kaggle quota ran out)
modal run experiments/fidelity_sweep/modal_pkg/fidelity_modal.py --bits 0 --t 3 --n-samples 30 --batch-size 8  # REF
modal run experiments/fidelity_sweep/modal_pkg/fidelity_modal.py --bits 4 --t 3 --n-samples 30 --batch-size 8  # INT4
# download both then analyze (see below). Modal A100 fp32 avoids the bf16 cast artifact.

# Fidelity sweep (Tier 2) — Kaggle T4 (free, when weekly quota is available)
./bin/push_fidelity_kernel.sh <bits> <T> <n_samples> <batch_size>
# e.g.  ./bin/push_fidelity_kernel.sh 0 3 50 4   # REF, T=3, n=50, b=4
#       ./bin/push_fidelity_kernel.sh 4 3 50 4   # INT4 (4-bit), T=3, n=50, b=4

# Analyze fidelity sweep results post-hoc
.venv/bin/python experiments/fidelity_sweep/analysis/analyze.py \
    --inputs <dir_with_downloaded_kernel_outputs> \
    --logit-dir <dir_with_per_kernel_NPZ_subdirs> \
    --out experiments/fidelity_sweep/analysis/results

# Local single-GPU backend (native bf16 on Ampere+; see local_pkg/README.md)
.venv/bin/python experiments/fidelity_sweep/local_pkg/run_cell.py \
    --style sequential_light --dataset math500 --n 250 \
    --ladder-batch 16 --cap-batch 2
```

## Key invariants (DO NOT VIOLATE)

1. **No secrets in tree.** `.kaggle/`, `.mcp.json`, `.env`, `*.env` are gitignored AND must not exist in the working tree. The previous cleanup removed a real Kaggle API token (`KGAT_…`) — never re-introduce it.
2. **No personal or machine-specific info (this repo is public).** The publication
   author name `Antonio Pastorelli` is explicitly approved; do not add an email,
   account username, home-directory path, or local cache path. Kaggle uses
   `<YOUR_KAGGLE_USERNAME>`. Scrub captured logs before committing.
3. **`external/` is gitignored.** Don't commit the upstream RecursiveMAS clone — agents must `git clone` it themselves.
4. **All scripts must be reproducible from a single command.** Helper scripts use `$PROJECT_ROOT` resolution, not absolute paths. Test it from a clean checkout.
5. **No edits to model or quantizer logic when adding measurement.** The fidelity sweep is **instrumentation only** — `src/quantizers/turboquant_honest.py` and the recursive pipeline are not modified. See `docs/reports/05_hardware_root_cause.md` §6 and `docs/reports/06_variant_b_in_loop_HEADLINE.md` for the methodology contract.
6. **Determinism.** Every new metric/bootstrap function accepts an explicit `seed` and uses `numpy.random.default_rng(seed)` so a single seed reproduces the entire distribution bit-for-bit.

## Hardware advisory (critical for any new experiment)

RecursiveMAS Sequential-Light **silently collapses to ~30% accuracy** on pre-Ampere GPUs with default `--dtype auto` (which loads bf16 from the checkpoint config and falls back to fp16 on non-bf16 hardware). This invalidated 22 P100 experiments and was the project's biggest reproducibility hazard.

**Safe configurations:**
- Ampere+ (sm_80+, e.g. A100, H100, RTX 30/40, L4, L40S) with `--dtype auto` (native bf16) ✓
- Pre-Ampere (T4 sm_75, V100, P100) with **`--dtype float32` EXPLICIT** (b=4 to fit memory) ✓
- Anything else: SILENT CORRUPTION. Do not trust accuracy numbers.

Full details in [docs/reports/05_hardware_root_cause.md](docs/reports/05_hardware_root_cause.md).

## Current state of research

**Closed:**
- ✓ Per-link rMSE matches TurboQuant Table 1 to 3rd decimal across synthetic, capture-replay, in-loop (reports 02, 03)
- ✓ Hardware/dtype advisory documented + figure generated (report 05 §15)
- ✓ Bit-rate ladder at n=250 on Kaggle T4 fp32 — no measurable accuracy change 4×-16× under sampled decoding (report 06; greedy nuance in report 07)

**Active / latest:**
- ✓ Modal Tier-2 cloud sweep complete: T∈{1,2,3,4}, powered n=250 T=3,
  REF-vs-REF control, and selective inner/outer runs (report 07).
- ✓ Local backend and four-cell extension complete (report 08). Raw logit NPZs remain
  local; compact correctness artifacts and summaries are public.
- ⚠️ Tier-2 pairing must exclude conditional answer-retry calls. `analyze.py` now
  accepts/derives a primary-batch limit; local captures are K=256, window=128.
- ⚠️ Matched-prefix KL is a top-K, selection-conditioned approximation. Do not infer
  a reliable depth trend or causal capacity mechanism from it.
- ⏳ Next: scaled×math500, multi-seed MBPP+ light/scaled, divergence hazard/length
  analysis, logit-margin mediation, and teacher-forced position-aligned KL.

**Open (post-Tier 2/local extension):**
- ⌛ multi-seed effect estimation and the remaining light/scaled matrix cells
- ⌛ right-censored first-divergence analysis and teacher-forced KL
- ⌛ QJL residual ablation and packed-transport systems measurement

## Where each thing is documented

| If you need to know… | Read |
|---|---|
| The main claim and its statistics | [docs/reports/06](docs/reports/06_variant_b_in_loop_HEADLINE.md) |
| Why pre-Ampere GPUs were a trap | [docs/reports/05 §15](docs/reports/05_hardware_root_cause.md) |
| Why 22 P100 experiments are retracted | [docs/reports/04](docs/reports/04_kaggle_p100_RETRACTED.md) |
| Per-link distortion math (TurboQuant theory match) | [docs/reports/02](docs/reports/02_variant_b_synthetic.md) + [03](docs/reports/03_capture_replay_solver.md) |
| Architectural decisions (patching, dtype routing) | [docs/design/architecture.md](docs/design/architecture.md) |
| External reproduction workflow | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) + [audit](docs/operations/external_reproducibility_audit.md) |
| Inventory of every kernel ever pushed | [docs/operations/experiments_log.md](docs/operations/experiments_log.md) |
| Tier 2 fidelity-sweep methodology | [experiments/fidelity_sweep/README.md](experiments/fidelity_sweep/README.md) |
| Local cross-task/tier results and corrected trajectory analysis | [docs/reports/08](docs/reports/08_local_cross_cell_generalization.md) |

## Don'ts

- ❌ Don't reintroduce `experiments/00_setup/`, `03_pristine_baseline_p100_FAILED/`, `05_variant_b_modal_a100_dtype_artifact/`, `retracted_p100_inloop/`, or `experiments/results/`. Their findings live in `docs/reports/`; the scripts were archived for repo cleanliness on 2026-06-01.
- ❌ Don't push to Kaggle without first scrubbing `<YOUR_KAGGLE_USERNAME>` placeholders.
- ❌ Don't write ASCII bar charts in reports. Use matplotlib → PNG → reference from markdown. See `docs/figures/_generate_figures.py`.
- ❌ Don't trust accuracy numbers from any experiment that didn't either (a) run on Ampere+ with `--dtype auto`, or (b) run on pre-Ampere with `--dtype float32` explicitly. Pascal/Turing with auto-dtype → silent collapse.
- ❌ Don't commit `external/` or `.venv/` (both gitignored, but be aware).
- ❌ Don't run `pip install kaggle` to get the Kaggle CLI — use `./bin/kaggle` which wraps `kaggle-cli 2.x` via `uvx` and exposes the modern fields (`enable_gpu`, `dataset_sources`, `--accelerator NvidiaTeslaT4`, etc.).

## Coding conventions

- Python ≥3.10, project venv at `.venv/`.
- Type hints + docstrings on every public function in `src/`.
- Tests are the verification mechanism — every metric / statistical helper has at minimum: (a) determinism test, (b) edge-case test, (c) sanity-check against known answer.
- Markdown for human docs; use `![alt](path)` for figures so they render on GitHub.
- ASCII tables in markdown are fine for small data; for anything that benefits from a plot, generate it via matplotlib (see `docs/figures/_generate_figures.py` for the style).

## Imported Claude Cowork project instructions
