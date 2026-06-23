# experiments/

This folder contains **only the scripts needed to reproduce the paper's results**. Historical experiments that hit dead-ends (P100 bf16-fallback collapse, bf16 boundary-cast artifact, infrastructure probes, …) had their _findings_ preserved in [`docs/reports/`](../docs/reports/) and their _scripts_ removed for repo cleanliness.

## Active experiments

| Folder | What it produces | Used by |
|---|---|---|
| `distortion_validation/` | Per-link rMSE/cosine/norm-ratio at 2/3/4/8 bits, synthetic Gaussian-on-sphere + real Solver-adapter capture-replay | [report 02](../docs/reports/02_variant_b_synthetic.md), [report 03](../docs/reports/03_capture_replay_solver.md), [figure: distortion_vs_bits.png](../docs/figures/distortion_vs_bits.png) |
| `solver_diagnostic/` | Solver-alone math500 accuracy (proves checkpoint is intact: 83% on n=100, greedy) | [report 05 §1.1](../docs/reports/05_hardware_root_cause.md) |
| `baseline_a100_modal/` | Upstream `run.py` pristine on Modal A100 bf16, reproduces paper baseline (84-86% on math500) | [report 05 §1.3](../docs/reports/05_hardware_root_cause.md) |
| `variant_b_ladder_t4_kaggle/` | Historical independent T4 fp32 Math500 ladder | [report 06](../docs/reports/06_variant_b_in_loop_HEADLINE.md) |
| `fidelity_sweep/` | **PRIMARY** local RTX 5070 Ti five-cell study plus historical Kaggle/Modal backends | [report 08](../docs/reports/08_local_cross_cell_generalization.md), `fidelity_sweep/local_pkg/` |

## What was removed (and where its finding lives)

| Removed folder | Why | Finding preserved in |
|---|---|---|
| `00_setup/` (gpu_probe, systemload_probe) | infrastructure verification only; not a paper result | [report 03](../docs/reports/03_capture_replay_solver.md) cites torch/Kaggle setup |
| `03_pristine_baseline_p100_FAILED/` | negative result — Kaggle P100 fails to reproduce due to bf16 fallback | [report 05 §1.2 + §15](../docs/reports/05_hardware_root_cause.md) |
| `05_variant_b_modal_a100_dtype_artifact/` | first Variant B attempt confounded by bf16/fp32 boundary cast | [report 05 §6](../docs/reports/05_hardware_root_cause.md), [report 06 §2.3](../docs/reports/06_variant_b_in_loop_HEADLINE.md) |
| `retracted_p100_inloop/` (20 sub-folders) | 22 retracted P100 in-loop runs (all hardware-collapse artifacts) | [report 04 (with retraction notice)](../docs/reports/04_kaggle_p100_RETRACTED.md) |
| `results/` (97 MB) | captured kernel logs / JSON from past runs | canonical numbers in markdown reports; fresh logs available by re-running |

If you need to re-run any of those removed experiments, the methodology is fully described in the cited reports.

## How to re-run an active experiment

Each folder contains either:
- `kernel_pkg/` (Kaggle kernel package) and optionally `dataset_pkg/` (Kaggle dataset package)
- A standalone Python script (Modal experiments)

### Prerequisites

1. Clone the upstream RecursiveMAS repo (used at runtime by some Modal scripts and as documentation reference):
   ```bash
   git clone https://github.com/RecursiveMAS/RecursiveMAS.git external/RecursiveMAS
   git -C external/RecursiveMAS checkout f95d512017fb713e9ac519248fbfd3d270dafd68
   ```
2. Configure your Kaggle credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`) per [Kaggle CLI docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/configuration.md).
3. (For Modal experiments) install Modal: `pip install modal && modal token new`.

### Kaggle T4 experiments (the HEADLINE)

```bash
# Push the Variant B Lloyd-Max + Haar quantizer src/ as a Kaggle dataset (one-time)
./bin/kaggle datasets create -p experiments/variant_b_ladder_t4_kaggle/dataset_pkg --dir-mode skip

# Push one or more kernels (each is a Variant B configuration)
./bin/push_kaggle_vb_kernel.sh 0 250 4   # baseline (no quantization)
./bin/push_kaggle_vb_kernel.sh 4 250 4   # Variant B 4-bit (8× compression)
./bin/push_kaggle_vb_kernel.sh 2 250 4   # Variant B 2-bit (16× compression)
```

Edit `experiments/variant_b_ladder_t4_kaggle/kernel_pkg/kernel-metadata.json` first to replace `<YOUR_KAGGLE_USERNAME>` with your actual Kaggle username.

Download each Kaggle output into a separate directory, then analyze the JSONs:

```bash
mkdir -p /tmp/rmas_ladder_outputs
./bin/kaggle kernels output "<YOUR_KAGGLE_USERNAME>/rmas-vbbaseline-n250-b4" -p /tmp/rmas_ladder_outputs/baseline
./bin/kaggle kernels output "<YOUR_KAGGLE_USERNAME>/rmas-vb8-n250-b4"        -p /tmp/rmas_ladder_outputs/vb8
./bin/kaggle kernels output "<YOUR_KAGGLE_USERNAME>/rmas-vb4-n250-b4"        -p /tmp/rmas_ladder_outputs/vb4
./bin/kaggle kernels output "<YOUR_KAGGLE_USERNAME>/rmas-vb2-n250-b4"        -p /tmp/rmas_ladder_outputs/vb2

.venv/bin/python bin/verify_artifacts.py /tmp/rmas_ladder_outputs \
    --manifest /tmp/rmas_ladder_outputs/SHA256SUMS.json

.venv/bin/python experiments/variant_b_ladder_t4_kaggle/analysis/analyze_ladder.py \
    --inputs /tmp/rmas_ladder_outputs \
    --n-samples 250 \
    --batch-size 4 \
    --out experiments/variant_b_ladder_t4_kaggle/analysis/results
```

### Modal A100 baseline

```bash
modal run experiments/baseline_a100_modal/phase0e_modal.py
```

### Modal A100 fidelity sweep

See [`fidelity_sweep/README.md`](fidelity_sweep/README.md) for the full paired
workflow. The essential post-hoc steps are:

```bash
modal volume get rmas-fidelity-out vb0_T3_n250 /tmp/fid_outputs/
modal volume get rmas-fidelity-out vb4_T3_n250 /tmp/fid_outputs/

.venv/bin/python bin/verify_artifacts.py /tmp/fid_outputs \
    --manifest /tmp/fid_outputs/SHA256SUMS.json

.venv/bin/python experiments/fidelity_sweep/analysis/analyze.py \
    --inputs /tmp/fid_outputs \
    --logit-dir /tmp/fid_outputs \
    --eps-pp 2.0 \
    --out experiments/fidelity_sweep/analysis/results_n250_T3
```

Analyze one comparison directory at a time. `analyze.py` intentionally rejects
duplicate `(bits, T)` runs so `n=50`, `n=250`, `inner`, `outer`, and REF-vs-REF
controls cannot be mixed accidentally.
