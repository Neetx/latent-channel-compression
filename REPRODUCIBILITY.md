# Reproducibility Guide

This guide is the entrypoint for reproducing the experiments and post-hoc
analyses behind the write-up. It separates two tasks:

1. **Artifact analysis**: starting from downloaded JSON/NPZ outputs, regenerate
   tables and figures locally.
2. **Cloud reruns**: rerun the GPU experiments on Kaggle or Modal to produce new
   artifacts.

The code is source-visible but not open-source; see `LICENSE`.

## 0. Local Setup

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional but recommended: enables patch tests against real upstream source.
git clone https://github.com/RecursiveMAS/RecursiveMAS.git external/RecursiveMAS
git -C external/RecursiveMAS checkout f95d512017fb713e9ac519248fbfd3d270dafd68

.venv/bin/python -m pytest tests/ -q
```

The cloud runners pin their own dependencies inside their scripts; local
`requirements.txt` is for tests and post-hoc analysis.

## 1. Verify Downloaded Artifacts

Always verify artifacts before analysis:

```bash
.venv/bin/python bin/verify_artifacts.py /path/to/artifacts \
    --manifest /path/to/artifacts/SHA256SUMS.json
```

This parses JSON, tests NPZ/ZIP CRCs, and writes a SHA256 manifest. It catches
interrupted `modal volume get` downloads that leave corrupt `fidelity_logits.npz`
files on disk.

## 2. Headline Sampled Bit-Rate Ladder

This is the sampled-decoding Kaggle T4 fp32 result used for the main accuracy
ladder.

### 2.1 Rerun on Kaggle

Configure Kaggle credentials first. Then create or update the source bundle
dataset:

```bash
./bin/kaggle datasets create \
    -p experiments/variant_b_ladder_t4_kaggle/dataset_pkg \
    --dir-mode skip
```

Replace `<YOUR_KAGGLE_USERNAME>` in generated Kaggle metadata with your username.
Then run the canonical n=250 ladder:

```bash
./bin/push_kaggle_vb_kernel.sh 0 250 4
./bin/push_kaggle_vb_kernel.sh 8 250 4
./bin/push_kaggle_vb_kernel.sh 4 250 4
./bin/push_kaggle_vb_kernel.sh 2 250 4
```

Each kernel writes one JSON:

```text
phase0g_vb{bits}_n250_b4.json
```

### 2.2 Download and Analyze

Download each kernel into its own subdirectory:

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

Expected outputs:

```text
experiments/variant_b_ladder_t4_kaggle/analysis/results/results.md
experiments/variant_b_ladder_t4_kaggle/analysis/results/summary.csv
experiments/variant_b_ladder_t4_kaggle/analysis/results/summary.json
experiments/variant_b_ladder_t4_kaggle/analysis/results/figures/bit_rate_ladder_n250.png
```

## 3. Greedy Fidelity Sweep

This is the paired REF-vs-INT4 Modal/Kaggle experiment used for channel fidelity,
trajectory divergence, matched-prefix logit metrics, and paired bootstrap/TOST.
The current analyzer pairs only fixed primary generate calls and excludes conditional
answer retries. It also corrects residual-tail accounting on the top-K union.

### 3.1 Modal A100 Path

For a powered T=3 n=250 comparison:

```bash
modal run --detach experiments/fidelity_sweep/modal_pkg/fidelity_modal.py::main \
    --bits 0 --t 3 --n-samples 250 --batch-size 8
modal run --detach experiments/fidelity_sweep/modal_pkg/fidelity_modal.py::main \
    --bits 4 --t 3 --n-samples 250 --batch-size 8
```

For the n=50 T-sweep:

```bash
modal run experiments/fidelity_sweep/modal_pkg/fidelity_modal.py::sweep \
    --t-values 1,2,3,4 --n-samples 50 --batch-size 8
```

Fetch one comparison layout at a time:

```bash
mkdir -p /tmp/fid_outputs
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

Do not point `analyze.py` at a parent containing multiple runs with the same
`(bits, T)` such as all-link, inner-only, outer-only, and n=50 together. The
script now rejects duplicates by default to prevent silent overwrites.

### 3.2 Kaggle T4 Path

When Kaggle quota is available:

```bash
for T in 1 2 3 4; do
    ./bin/push_fidelity_kernel.sh 0 $T 50 4
    ./bin/push_fidelity_kernel.sh 4 $T 50 4
done
```

Then download each kernel into one subdirectory and run the same verifier and
`analyze.py` command as above.

## 4. Local Single-GPU Cross-Cell Extension

The portable local backend is documented in
[`experiments/fidelity_sweep/local_pkg/README.md`](experiments/fidelity_sweep/local_pkg/README.md).
It produced four n=250 cells in bf16 on one 16 GB GPU. From the `local_pkg` directory:

```bash
bash run_step0.sh
bash run_step1_mbppplus.sh
bash run_step2_scaled_mbppplus.sh
python run_cell.py --style sequential_light --dataset medqa \
  --ladder-batch 16 --cap-batch 2 --n 250

python analysis/flip_churn_tost.py
python analysis/compare_cells.py
python analysis/tier2_logit_fidelity.py
```

Set `LCC_RUN_ROOT` to choose the raw-output directory and `HF_HOME` to choose the
checkpoint cache. Raw prompts, traces, logs, and `fidelity_logits.npz` are not part
of the public repository. Compact JSONL records under `local_pkg/results/` are enough
to reproduce paired answer statistics; Tier-2 requires regenerated or archived NPZs.

## 5. Regenerate Figures and Write-Up

After running the analyses:

```bash
.venv/bin/python docs/figures/_generate_figures.py

cd writeup
tectonic main.tex
```

`docs/figures/_generate_figures.py` reads
`experiments/variant_b_ladder_t4_kaggle/analysis/results/summary.json` for the
headline ladder when present. If that file is absent, it falls back to the
historical report values and says so in the generated title.

## 6. Known Reproducibility Boundaries

- Cloud credentials and private Modal/Kaggle volumes are not included.
- Raw historical cloud outputs are not committed; rerun the jobs or use an
  external artifact archive with a SHA256 manifest.
- The RecursiveMAS upstream commit used by this package is pinned to
  `f95d512017fb713e9ac519248fbfd3d270dafd68` in local instructions. The cloud
  runners should be kept on that commit for strict reruns.
- The headline ladder uses sampled decoding and unpaired runs. The fidelity
  sweep uses greedy decoding and paired runs. Do not merge those baselines.
- Local Tier-2 captures use top-K=256 and a 128-position window. Divergence is
  windowed and matched-prefix KL is approximate and selection-conditioned.
- REPORT_07 cloud KL/divergence values were produced by the legacy all-call/tail
  estimator. Reanalyze the raw cloud NPZs with the current analyzer before comparing
  them numerically with the corrected local table.
- Fake quantization measures information-theoretic channel compression, not
  wall-clock network bandwidth or latency.
