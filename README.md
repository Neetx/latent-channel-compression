# Latent Space Compression Research — RecursiveMAS x TurboQuant

This repository studies whether the inter-agent latent channel of
[RecursiveMAS](https://github.com/RecursiveMAS/RecursiveMAS) can tolerate the
data-oblivious MSE core of
[TurboQuant](https://arxiv.org/abs/2504.19874): Haar rotation followed by a
Lloyd--Max scalar quantizer, without the QJL residual.

## Main result: four local cells on one RTX 5070 Ti

The primary study runs locally on an RTX 5070 Ti 16 GB in native bf16. It evaluates
five $n=250$ task/tier cells — the 2×2 {Math500, MBPP+} × {light, scaled} plus MedQA —
seed 42, three recursive rounds, with a sampled REF/8/4/2-bit ladder and paired greedy
REF/INT4 capture.

| cell | sampled REF/8/4/2 bit accuracy | greedy delta (95% CI) | answer churn | divergence within 128 positions |
|---|---:|---:|---:|---:|
| Math500 / light | 77.6 / 76.0 / 78.8 / 78.0 | +2.0 pp [-2.0,+6.0] | 10.0% | 86.4% |
| Math500 / scaled | 82.8 / 85.6 / 84.0 / 84.8 | -2.4 pp [-6.0,+1.2] | 8.8% | 80.4% |
| MBPP+ / light | 32.8 / 33.6 / 38.8 / 34.8 | 0.0 pp [-4.0,+4.0] | 9.6% | 92.8% |
| MBPP+ / scaled | 70.8 / 73.6 / 72.0 / 71.2 | -2.0 pp [-4.8,+0.4] | 4.4% | 51.2% |
| MedQA / light | 34.4 / 26.4 / 32.8 / 30.0 | confounded | 30.4% | 96.4% |

The defensible conclusion is:

> At the tested sample size, aggressive fake quantization produces no detected
> monotonic aggregate-accuracy degradation in the five sampled ladders. It is not
> behaviorally lossless: individual answers change and most light-tier greedy
> trajectories diverge within the capture window.

The MBPP+ same-task contrast is especially interesting: scaled diverges much less
than light (51.2% versus 92.8%) and has lower matched-prefix KL (0.059 versus 0.113
nats) at nearly identical mean channel cosine. **But this does not generalize across
tasks:** on Math500 the same light→scaled change barely moves divergence (86.4% →
80.4%) and the matched-prefix KL even rises (0.079 → 0.147). The MBPP+ gap is therefore
a **task-specific tier association**, not a causal law of model capacity; architecture,
checkpoint family, competence, length, and token margins also differ.

The MBPP+ gap is, however, **robust to the quantizer rotation**: a five-rotation matrix
(quantizer seeds 42, 7, 17, 73, 101) gives a problem-clustered light−scaled contrast of
**+40.2 pp [+34.9, +45.7]** (divergence within 128) and **+31.8 pp [+25.9, +37.6]**
(within 25), so it is not a single-rotation artifact
([results/rotation_matrix_SUMMARY.md](experiments/fidelity_sweep/local_pkg/results/rotation_matrix_SUMMARY.md)).

MedQA greedy REF develops a pathological first-option bias. Its apparent +15.2 pp
INT4 gain is a diagnostic failure case, not evidence that quantization improves
medical reasoning. Use its non-monotonic sampled ladder for task-level interpretation.

Full statistics and methodology are in
[REPORT_08](docs/reports/08_local_cross_cell_generalization.md).

## Reproduce locally

The primary reproduction guide is
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). It covers:

1. the exact Python 3.12 / PyTorch 2.9.0+cu128 environment;
2. cloning and pinning the read-only RecursiveMAS upstream;
3. checkpoint download and adapter-manifest validation;
4. a four-sample end-to-end GPU smoke test;
5. the four full local cells and their deterministic output layout;
6. answer-level and corrected Tier-2 analysis from the generated artifacts.

Minimal shape of the workflow:

```bash
git clone https://github.com/RecursiveMAS/RecursiveMAS.git external/RecursiveMAS
git -C external/RecursiveMAS checkout f95d512017fb713e9ac519248fbfd3d270dafd68

# Create .venv and install the cu128 torch wheel + requirements.txt first.
export HF_HOME="$HOME/.cache/huggingface"
export LCC_RUN_ROOT="$HOME/lcc/runs"

.venv/bin/python experiments/fidelity_sweep/local_pkg/setup/verify_gpu.py
.venv/bin/python experiments/fidelity_sweep/local_pkg/setup/download_checkpoints.py
.venv/bin/python experiments/fidelity_sweep/local_pkg/setup/download_scaled.py

.venv/bin/python experiments/fidelity_sweep/local_pkg/run_cell.py \
  --style sequential_light --dataset math500 --n 250 \
  --ladder-batch 16 --cap-batch 2
```

The runner treats `external/RecursiveMAS` as read-only. It verifies the pinned commit,
copies the upstream source into the run directory, instruments only the disposable
copy, and removes that copy after execution.

## What the experiment measures

- **Channel fidelity:** relative L2/MSE, cosine, norm ratio, and call counts.
- **Answer behavior:** sampled accuracy ladder, paired greedy delta, discordant pairs,
  McNemar test, bootstrap interval, TOST, and answer churn.
- **Trajectory behavior:** divergence within 128 captured positions, common-prefix
  length, and approximate matched-prefix KL/JS on a corrected top-K union.

The 4x--16x ratios are nominal packed-payload ratios. The study uses fake
quantization and does not yet measure real serialized bytes, network latency, codec
overhead, or end-to-end throughput.

## Independent cloud history

Kaggle T4 fp32 produced the original $n=250$ Math500 ladder, and Modal A100 fp32
produced deterministic controls and the original depth sweep. They independently
motivated and validated the safe dtype path, but they are now secondary historical
evidence rather than the primary reproduction target:

- [REPORT_05](docs/reports/05_hardware_root_cause.md): hardware/dtype failures;
- [REPORT_06](docs/reports/06_variant_b_in_loop_HEADLINE.md): original cloud ladder;
- [REPORT_07](docs/reports/07_fidelity_sweep_modal.md): historical cloud fidelity;
- [REPORT_08](docs/reports/08_local_cross_cell_generalization.md): current local result.

REPORT_07's KL/divergence values used the legacy estimator and should not be compared
numerically with corrected local Tier-2 values without reanalyzing the raw cloud NPZs.

## Repository map

| path | purpose |
|---|---|
| [docs/RESEARCH.md](docs/RESEARCH.md) | current claims, design, limits, and evidence authority |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | primary clean-clone RTX 5070 Ti workflow |
| [experiments/fidelity_sweep/local_pkg/](experiments/fidelity_sweep/local_pkg/) | local runner, setup, analyzers, and compact result artifacts |
| [docs/reports/08_local_cross_cell_generalization.md](docs/reports/08_local_cross_cell_generalization.md) | canonical five-cell report |
| [ROADMAP.md](ROADMAP.md) | prioritized remaining research |
| [writeup/main.pdf](writeup/main.pdf) | current paper |

## What is not committed

- the third-party `external/RecursiveMAS` clone;
- model checkpoints and Hugging Face cache;
- raw logit NPZs, prompts, generations, and machine logs;
- cloud credentials or private cloud volumes.

Compact per-problem correctness artifacts are committed so the answer-level table can
be recomputed without a GPU. Exact Tier-2 reproduction requires regenerating the raw
local captures or obtaining a separately archived checksum bundle.

## License

**All rights reserved** — see [LICENSE](LICENSE). This repository is public for
transparency and reproducibility, but it is not open-source. RecursiveMAS and
TurboQuant remain governed by their own repositories and licenses.
