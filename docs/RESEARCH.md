# Research specification and current evidence

**Status (2026-06-24):** primary five-cell RTX 5070 Ti study complete (the 2×2
{math500, mbppplus} × {light, scaled} plus MedQA/light); independent cloud validation
complete; causal and teacher-forced follow-ups open.

## Research question

Can inter-agent latent messages in a recursive multi-agent language-model system be
compressed aggressively without materially changing task performance, and how does
small channel distortion propagate into token-level behavior?

This repository evaluates the data-oblivious MSE core of TurboQuant (Variant B): a
randomized orthogonal transform followed by scalar Lloyd-Max-Gaussian quantization.
It does not yet implement or evaluate the full QJL inner-product residual.

## Claims supported by current evidence

1. On RecursiveMAS `sequential_light x math500`, sampled answer accuracy shows no
   detected degradation from 8 to 2 bits per coordinate (nominal 4x-16x payload
   compression) at n=250.
2. The same flat sampled pattern appears locally on scaled math500, light MBPP+,
   scaled MBPP+, and light MedQA. Paired greedy estimates for the four non-confounded
   cells are +2.0, -2.4, 0.0, and -2.0 percentage points, with confidence intervals
   spanning zero.
3. Aggregate accuracy robustness is not trajectory identity. Individual answer churn
   is 4.4-10% in the four usable paired cells, and 51.2-92.8% of greedy primary
   sequences diverge within the first 128 captured positions.
4. On the MBPP+ task, the scaled tier is substantially more trajectory-robust than the
   light tier (51.2% vs 92.8% windowed divergence; matched-prefix KL 0.059 vs 0.113
   nats). On math500, however, the same light→scaled change is small (80.4% vs 86.4%
   divergence, KL 0.147 vs 0.079), so the contrast is **task-specific**. On MBPP+ it is
   **robust to the quantizer rotation**: a five-rotation matrix (seeds 42, 7, 17, 73, 101)
   gives a problem-clustered light−scaled contrast of +40.2 pp [+34.9, +45.7] (within 128)
   and +31.8 pp [+25.9, +37.6] (within 25). It remains a task-specific tier association,
   not a causal model-size law.
5. Light-MedQA greedy REF is pathologically biased toward option A. Its +15.2 pp
   INT4 delta is a diagnostic failure case, not evidence that compression improves
   medicine performance; sampled decoding is the valid task-level evidence.

The terms **lossless**, **equivalent**, and **capacity law** are not justified by the
current experiment. The strongest wording is "no detected aggregate accuracy change
at the tested sample sizes" and "tier-associated trajectory robustness."

## Canonical result tables

### Local answer-level study (n=250, seed 42, T=3, bf16)

| cell | sampled ladder REF/8/4/2 bit | greedy REF/INT4 | delta (95% CI) | churn |
|---|---:|---:|---:|---:|
| math500 / light | 77.6 / 76.0 / 78.8 / 78.0 | 76.8 / 78.8 | +2.0 pp [-2.0,+6.0] | 10.0% |
| math500 / scaled | 82.8 / 85.6 / 84.0 / 84.8 | 87.6 / 85.2 | -2.4 pp [-6.0,+1.2] | 8.8% |
| mbppplus / light | 32.8 / 33.6 / 38.8 / 34.8 | 36.4 / 36.4 | 0.0 pp [-4.0,+4.0] | 9.6% |
| mbppplus / scaled | 70.8 / 73.6 / 72.0 / 71.2 | 74.4 / 72.4 | -2.0 pp [-4.8,+0.4] | 4.4% |
| medqa / light | 34.4 / 26.4 / 32.8 / 30.0 | 21.2 / 36.4 | +15.2 pp [+8.8,+21.6] | 30.4% |

The ladder order is REF, 8-bit, 4-bit, 2-bit. No completed ladder is monotonic in
bit rate. MedQA's 2-bit endpoint is lower but the sequence is non-monotonic and a
single-seed n=250 run cannot establish a rate effect.

### Corrected local Tier-2 study

| cell | divergence in 128-token window | common prefix | KL on matched prefix | channel cosine |
|---|---:|---:|---:|---:|
| math500 / light | 86.4% | 53.7 | 0.079 | 0.9952 |
| math500 / scaled | 80.4% | 54.6 | 0.147 | 0.9953 |
| mbppplus / light | 92.8% | 35.8 | 0.113 | 0.9953 |
| mbppplus / scaled | 51.2% | 65.7 | 0.059 | 0.9953 |
| medqa / light | 96.4% | 32.4 | 0.045 | 0.9952 |

These values supersede the first local summary that included condition-dependent
retry calls and double-counted some top-K union mass in the residual tail. The
correct analyzer pairs only `ceil(n_samples / batch_size)` primary calls. Local
capture used top-K=256 and at most 128 positions.

## Method

### System and intervention

- System: RecursiveMAS, pinned at commit `f95d512`.
- Primary style: `sequential_light`; two scaled contrasts (MBPP+ and math500) use
  `sequential_scaled`.
- Injection sites: all inner and outer cross-model adapters.
- REF: unchanged bf16/fp32 channel (`bits=0`).
- INT4: Variant B at 4 bits per coordinate.
- Sampled ladder: bits in {0, 8, 4, 2}, upstream sampling settings.
- Paired fidelity: deterministic greedy decoding, same sample order and seed.

### Metrics

- task accuracy and paired accuracy delta;
- discordant pairs, exact McNemar test, bootstrap interval, and answer churn;
- channel cosine, relative L2 error, and norm ratio per adapter call;
- windowed greedy divergence and common-prefix length;
- approximate KL/JS on the union of captured top-K supports plus residual tail.

Matched-prefix KL is selection-conditioned: positions after the first token mismatch
are not comparable because the contexts differ. A teacher-forced analysis is the
planned replacement for causal per-position conclusions.

### Hardware and dtype

- Primary study: RTX 5070 Ti 16 GB in native bf16 under WSL2.
- Independent historical ladder: Kaggle T4 in fp32.
- Independent historical controls/depth sweep: Modal A100 in fp32.
- Pre-Ampere bf16/autocast runs are invalid for this recursive pipeline; REPORT_05
  documents the hardware precision collapse.

The pipeline loads one agent at a time. Local safe batches are 16 sampled / 2 capture
for light, and 4 sampled / 1 capture for scaled. Batch 8 scaled thrashes a 16 GB GPU.

## Statistical interpretation

The experiments estimate effects; they do not prove exact equality. The original
TOST margin was +/-2 pp, but with n=250 and few discordant pairs most useful cells are
underpowered for that threshold. Report confidence intervals and paired counts, and
pool independent seeds only with a pre-specified hierarchical analysis.

The sampled ladders are unpaired across stochastic generations. The greedy runs are
paired at the problem level. REF-vs-REF control on cloud is exactly deterministic,
which attributes nonzero paired churn to the intervention under that configuration.

## Reproducibility and artifact authority

Evidence authority, highest first:

1. raw run artifacts and NPZ captures held outside git;
2. compact result JSON/JSONL and call statistics under `experiments/`;
3. canonical reports under `docs/reports/`;
4. this living research specification and the paper;
5. historical reports, which retain superseded interpretations for auditability.

Public per-problem JSONL files are minimized to correctness and parsed option fields.
They intentionally omit prompts, machine paths, traces, and raw generations. Large
NPZ captures and logs are regenerated rather than committed.

## What failed, and what remains valid

- Variant A (Hadamard + uniform quantization) is a baseline, not honest TurboQuant;
  it has roughly twice the synthetic 4-bit rMSE of Variant B.
- Early P100 task accuracies were invalid because the recursive latent pipeline
  collapsed numerically. Single-call distortion measurements from those runs remain
  useful.
- The A100 bf16 in-loop -19 pp result was a dtype-boundary cast artifact. It is not a
  quantizer effect and must not be quoted as the current answer.
- The cloud REPORT_07 Tier-2 table used the legacy all-call/tail estimator. Its raw
  accuracy and channel-fidelity observations remain historical evidence, but its KL
  values should not be compared numerically with corrected local Tier-2 values until
  the cloud NPZs are reanalyzed.

## Next experiments

1. Repeat light/scaled MBPP+ at 3-5 independent seeds. (Scaled math500 is done: it
   showed the MBPP+ tier gap is task-specific, so seed replication of MBPP+ is the
   priority, not more tiers.)
2. Implement teacher-forced aligned logit fidelity with token-margin diagnostics.
3. Sweep 2/3/4/6/8 bits and add an unrotated scalar baseline.
4. Implement and ablate the QJL residual.
5. Test a tool-sensitive deliberation topology and a high-baseline non-math task.
6. Measure serialized bytes, metadata, codec latency, wall-clock overhead, and VRAM.
7. Pre-register primary estimands, margins, seed pooling, and multiple-comparison rules.

Detailed prioritization is in [`../ROADMAP.md`](../ROADMAP.md).

## Reports

1. [`reports/01_variant_a_synthetic.md`](reports/01_variant_a_synthetic.md)
2. [`reports/02_variant_b_synthetic.md`](reports/02_variant_b_synthetic.md)
3. [`reports/03_capture_replay_solver.md`](reports/03_capture_replay_solver.md)
4. [`reports/04_phase0b_real_inference.md`](reports/04_phase0b_real_inference.md)
5. [`reports/05_hardware_root_cause.md`](reports/05_hardware_root_cause.md)
6. [`reports/06_variant_b_in_loop_HEADLINE.md`](reports/06_variant_b_in_loop_HEADLINE.md)
7. [`reports/07_fidelity_sweep_modal.md`](reports/07_fidelity_sweep_modal.md)
8. [`reports/08_local_cross_cell_generalization.md`](reports/08_local_cross_cell_generalization.md)

## References

- Zandieh et al., *TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate*, NeurIPS 2024.
- RecursiveMAS upstream repository and released checkpoints, pinned in this project.
