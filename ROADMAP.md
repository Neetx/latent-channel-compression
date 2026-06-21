# Roadmap

## Current result (2026-06-21)

Variant B, the data-oblivious TurboQuant MSE core used in this repository,
compresses RecursiveMAS latent messages by 4x-16x with no detected monotonic
accuracy degradation in the four completed local cells. The strongest supported
description is:

> Aggregate answer accuracy is robust at the tested sample sizes, while individual
> answers and greedy token trajectories can change substantially.

This is deliberately narrower than "lossless" or "trajectory preserving". The
paired accuracy confidence intervals still allow effects of a few percentage points,
and 5-30% of individual answers change depending on the cell.

### Completed local matrix

All runs used seed 42, three recursive rounds, native bf16 on one RTX 5070 Ti,
and the upstream RecursiveMAS checkout pinned at `f95d512`.

| cell | sampled REF/INT4 | greedy REF/INT4 | paired delta (95% CI) | answer churn |
|---|---:|---:|---:|---:|
| math500 / light | 77.6 / 78.8 | 76.8 / 78.8 | +2.0 pp [-2.0,+6.0] | 10.0% |
| mbppplus / light | 32.8 / 32.8 | 36.4 / 36.4 | 0.0 pp [-4.0,+4.0] | 9.6% |
| mbppplus / scaled | 70.8 / 72.0 | 74.4 / 72.4 | -2.0 pp [-4.8,+0.4] | 4.4% |
| medqa / light | 34.4 / 32.8 | 21.2 / 36.4 | +15.2 pp [+8.8,+21.6] | 30.4% |

The MedQA greedy comparison is not evidence that quantization improves medicine:
the REF run develops a pathological first-option bias that sampled decoding does
not show. Use the flat sampled ladder as the primary MedQA evidence.

### Corrected Tier-2 trajectory result

The public analyzer pairs only the fixed primary generate calls and excludes
condition-dependent answer-retry calls. Local captures use top-K=256 and a
128-position observation window. The KL estimate expands the union support without
double-counting union-token mass in the residual tail.

| cell | diverged within window | mean common prefix / 128 | matched-prefix KL (nats) | channel cosine |
|---|---:|---:|---:|---:|
| math500 / light | 86.4% | 53.7 | 0.079 | 0.9952 |
| mbppplus / light | 92.8% | 35.8 | 0.113 | 0.9953 |
| mbppplus / scaled | 51.2% | 65.7 | 0.059 | 0.9953 |
| medqa / light | 96.4% | 32.4 | 0.045 | 0.9952 |

The same-task MBPP comparison is the cleanest observation: the scaled tier is much
more trajectory-robust than the light tier under nearly identical measured channel
cosine. This is a tier association, not yet a causal capacity law: architecture,
checkpoint family, baseline competence, output length, and token margins also change.
Matched-prefix KL is approximate and selection-conditioned; teacher forcing is needed
for a position-aligned causal measurement.

Canonical details and artifact provenance are in
[`docs/reports/08_local_cross_cell_generalization.md`](docs/reports/08_local_cross_cell_generalization.md).

## Priorities

### 1. Replicate the tier contrast

Run at least three seeds for both MBPP cells and add `scaled x math500`. Report
hierarchical or cluster-bootstrap intervals across seeds, not eight isolated p-values.
This decides whether the 92.8% vs 51.2% gap is stable and whether it follows tier
rather than task.

### 2. Replace matched-prefix KL with teacher-forced fidelity

Force both conditions along the REF token sequence and record per-position KL/JS,
top-1 margin, rank changes, and hidden-state norm. This removes the post-divergence
selection problem and can localize where perturbations are amplified or absorbed.

### 3. Map the rate-distortion frontier

Repeat paired captures at 2, 3, 4, 6, and 8 bits and include an unrotated uniform
baseline. Add the QJL inner-product residual as a separate ablation. The paper
currently establishes robustness at selected rates, not the full frontier nor the
contribution of each algorithmic component.

### 4. Expand architecture and task coverage

Prioritize `deliberation` because tool-caller decisions make trajectory drift
operationally meaningful. Then test mixture/distillation where memory permits. Add a
high-baseline multiple-choice task to replace the confounded light-MedQA cell; GPQA
remains gated by dataset access.

### 5. Measure system value

Report encoded bytes, quantize/dequantize latency, wall-clock overhead, peak VRAM,
and end-to-end communication savings. The present 4x-16x figures are nominal payload
compression ratios and do not yet include metadata or compute overhead.

### 6. Strengthen inference

Pre-register the primary estimand and equivalence margin; use multiple seeds; report
paired effect estimates and uncertainty; correct secondary multiple comparisons; and
separate confirmatory from exploratory analyses. Keep MedQA's greedy pathology as a
methodological failure case, not a headline result.

## Working configuration

| tier | sampled ladder batch | greedy capture batch |
|---|---:|---:|
| light (~1.5B) | 16 | 2 |
| scaled (~4B) | 4 | 1 |

Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for scaled, `--dtype auto`
on Ampere-or-newer GPUs, seed 42, T=3, and the portable `local_pkg` drivers. Batch 8
on the 16 GB card thrashes the CUDA allocator; the pipeline otherwise loads one agent
at a time. Raw logits remain local and are regenerated from the documented commands.

## Publication threshold

Before claiming a general capacity effect, require: (1) replicated seeds, (2) a
second same-task light/scaled comparison, (3) teacher-forced aligned fidelity, and
(4) a model/tier description that does not confound size with architecture. Until
then, the publishable claim is a robust cross-cell dissociation between aggregate
answer accuracy and individual trajectory identity, plus an exploratory tier contrast.
