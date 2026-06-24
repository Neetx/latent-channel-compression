# Roadmap

## Current result (2026-06-24)

Variant B, the data-oblivious TurboQuant MSE core used in this repository,
compresses RecursiveMAS latent messages by 4x-16x with no detected monotonic
accuracy degradation in the five completed local cells. The strongest supported
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
| math500 / scaled | 82.8 / 84.0 | 87.6 / 85.2 | -2.4 pp [-6.0,+1.2] | 8.8% |
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
| math500 / scaled | 80.4% | 54.6 | 0.147 | 0.9953 |
| mbppplus / light | 92.8% | 35.8 | 0.113 | 0.9953 |
| mbppplus / scaled | 51.2% | 65.7 | 0.059 | 0.9953 |
| medqa / light | 96.4% | 32.4 | 0.045 | 0.9952 |

The same-task MBPP comparison is the cleanest single observation: on mbppplus the scaled
tier is much more trajectory-robust than light (92.8% -> 51.2% divergence, common prefix
35.8 -> 65.7) under nearly identical measured channel cosine. **It does not generalise,
though:** on math500 the same light->scaled change barely moves divergence (86.4% ->
80.4%), leaves the prefix essentially flat (53.7 -> 54.6), and the matched-prefix KL
actually *rises* (0.079 -> 0.147). So the effect is a task-and-tier-conditioned
association, not a causal capacity law: architecture, checkpoint family, baseline
competence, output length, and token margins all change too, and the sign/magnitude of the
contrast depends on the task. Matched-prefix KL is approximate and selection-conditioned;
teacher forcing is needed for a position-aligned causal measurement. The two scaled cells
completing the 2x2 are `results/step2_scaled_mbppplus/` and `results/step4_scaled_math500/`.

A length-censored first-divergence hazard analysis
([`results/divergence_hazard_SUMMARY.md`](experiments/fidelity_sweep/local_pkg/results/divergence_hazard_SUMMARY.md),
`analysis/divergence_hazard.py`) confirms the cross-task difference is **not** a
generation-length artifact. Only MBPP+/scaled finishes early (median 117 tokens, 60% under
the 128 window); the other cells run to the cap. Yet at position 25 — before censoring
matters — MBPP+/scaled has diverged on only 15.4% of problems versus 50.0% for MBPP+/light
(early per-position hazard 0.0066 versus 0.0271, ~4x), while on Math500 the tiers are
indistinguishable early and scaled diverges slightly *more* (38.4% versus 32.4%). The
Math500 full-window "advantage" (80.4% versus 86.4%) is therefore a late-position effect,
and the MBPP+ contrast is a genuine early, per-token effect. The first-divergence-hazard
part of the closure-package mechanism test is done for the single-seed captures; the
teacher-forced and multi-seed parts remain.

Canonical details and artifact provenance are in
[`docs/reports/08_local_cross_cell_generalization.md`](docs/reports/08_local_cross_cell_generalization.md).

## Scope decision — two separate papers (locked 2026-06-21)

### Paper 1: behavioral fidelity and trajectory robustness (this repository/write-up)

The current paper studies the **scientific effect of low-bit perturbation** on a
recursive latent channel:

- channel reconstruction fidelity;
- aggregate answer accuracy and per-problem churn;
- greedy trajectory divergence;
- dependence on task and system tier;
- mechanisms based on token margins, length, and teacher-forced distributions.

Paper 1 continues to use fake quantization because that cleanly isolates the
information-loss intervention. It must describe 4x--16x as **nominal packed-payload
ratios**, not measured network speedups. A tiny pack/unpack correctness check may be
mentioned if useful, but real kernels and performance optimization are **not a
closure requirement** for this paper.

### Paper 2: real codec, kernel, and distributed performance (future work)

The implementation/systems paper will build and benchmark the actual transport path:

- packed low-bit representation, norms, codebooks, seeds/rotation metadata;
- fused CUDA or Triton quantize/dequantize kernels;
- GPU memory traffic and kernel latency;
- overlap of codec, communication, and model execution;
- real transmitted bytes, bandwidth, end-to-end latency, throughput, and VRAM;
- single-GPU simulation followed by multi-GPU or networked-agent evaluation;
- fidelity parity between fake quantization and the packed implementation.

This should be a separate report/paper because its research question, baselines,
engineering work, and evaluation methodology are different. Do not delay Paper 1
while waiting for the real kernel. Paper 2 may reuse the current fidelity tests as
its numerical-correctness oracle.

## Priorities

### 1. Confirm whether scaled is genuinely more trajectory-robust — NEXT ACTION

This is the highest-priority scientific question. The 92.8% (light) versus 51.2%
(scaled) MBPP+ divergence gap is the most novel result, but currently uses one
quantizer rotation, one problem subset/order, one task contrast, and a 128-position
window.

#### 1A. Make seeds explicit before running — IMPLEMENTED

`quantizer_seed` is now a first-class, recorded field distinct from the generation seed.
It is plumbed CLI -> child env `QUANTIZER_SEED` -> injected head `_VB_QSEED` -> the
`TurboQuantHonest` factory, and recorded in the config tag, output dir, result JSON, and
cell manifest. Seed 42 is the default and keeps the original (unsuffixed) tags and log
names; any other seed `N` writes to a `..._qsN` tag, so rotations never collide and the
existing seed-42 results/analyzers resolve unchanged. Tests cover propagation, tag
suffixing, backward compatibility, and quantizer seed determinism/sensitivity
(`tests/test_quantizer_seed.py`, `tests/test_fidelity_kernel.py`); an end-to-end GPU smoke
at `--quantizer-seed 7` was verified.

Still distinct and not yet swept: `generation_seed` (sampled decoding only, fixed at 42)
and the problem indices/order or subset manifest. Greedy REF does not depend on the
quantizer rotation and can be reused when every other condition is identical; INT4 must be
rerun for each rotation. Pre-register the additional rotation seeds before inspecting
results.

#### 1B. Minimum confirmatory matrix

1. MBPP+ / light: INT4 at at least 5 quantizer rotations, same 250 problems.
2. MBPP+ / scaled: the same rotations and problem indices.
3. Math500 / scaled: REF plus the same INT4 rotations.
4. Math500 / light: reuse the compatible REF and rerun INT4 rotations as needed.

Use top-K=256 and capture at least 256 positions if memory/disk permit. Always report
the original 128-position estimand as a fixed comparable slice; treat longer-window
results as an additional survival/censoring analysis.

#### 1C. Primary estimands and analysis

- divergence probability within 128 positions;
- first-divergence survival/hazard, accounting for output length and right-censoring;
- common-prefix length;
- answer churn and paired accuracy delta;
- teacher-forced KL/JS, top-1/top-2 margin, chosen-token rank, and margin crossings.

Report the light-minus-scaled contrast with a problem-clustered/hierarchical bootstrap
over problems and rotations. Do not pool repeated rotations as if they were independent
new benchmark problems. Check whether the tier coefficient survives adjustment for
generation length, baseline correctness, and REF token margin.

#### 1D. Interpretation gate

- If scaled is lower-divergence across rotations and on both MBPP+ and Math500, call
  it a **replicated tier-associated trajectory-robustness effect**.
- If the difference disappears after length/margin adjustment, report the mediator
  rather than a tier effect.
- Do not call it a causal parameter-count/capacity law unless model family and
  architecture are controlled more tightly than Sequential-Light versus Scaled.

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

### 5. Strengthen inference

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

## Minimum closure package

Before freezing the study, complete these in order:

1. **Confirmatory tier replication:** MBPP+ light/scaled across at least three
   quantizer rotations, plus scaled Math500 at the same rotations.
2. **Aligned mechanism test:** teacher-forced KL/rank/margin analysis and a
   right-censored first-divergence hazard.
3. **One algorithmic control:** unrotated scalar quantization at 4 bit; ideally add
   the QJL residual and at least one lower/higher rate.
4. **One operationally sensitive task:** deliberation/tool-calling or another task
   where a changed intermediate action matters.
5. **Reproducibility release:** second-machine clean rerun and checksumed raw local
   artifact archive with environment/model hashes.

Items 1--2 are required for a stronger Paper 1. Items 3--4 determine whether the
result is specific to this quantizer/benchmark or generalizes mechanistically. Item 5
is required for a strong reproducibility claim. Real packed transport and performance
belong to Paper 2 and are intentionally excluded from the Paper 1 closure gate.

## Handoff for the next Claude Code session

Start here, in order:

1. Read `AGENTS.md`, this roadmap, REPORT_08, and `REPRODUCIBILITY.md`.
2. Do not modify `external/RecursiveMAS`; it is read-only third-party upstream.
3. Add explicit `quantizer_seed` plumbing through the local driver and injected
   quantizer factory, plus unit tests and manifest recording.
4. Design the teacher-forced capture format before spending GPU time.
5. Pre-register the rotation seeds, problem indices, primary 128-position estimand,
   and bootstrap analysis in this roadmap/report.
6. Run a tiny light/scaled smoke first, then launch the MBPP+ rotation matrix.
7. Only after MBPP+ validates end-to-end, run scaled Math500.

The immediate objective is not another broad benchmark sweep. It is to decide whether
the scaled trajectory-stability gap is reproducible and to explain it with aligned
margin/distribution measurements.
