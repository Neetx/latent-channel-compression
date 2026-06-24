# REPORT 08 — Local cross-cell replication and tier-associated trajectory robustness

**Date:** 2026-06-19 to 2026-06-24  
**Hardware:** NVIDIA RTX 5070 Ti 16 GB (Blackwell, sm_120), native bf16  
**Backend:** `experiments/fidelity_sweep/local_pkg/`  
**Upstream:** RecursiveMAS commit `f95d512017fb713e9ac519248fbfd3d270dafd68`  
**Protocol:** seed 42, `num_recursive_rounds=3`, Variant B on all inner and outer
links; sampled bit-rate ladder plus paired greedy REF/INT4 capture  
**Status:** five cells complete — the 2×2 {math500, mbppplus} × {light, scaled} plus
MedQA/light; Tier-2 analysis corrected to exclude conditional answer-retry calls

---

## TL;DR

The original math500 / Sequential-Light result reproduces on a consumer Blackwell
GPU in native bf16, and aggregate answer robustness extends to code and to the larger
Sequential-Scaled constellation. Across the four clean math/code cells, sampled
bit-rate ladders show no monotonic degradation from 8 to 2 bits, and paired greedy
accuracy deltas are small and non-significant. Individual correctness outcomes still
change in 4.4--10% of problems.

The corrected top-K trajectory analysis finds a large same-task tier contrast on
MBPP+: divergence within the first 128 captured positions is 92.8% for
Sequential-Light and 51.2% for Sequential-Scaled, despite the same mean per-call
channel cosine (0.9953). This is strong evidence that the scaled constellation is
more trajectory-robust in this cell. **It does not generalize across tasks, however:**
completing the 2×2 with Sequential-Scaled × math500 shows that the same light→scaled
change on math500 barely moves divergence (86.4% → 80.4%) and actually *raises*
matched-prefix KL (0.079 → 0.147). The MBPP+ gap is therefore a task-and-tier-conditioned
association, **not yet a causal law of model capacity**: the tiers differ in models,
tokenizers, adapters, hidden dimensions, generation lengths, and logit margins; and the
contrast's sign and magnitude depend on the task. On MBPP+ the gap is, however, **robust
to the quantizer rotation** — a five-rotation matrix (seeds 42, 7, 17, 73, 101) gives a
problem-clustered light−scaled contrast of +40.2 pp [+34.9, +45.7] (within 128) and
+31.8 pp [+25.9, +37.6] (within 25) — so a single quantizer rotation is ruled out as the
explanation, though a single generation seed and problem order remain.

MedQA exposes a separate methodological failure mode. Under greedy decoding the weak
unquantized REF develops a strong first-option bias; INT4 acts like dither and raises
accuracy by 15.2 pp. That comparison is not a clean compression effect and is reported
through its sampled ladder instead.

---

## 1. Why the local study is primary

The local backend serves three purposes:

1. establish a reproducible single-consumer-GPU experiment matrix using the native
   bf16 path actually available to the project;
2. independently reproduce the earlier cloud math500 result;
3. test whether answer robustness generalizes beyond math500 to functional code and
   multiple-choice medicine;
4. remove the low-baseline floor effect by running the larger Sequential-Scaled tier
   on MBPP+.

The pipeline loads one agent at a time. Sequential-Scaled therefore fits 16 GB at
small batch sizes: batch 4 for the sampled ladder and batch 1 for logit capture.

## 2. Sampled bit-rate ladders

All cells use `n=250`. Values are accuracy percentages for REF / 8 / 4 / 2 bits.

| cell | REF | 8-bit | 4-bit | 2-bit | reading |
|---|---:|---:|---:|---:|---|
| math500 / light | 77.6 | 76.0 | 78.8 | 78.0 | flat around the local baseline |
| math500 / scaled | 82.8 | 85.6 | 84.0 | 84.8 | flat at a high baseline |
| mbppplus / light | 32.8 | 33.6 | 38.8 | 34.8 | no monotonic degradation; low-baseline floor |
| mbppplus / scaled | 70.8 | 73.6 | 72.0 | 71.2 | flat at a high baseline |
| medqa / light | 34.4 | 26.4 | 32.8 | 30.0 | scattered; no bit-rate dose response |

These are sampled, unpaired single-seed runs. “No monotonic degradation” is the
appropriate statement; the ladders do not by themselves establish equivalence.

## 3. Paired greedy answer-level results

| cell | REF | INT4 | delta pp (95% bootstrap CI) | losses/gains | churn | McNemar p |
|---|---:|---:|:---:|:---:|---:|---:|
| math500 / light | 76.8 | 78.8 | +2.0 [-2.0,+6.0] | 10/15 | 10.0% | 0.42 |
| math500 / scaled | 87.6 | 85.2 | -2.4 [-6.0,+1.2] | 14/8 | 8.8% | 0.29 |
| mbppplus / light | 36.4 | 36.4 | 0.0 [-4.0,+4.0] | 12/12 | 9.6% | 1.00 |
| mbppplus / scaled | 74.4 | 72.4 | -2.0 [-4.8,+0.4] | 8/3 | 4.4% | 0.23 |
| medqa / light | 21.2 | 36.4 | +15.2 [+8.8,+21.6] | 19/57 | 30.4% | <0.001 |

For the four clean math/code cells, every delta CI straddles zero and every exact
McNemar test is non-significant. The result is aggregate answer robustness with real
per-problem churn, not bit-exact preservation and not formal equivalence at a
pre-specified ±2 pp margin. The two scaled cells both lean slightly negative (-2.0,
-2.4 pp) but neither is statistically resolved.

### MedQA greedy confound

The MedQA REF chooses option A 116/250 times (46.4%) and reaches only 21.2% accuracy.
INT4 is more balanced and reaches 36.4%; sampled REF is also normal at 34.4%. Clean
A/B/C/D parsing rules out an extraction bug. The +15.2 pp greedy result is therefore
a first-option degeneration/dither interaction, not evidence that compression
improves the channel. It also demonstrates that paired greedy analysis requires a
well-behaved REF.

## 4. Corrected Tier-2 trajectory analysis

### 4.1 Correction to the first local analysis

The capture hook records every call to `GenerationMixin.generate`. After the fixed
primary solver generation, RecursiveMAS conditionally retries examples whose answer
cannot be parsed. REF and INT4 can retry different examples, so those calls are not
positionally pairable. The initial local analysis paired all captured calls, including
retries. It also described the local capture as if it were a complete trajectory.

The corrected analysis:

- pairs exactly the first `ceil(250 / batch_size)` primary solver batches;
- excludes 3--35 conditional retry batches per run;
- reports divergence only within the local 128-position capture window;
- counts common-prefix positions strictly before the first mismatch;
- uses top-K=256 locally, not the cloud K=512;
- subtracts approximated missing-union-token mass from the residual tail bucket to
  avoid double-counting.

### 4.2 Results

| cell | divergence within 128 | common prefix | matched-prefix KL (95% CI) | JS | channel cosine |
|---|:---:|:---:|:---:|:---:|:---:|
| math500 / light | 86.4% | 53.7 | 0.079 [0.043,0.124] | 0.009 | 0.9952 |
| math500 / scaled | 80.4% | 54.6 | 0.147 [0.072,0.251] | 0.013 | 0.9953 |
| mbppplus / light | 92.8% | 35.8 | 0.113 [0.078,0.156] | 0.015 | 0.9953 |
| **mbppplus / scaled** | **51.2%** | **65.7** | **0.059 [0.031,0.092]** | **0.003** | 0.9953 |
| medqa / light | 96.4% | 32.4 | 0.045 [0.030,0.062] | 0.005 | 0.9952 |

The divergence contrast on MBPP+ is too large to be explained by the few excluded
retry calls. The channel cosine is almost identical, but cosine is only an average
geometric distortion measure; it does not establish that the perturbations are
equivalent relative to the two systems' token-decision boundaries.

**Crucially, the contrast is task-specific.** The MBPP+ light→scaled change nearly
halves divergence (92.8% → 51.2%), but the math500 light→scaled change moves it only
86.4% → 80.4%, leaves the common prefix essentially flat (53.7 → 54.6), and *raises*
matched-prefix KL (0.079 → 0.147). A general "scaled is more trajectory-robust" law
would predict a comparable drop on math500; it does not appear. This is the strongest
single reason the MBPP+ gap must be read as a task-and-tier association rather than a
capacity law, and it sharpens the mechanism question: what about the MBPP+ task (output
length, answer redundancy, token margins) makes the scaled constellation absorb the same
channel distortion so differently there but not on math500?

A length-censored first-divergence hazard analysis rules out generation length as that
explanation (`analysis/divergence_hazard.py`,
`results/divergence_hazard_SUMMARY.md`). Treating first-divergence as a right-censored
survival process — observing each problem only while both sequences still generate real
tokens — shows that MBPP+/scaled is the *only* cell that finishes early (median 117
tokens, 60% under the 128 window), yet at position 25, before censoring matters, it has
diverged on only 15.4% of problems versus 50.0% for MBPP+/light (early per-position
hazard ~4x lower). On math500 the tiers are indistinguishable that early (scaled even
slightly higher, 38.4% versus 32.4%), so its small full-window gap is a late-position
effect. The MBPP+ contrast is thus a genuine early, per-token effect, not an artifact of
the shorter scaled-code generations.

Matched-prefix KL is less robust than divergence. It is a top-K approximation,
conditions on trajectories that have not yet diverged, averages over variable-length
prefixes, and includes the mismatch position in the distributional calculation. It
supports the same qualitative tier contrast on MBPP+, but it is not the basis for a
depth or causal-capacity claim.

### 4.3 Defensible interpretation

The current evidence supports:

> On MBPP+, across five quantizer rotations, Sequential-Scaled is substantially less
> likely than Sequential-Light to diverge within the first 128 greedy decode positions
> under the same 4-bit quantizer and nearly identical mean channel cosine (problem-clustered
> light−scaled gap +40.2 pp [+34.9, +45.7]).

It does not yet support:

- “trajectory preservation scales with parameter count” as a general law;
- “48.8% of full generations never diverge” (the observations are right-censored);
- a mechanism based on task redundancy, which MBPP+ answer robustness already
  failed to confirm;
- a clean teacher-forced per-position KL or a reliable KL-vs-depth trend.

## 5. Reproducibility

The public artifacts contain compact per-problem correctness records, small result
JSONs, analysis scripts, and summaries. Full logit NPZs and verbose model outputs are
not committed; they are regenerated by the local runner. RecursiveMAS is read-only
third-party upstream: the runner verifies its pinned commit and instruments only a
disposable source copy inside the run output. The clean-clone workflow, exact runtime,
expected output tree, measured duration, and analysis commands are in the root
`REPRODUCIBILITY.md`.

Key files:

- `experiments/fidelity_sweep/local_pkg/fidelity_local.py`
- `experiments/fidelity_sweep/local_pkg/run_cell.py`
- `experiments/fidelity_sweep/local_pkg/analysis/compare_cells.py`
- `experiments/fidelity_sweep/local_pkg/analysis/tier2_logit_fidelity.py`
- `experiments/fidelity_sweep/local_pkg/results/`

## 6. Next experiments implied by this report

1. ~~Complete the tier contrast with Sequential-Scaled × math500.~~ **Done in this
   report:** the math500 tier contrast is small (86.4% → 80.4%), so the MBPP+ gap is
   task-specific. The priority shifts from "measure more tiers" to "explain the MBPP+
   gap" — items 3–4 below.
2. ~~Repeat MBPP+ light/scaled across rotations and pool discordances.~~ **Done for the
   quantizer rotation** (5-rotation matrix, gap +40.2 pp [+34.9, +45.7]; see
   `results/rotation_matrix_SUMMARY.md`). The generation-seed and problem-subset axes remain.
3. Compute output-length distributions and a per-token first-divergence hazard with
   right-censoring.
4. Test whether REF top-1/top-2 logit margins explain the light/scaled contrast.
5. Implement teacher-forced position-aligned logit capture for stable KL and link
   localization.
6. Evaluate deliberation/tool-calling, where a changed intermediate action is itself
   consequential.
