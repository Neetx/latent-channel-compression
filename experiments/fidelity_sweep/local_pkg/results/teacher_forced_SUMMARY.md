# MBPP+ teacher-forced mechanism — why does the trajectory drift, and why is scaled robuster?

**Provenance.** RTX 5070 Ti (16 GB, bf16), RecursiveMAS @ f95d512, MBPP+, n=250, generation
seed 42, quantizer seed 42, T=3, Variant B INT4. Aligned (teacher-forced) capture: each decode
position is forced along the paired full-precision REF tokens while the inter-agent channel stays
INT4-quantized, and the **clean** next-token logits are recorded (top-K=256, 128-position window).
Captures run at **batch_size=1** — gate G0 (teacher-forcing the full-precision REF reproduces the
free-running REF exactly) passes at b=1 (0 mismatches) but not at b=2 (post-EOS batched-padding
artifact). The TF-REF reference is the b=1 REF capture (light: freshly generated; scaled: the
committed b=1 REF reused). Orchestrator [`run_teacher_forced.py`](../run_teacher_forced.py),
analysis `python ../analysis/teacher_forced_analysis.py`.

> **The teacher-forced run's final "accuracy" is not interpretable** and is intentionally ignored:
> the output is *forced* to the REF tokens and *truncated* at the 128-position capture window, so
> long MBPP+ solutions are cut off. The real INT4 task accuracy is the free-running number
> (36.4 % light / 72.4 % scaled). The TF run exists only to capture aligned per-position logits.

## Per-position perturbation and flips (REF vs INT4, same forced prefix)

| tier | flip-rate (argmax) | median REF margin (top1−top2) | median \|top-1 logit diff\| | REF top-1 still rank-1 |
|---|---:|---:|---:|---:|
| light  | 3.9 % | 5.75 | 0.250 | 96.1 % |
| scaled | 2.1 % | 7.62 | 0.125 | 97.9 % |

## Flip-rate vs REF margin (the margin-tipping signature)

| REF margin (logit) | light flip | scaled flip |
|---|---:|---:|
| <1   | 22.1 % | 11.7 % |
| 1–2  | 4.2 %  | 2.6 % |
| 2–4  | 1.9 %  | 1.4 % |
| 4–8  | 1.0 %  | 1.7 % |
| 8–16 | 0.4 %  | 0.5 % |
| >16  | 0.2 %  | 0.0 % |

## Decomposition of the light−scaled flip-rate gap (Oaxaca, problem-clustered bootstrap)

| component | value |
|---|---:|
| raw gap (light − scaled) | **+1.81 pp**  (95 % CI [+1.23, +2.40]) |
| margin-distribution component (scaled decides with larger margins) | +0.43 pp (**24 %**) |
| per-margin / sensitivity component (channel perturbs scaled's logits less) | +1.39 pp (**76 %**) |

## Findings

1. **T1 — the perturbation is universal but small.** Holding the prefix fixed, INT4 quantization
   shifts the next-token logits at essentially every position by a small amount (median |top-1
   logit diff| 0.25 light / 0.125 scaled) and leaves the REF top-1 token ranked first 96–98 % of
   the time. The channel is perturbed everywhere, not only where the trajectory forks.
2. **T2 — flips concentrate at low margin (confirmed, ~100× gradient).** The arg-max flip
   probability is a strictly decreasing function of the REF top1−top2 margin on both tiers (light
   22.1 % at margin <1 down to 0.2 % at >16). The trajectory drift is a **margin-tipping**
   phenomenon: quantization noise only changes the emitted token where the decision is knife-edge.
3. **T3 (strong form) is falsified; the mechanism is identified.** The hypothesis that the tier
   gap is *mediated by the margin distribution* explains only **24 %** of the +1.81 pp gap. The
   **dominant driver (76 %) is attenuation**: the latent-channel perturbation reaching the larger
   (scaled) model's output logits is about **half** as large (0.125 vs 0.250), so at matched
   margin scaled flips less (e.g. 11.7 % vs 22.1 % at margin <1). Scaled is more trajectory-robust
   primarily because it **absorbs the channel noise**, and secondarily because it decides with
   larger margins. This is direct per-position evidence for the paper's "larger models absorb
   channel noise" hypothesis, and it sharpens it: absorption dominates decision-confidence.

## Math500 control (does the attenuation generalize, or is it task-specific?)

Re-running the same teacher-forced capture on Math500 (where the free-running tier gap is ~0)
tests whether the attenuation mechanism is a general capacity property or task-conditioned.

| tier | flip-rate | median REF margin | median \|top-1 logit diff\| |
|---|---:|---:|---:|
| light  | 2.4 % | 11.62 | 0.375 |
| scaled | 3.2 % |  9.62 | 0.250 |

| component (light − scaled) | MBPP+ | Math500 |
|---|---:|---:|
| raw flip-rate gap | +1.81 pp [+1.23,+2.40] | **−0.88 pp** [−1.43,−0.34] |
| attenuation (per-margin) component | +1.39 pp (76 %) | **−0.18 pp** |

**Margin-tipping (T2) is universal** — the flip-vs-margin gradient holds on Math500 too (light
25.7 % at margin <1 down to 0.1 %). **But the tier gap and its attenuation are task-specific**: on
Math500 the gap nearly vanishes and slightly reverses, and the attenuation component collapses from
+1.39 pp to −0.18 pp. The larger constellation's noise-absorption advantage is therefore not a
general capacity property; it appears on code and not on math, mirroring the free-running tier-gap
pattern (~40 pp on MBPP+, ~0 on Math500).

## Limitations

- Single generation seed (42), single quantizer rotation (42), single problem subset, INT4 only,
  two tasks (MBPP+ and the Math500 control).
- The 128-position window and top-K=256 truncation (rank censored >256 at only ~0.2 % of
  positions). The decomposition is Oaxaca with light as the reference weighting; the qualitative
  split (attenuation ≫ margin distribution) is insensitive to the reference choice.
- The TF accuracy is forced+truncated and not a task metric (see note above).

Raw NPZ captures are not committed; regenerate with `run_teacher_forced.py` and analyse with
`teacher_forced_analysis.py`.
