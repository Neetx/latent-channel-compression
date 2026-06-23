# Tier-2 logit fidelity and trajectory analysis (corrected)

Post-hoc analysis of the local greedy REF/INT4 top-K logit captures. The corrected
analysis pairs only the **250 fixed primary solver sequences** in each cell and excludes
conditional answer-retry calls, because REF and INT4 can retry different examples.
Local captures use top-K=256 and are right-censored at **128 decode positions**.

The probability estimator works on the union of both top-K supports, uses the captured
full-vocabulary log-sum-exp for normalization, assigns a boundary-probability estimate
to union tokens missing from one side, and places the remaining probability in one tail
bucket. Missing-token estimates are subtracted from the tail so they are not counted
twice. KL/JS remain matched-prefix approximations, not full-vocabulary teacher-forced
measurements.

| cell | divergence within 128 | common prefix | KL nats (95% CI) | JS | channel cosine |
|---|:---:|:---:|:---:|:---:|:---:|
| math500 / light | 86.4% | 53.7 | 0.079 [0.043, 0.124] | 0.009 | 0.9952 |
| math500 / scaled | 80.4% | 54.6 | 0.147 [0.072, 0.251] | 0.013 | 0.9953 |
| mbppplus / light | 92.8% | 35.8 | 0.113 [0.078, 0.156] | 0.015 | 0.9953 |
| **mbppplus / scaled** | **51.2%** | **65.7** | **0.059 [0.031, 0.092]** | **0.003** | 0.9953 |
| medqa / light | 96.4% | 32.4 | 0.045 [0.030, 0.062] | 0.005 | 0.9952 |

`common prefix` counts positions strictly before the first mismatching token. A
non-diverged sequence is censored at its captured length, at most 128 positions.

## Findings

1. **Channel-level distortion is stable across cells.** Mean cosine is
   approximately 0.9953 everywhere, as expected for the same data-oblivious 4-bit
   quantizer. This is a consistency check; equal mean cosine does not imply equal
   semantic perturbations relative to each model's decision boundary.

2. **The same-task light/scaled contrast is large on mbppplus but task-specific.** On
   mbppplus, divergence within the capture window falls from 92.8% (light) to 51.2%
   (scaled), the common prefix rises from 35.8 to 65.7 positions, and matched-prefix KL
   falls (0.113 to 0.059). **On math500 the same light→scaled change is small:** divergence
   86.4% → 80.4%, common prefix essentially flat (53.7 → 54.6), and KL actually *rises*
   (0.079 → 0.147). The large mbppplus gap therefore does not generalize across tasks.

3. **Interpretation: task-specific tier association, not a causal capacity law.**
   Sequential-Light and Sequential-Scaled differ in models, tokenizers, adapters,
   hidden dimensions, output lengths, and logit margins as well as parameter count, and
   the contrast's sign/magnitude depends on the task. A general capacity law would predict
   a comparable divergence drop on math500; none appears. The result suggests the scaled
   system is more trajectory-robust *on mbppplus*; it does not prove that model capacity
   itself causes the difference.

4. **The divergence metric is windowed and length-dependent.** “51.2% divergence”
   means 128-position-window divergence, not full-generation divergence. Shorter
   outputs have fewer opportunities to diverge. A per-token divergence hazard or
   survival analysis is needed before comparing uncensored trajectory stability.

5. **MedQA remains a diagnostic/confounded cell.** Its weak greedy REF has the
   documented first-option bias. The high divergence is real within the capture
   window, but it should not be used as clean evidence about answer preservation or
   capacity.

## Relationship to the answer-level results

Across the four clean math/code cells, aggregate accuracy deltas are small and not
statistically distinguishable from zero, while 4.4--10% of individual correctness
outcomes flip. This is **aggregate answer robustness with per-problem churn**, not
bit-exact answer preservation. MedQA's greedy +15.2 pp result is excluded from that
claim because its REF is pathological; its sampled ladder is the appropriate result.

The defensible current headline is:

> Aggregate answer accuracy is robust in the clean cells, while trajectory stability
> varies by system tier *and task*: on mbppplus the scaled constellation is much less
> likely to diverge within the first 128 positions than light, but on math500 the same
> tier change is small — so the contrast is a task-specific association, not a capacity law.

## Required follow-up before a mechanistic claim

- repeat light/scaled comparisons across seeds and at least one additional task;
- report generation-length distributions and per-token divergence hazard;
- test whether REF top-1/top-2 logit margins explain the tier contrast;
- run teacher-forced, position-aligned full-vocabulary or high-K KL capture.
