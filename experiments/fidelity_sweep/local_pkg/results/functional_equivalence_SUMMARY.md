# Is the trajectory divergence functionally inert?

**Provenance.** Same four non-confounded local cells (RecursiveMAS @ f95d512, n=250, seed 42, INT4).
Per-problem *trajectory divergence within 128* comes from the paired REF/INT4 captures; per-problem
*outcome* (unit-test pass/fail on MBPP+, boxed-answer correctness on Math500) comes from the
committed per-problem JSONLs. Reproduce: `python ../analysis/functional_equivalence.py` (needs the
local NPZ captures + the committed JSONLs).

A high divergence rate can read as alarming, but a different greedy token path need not change the
task outcome: divergent code can still pass the same tests, a divergent trace can still reach the
same answer. We therefore condition the correctness flip on whether the trajectory diverged.

| cell | diverged within 128 | answer churn | P(outcome flip \| diverged) | **outcome preserved \| diverged** |
|---|---:|---:|---:|---:|
| math500 / light  | 86.4% | 10.0% | 10.6% | **89.4%** |
| mbppplus / light | 92.8% |  9.6% | 10.3% | **89.7%** |
| mbppplus / scaled| 51.2% |  4.4% |  8.6% | **91.4%** |
| math500 / scaled | 80.4% |  8.8% | 10.4% | **89.6%** |

## Findings

1. **The divergence is largely functionally inert.** Across all four cells, **89--91% of the
   diverged trajectories still produce the same final outcome** (pass/fail). The wholesale
   token-path change (51--93% of trajectories) only rarely crosses a task-relevant decision
   boundary: the per-problem probability that divergence comes with an outcome flip is just
   8.6--10.6%.
2. **It sharpens the dissociation in both directions.** "Answer-preserving" remains a
   population-level shorthand (real per-problem churn exists), but the headline divergence rate
   should not be read as 51--93% of *behaviour* changing -- only ~10% of diverged problems change
   the outcome. The interesting object is precisely the small low-margin set where the perturbation
   *does* tip a task-relevant decision (the teacher-forced mechanism, `teacher_forced_SUMMARY.md`).
3. **The preserved fraction is remarkably stable (~90%) across tasks and tiers**, even though the
   raw divergence rate ranges from 51% to 93% -- consistent with a margin-tipping mechanism whose
   per-decision flip probability, not the cumulative path divergence, controls the outcome.

## Limitations

- "Outcome" is unit-test pass/fail (MBPP+) or boxed-answer correctness (Math500) -- a coarser
  notion than full functional/semantic equivalence (e.g. identical outputs on held-out inputs).
  It is the operationally relevant outcome for these benchmarks.
- Divergence is windowed at 128 positions; an answer can still differ from generation beyond the
  window. The conditional is on *windowed* divergence.
