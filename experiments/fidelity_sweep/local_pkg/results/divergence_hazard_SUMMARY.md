# First-divergence hazard — is the trajectory contrast a generation-length artifact?

The Tier-2 "divergence within 128 positions" is **windowed and padding-inclusive**: every
capture stores a fixed 128 positions, zero-padded once a sequence finishes (detected as
`full_lse == 0`). That conflates "the two greedy sequences were identical and one simply
stopped sooner" with a genuine divergence, and makes a cell's apparent robustness depend on
how long its generations happen to be. This analysis
(`analysis/divergence_hazard.py`) re-frames first-divergence as a **right-censored survival
process**: each problem is observed only while *both* sequences still generate real tokens
(up to `L = min(len_REF, len_INT4, 128)`); the event is the first greedy top-1 mismatch;
otherwise it is censored at `L`. Only the 250 primary solver batches are used.

Reproduce: `python ../analysis/divergence_hazard.py --roots "$LCC_RUN_ROOT" ~/lcc/fid_out`
(raw NPZ captures required; regenerate them with the canonical local workflow).

| cell | n | med. len | %len<128 | 1−S(10) | 1−S(25) | 1−S(50) | 1−S(100) | event% | early hazard (pos 1–25) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math500 / light | 250 | 128 | 0% | 0.184 | 0.324 | 0.548 | 0.808 | 86.4% | 0.0154 |
| math500 / scaled | 250 | 128 | 0% | 0.192 | **0.384** | 0.572 | 0.752 | 80.4% | **0.0191** |
| mbppplus / light | 250 | 128 | 0% | 0.248 | 0.500 | 0.748 | 0.912 | 92.8% | 0.0271 |
| **mbppplus / scaled** | 250 | 117 | **60%** | 0.020 | **0.154** | 0.321 | 0.554 | 51.2% | **0.0066** |
| medqa / light | 250 | 128 | 0% | 0.176 | 0.528 | 0.808 | 0.944 | 96.4% | 0.0295 |

`1−S(t)` is the fraction of problems whose greedy path has diverged by position `t`;
`event%` is the overall divergence within the observation window; the `early hazard` is the
mean per-position divergence probability over positions 1–25 (a length-independent regime,
since essentially every problem is still at risk that early).

## Findings

1. **Validation.** The overall `event%` reproduces the independent
   `tier2_logit_fidelity.py` divergence rates exactly (86.4 / 80.4 / 92.8 / 51.2 / 96.4),
   confirming the per-problem extraction is correct.

2. **Length variation is confined to one cell.** Only `mbppplus / scaled` finishes early
   (median length 117, 60% of problems shorter than 128); the capable model writes more
   concise code and stops sooner. Every other cell runs to the 128-position cap (0% short),
   so for them the length-censored and the padding-inclusive metrics coincide.

3. **The MBPP+ tier contrast is real, early, and NOT a length artifact.** At position 25 —
   long before any censoring matters — MBPP+/light has already diverged on 50.0% of
   problems but MBPP+/scaled on only 15.4%, and the early per-position hazard differs about
   **4×** (0.0271 vs 0.0066). The gap is present from the first tokens, so it is a genuine
   lower per-token divergence propensity, not an artifact of mbppplus/scaled's shorter
   generations.

4. **The Math500 tier "advantage" is a late-window effect that vanishes (even reverses)
   early.** The full-window divergence makes scaled look slightly more robust on Math500
   (80.4% vs 86.4%), but early it is the opposite: at position 25 scaled has diverged *more*
   (38.4% vs 32.4%) and its early hazard is higher (0.0191 vs 0.0154). The modest full-window
   gap comes entirely from the last positions (1−S(100): 0.752 vs 0.808).

5. **Conclusion: the task-specificity is strengthened, not explained away.** The dramatic
   scaled trajectory-robustness exists on MBPP+ from the very first tokens and is essentially
   absent on Math500. Generation length does not account for the cross-task difference. This
   sharpens the open mechanism question: what about the MBPP+ task (token margins, answer
   structure) lets the scaled constellation absorb the same channel distortion so much more
   per token there, but not on math?

## Limitations

- Single seed (one quantizer rotation, one problem order); the multi-seed replication in
  `ROADMAP.md` §1 remains required.
- The early/late decomposition is descriptive; a formal log-rank / Cox comparison with a
  length covariate would quantify it. The early-hazard contrast is already length-robust by
  construction.
- Censoring at `min(len_REF, len_INT4)` treats a pure length difference (identical tokens,
  one sequence longer) as censoring rather than as an event — the conservative choice for a
  "when does the path first differ" question.
- Capture is right-censored at 128 positions, so `1−S(100)` and `event%` still understate
  full-generation divergence; only the early probes are window-independent.
