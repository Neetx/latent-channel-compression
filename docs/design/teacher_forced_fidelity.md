# TEACHER_FORCED_FIDELITY — design plan for the aligned mechanism test

**Status:** design (not implemented)
**Created:** 2026-06-25
**Predecessors:**
[ROADMAP.md](../../ROADMAP.md) §2 (aligned mechanism test, closure-package item 2),
[divergence_hazard_SUMMARY.md](../../experiments/fidelity_sweep/local_pkg/results/divergence_hazard_SUMMARY.md)
(the free-running first-divergence result),
[links_ablation_SUMMARY.md](../../experiments/fidelity_sweep/local_pkg/results/links_ablation_SUMMARY.md)
(drift is not localized to a link type).
**Successor:** kernel + driver + analyzer implementation, then `docs/reports/09_teacher_forced_mechanism.md`.

This document freezes the design for the teacher-forced, position-aligned fidelity measurement
*before* code. It is the one experiment the paper repeatedly names as missing ("not a
teacher-forced causal comparison"). It upgrades the paper from a **behavioural** dissociation
result to one that can also say **why** the trajectory drifts.

> **Implementation status (2026-06-25).** Kernel + driver implemented. The forcing is done with
> a `LogitsProcessor` that records the clean logits then forces the REF token (reusing
> `generate`'s machinery, so it is agnostic to `input_ids` vs `inputs_embeds`).
> **Gate G0 PASSED at batch_size=1** (TF-REF reproduces the free-running REF exactly: 0/256
> top-1 mismatches, max|logit diff| = 0). G0 also *caught* a real defect: at **batch_size=2**
> the post-EOS dynamics of variable-length batched generation diverge under forcing (13/512
> mismatches). **Resolution: teacher-forced captures run at batch_size=1**, where each sequence
> stops at its own EOS and there is no cross-sequence padding interaction. This is the intended
> use of G0 — a cheap gate that localizes a subtle correctness bug before any INT4 GPU spend.

---

## 1. Why this, and an honest self-critique

Two turns ago I argued teacher-forcing is **not required to close** the paper. I still believe
that: the paper is internally consistent as a behavioural dissociation study and disclaims every
mechanism/causal claim (verified by grep — all margin/capacity/mechanism mentions are hedged or
flagged as future work). So this is **not** a closure patch; it is a deliberate upgrade to a
**stronger** paper that makes a *mechanistic* claim.

What changes my cost/benefit, and weakens my earlier objection, is the design itself:

- My earlier worry was "expensive instrumentation + GPU for a non-essential addition." But the
  measurement here is a **single forced forward pass** per primary call, not autoregressive
  decoding, and the full-precision side (TF-REF) is **provably identical to the REF capture we
  already have** (§5.3) — so it is **cheaper** than the free-running captures, not more
  expensive. The dominant cost is implementation + validation, not GPU.
- The payoff is a **decisive, falsifiable** test of the margin-mediation hypothesis (§2), which
  is the single most reviewer-exposed soft spot ("you observe divergence but the matched-prefix
  KL is selection-biased and you never explain it").

The honest residual risk is the opposite one: teacher-forcing might **fail to find** a clean
margin mechanism (§2 falsification). That is still a publishable, informative outcome, and the
design is pre-registered to report it either way (§12). It is not biased toward confirming the
hypothesis.

---

## 2. Research question as a falsifiable claim

The headline behavioural fact: under INT4 latent quantization, aggregate accuracy is preserved
but greedy trajectories diverge a lot, and far more on MBPP+/light (92.8%) than MBPP+/scaled
(51.2%) — a +40.2 pp, rotation-robust, task-specific gap. The open question is the **mechanism**.

> **Claim T (margin mediation).** Holding the decoded text prefix fixed (teacher-forced along the
> full-precision REF tokens), INT4 latent quantization perturbs the next-token distribution at
> *every* position, but only flips the arg-max where the REF top-1/top-2 logit **margin** is
> small. Concretely:
> 1. **T1 — local perturbation is near-universal but small.** Median per-position symmetric KL
>    between REF and INT4 next-token distributions is small (e.g. ≲ 0.05 nat) and *non-zero* at
>    essentially every position — the channel is perturbed everywhere, not just where it flips.
> 2. **T2 — flips concentrate at low margin.** The per-position arg-max flip probability is a
>    strictly decreasing function of the REF top-1−top-2 margin; high-margin positions almost
>    never flip.
> 3. **T3 — margin explains the tier gap.** The light−scaled flip-rate gap shrinks substantially
>    after conditioning on margin: scaled flips less *because* it makes higher-margin (more
>    confident) token decisions, not because the same decision is more robust at matched margin.

**What falsifies each part (pre-registered):**
- **¬T1** if median per-position KL is ≈ 0 (no perturbation; the free-running divergence would
  then be a sampling/length artifact, contradicting the hazard result) **or** is large at most
  positions (then the answer-robustness story is itself in question).
- **¬T2** if flip probability is flat in margin, or flips occur at high-margin positions — then
  the perturbation is not a margin-tipping phenomenon and some other geometry drives divergence.
- **¬T3** if, at **matched margin**, light still flips materially more than scaled — then margin
  is *not* the mediator of the tier gap, and the tier effect lives elsewhere (e.g. larger latent
  perturbation reaching the solver, or different logit-Jacobian sensitivity). This is the most
  important and most falsifiable sub-claim; T3 is the one that would let the paper *explain* the
  flagship contrast rather than merely report it.

T3 is the load-bearing claim. T1/T2 can hold while T3 fails — that asymmetry is exactly what a
clean design must be able to detect.

---

## 3. What "done" looks like

A per-cell mechanism table plus one mediation figure:

| cell | bits | median pos. KL | flip-rate (TF) | mean REF margin | flip-rate at matched margin (light−scaled) |
|---|---:|---:|---:|---:|---:|
| MBPP+ / light | 4 | ? | ? | ? | — |
| MBPP+ / scaled | 4 | ? | ? | ? | **? (T3 estimand)** |
| Math500 / light | 4 | ? | ? | ? | — |
| Math500 / scaled | 4 | ? | ? | ? | (control: expect ≈ 0 gap either way) |

Plus a figure: per-position **flip probability vs REF margin**, one curve per tier, on MBPP+.
T3 is read off the vertical gap between the two curves: if margin mediates, the curves
**collapse onto each other** (the tier gap is explained by *where* each tier sits on a shared
curve, i.e. its margin distribution); if they stay separated at matched margin, T3 is falsified.

Done = this table + figure exist with problem-clustered bootstrap CIs, the validation gates in §9
pass, and we can state for each of T1/T2/T3 whether it holds, with uncertainty.

---

## 4. Why the free-running capture cannot answer this

In the current capture each generated step conditions on the model's **own** previously chosen
tokens (`out.scores` from a monkey-patched `GenerationMixin.generate`, greedy). After the first
arg-max disagreement at position *k*, the REF and INT4 runs condition on **different prefixes**,
so every later position compares two already-diverged texts — divergence **compounds** and is no
longer attributable to the per-step channel perturbation. The matched-prefix KL sidesteps this by
looking only at the common prefix, which is a **selection-conditioned** (biased) subset of
positions. Teacher-forcing removes the confound by construction: both conditions consume the same
fixed REF prefix at every position, so each position is an independent, controlled paired
comparison of the channel perturbation alone, over the **entire** sequence.

---

## 5. Intervention design

### 5.1 Constraint recap

`external/RecursiveMAS` is read-only. The existing pipeline already intervenes only via (a)
monkey-patching the transformers method `GenerationMixin.generate`, and (b) injecting the
`VARIANT_B_HEAD` into a **working copy** of the upstream source (`patch_inference_mas` /
`patch_run_py`). Teacher-forcing must stay inside this envelope.

### 5.2 The forced-forward, inside the same generate hook

Add a `TEACHER_FORCED=1` mode to the kernel's `VARIANT_B_HEAD` generate hook. In TF mode the
patched `generate`, for each **primary** call *c* (the same primary calls the analysis already
pairs), does **not** free-run. Instead it:

1. retrieves the REF token sequence for call *c* (§5.3) — `ref_idx[c]` of shape `(T_c, B)`;
2. concatenates the call's own prompt `input_ids` (from the intercepted `generate` args) with
   `ref_idx[c]` and runs **one** model forward with the **quantized adapters still active**
   (the latent context the solver/critic/planner sees is the INT4 channel);
3. reads the logits at the `T_c` answer positions — these are
   `p_int(· | REF prefix, quantized latent context)` — and records them into the existing
   `_vb_logit_buffer` in the **same** `{vals, idxs, full_lse, tail_log}` top-K format, so the
   downstream NPZ/analysis pipeline is unchanged;
4. returns a `sequences` tensor equal to `[prompt, ref_idx[c]]` so upstream call sites that
   consume the text continue along the REF trajectory (keeping the whole pipeline aligned to REF
   and eliminating condition-dependent retry calls — a bonus over free-running pairing).

A causal decoder-only forward over `[prompt, REF]` yields `logits[i] = p(token_{i+1} |
tokens[:i+1])`, exactly the teacher-forced distribution, in one pass instead of `T_c` decode
steps.

### 5.3 TF-REF is the existing REF capture (no new full-precision run)

Greedy REF already picks the arg-max at every step, i.e. it already *followed* `ref_idx`. Forcing
REF along its own tokens feeds an identical prefix at every position, so the forced logits equal
the free-running REF logits (deterministic forward). Therefore **TF-REF ≡ the committed REF
capture** and is reused as the full-precision side; only **TF-INT4** is a new run per cell. (This
identity is also our numerical validation gate, §9.)

### 5.4 REF-token source and pairing

`ref_idx[c] = REF_npz[f"batch{c}_idxs"][:, :, 0]` (top-1 column = the greedy tokens) over the
captured ≤256-position window. The pipeline issues primary calls in a fixed order and processes
the same problems in the same batches (seed 42, same subset), so buffer index *c* aligns between
the REF and TF-INT4 runs. Because TF forces REF content, the TF-INT4 run is **structurally
identical** to REF (same call count, same `T_c`), so pairing is exact and length-divergence
censoring (the free-running headache) does not arise.

---

## 6. Metrics and the mediation analysis

All computed per aligned position from the paired TF-REF and TF-INT4 top-K captures (tail
corrected via `full_lse`/`tail_log`, as the existing matched-prefix KL already does):

1. **Per-position symmetric KL / JS** `D(p_ref ‖ p_int)` — magnitude of the local perturbation
   (tests T1).
2. **REF top-1−top-2 margin** = `vals_ref[...,0] − vals_ref[...,1]` (logit gap = decision
   confidence).
3. **Rank of the REF top-1 token under INT4** — where REF's chosen token falls in INT4's ordering
   (1 = unchanged; >256 = censored, report rate).
4. **Arg-max flip indicator** = `idxs_int[...,0] ≠ idxs_ref[...,0]` — the per-position,
   context-controlled analog of a divergence event (tests T2/T3).
5. **(optional) solver hidden-state norm** at the forced positions — needs a separate activation
   hook; deferred unless T1–T3 are ambiguous.

**Primary mechanism analysis (T2/T3):** bin positions by REF margin and estimate flip probability
per bin → the flip-vs-margin curve, per tier, on MBPP+. T3 estimand = the light−scaled flip-rate
difference **after** conditioning on margin (stratified / logistic with margin as covariate),
with a problem-clustered bootstrap (resample problems; positions nested in problems are repeated
measures, never pooled as independent — same discipline as the rotation matrix). Report the
*unconditioned* tier gap and the *margin-conditioned* gap side by side: their difference is how
much margin explains.

---

## 7. Experiments (pre-registered)

Cells, each = one new TF-INT4 capture (REF reused), INT4 (b=4) primary:

| # | cell | role | pre-registered prediction |
|---|---|---|---|
| E1 | MBPP+ / light | tier-contrast target (fragile) | high flip-rate, many low-margin positions |
| E2 | MBPP+ / scaled | tier-contrast target (robust) | lower flip-rate; **T3:** gap vs E1 collapses at matched margin |
| E3 | Math500 / light | control (no tier gap) | flip-rate similar to E4 |
| E4 | Math500 / scaled | control (no tier gap) | **margin gap vs E3 ≈ 0**, consistent with the ~0 behavioural tier gap on math |

The MBPP+ pair (E1, E2) is the decisive test of T3. The Math500 pair (E3, E4) is the **control**:
the behavioural tier gap is ~0 on math, so margin mediation predicts ≈ 0 flip-rate gap there too;
a large unexplained math gap would itself complicate T3.

**Optional dose-response (post-primary, only if T1–T3 hold):** repeat E1/E2 at b=3 and b=6 to
check that flip-rate tracks perturbation magnitude monotonically (a second, independent line of
support for the margin-tipping mechanism). Not in the primary gate.

GPU budget: TF-INT4 is a forced forward (no autoregressive decode) over the same pipeline, so
per-cell cost is ≤ the corresponding free-running capture (light ≲ 1 h, scaled ≲ 2.5 h). Four
primary cells ≈ the same ~7 h envelope as the links ablation, REF reused.

---

## 8. Implementation plan (repo-native)

Following the existing folder/file standards (mirrors `run_links_ablation` + `divergence_hazard`):

1. **`experiments/fidelity_sweep/kernel_pkg/fidelity_kernel.py`** — add `TEACHER_FORCED` env +
   the forced-forward branch in the generate hook (§5.2); load the paired REF NPZ path via a new
   env `TF_REF_NPZ`. Keep the `{vals,idxs,full_lse,tail_log}` schema unchanged.
2. **`experiments/fidelity_sweep/local_pkg/fidelity_local.py`** — add `--teacher-forced` (sets
   `TEACHER_FORCED=1`, requires/propagates `--tf-ref-npz`); extend `build_config_tag` with a
   `_tf` suffix (single source of truth; mirrors the `_qs` / `_li`/`_lo` pattern) so TF captures
   never collide with free-running ones.
3. **`experiments/fidelity_sweep/local_pkg/run_teacher_forced.py`** — resumable, lock-guarded
   orchestrator (sibling of `run_links_ablation.py`); resolves each cell's REF NPZ, runs the
   TF-INT4 capture, writes a `TEACHER_FORCED_DONE` marker.
4. **`experiments/fidelity_sweep/local_pkg/analysis/teacher_forced_analysis.py`** — per-position
   KL/JS, margin, rank, flip; the flip-vs-margin curve; the margin-conditioned tier gap with a
   problem-clustered bootstrap. Reuses `divergence_hazard` tail-corrected KL helpers.
5. **`tests/test_teacher_forced.py`** — tag suffix + orchestrator/driver tag agreement (like
   `test_links_ablation`); a **TF-REF≡REF numerical-equivalence** unit on a tiny synthetic capture
   (the §9 gate); flip/margin estimator correctness on a hand-built fixture.
6. **`experiments/fidelity_sweep/local_pkg/results/teacher_forced_SUMMARY.md`** — provenance +
   the §3 table + the verdict on T1/T2/T3.
7. **`docs/reports/09_teacher_forced_mechanism.md`** — the writeup; feeds a new paper subsection
   replacing the "matched-prefix KL is approximate" hedge with the aligned result.

---

## 9. Validation gates (must pass before trusting any cell)

| Gate | Condition | If fail |
|---|---|---|
| **G0 — forced-forward correctness** | TF-REF logits reproduce the free-running REF capture to high precision (e.g. max abs top-1 logit diff < 1e-3, identical top-1 idx at every position). | The forced forward mis-reconstructs the generate context (inputs_embeds / attention_mask / latent prefix / position_ids). Fix before any INT4 cell — this is the make-or-break gate. |
| **G1 — pairing alignment** | TF-INT4 produces the same call count and per-call `T_c` as the paired REF. | Primary-call identification or batch order drifted; debug the call counter. |
| **G2 — consistency with free-running** | TF per-position flip-rate is rank-correlated with, and an upper-envelope of, the free-running first-divergence rate across cells. | The two regimes disagree qualitatively; investigate before interpreting T2/T3. |

G0 is decisive: it is a single numerical assertion that the whole forced-forward path is faithful,
and it costs almost nothing (REF is already captured).

---

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Forced forward can't reconstruct the exact `generate` context (latent-prefix injection, `inputs_embeds`, mask) | medium | high | G0 gate on a tiny run first; if needed, intercept at `model.forward` rather than `generate`, still via monkey-patch, still upstream-read-only |
| Top-K=256 censors REF-token rank / tail KL on high-entropy positions | medium | low | tail-corrected KL already handles mass; report rank-censoring rate; raise K for the mechanism cells if disk permits |
| Multiple primary calls per problem (planner/critic/solver) — which to force? | medium | medium | force exactly the calls the divergence analysis already pairs; verify against the existing primary-call pairing before the full run |
| Margin defined on logits vs log-probs changes the curve shape | low | low | pre-register margin = top1−top2 **logit** gap; report log-prob margin as a secondary, expect monotone-equivalent |
| T3 is null (margin doesn't explain the tier gap) | medium | low (still informative) | pre-registered as a reportable outcome (§12); triggers the secondary "perturbation-magnitude / Jacobian" probe, not a silent drop |

---

## 11. What this design intentionally does NOT do

- **No new full-precision runs.** TF-REF ≡ the committed REF captures.
- **No autoregressive teacher-forcing.** A single forced forward per call; no per-step
  `LogitsProcessor` games that would corrupt the recorded logits.
- **No change to `external/RecursiveMAS`.** Intervention via the same monkey-patch + working-copy
  source injection as the existing pipeline.
- **No new benchmarks or topologies.** Only the four existing tier/task cells; deliberation and
  other styles remain Paper-2 / future work.
- **No causal *capacity* claim.** T3, if it holds, explains the tier gap *via margins* on these
  tasks; it does not control architecture/family and is not a parameter-count law.

---

## 12. What we'll know afterwards (both outcomes pre-registered)

- **If T1–T3 hold:** "The latent-quantization trajectory drift is a margin-tipping phenomenon:
  the channel perturbs every next-token distribution slightly, flipping the arg-max only at
  low-margin decisions; the scaled tier diverges less *because* it decides with larger margins.
  This mechanistically explains the rotation-robust, task-specific MBPP+ tier contrast." — a
  genuine mechanism result; removes the paper's most-cited limitation.
- **If T3 fails (margin does not explain the tier gap):** "Quantization tips low-margin decisions
  (T1/T2), but matched-margin flip rates still differ by tier, so token margin is **not** the
  mediator of the tier contrast — the tier effect must act through the size of the latent
  perturbation reaching the solver or the logit sensitivity to it, not through decision
  confidence." — a sharper, falsified-hypothesis result that redirects the mechanism search.

Either way the paper gains an aligned, selection-unbiased fidelity measurement that the current
matched-prefix KL cannot provide.
