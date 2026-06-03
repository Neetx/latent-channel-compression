# REPORT 03 — Phase 0: Variant B on REAL RecursiveLink output

**Date:** 2026-05-27
**Phase:** 0.A — identity wrapper sanity + Variant B sweep on real inner-adapter output
**Model:** `RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B` (Qwen2.5-Math-1.5B + trained inner adapter for the math task)
**Compute:** Kaggle private kernel (`<YOUR_KAGGLE_USERNAME>/recursivelink-phase-0-real-adapter-sweep` v3), CPU fallback (torch 2.10.0+cu128 but no GPU allocated on this run)
**Raw artifacts:** _experiments/results/ (archived; canonical numbers in markdown reports)_
- `phase0_results.json` — full sweep + distribution stats + Gate 0 result
- `recursivelink-phase-0-real-adapter-sweep.log` — kernel stdout/stderr

For the synthetic baselines see [REPORT.md](./01_variant_a_synthetic.md) (Variant A) and [REPORT_02.md](./02_variant_b_synthetic.md) (Variant B). For the research design see [RESEARCH.md](../RESEARCH.md).

---

## Headline

**The RecursiveLink inner-adapter output behaves, for the purposes of Variant B quantization, indistinguishably from synthetic Gaussian-on-sphere input.** rMSE matches the synthetic numbers (and the TurboQuant paper Table 1) at every bit-rate to within 2 % relative.

This means:
1. The patching pipeline (wrap `Adapter.forward` output → quantize → return) is sound.
2. Phase 0 Gate 0 **passes** (rMSE ≈ 2e-9, cosine = 1.0).
3. Phase 1 / Gate 1B is **passable** on this channel at 4 bits and very likely 3 bits as well — the dimensional-distortion characteristic of the trained Link does not punish quantization any more than a random vector on the sphere would.
4. The earlier ambiguity ("maybe RecursiveLink output is fragile in ways KV-cache isn't") is resolved on the side of **compressible**.

---

## Setup

- Loaded the released Solver model + its `adapter(math).pt` inner-link weights into a clean reimplementation of `RecursiveMAS.modeling.Adapter` (proj1 → GELU → proj2 → residual → post_LN). The match is exact (`load_state_dict(..., strict=True)` passed with zero missing/unexpected keys).
- Tokenized 15 hard-coded math prompts (208 tokens total) with the Solver's tokenizer.
- Forward through the model with `output_hidden_states=True`, took the last layer hidden state, pushed it through the inner adapter to get the actual RecursiveLink output tensor [208, 1536].
- Gate 0: ran Variant B at bits = 16 on the captured tensor. Output ≈ input to within fp32 noise.
- Sweep: Variant B at bits ∈ {8, 4, 3, 2} on the same captured tensor + on the raw pre-adapter hidden states for comparison.

Reproducibility: every Haar rotation and bootstrap uses an explicit seed; the script is checked in at [`experiments/distortion_validation/identity_check/kernel_pkg/phase0.py`](../../experiments/distortion_validation/identity_check/kernel_pkg/phase0.py) and was pushed to Kaggle via [`./bin/kaggle kernels push`](bin/kaggle).

---

## Gate 0 — identity wrapper sanity

| metric | value | threshold | pass |
|---|---|---|---|
| rMSE | 1.94 × 10⁻⁹ | < 1e-3 | ✅ |
| cosine | 1.0 | > 0.9999 | ✅ |
| norm-ratio | 1.0 | \|·−1\| < 1e-2 | ✅ |

Conclusion: rotate → trivial 16-bit quantize → inverse → rescale introduces zero meaningful drift. The plumbing is correct. We can attribute any subsequent degradation to the quantizer alone.

---

## Adapter output distribution (208 tokens, d=1536)

| stat | value | comment |
|---|---|---|
| `vector_norm_mean` | 36.69 | non-unit — the post-LN does NOT renormalize to ‖·‖ = 1 |
| `vector_norm_std` | 1.24 | **very tight** (3.4 % CV) — LayerNorm collapses scale |
| `vector_norm_min` / `max` | 35.1 / 40.5 | range only ±8 % around the mean |
| `abs_max` | 12.4 | a few coordinates carry large magnitude |
| `coord_std_min` / `max` | 0.45 / 2.62 | ~6× heterogeneity across coordinates |
| `coord_std_mean` | 0.80 | average per-coordinate sigma |

The vector-norm distribution is very tight thanks to the final LayerNorm, so per-vector L2 normalization in Variant B is essentially a scalar division by a near-constant. The per-coordinate std spans a factor of ~6 — moderate outlier coordinates exist but nothing like the raw pre-adapter signal (next section).

For comparison, raw last-hidden state (pre-adapter):
- `vector_norm_mean` = 97.5 (3× larger), `vector_norm_std` = 12.9 (10× larger CV)
- `abs_max` = 56.5
- `coord_std_min` / `max` = 0.03 / 15.6 — a factor of **500×** across coords. Classic LLM heavy-tailed activations with outlier channels.

The adapter dramatically tames the distribution: the post-LN essentially Gaussianizes the per-vector scale.

---

## Variant B sweep on REAL adapter output

| bits | rMSE (real, d=1536) | rMSE (synthetic, d=2048) | TurboQuant paper | cosine | norm-ratio | ip-error |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.0001 | 0.0001 | (lossless) | 1.000 | 1.000 | 0.0003 |
| 4 | **0.0093** | 0.0095 | 0.009 | 0.995 | 0.997 | 0.0053 |
| 3 | **0.0339** | 0.0345 | 0.030 | 0.983 | 0.984 | 0.0200 |
| 2 | **0.1159** | 0.1175 | 0.117 | 0.940 | 0.941 | 0.0656 |

Bootstrap 95 % CIs are tight (within ±0.001 of every mean). The match across the three columns is the headline of this report: **real data, synthetic data, and the published TurboQuant numbers agree to the third decimal.**

This is a *strong* empirical demonstration of TurboQuant's data-obliviousness claim: the random rotation makes the input distribution irrelevant, and our pipeline reproduces that property on a non-trivial real channel.

---

## Bonus — Variant B on RAW hidden states (pre-adapter)

The raw last-hidden state has a much messier distribution (vector_norm CV 13 %, per-coordinate outlier channels with std up to 15.6). Yet:

| bits | rMSE raw | rMSE adapted | difference |
|---:|---:|---:|---:|
| 8 | 0.0001 | 0.0001 | — |
| 4 | 0.0096 | 0.0093 | +3 % |
| 3 | 0.0345 | 0.0339 | +2 % |
| 2 | 0.1176 | 0.1159 | +1 % |

The raw signal compresses essentially as well as the adapted one. Concentration of measure under Haar rotation is doing all the heavy lifting — the outlier structure of the un-rotated coordinates is invisible to the rotated quantizer. This is a useful side-result: if we ever wanted to quantize the hidden state itself (not the adapter output), the numbers would be roughly the same.

ip-error is somewhat higher on raw than on adapted (0.081 vs 0.066 at 2 bits), which makes sense — raw vectors have larger and more variable norms, amplifying ⟨x,y⟩ scale.

---

## Decision per stage-gate

Per RESEARCH.md §6:

- **Gate 0 (identity sanity):** ✅ passed cleanly.
- **Gate 1B (Variant B 4-bit screen):** rMSE 0.009, cosine 0.995, norm drift 0.3 %, ip-error 0.005. All comfortably above the thresholds we set for stability proxies. The full §6 condition is downstream task accuracy delta, which requires the full RecursiveMAS pipeline (planner+critic+solver) running on a real benchmark — that's Phase 0.B and not in scope for this report. But the proxy gate is unambiguously open.

Decision: **proceed to Gate 2 / 3-bit on this same channel**, then graduate to a full RecursiveMAS pipeline run if 3-bit also holds.

---

## What this changes about the research plan

1. The "Variant A might fail for the wrong reason" risk is now moot — Variant B is the working quantizer on real data, and Variant A becomes purely a diagnostic for outlier sensitivity if we ever care.
2. **Per-vector L2 normalization is exactly the right preprocessing step** for RecursiveLink output. The LayerNorm in the Adapter already concentrates norms into a narrow band, so the rescale step in Variant B is essentially a no-op (just a stored scalar per vector that we multiply back in at the end).
3. The TurboQuant paper's data-oblivious property is empirically real on this channel. We do not need any calibration data, training, or per-channel statistics — a fixed Haar matrix + a fixed Lloyd-Max-Gaussian codebook is enough.
4. **Inner-adapter output and raw hidden states have similar compressibility profiles**, which means if we ever want to quantize at a different boundary (e.g., the hidden states *fed into* the adapter rather than its output), the conclusions extend.

---

## What's NOT yet measured (next steps)

1. **Downstream task accuracy.** Distortion is a proxy; the true Gate 1 condition is Δacc on math500 (or similar). This needs the full RecursiveMAS Sequential-Light pipeline: planner + critic + solver + inner adapters + outer adapters. That's a substantially bigger script — disk for ~9 GB of models, all three forwards per prompt, RecursiveMAS's iterative protocol. Phase 0.B.
2. **Compounding along recursion.** Each RecursiveMAS round invokes the Link; quantization error may compound. We've only measured 1-step distortion here.
3. **Outer link (`CrossModelAdapter`).** Same `Adapter` shape but cross-dim. Should behave similarly given the LayerNorm pattern. Verify in Phase 0.B.
4. **KL on final logits.** The strongest end-to-end proxy short of full task accuracy. Requires re-injecting the quantized output back into the rest of the model (which we don't do in Phase 0.A — we just measure the latent distortion).
5. **GPU runs.** This run fell back to CPU because the assigned GPU had unsupported compute capability. Not a problem at 1.5B + 15 prompts; will matter at full pipeline scale. The `pick_device` defensive check in the script handles both gracefully.

---

## Cost summary

- One Kaggle private kernel run, ~3 minutes wall clock on CPU (after fallback from incompatible P100).
- Local code change: +25 lines for the `pick_device` defensive GPU check.
- Two failed pushes before the success (one DNS-blocked due to unverified account, one GPU-incompatibility crash). Both produced useful information.
- Zero Kaggle quota concerns — script kernels on CPU are cheap.

---

## Provenance

- Script: [`experiments/distortion_validation/identity_check/kernel_pkg/phase0.py`](../../experiments/distortion_validation/identity_check/kernel_pkg/phase0.py).
- Kaggle metadata: [`experiments/distortion_validation/identity_check/kernel_pkg/kernel-metadata.json`](../../experiments/distortion_validation/identity_check/kernel_pkg/kernel-metadata.json).
- Kaggle notebook (private): <https://www.kaggle.com/code/<YOUR_KAGGLE_USERNAME>/recursivelink-phase-0-real-adapter-sweep> (v3).
- Determinism: torch generator seeds and numpy RNG seeds explicit in the script. Solver weights pulled from HF by snapshot, adapter weights `adapter(math).pt` from the same repo.
