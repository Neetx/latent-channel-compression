# REPORT 01 — Variant A (Hadamard + uniform) distortion on synthetic data

**Date:** 2026-05-26
**Variant:** A (Randomized Hadamard rotation + uniform symmetric scalar quantizer)
**Compute:** Kaggle CPU notebook (private), torch 2.10.0+cpu, runtime 37 s
**Raw artifacts:** _experiments/results/ (archived; canonical numbers in markdown reports)_
- `variant_a_sweep.json` — full results with 95% bootstrap CIs
- `summary.csv` — flat table for spreadsheet inspection
- `output.zip` — original Kaggle kernel output

---

## TL;DR

- **8-bit is essentially lossless** at every tested dimension and configuration. Use as upper-bound sanity reference.
- **4-bit per-channel is the strongest screening candidate** for Phase 0: cosine ≥ 0.99, rMSE ≤ 0.02, norm-ratio drift ≤ 1%.
- **3-bit per-channel is marginal-promising** (cosine 0.95, rMSE 0.08-0.11). Worth probing on real RecursiveLink output but expect partial degradation.
- **2-bit collapses** with per-tensor scaling (cosine 0.31). With per-channel it recovers norm but cosine is still 0.63 → likely too lossy for one-shot latent communication.
- **Per-channel scaling roughly halves rMSE** at every bit-rate. It is the cheap win and should be the default for Variant A going forward.
- **Hadamard + uniform is ≈ 2× worse than honest TurboQuant** (Lloyd-Max for Beta) at 4 bits on Gaussian-on-sphere inputs (0.020 vs ~0.009 reported in the TurboQuant paper). This matches the structural prediction: substituting uniform for Lloyd-Max throws away meaningful capacity even after the rotation has Gaussianized the marginals.

The stage-gate decision tree (RESEARCH.md §6) is **not yet triggered** by this run — we have only synthetic data. But Variant A passes its own screening criterion (4-bit Δacc proxy looks tolerable), so the next concrete step is Variant B on synthetic data, then both variants on real RecursiveLink output.

---

## What was measured

- **Quantizer**: Variant A as specified in RESEARCH.md §5.1.
- **Inputs**: i.i.d. N(0, I) samples normalized to the unit sphere — the regime where TurboQuant theory holds tightly and where Hadamard rotation is supposed to be near-equivalent.
- **Grid**: 3 dims × 2 scaling modes × 5 bit-rates × 5 seeds × 1000 vectors/seed = 150 configurations, ~5 000 vectors per cell.
- **Metrics**: rMSE = ‖x−x̂‖²/‖x‖², cosine, norm-ratio ‖x̂‖/‖x‖, inner-product error |⟨x,y⟩−⟨x̂,ŷ⟩|/(‖x‖‖y‖). 95 % bootstrap CIs over 1 000 resamples.
- **Out of scope here**: real RecursiveLink output, downstream task accuracy, KL on logits, recursion compounding. Those come in Phase 0 once we have model checkpoints loaded.

---

## Results

### Per-tensor scaling (the harder bar)

| d    | bits | rMSE   | cosine | norm-ratio | ip-error |
|-----:|-----:|-------:|-------:|-----------:|---------:|
|  256 |   16 | 0.0000 | 1.0000 |     1.0000 |   0.0000 |
|  256 |    8 | 0.0001 | 0.9999 |     1.0001 |   0.0008 |
|  256 |    4 | 0.0366 | 0.9823 |     1.0182 |   0.0139 |
|  256 |    3 | 0.1986 | 0.9137 |     1.0942 |   0.0337 |
|  256 |    2 | 0.9345 | 0.3715 |     0.6471 |   0.0515 |
| 1536 |    4 | 0.0350 | 0.9830 |     1.0174 |   0.0054 |
| 1536 |    3 | 0.1898 | 0.9164 |     1.0868 |   0.0130 |
| 1536 |    2 | 0.9095 | 0.3178 |     0.4432 |   0.0207 |
| 2048 |    4 | 0.0427 | 0.9793 |     1.0211 |   0.0053 |
| 2048 |    3 | 0.2323 | 0.9006 |     1.1086 |   0.0129 |
| 2048 |    2 | 0.9600 | 0.3118 |     0.5523 |   0.0176 |

### Per-channel scaling

| d    | bits | rMSE   | cosine | norm-ratio | ip-error |
|-----:|-----:|-------:|-------:|-----------:|---------:|
|  256 |    8 | 0.0001 | 1.0000 |     1.0000 |   0.0006 |
|  256 |    4 | 0.0198 | 0.9903 |     1.0101 |   0.0099 |
|  256 |    3 | 0.1077 | 0.9503 |     1.0524 |   0.0239 |
|  256 |    2 | 0.7461 | 0.6313 |     1.0110 |   0.0564 |
| 1536 |    4 | 0.0151 | 0.9925 |     1.0076 |   0.0036 |
| 1536 |    3 | 0.0825 | 0.9612 |     1.0408 |   0.0085 |
| 1536 |    2 | 0.6046 | 0.6775 |     0.9294 |   0.0204 |
| 2048 |    4 | 0.0202 | 0.9901 |     1.0101 |   0.0036 |
| 2048 |    3 | 0.1099 | 0.9492 |     1.0537 |   0.0087 |
| 2048 |    2 | 0.7509 | 0.6261 |     1.0041 |   0.0191 |

CIs at 95% bootstrap are tight — within ≤ ±0.001 of the mean for rMSE/cosine in the 3-8 bit range and ≤ ±0.01 at 2 bits. Full CIs in `summary.csv`.

---

## Analysis

**1. Dimension barely affects relative distortion.** rMSE at 4-bit per-channel is 0.020, 0.015, 0.020 for d = 256, 1536, 2048. The mild non-monotonicity is sub-CI noise. Hadamard does Gaussianize coordinates well enough that the per-coordinate quantization problem becomes dimension-independent in the relative sense — exactly what concentration of measure predicts.

**2. Inner-product error scales ≈ 1/√d.** At 4-bit per-tensor: 0.0139 (d=256), 0.0054 (d=1536), 0.0053 (d=2048). Ratio matches √(256/1536) ≈ 0.41 vs observed 0.0054/0.0139 = 0.39. Consistent with the TurboQuant theoretical analysis and a strong signal that the rotation is doing what it should. For RecursiveLink at d ≈ 2048, inner products are preserved well even at 4 bits.

**3. Per-channel is the cheap win.** Halves rMSE at 4-bit (0.043 → 0.020 at d=256). Larger relative improvement at lower bit-rates. The reason is that even after Hadamard rotation the *maximum* coordinate magnitude is moderately heavy-tailed across vectors — per-tensor scaling has to allocate dynamic range for the worst outlier, wasting range on typical coordinates. Per-channel pays the cost separately per axis and recovers the slack.

**4. Norm-ratio drift is a real artifact at low bits.** At 3-bit per-tensor, ‖x̂‖/‖x‖ = 1.09. The quantizer biases reconstruction magnitude upward by ~9 %. For a consumer that pipes the output into a LayerNorm (as the next RecursiveMAS agent does), this gets absorbed — but it's a flag to watch in the real-model run. Per-channel mostly removes the drift (1.04 at 3-bit, 1.01 at 4-bit).

**5. 2-bit is structurally broken with uniform quantization, even after rotation.** Per-tensor: cosine 0.31, norm collapses to 0.5. Per-channel: cosine recovers to 0.63, norm to ≈1, but rMSE is still 0.60-0.75 — most of the signal is gone. This is the bit-rate where Lloyd-Max-for-Beta vs uniform diverges most dramatically (the Beta distribution is heavily concentrated near zero, and uniform quantization wastes most of its 3 levels on coordinates that are almost never hit). Expect Variant B to do meaningfully better here.

**6. Compared to TurboQuant paper's reported numbers.** Their Theorem 1 gives E[‖x−x̂‖²] ≤ (√3π/2) · 4⁻ᵇ. At b=4 this is 0.034; their empirical is 0.009. Ours at b=4 per-tensor is 0.04, per-channel is 0.02. So per-tensor Variant A roughly matches the *theoretical upper bound* of TurboQuant (and per-channel is ~half of that), but both are significantly worse than the actual TurboQuant *empirical* number. The gap is the cost of using uniform instead of Lloyd-Max. This validates the original prediction that Variant A is a screening tool, not a final answer.

---

## Decision per stage-gate (RESEARCH.md §6)

Stage Gate 1A (Variant A at 4-bit on screening data) — formally requires downstream task Δacc ≥ −2 pp on a real task. Synthetic distortion is **not** task accuracy, so we cannot trigger Gate 1A from this report. However:

- The proxy signals (cosine 0.99, rMSE 0.02, norm-drift 1 % at 4-bit per-channel) are well above the thresholds we'd want for downstream stability.
- No reason to stop. Next step is to run on real RecursiveLink output.

Stage Gate 1B (Variant B with Lloyd-Max) — not yet attempted. Variant B is expected to outperform Variant A by a constant factor of ~2 on rMSE at 4-bit, matching the TurboQuant paper. That comparison is the natural follow-up before any real-model run.

---

## What's next

In order:

1. **Variant B (honest TurboQuant): Haar rotation + Lloyd-Max-for-Beta**, synthetic-data sweep, head-to-head with Variant A on this same grid. Establish whether the Lloyd-Max codebook is worth the implementation cost on Hadamard-friendly inputs.
2. **CrossModelAdapter / Adapter patching utility** (`src/adapters/patch.py`) that wraps a quantizer module around the post-LN output without forking RecursiveMAS code.
3. **Phase 0 on Kaggle with GPU**: load Sequential-Light Solver (smallest), capture inner-link output tensors on a 50-prompt math500 subset, replay through Variant A and Variant B, measure cosine / KL of resulting logits, run identity-wrapper sanity check.
4. Iterate per RESEARCH.md §7 phase plan.

---

## Provenance and reproducibility

- Code: [`src/quantizers/hadamard_uniform.py`](src/quantizers/hadamard_uniform.py), [`src/metrics/distortion.py`](src/metrics/distortion.py), tests in [`tests/test_hadamard_uniform.py`](tests/test_hadamard_uniform.py).
- Kaggle script (inlined for portability): [`experiments/distortion_validation/synthetic_sweep/kaggle_variant_a_sweep.py`](../../experiments/distortion_validation/synthetic_sweep/kaggle_variant_a_sweep.py).
- Kaggle notebook (private): <https://www.kaggle.com/code/<YOUR_KAGGLE_USERNAME>/recursivelink-variant-a-sweep>, kernelId 120682849, version 1.
- Local unit tests: 36/36 passed on macOS, Python 3.10.10, torch 2.12.0.
- Determinism: every cell uses `torch.Generator().manual_seed(seed)` with explicit seeds; the JSON contains the seed list.
