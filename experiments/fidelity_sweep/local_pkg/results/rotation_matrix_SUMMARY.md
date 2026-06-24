# MBPP+ quantizer-rotation matrix — is the light/scaled gap rotation-robust?

**Provenance.** RTX 5070 Ti (16 GB, bf16), RecursiveMAS @ f95d512, MBPP+, n=250,
generation seed 42, T=3, Variant B at all links. INT4 greedy captures at five quantizer
rotations **`quantizer_seed` ∈ {42, 7, 17, 73, 101}** for both tiers; the rotation-
independent greedy REF is the seed-42 capture, reused for every rotation (only INT4 was
swept). 8 new INT4 runs took ~14 h total (light ~57–72 min, scaled ~140–156 min) and were
produced by the resumable detached orchestrator
[`run_rotation_matrix.py`](../run_rotation_matrix.py).
Reproduce the analysis: `python ../analysis/rotation_matrix_analysis.py`.

## Per-rotation divergence (greedy REF vs INT4)

| quantizer_seed | light div/128 | scaled div/128 | light div/25 | scaled div/25 |
|---:|:---:|:---:|:---:|:---:|
| 42 (original) | 92.8% | 51.2% | 50.0% | 15.2% |
| 7 | 91.6% | 53.6% | 50.0% | 19.2% |
| 17 | 93.2% | 52.4% | 54.0% | 17.2% |
| 73 | 92.8% | 50.4% | 45.2% | 18.8% |
| 101 | 93.6% | 55.2% | 48.4% | 18.4% |
| **mean** | **92.8%** | **52.6%** | **49.5%** | **17.8%** |

`div/128` is divergence within the 128-position capture window; `div/25` is divergence
within the first 25 positions (the length-robust early estimand from
[`divergence_hazard_SUMMARY.md`](divergence_hazard_SUMMARY.md)). The light and scaled
ranges never overlap on either metric (light 91.6–93.6%, scaled 50.4–55.2% within 128).

## Light − scaled contrast (problem-clustered bootstrap)

Each problem's divergence propensity is averaged over the five rotations; problems (the
resampling unit) are resampled with replacement. Rotations are repeated measures on the
same 250 problems and are **not** pooled as independent benchmark items.

| estimand | light | scaled | Δ (light − scaled) | 95% CI |
|---|:---:|:---:|:---:|:---:|
| divergence within 128 | 92.8% | 52.6% | **+40.2 pp** | **[+34.9, +45.7]** |
| divergence within 25 | 49.5% | 17.8% | **+31.8 pp** | **[+25.9, +37.6]** |

## Findings

1. **The MBPP+ light/scaled trajectory-divergence gap is robust to the quantizer
   rotation.** Across five independent rotations the contrast is ~+40 pp (full window) and
   ~+32 pp (length-robust early window), with bootstrap CIs that exclude zero by a wide
   margin. It is not a single-rotation artifact.
2. **The between-rotation variance is tiny** (light 92.8% ± ~1, scaled 52.6% ± ~2), much
   smaller than the tier gap — the effect size dwarfs the rotation noise.
3. **This strengthens, but does not change, the interpretation.** It remains a
   *task-specific tier association*, not a causal capacity law: on Math500 the same
   light→scaled change barely moves divergence (`step4_scaled_math500/SUMMARY.md`), and the
   tiers differ in architecture, tokenizer, hidden size, output length, and token margins
   as well as parameter count. Rotation-robustness rules out the *quantizer rotation* as a
   confound; it does not isolate model capacity.

## Limitations

- A single **generation seed (42)** and a single problem subset/order; only the quantizer
  rotation was swept. The generation-seed and subset axes remain.
- The bootstrap CI is on divergence-rate contrasts (clean per-problem binaries). Early
  per-position hazard is reported descriptively per rotation (mean light ≈ 0.027, scaled
  ≈ 0.008, ~3.4× lower at every rotation).
- Still not a teacher-forced, position-aligned mechanism test — that (and a length/margin
  mediation analysis) is the remaining `ROADMAP.md` §1C/§2 work needed before a mechanistic
  claim.

Raw NPZ captures and machine logs are not committed; regenerate them with
`run_rotation_matrix.py` and analyse with `rotation_matrix_analysis.py`.
