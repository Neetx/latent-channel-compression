# Experiments inventory

Complete catalog of every experiment ever run. Status legend:

- ✅ **active** — result currently cited in REPORT_05/06/RESEARCH
- 🟡 **superseded** — earlier version of a working experiment; preserved for history
- ⚠️ **retracted** — numbers no longer trusted due to subsequently-discovered confound; preserved as evidence of the discovery process
- 🔵 **distortion-only** — accuracy claims retracted (broken-pipeline artifact) but per-link distortion measurements (rMSE, cosine, norm-ratio) remain valid and feed REPORT_02/03's TurboQuant theory match

---

## Phase 0 — single-component validation

### `experiments/distortion_validation/identity_check/` — Gate 0: identity wrapper on Solver inner adapter
- **Status:** ✅ active
- **Hardware:** Kaggle CPU + P100
- **Finding:** rMSE 2e-9, cosine 1.0 → patching infrastructure is bit-exact on real adapter
- **Cited in:** REPORT_03 §2

### _experiments/00_setup/ (archived; setup probes; details in docs/reports/03_capture_replay_solver.md)_ — Kaggle GPU compatibility probe
- **Status:** ✅ active (used as setup precondition)
- **Hardware:** Kaggle (whatever was assigned)
- **Finding:** Verified torch 2.4.1+cu121 supports P100 sm_60 (the Kaggle default at the time)

---

## Phase 0.B — in-loop quantization on Kaggle P100 (RETRACTED for accuracy, valid for distortion)

All Phase 0.B experiments share the same fatal confound: **Kaggle P100 lacks native bf16 hardware, and the released RecursiveMAS checkpoints default to bf16 (`config.json: "dtype": "bfloat16"`). PyTorch silently downconverts to fp16 on Pascal, collapsing the recursive latent rollouts to ~30-35% accuracy regardless of quantization (REPORT_05 §15).**

→ **All accuracy numbers in these directories are retracted.** They are NOT failure of Variant B; they are pipeline-collapse artifacts.
→ **Per-link rMSE/cosine measurements ARE valid** (single-call ops don't accumulate enough fp16 error to corrupt them) and feed REPORT_02/03 TurboQuant theory validation.

| Directory | Date | Setup | Reported accuracy | Truth |
|---|---|---|---|---|
| `phase0b_baseline/` | 2026-05-27 | smoke n=5, baseline | 60% (3/5) | retracted (P100 collapse) |
| `phase0b_systemload_probe/` | 2026-05-27 | system probe | n/a | ✅ active (verified system loads) |
| `phase0b_smoke_2b/` | 2026-05-27 | smoke n=5, Variant B 4-bit | 80% (4/5) | retracted |
| `phase0b_optB_8bit/` | 2026-05-27 | n=25 r=3 b=8 | 32% | retracted |
| `phase0b_optB_4bit/` | 2026-05-27 | n=25 r=3 b=4 | 32% | retracted |
| `phase0b_optB_3bit/` | 2026-05-27 | n=25 r=3 b=3 | 32% | retracted |
| `phase0b_optB_2bit/` | 2026-05-27 | n=25 r=3 b=2 | 24% | retracted |
| `phase0b_optB_baseline/` | 2026-05-27 | n=25 r=3 baseline | 40% | retracted |
| `phase0b_optB_r1_8bit/` | 2026-05-27 | n=25 r=1 b=8 | 40% | retracted |
| `phase0b_optB_r1_4bit/` | 2026-05-27 | n=25 r=1 b=4 | 40% | retracted |
| `phase0b_optB_r1_3bit/` | 2026-05-27 | n=25 r=1 b=3 | 44% | retracted |
| `phase0b_optB_r1_2bit/` | 2026-05-27 | n=25 r=1 b=2 | 24% | retracted |
| `phase0b_optB_r1_baseline/` | 2026-05-27 | n=25 r=1 baseline | 40% | retracted |
| `phase0b_optB_r2_baseline/` | 2026-05-27 | n=25 r=2 baseline | 40% | retracted |
| `phase0b_optB_n100_shuf_r1_baseline/` | 2026-05-27 | n=100 shuf | 31% | retracted |
| `phase0b_optB_n100_shuf_r1_4bit/` | 2026-05-27 | n=100 shuf Variant B 4-bit | 25% | retracted |
| `phase0b_optB_paper_n100_baseline/` | 2026-05-27 | n=100 paper-like | 34% | retracted |
| `phase0b_optB_paper_n100_4bit/` | 2026-05-27 | n=100 paper-like 4-bit | 30% | retracted |
| `phase0b_optB_greedy_n100_baseline/` | 2026-05-27 | n=100 greedy | 30% | retracted |
| `phase0b_optB_greedy_n100_4bit/` | 2026-05-27 | n=100 greedy 4-bit | 31% | retracted |
| `phase0b_optB_n500_full_r1_baseline/` | 2026-05-27 | n=500 baseline | 28.4% | retracted |
| `phase0b_optB_n500_full_r1_4bit/` | 2026-05-27 | n=500 Variant B 4-bit | 25.6% | retracted |

**Lessons preserved from Phase 0.B (positive learnings despite retracted accuracy):**
- Patching infrastructure `src/adapters/patch.py` works end-to-end on real adapters: ✅ proven by 16+ in-loop runs without exceptions
- Per-link rMSE 0.009 at 4-bit reproduces TurboQuant paper Table 1 to the third decimal in-loop: ✅ confirms quantizer correctness in a real pipeline
- bf16 default on non-bf16 hardware is the root cause of pipeline collapse: discovered via Phase 0.D/E/G/H bisection

---

## Phase 0.C — Solver-only diagnostic

### `experiments/solver_diagnostic/`
- **Status:** ✅ active
- **Hardware:** Kaggle P100
- **Finding:** **83% accuracy** on math500 first 100 problems with just the Solver checkpoint + chat template + greedy decoding. Confirms the released Solver checkpoint preserves Qwen2.5-Math-1.5B behavior, ruling out checkpoint corruption as a cause of the Phase 0.B 30% baseline.
- **4 kernel iterations:**
  - v1 — failed (transcript lost in compaction)
  - v2 — failed: torchvision::nms operator missing (no `MAS_FORCE_DISABLE_TORCHVISION` env var)
  - v3 — failed: `AttributeError: 'list' object has no attribute 'keys'` on tokenizer (transformers 4.50.0 bug with Solver's `extra_special_tokens`)
  - v4 — success (current `solver_only.py`)
- **Cited in:** REPORT_05 §1.1

---

## Phase 0.D — Upstream `run.py` pristine on Kaggle P100

### _experiments/03_pristine_baseline_p100_FAILED/ (archived; finding documented in docs/reports/05_hardware_root_cause.md §1.2)_
- **Status:** ✅ active
- **Hardware:** Kaggle P100 (sm_60, no bf16 hardware)
- **Finding:** **35% accuracy** on math500 n=100 b=8 with the official `python run.py ...` and zero modifications (except 2 surgical patches for num_samples & batch_size). Confirms our `baseline.py` wrapper is NOT the bug — the pipeline collapses identically.
- **Cited in:** REPORT_05 §1.2

---

## Phase 0.E — Modal A100 baseline reproduction

### `experiments/baseline_a100_modal/`
- **Status:** ✅ active
- **Hardware:** Modal A100 40GB (sm_80, native bf16)
- **Findings:**
  - Smoke n=5 b=4: 100% (5/5), 3.7 min, $0.14
  - Main n=50 b=32: **84%** (42/50), 5.4 min, $0.20 — **PAPER REPRODUCED (paper 75.8% within n=50 sample variance)**
  - Ablation n=50 b=8: **86%** (43/50), 7.3 min, $0.24 — batch_size irrelevant
- **Cited in:** REPORT_05 §1.3

---

## Phase 0.F — Modal A100 Variant B 4-bit (REINTERPRETED)

### _experiments/05_variant_b_modal_a100_dtype_artifact/ (archived; finding documented in docs/reports/05_hardware_root_cause.md §6 + docs/reports/06_variant_b_in_loop_HEADLINE.md §2.3)_
- **Status:** 🟡 superseded by Phase 0.I (REPORT_06); reinterpretation in REPORT_05 §6
- **Hardware:** Modal A100 40GB
- **Original finding:** 66.67% (n=30) with Variant B 4-bit injected — initially interpreted as −19pp catastrophic depth-amplification of quantization error
- **Current interpretation:** the −19pp was caused by **bf16/fp32 boundary cast amplification at the quantizer call sites**, NOT by Variant B itself. Phase 0.I on T4 fp32 with the same VB=4 gives only −4pp. The 14pp difference is the cast contribution.
- **4 kernel iterations:**
  - v1 — IndexError on `Path(__file__).parents[2]` (resolved by relative traversal)
  - v2 — 0 patches applied (sitecustomize.py via PYTHONPATH didn't load in subprocess; replaced with in-place source patch)
  - v3 — src.utils import path error (Modal mounted at `/opt/lqc/` instead of `/opt/lqc/src/`)
  - v4 — final, ran successfully but with ambiguous stdout markers (resolved offline via mock execution proving patches DO fire)
- **Cited in:** REPORT_05 §6, REPORT_06 §2.3, RESEARCH.md §12.5.D

---

## Phase 0.G + 0.H — Kaggle T4 dtype ablation

### `experiments/variant_b_ladder_t4_kaggle/` (same kernel script reused for both phases)
- **Status:** ✅ active (both phases produced canonical findings)
- **Hardware:** Kaggle Tesla T4 (sm_75, no native bf16)

**Phase 0.G — baseline T4 with `--dtype auto` (= bf16 fallback to fp16):**
- n=50 b=8 default: **30%** ❌ → confirms bf16-on-non-bf16-HW collapses pipeline
- **Cited in:** REPORT_05 §15.3

**Phase 0.H — baseline T4 with `--dtype float32` explicit (no fallback):**
- n=50 b=4 (b=4 for fp32 memory safety): **84%** ✅ → workaround restores accuracy
- Same kernel slug: `rmas-phase0h-t4-fp32-ablation`
- **Cited in:** REPORT_05 §15.3, REPORT_06 §1

---

## Phase 0.I — Variant B bit-rate ladder on T4 fp32 (n=50 exploration)

### `experiments/variant_b_ladder_t4_kaggle/` (same kernel folder, multiple bit-rate slugs)
- **Status:** 🟡 superseded by Phase 0.J n=250 canonical numbers
- **Hardware:** Kaggle T4 sm_75, fp32 forced, b=4
- **Setup:** n=50, seed=42, sampled (temp=0.6 top_p=0.95), num_recursive_rounds=3, latent_length=48, num_rollouts=1
- **Kernel slugs (4 runs):**
  - `rmas-phase0i-vb8` — 86.0% (n=50)
  - `rmas-phase0i-vb6` — 82.0% (n=50)
  - `rmas-phase0i-vb4` — 80.0% (n=50)
  - `rmas-phase0i-vb2` — 74.0% (n=50)
- **(Cancelled:** `rmas-phase0i-vb3` — pushed accidentally during pipeline iteration; manually stopped via web UI)
- **Why superseded:** n=50 SE ≈ 5pp produced a misleading monotonic decline. The Phase 0.J n=250 numbers show the n=50 baseline (84%) was a lucky subset; true baseline ≈ 75% and the bit-rate curve is FLAT.
- **Cited in:** REPORT_06 §2.3 (cautionary discussion of how small-n bias misled us)

## Phase 0.J — Variant B at n=250 (CANONICAL HEADLINE FINDING)

### `experiments/variant_b_ladder_t4_kaggle/` (same kernel folder, different bit-rate + N_SAMPLES)
- **Status:** ✅ active — **CANONICAL TABLE 1 of the paper**
- **Hardware:** Kaggle T4 sm_75, fp32 forced, b=4
- **Setup:** n=250, seed=42, sampled, num_recursive_rounds=3, latent_length=48, num_rollouts=1
- **Kernel slugs (4 runs):**
  - `rmas-phase0j-baseline-n250-b4` — **75.2%** (188/250) — true baseline
  - `rmas-phase0j-8-n250-b4` — **78.4%** (196/250) — +3.2pp, p>0.4 (lossless)
  - `rmas-phase0j-vb4-n250-b4` — **76.8%** (192/250) — +1.6pp, p>0.5 (lossless)
  - `rmas-phase0j-vb2-n250-b4` — **75.2%** (188/250) — 0.0pp (identical to baseline!)
- **(Earlier crashed attempts at b=8:** `rmas-phase0j-vb4-n250-b8` and `rmas-phase0j-vb2-n250-b8` — both OOMed at b=8 fp32 on T4 16GB. Dropped to b=4 for the working runs.)
- **Cited in:** REPORT_06 §2.1 (canonical bit-rate table), RESEARCH.md §12.5.C (headline finding)

**Headline:** TurboQuant Variant B compresses the RecursiveMAS Sequential-Light inter-agent latent channel **4× to 16× with no measurable accuracy change under sampled decoding**. All n=250 sampled measurements are statistically indistinguishable from baseline (two-proportion z-tests, all p > 0.4); 2-bit = baseline problem-for-problem. (Greedy ±2pp TOST is inconclusive — see REPORT_07; "drop-in lossless" is a sampled-decoding statement.)

---

## Phase 1 — bit-rate sweep (synthetic, deprecated)

### `experiments/distortion_validation/synthetic_sweep/`
- **Status:** 🟡 superseded by Phase 0.I (in-loop sweep replaces the synthetic sweep target)
- **Hardware:** local CPU
- Note: original Phase 1 was a stand-alone synthetic-Gaussian sweep that fed Variant A/B distortion validation. Numbers in REPORT.md / REPORT_02.md.

---

## Datasets created on Kaggle

| Dataset | Status | Purpose | Notes |
|---|---|---|---|
| `<YOUR_KAGGLE_USERNAME>/lqc-src` | ❌ deleted 2026-05-31 | First Variant B src/ bundle | Built with `--dir-mode zip`; Kaggle stripped the `src/` wrapper → kernel imports failed. Replaced. |
| `<YOUR_KAGGLE_USERNAME>/lqc-src-v2` | ❌ deleted 2026-05-31 | Re-attempt, same problem | Identical structure issue. Replaced. |
| `<YOUR_KAGGLE_USERNAME>/lqc-src-bundle` | ✅ active | Current Variant B src/ bundle | Manually built `lqc_src.tar.gz` with `tar czf` (preserves `src/` wrapper) + `--dir-mode skip` (Kaggle uploads as-is). Auto-detected at `/kaggle/input/datasets/.../lqc-src-bundle/src/...` by kernel's walk-based discovery. |

---

## Modal volumes

| Volume | Purpose |
|---|---|
| `rmas-hf-cache` | HF model cache (Sequential-Light planner+critic+solver+outerlinks, 9 GB). Reused across Phase 0.E/0.F AND the fidelity sweep — no re-download. |
| `rmas-phase0e-out` | JSON outputs for Phase 0.E + 0.F runs. |
| `rmas-fidelity-out` | Tier 2 fidelity sweep outputs. Per-config subdirs `vb{bits}_T{T}/` each hold `fidelity_*.json` + `fidelity_logits.npz` + `fidelity_call_stats.json`. |

---

## Tier 2 fidelity sweep — Modal A100 fp32 (2026-06-02)

Driver `experiments/fidelity_sweep/modal_pkg/fidelity_modal.py` reuses the Kaggle
kernel's tested patch functions (single source of truth). fp32 on A100 avoids both the
bf16 collapse and the Phase 0.F cast artifact. See [REPORT_07](../reports/07_fidelity_sweep_modal.md).

| Run | config | result | cost | notes |
|---|---|---|---|---|
| dry-run | INT4 T=3 n=8 b=4 | rc=0, 16 patches, 8/8 correctness, 2 logit batches, cos 0.995/rMSE 0.009 | ~$0.22 | validated full path (also caught `is_local`/`scipy`) |
| REF (early) | bits=0 T=3 n=30 b=8 | 80.0% (24/30) | ~$0.35 | hit the then-$1/day cap; superseded by the n=50 sweep |
| **full sweep** | **{REF,INT4} × T∈{1,2,3,4}, n=50 b=8** | **8/8 rc=0** — see table below + [REPORT_07](../reports/07_fidelity_sweep_modal.md) | **~$4–5** | concurrent A100s after budget raised to ~$28.85 |

Full-sweep headline (n=50, greedy, fp32):
- **Channel fidelity flat across depth**: cos 0.9954, rMSE ≈ 0.0093 at every T=1→4.
- **Per-step egress KL small + non-amplifying** (matched-prefix): 0.03–0.6 nats, JS < 0.04.
- **Accuracy TOST underpowered at n=50** (Δacc within ±6–8pp noise; CIs ≫ ±2pp) → needs n≈250.
- Trajectory `div_rate` 0.72–0.92: 4-bit flips a greedy token in most sequences over ~70–120 tokens (accuracy still holds).

Pre-validation failures (~$0.01) fixed: remote `parents[3]` crash → `modal.is_local()`
guard; missing `scipy` in `debian_slim`.

---

## Local cross-cell extension — RTX 5070 Ti bf16 (2026-06-19 to 2026-06-21)

The portable `experiments/fidelity_sweep/local_pkg/` backend completed four n=250,
seed-42, T=3 cells: light x {math500, MBPP+, MedQA} and scaled x MBPP+.
Sampled ladders are non-monotonic with no detected rate-dependent decline. The three
clean paired-greedy math/code deltas are +2.0, 0.0, and -2.0 pp with 4.4--10.0%
answer churn. MedQA greedy is confounded by a REF first-option bias.

Corrected Tier-2 analysis pairs only primary calls, excludes conditional retries,
uses local top-K=256 and a 128-position window, and corrects residual-tail mass.
Windowed divergence is 86.4% math/light, 92.8% MBPP+/light, 51.2% MBPP+/scaled,
and 96.4% MedQA/light. See [REPORT_08](../reports/08_local_cross_cell_generalization.md).

Raw NPZs and verbose logs remain outside git. Public JSONL artifacts are minimized
to fields needed for paired answer analysis.

---

## What we have NOT done yet (open work)

Open work (larger systems, more seeds, a QJL-residual ablation, teacher-forced per-step fidelity) is summarized in the write-up's Discussion and Limitations.

Briefly: multi-seed light/scaled replication, scaled Math500, teacher-forced
position-aligned fidelity, QJL residual ablation, additional topologies/tasks, and
real packed-transport latency/byte measurements.
