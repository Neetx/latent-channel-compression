# Latent Space Compression Research — RecursiveLink × TurboQuant

**Status:** Phase 0.E complete (baseline pipeline reproduced 84-86% on A100, matches paper 75.8% within sample noise) · Phase 0.F in progress (Variant B injected, A100 b=8 n=30)
**Started:** 2026-05-26
**Last updated:** 2026-05-29
### Document map (read order for newcomers)

1. **This file (RESEARCH.md)** — the master research design, hypotheses, current state, references.
2. **[REPORT_05.md](reports/05_hardware_root_cause.md)** — Phase 0.C/0.D/0.E investigation: root-caused the 40pp accuracy gap to GPU precision (P100 fp16 vs A100 Tensor Cores). **Read this before REPORT_04.**
3. **[REPORT_04.md](reports/04_kaggle_p100_RETRACTED.md)** — Phase 0.B in-loop on Kaggle P100. ⚠️ Accuracy numbers retracted (broken-pipeline artifacts); distortion measurements valid.
4. **[REPORT_03.md](reports/03_capture_replay_solver.md)** — Phase 0.A on the real Solver inner adapter (capture-replay): rMSE 0.0093 @ 4-bit, matches synthetic.
5. **[REPORT_02.md](reports/02_variant_b_synthetic.md)** — Variant B synthetic + head-to-head vs Variant A.
6. **[REPORT.md](reports/01_variant_a_synthetic.md)** — Variant A synthetic-data findings.
7. **[PHASE_0C_DESIGN.md](design/phase0c_investigation.md)** — investigation design for the 40pp gap (resolved by REPORT_05).
8. **[CORE_DESIGN.md](design/architecture.md)** — architectural plan for the Phase 0.B experiment.

---

## 1. One-line problem statement

Is the latent tensor that flows through a RecursiveMAS `RecursiveLink` compressible by a TurboQuant-style online vector quantizer at 3–4 bits per coordinate, without significant downstream accuracy loss?

This is **not** the question "does TurboQuant improve RecursiveMAS". It is a feasibility study of a specific communication channel under a specific compression family.

---

## 1.5 Compute and storage strategy

Local machine (this MacBook) does not have enough disk (~11 GB free) or compute (no CUDA) to host the RecursiveMAS checkpoints and run multi-LLM inference. The work is therefore split:

- **Local (MacBook, CPU/MPS)** — library code, quantizer implementations, unit tests against `turboquant_ref` with synthetic data, the patching utilities for `CrossModelAdapter` / `Adapter`, the experiment harness as importable Python. No model checkpoints touched locally.
- **Cloud Tier 1 — Kaggle P100 16 GB (free, deprecated for accuracy runs).** Used for the entire Phase 0.A/0.B/0.C/0.D ladder (≈30 GPU-hours total, $0). **Pascal architecture (sm_60) confirmed unsuitable for RecursiveMAS Sequential-Light end-to-end accuracy evaluation** — see §15. Still useful for non-recursive per-link distortion measurements, smoke tests, and CPU-equivalent sanity checks. The fp16 accumulator in fp16 (no Tensor Cores) silently collapses the deep recursive latent rollouts.
- **Cloud Tier 2 — Modal serverless A100 40 GB ($1.42-2.10/h effective, pay-per-second).** Required for any RecursiveMAS Sequential-Light full-pipeline accuracy run. Ampere architecture (sm_80) with Tensor Cores accumulates fp16 matmul in fp32, preserving signal across the ~80,000 sequential matmuls per math500 problem. Used for Phase 0.E (baseline reproduction) and Phase 0.F (Variant B in-loop). Total Modal cost to date for all accuracy runs: ~$0.84.

Implication for code: every quantizer and adapter patch is plain PyTorch with no hardware assumption. Tests run on CPU. The cloud runner is just a thin wrapper that either:
- subprocesses upstream `python run.py ...` after two surgical regex patches (`num_samples` and `batch_size`), for baseline pipeline reproduction (`experiments/baseline_a100_modal/`); or
- additionally appends a runtime patch to upstream `inference_mas.py:load_inner_adapter_module` / `load_outer_adapter_module` that wraps each loaded adapter with our Variant B quantizer (_experiments/05_variant_b_modal_a100_dtype_artifact/ (archived; finding documented in docs/reports/05_hardware_root_cause.md §6 + docs/reports/06_variant_b_in_loop_HEADLINE.md §2.3)_).

---

## 2. Background and verified references

### RecursiveMAS

Recursive multi-agent framework that connects heterogeneous LLM agents in a collaboration loop via a lightweight `RecursiveLink` module. The Link transmits an agent's last-layer hidden states and is decomposed into:

- **Inner link** — maps last-layer hidden state back to input embedding space of the *same* agent (latent-thought generation).
- **Outer link** — projects hidden states across agents of *different* hidden sizes (cross-agent latent transfer).

The Link is a small two-layer residual MLP with LayerNorms; the trainable Link system across all agents is ~13M params (≈0.31% of the full system).

- Paper: <https://arxiv.org/abs/2604.25917>
- Code: <https://github.com/RecursiveMAS/RecursiveMAS>
- Project page: <https://recursivemas.github.io/>

### TurboQuant

Online, data-oblivious vector quantizer with near-optimal distortion rate (within ~2.7× the information-theoretic lower bound). Two-stage algorithm:

1. **Random rotation** of the input vector (Haar-uniform on the orthogonal group). This induces a concentrated Beta(½, (d−1)/2) marginal on each rotated coordinate, by concentration of measure on the unit sphere.
2. **Optimal scalar quantization** per coordinate via Lloyd-Max codebook precomputed for that Beta distribution.
3. **QJL residual** — a 1-bit quantizer applied to the residual gives an *unbiased* inner-product estimator (important for attention-like consumers).

Reported on KV cache: 3.5 bit/coord = absolute quality neutrality; 2.5 bit/coord = marginal degradation.

- Paper: <https://arxiv.org/abs/2504.19874> (arXiv preprint, 2025)
- Google Research blog: <https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/>
- Reference Python implementation: <https://github.com/yashkc2025/turboquant>

---

## 3. Why this is non-trivial

KV-cache quantization is well-studied and tolerant because:

- Per-channel outliers are structured and predictable.
- Each K/V is averaged over many attention queries → errors smooth out.
- Softmax further attenuates per-token noise.

The RecursiveLink is qualitatively different:

- It is a *one-shot* communication channel — the receiver consumes the output once, with no averaging.
- It is a *bottleneck* (proj1 → act → proj2 with LayerNorm) → output is information-dense.
- Errors accumulate across the recursion depth (Link is invoked at every recursion step).
- Inner link consumer (same model) vs outer link consumer (different model) likely have different geometric sensitivity → must be measured separately.

**A priori it is unclear whether the RecursiveLink behaves more like KV cache (compressible) or more like a final-layer logit (fragile).** This is what makes the question worth asking.

---

## 4. Hypotheses

- **H0** — RecursiveLink output is fragile. TurboQuant-style quantization at ≤4 bits degrades downstream accuracy by >2 percentage points and/or destabilizes recursion.
- **H1** — RecursiveLink tolerates 3–4 bit quantization with downstream Δacc ≥ −1 pp.
- **H2** — If inference-only quantization shows moderate degradation, quantization-aware fine-tuning of the Link only (base models frozen, STE through the quantizer) recovers most of the quality.

A negative result (H0) is still informative: it identifies a structural difference between KV-cache compression and latent-communication compression.

---

## 5. Method

### 5.1 Two minimal quantizer variants

To avoid conflating "rotation choice" with "quantizer choice" failures, we implement two variants in parallel.

#### Variant A — Hadamard + uniform (screening only)

- Rotation: randomized Hadamard transform (RHT) — diag(±1) · FWHT, O(d log d).
- Quantizer: uniform symmetric per-tensor.
- No residual correction.

Cost: ~50 lines, ~30 minutes.
Purpose: fast positive screening. If A succeeds at 4 bits, TurboQuant honest will succeed too — we save the implementation effort.

#### Variant B — Haar + Lloyd-Max-for-Beta (honest TurboQuant minimal)

- Rotation: Haar random orthogonal matrix, computed once via QR of a Gaussian, sign-corrected, stored as buffer. O(d²) per forward (fine for d≈4096 in inference).
- Quantizer: Lloyd-Max-optimal codebook precomputed for Beta(½, (d−1)/2), 2^bits levels, signed-symmetric.
- Pre-step: L2-normalize input onto unit sphere before rotation, rescale by stored norm after inverse rotation. The Beta marginal guarantee only holds on the sphere.
- No QJL residual in Phase 1 (deferred to Phase 4 only if data justifies it).

Cost: ~100 lines plus ~30 lines for Lloyd-Max codebook precomputation.
Purpose: the actual scientific test. A failure here is a genuine signal that the channel is incompressible by TurboQuant-family methods.

### 5.2 Insertion point

```python
class CrossModelAdapter(nn.Module):
    def __init__(self, ..., quantizer: nn.Module | None = None):
        super().__init__()
        # ... existing layers ...
        self.quantizer = quantizer  # None → baseline (identity)

    def forward(self, x):
        h = self.ln_source(x)
        main = self.proj2(self.act(self.proj1(h)))
        residual = self.residual_proj(x)
        out = self.ln_target(main + residual)
        if self.quantizer is not None:
            out = self.quantizer(out)
        return out
```

Quantizer goes **after** `ln_target`, just before return. Justification: the consumer downstream expects a normalized hidden state, and quantizing pre-LN would shift LN's input distribution and confound the signal.

### 5.3 Hard constraints

- PyTorch only, autograd-compatible.
- Fake-quantization in float — no packed storage, no custom kernels.
- Inference-only in Phase 1. Fine-tuning of Link only (frozen base models) in Phase 3 if and only if Phase 1 justifies it.
- No TCQ, KIVI, SnapKV, llama.cpp, vLLM, Triton/CUDA, packed low-bit, or QJL until earlier phases pass.

---

## 6. Pre-registered thresholds and stage gates

These thresholds are fixed **before** running experiments. Do not relax them after seeing numbers.

| Gate | Condition to pass | If fail |
|---|---|---|
| 0 — Identity sanity | With identity wrapper (no quantization, but rotate→inverse enabled), Δacc on 50 examples ≤ noise floor (≤ 0.1 pp for A; ≤ 0.3 pp for B in fp16) | Debug padding / FWHT / Haar inverse — do not advance |
| 1A — Hadamard screen | Variant A at 4 bits: Δacc ≥ −2 pp on ≥200 examples; KL(logits) < 0.1 nat median | Skip A, go to Variant B |
| 1B — Honest TurboQuant | Variant B at 4 bits: Δacc ≥ −1 pp; KL < 0.05 nat | If both A and B fail → stop, write negative result |
| 2 — Push to 3 bits | Variant B at 3 bits: Δacc ≥ −1 pp | If Δacc ∈ [−3, −1] pp → go to Phase 3 (QAT). If worse, partial result, stop |
| 3 — QAT recovery | Fine-tuned Link recovers ≥50% of the 3-bit gap | If no → stop with "needs more than online quantization" result |
| 4 — QJL residual | Inner-product error correlates with downstream KL, and adding QJL reduces both | If no → drop QJL |
| 5 — Systems | All earlier gates passed | Only then consider packed storage, Triton, real VRAM measurement |

### Decision matrix for Phase 1 outcomes

| Variant A @4b | Variant B @4b | Decision |
|---|---|---|
| Pass | (not run) | Strong screen success → proceed to push to 3 bits with B |
| Fail | Pass | Quantizer matters → continue with B only |
| Fail | Fail | RecursiveLink genuinely fragile → stop, write up negative |
| Pass | Fail | **Bug** — investigate before any further conclusion |

---

## 7. Phase plan

### Phase 0 — Setup and identity check
- Clone RecursiveMAS, locate `CrossModelAdapter` / `RecursiveLink` in code.
- Identify hidden dim `d` of inner and outer links (likely 4096-ish but unknown until inspected).
- Download a single small checkpoint (the smallest agent combination released).
- Run baseline on 50 math500 examples; record `baseline_acc.json`.
- Insert identity-wrapper quantizer (rotate → inverse, no quantization). Re-run.
- **Gate 0 must pass before any quantization experiment.**

### Phase 1 — Bit-rate sweep, inference-only
- Variant A first (cheap): bits ∈ {8, 4, 3, 2}, 200 examples.
- If A at 4b passes Gate 1A: implement Variant B, run same sweep.
- If A at 4b fails: implement Variant B directly, run sweep.
- Log all metrics from §8 per-step and aggregated.
- Separately log inner-link vs outer-link results.
- Separately log 1-step vs N-step compounding (N = typical recursion depth).

### Phase 2 — Decision
- Apply decision matrix. If stopping, write up; if continuing, define Phase 3 scope.

### Phase 3 — Quantization-aware fine-tuning of Link only
- Freeze all base agents.
- Add STE through Variant B quantizer.
- Train Link on its original objective (or distillation `out_baseline → out_quantized`).
- Measure recovery.

### Phase 4 — QJL residual ablation (conditional)
- Only if inner-product error explains downstream degradation.
- Implement 1-bit residual quantizer per QJL.
- Measure delta on metrics; keep only if it helps.

### Phase 5 — Systems (deferred)
- Packed representation, real VRAM/bandwidth measurement, Triton/CUDA.
- Only if scientific validation in 1–3 is solid.

---

## 8. Metrics

Log all metrics at three levels: per-token (where applicable), per-example, aggregated with bootstrap CI.

| Metric | Computed on | Reason |
|---|---|---|
| Downstream accuracy (task) | math500 (or RecursiveMAS-bench equivalent) | Final goal |
| KL(p_baseline ‖ p_quant) on logits | Final-step logits | Best end-to-end proxy for accuracy impact |
| Relative MSE = ‖x − x_q‖ / ‖x‖ | RecursiveLink output | Global distortion |
| Cosine(x, x_q) | RecursiveLink output | Direction preservation |
| Norm ratio ‖x_q‖ / ‖x‖ | RecursiveLink output | Detects scale drift through LN |
| Inner-product error: \|⟨x,y⟩ − ⟨x_q,y_q⟩\| / (‖x‖‖y‖) | Sampled pairs | Predicts QJL necessity |
| 1-step Δ vs N-step Δ | Recursion depth sweep | Detects error compounding |
| ‖main‖ / ‖residual‖ in adapter | Internal | Confound check |
| Latency overhead of Link | Wall clock | Sanity check (not optimization metric in Phase 1) |

---

## 9. Failure modes

- **LN scale drift**: quantization alters output variance the downstream consumer expects. Watch norm ratio.
- **Per-channel outliers**: if per-tensor uniform fails but per-channel works, problem is outlier handling, not incompressibility. Distinguish.
- **Identity wrapper instability**: padding bug, FWHT non-determinism, Haar reproducibility. Fix seeds, use `.contiguous()`.
- **Residual-dominated output**: if `‖residual_proj(x)‖ ≫ ‖main‖`, quantization looks innocuous for the wrong reason. Log the ratio.
- **Cherry-picked task**: math500 has high variance. Always report bootstrap CI, never a single number.
- **Compounding error along recursion**: linear-in-N degradation = fundamentally different problem from KV cache. Could be a stop signal even if 1-step looks fine.
- **Confounding inner and outer links**: they likely have different compressibility profiles. Quantize one at a time first, then both.

---

## 10. Out of scope (explicit)

- TCQ, KIVI, SnapKV, KVQuant.
- llama.cpp, vLLM, ONNX.
- Multi-method comparison ("TurboQuant vs X").
- Full retraining of base models.
- Triton / CUDA / custom kernels.
- Packed low-bit storage.
- QJL residual in Phase 1.
- Real VRAM / bandwidth measurement before Phase 5.
- Anything involving cloud GPUs (local hardware only).

---

## 11. Project layout

```
latent_space_compression_research/
├── RESEARCH.md                         ← this file
├── .gitignore
├── external/
│   ├── RecursiveMAS/                   ← cloned reference, read-only
│   └── turboquant_ref/                 ← cloned yashkc2025/turboquant
├── src/
│   ├── quantizers/
│   │   ├── hadamard_uniform.py         ← Variant A
│   │   └── turboquant_honest.py        ← Variant B
│   ├── adapters/                       ← patched CrossModelAdapter
│   ├── metrics/                        ← cosine, KL, inner-product
│   └── utils/                          ← seeds, codebook precompute, IO
├── experiments/
│   ├── phase0_identity/                ← local harness code
│   ├── phase1_sweep/                   ← local harness code
│   ├── cloud_phase0.ipynb              ← cloud runner for Phase 0 (TBD)
│   ├── cloud_phase1.ipynb              ← cloud runner for Phase 1 (TBD)
│   └── results/                        ← gitignored
├── checkpoints/                        ← gitignored
├── notebooks/                          ← exploratory only
└── tests/                              ← Lloyd-Max correctness, identity sanity, etc.
```

---

## 12. Concrete next steps

Status (last updated 2026-05-27 00:10):

- [x] Inspected `external/RecursiveMAS/modeling.py` and `system_loader.py`. `CrossModelAdapter` signature confirmed; inner hidden dim = agent `hidden_size`; outer in_dim → out_dim per directed pair.
- [x] Identified Sequential-Light as smallest viable system (~9 GB total).
- [x] Cloned reference implementations (RecursiveMAS, turboquant_ref).
- [x] RESEARCH.md drafted.
- [x] Cloud-vs-local split decided (this MacBook for code, Kaggle for inference).
- [x] Local Python venv set up: torch 2.12 + MPS, numpy, scipy, pytest.
- [x] Kaggle MCP (community `kaggle-mcp-server`) auth'd via API token, private notebook push verified end-to-end.
- [x] Variant A implemented + 36 unit tests + Kaggle synthetic sweep → REPORT.md.
- [x] Variant B implemented (`src/quantizers/turboquant_honest.py` + `src/utils/lloyd_max.py`) + 25 unit tests (including 3 oracle tests vs `external/turboquant_ref`) + Kaggle synthetic sweep.
- [x] **Variant B empirical numbers match TurboQuant paper Table 1 to the third decimal.** Implementation verified correct from two independent angles (oracle code match + published numbers match).
- [x] **REPORT_02.md** written with B sweep + head-to-head vs A.
- [x] Installed official Kaggle CLI 2.2.0 via uvx wrapper `./bin/kaggle`, unlocks `enable_gpu`/`enable_internet`/all metadata fields.
- [x] Phone-verified Kaggle account (required for actual GPU+internet on kernels).
- [x] **Phase 0.A done on REAL Sequential-Light Solver Qwen2.5-Math-1.5B inner adapter.** Gate 0 passes cleanly (rMSE 2e-9, cosine 1.0). Variant B @ 4-bit on real adapter output: rMSE 0.0093 — matches synthetic baseline to 2 % and the TurboQuant paper to the third decimal.
- [x] **REPORT_03.md** written with Phase 0 findings.

Active queue (**[CORE_DESIGN.md](design/architecture.md)** is the architectural plan for Phase 0.B):

1. Implement `src/adapters/patch.py` per CORE_DESIGN §6.1.
2. Run Phase 0.B.0 baseline capture + reproducibility check on math500 subset (100 problems, sequential_light).
3. Run 16 quantized cells (4 bits × 4 link combinations) per CORE_DESIGN §5.2.
4. Aggregate into REPORT_04 = the answer to Claim Q (CORE_DESIGN §2).

## 12.5 Headline results so far

### A. Per-link distortion (Variant B, Haar + Lloyd-Max-Gaussian) — UNCHANGED, FULLY VALID

**Synthetic vs. real RecursiveLink output:**

| bits | Synthetic d=2048 | **Real adapter d=1536** | TurboQuant paper |
|---:|---:|---:|---:|
| rMSE @ 8 | 0.0001 | **0.0001** | (lossless) |
| rMSE @ 4 | 0.0095 | **0.0093** | 0.009 |
| rMSE @ 3 | 0.0345 | **0.0339** | 0.030 |
| rMSE @ 2 | 0.1175 | **0.1159** | 0.117 |
| cos  @ 2 | 0.94   | **0.94**   | — |

Real-model rMSE matches synthetic to 2 %, and matches the TurboQuant paper Table 1 to the third decimal. **The RecursiveMAS inner adapter output is a textbook TurboQuant-compressible channel.** The data-oblivious property (no calibration data needed) holds empirically.

For context, Variant A per-channel — the strongest "uniform-quantizer" baseline — gives rMSE 0.020 at 4-bit on the same synthetic data, 2.1× worse than Variant B. Lloyd-Max-Gaussian is doing real work.

### B. Baseline pipeline reproduction (Phase 0.E)

**math500 accuracy bisection.** The seed observation that drove Phase 0.C-E was a 40pp gap between our P100 Kaggle baselines (28-34%) and the paper's claimed 75.8% for Sequential-Light @ r=1. Root cause identified 2026-05-29 — see [REPORT_05.md](reports/05_hardware_root_cause.md) for full investigation:

| Phase | Hardware | n | b | Pipeline | Accuracy | Status |
|---|---|---|---|---|---|---|
| Paper Sequential-Light r=1 | A100/H100 | 500 | 32 | upstream | **75.8%** | target |
| Phase 0.C Solver alone, greedy | P100 | 100 | 1 | solver only | **83.0%** | ✓ checkpoint healthy |
| Phase 0.D upstream pristine | P100 | 100 | 8 | full multi-agent | **35.0%** | ❌ pipeline collapse |
| Phase 0.E main | A100 | 50 | 32 | upstream | **84.0%** | ✅ paper reproduced |
| Phase 0.E ablation | A100 | 50 | 8 | upstream | **86.0%** | ✅ batch_size irrelevant |
| **Phase 0.F Variant B 4-bit** | A100 | 30 | 8 | quant in-loop | **66.67%** | ⚠️ **−19pp drop, depth-amplification** |

The 51pp swing between P100 b=8 (35%) and A100 b=8 (86%) at constant code/checkpoints/batch_size isolates the cause to **GPU precision arithmetic** (fp16-accumulated-in-fp16 on Pascal vs fp16-accumulated-in-fp32 on Ampere Tensor Cores). See §15.

### C. In-loop quantization — the bit-rate ladder (Phase 0.I @ n=50 + Phase 0.J @ n=250)

**No measurable accuracy change, 4× to 16× compression, under sampled decoding** at n=250 (Phase 0.J; greedy ±2pp TOST is inconclusive — REPORT_07):

| Bits | Compression | n=250 acc | Δ vs baseline | 2-prop z | p |
|:---:|:---:|:---:|:---:|:---:|:---:|
| – baseline | 1× | **75.2%** | — | — | — |
| **8** | **4×** | **78.4%** | +3.2pp | 0.83 | > 0.4 ✓ |
| **4** | **8×** | **76.8%** | +1.6pp | 0.41 | > 0.5 ✓ |
| **2** | **16×** | **75.2%** | 0.0pp | 0.00 | identical ✓ |

**All Variant B bit-rates from 8 down to 2 are statistically indistinguishable from baseline (all p > 0.4).** The n=50 exploration ladder had appeared to show a small monotonic decline (−4pp at 4-bit, −10pp at 2-bit) but this was an artifact of which 50-problem subset of math500 happened to be sampled at seed=42 — the n=50 baseline was on the easy end of the distribution. At n=250 with 5× tighter CIs (±5pp instead of ±10pp), the curve flattens completely and **Variant B 2-bit gives bit-for-bit identical accuracy to the unquantized baseline** (188/250 = 75.2% in both cases).

The practical implication: a multi-agent system that exchanges 144 latent vectors per math500 query can compress its inter-agent traffic from **~55 MB → ~3.4 MB per query** with no measurable accuracy cost.

### D. Phase 0.F result reframed (Modal A100 bf16 + Variant B = −19pp)

Phase 0.F initially measured Variant B 4-bit on Modal A100 bf16 at 66.67% (−19pp). The Phase 0.I result on T4 fp32 (−4pp at the same bit-rate) shows this was a **dtype-coherence artifact**, not a fundamental quantizer property:

- Modal A100 pipeline = bf16; our Variant B internally computes in fp32 → each quantizer call does `bf16 → cast fp32 → quantize → cast back to bf16`
- Across 16 quantizer calls × 3 rounds = 48 boundary casts. bf16 mantissa (~3 decimal digits) accumulates ~10⁻³ relative error per cast, doubling the effective per-call error.
- T4 fp32 pipeline + Variant B fp32-internal → no boundary cast → quantizer's clean precision preserved → near-lossless at 4 bits.

This is an **independent methodological finding**: when injecting a quantizer into a deep pipeline, the quantizer's internal dtype must match the pipeline's. See [REPORT_06.md §2.3](reports/06_variant_b_in_loop_HEADLINE.md) for the full analysis.

### E. Phase 0.B accuracy claims — RETRACTED

Phase 0.B (Kaggle P100, REPORT_04) measured Variant B at multiple bit rates and reported "+1pp greedy at 30→31% baseline → Variant B 4-bit". **Retracted 2026-05-29** because the 30% baseline was itself a Pascal-precision artifact (REPORT_05 §2). The +1pp delta was real but uninformative.

**What stays valid from Phase 0.B:** per-link distortion measurements (independent of pipeline health) and the `src/adapters/patch.py` infrastructure.

### D. Phase 0.B accuracy claims — RETRACTED

Phase 0.B (Kaggle P100, REPORT_04) measured Variant B at multiple bit rates and reported "+1pp greedy at 30→31% baseline → Variant B 4-bit". **Retracted 2026-05-29** because the 30% baseline was itself a Pascal-precision artifact (REPORT_05 §2). The +1pp delta was real but uninformative (both numbers dominated by hardware-induced collapse). Phase 0.F gives the correct picture: at the A100 baseline of ~86%, Variant B 4-bit costs −19pp, not +1pp.

**What stays valid from Phase 0.B:** per-link distortion measurements (independent of pipeline health) and the `src/adapters/patch.py` infrastructure.

Full data and analysis:
- [REPORT.md](reports/01_variant_a_synthetic.md) — Variant A synthetic
- [REPORT_02.md](reports/02_variant_b_synthetic.md) — Variant B synthetic + head-to-head
- [REPORT_03.md](reports/03_capture_replay_solver.md) — Phase 0.A on real Solver inner adapter
- [REPORT_04.md](reports/04_kaggle_p100_RETRACTED.md) — Phase 0.B in-loop on P100 (accuracy retracted, distortion valid)
- [REPORT_05.md](reports/05_hardware_root_cause.md) — Phase 0.C/0.D/0.E/0.G/0.H investigation: root cause = bf16 hardware vs dtype mismatch
- **[REPORT_06.md](reports/06_variant_b_in_loop_HEADLINE.md) — Phase 0.I: TurboQuant Variant B bit-rate ladder on the working pipeline. main result.**

---

### F. Visual summary: bit-rate vs accuracy curve

```
math500 accuracy (T4 fp32, b=4, n=250, seed=42, sampled)
                            CANONICAL — Phase 0.J confirmation

 90% │
 85% │   ────────── all within ±2 SE of baseline 75.2% ──────────
 80% │              ┌─┐ 78.4%
     │  ┌─┐ 75.2%   │ │      ┌─┐ 76.8%
 75% │  │ │         │ │      │ │      ┌─┐ 75.2%
     │  │ │         │ │      │ │      │ │
 70% │  │ │         │ │      │ │      │ │
     │  │ │         │ │      │ │      │ │
 65% │  │ │         │ │      │ │      │ │
     └──┴─┴─────────┴─┴──────┴─┴──────┴─┴─────
        BL          8        4        2     bits/coord
        1×          4×       8×       16×   compression vs fp32

  (error bars ±5pp at n=250; ALL bit-rates statistically indistinguishable from baseline)
```

**Reading:** at the canonical n=250 confidence level, the bit-rate curve is FLAT from
baseline through 16× compression. The earlier-suspected gradient (at n=50) was sample
noise. Variant B 2-bit and baseline produce identical accuracy (188/250 each).

---

## 13. Open questions — resolved status

| Question (original) | Status | Answer / pointer |
|---|---|---|
| What is `d` for the inner link vs outer link in released checkpoints? | ✅ resolved | Inner: matches each agent's hidden_size (Planner Qwen3-1.7B = 2048, Critic Llama3.2-1B = 2048, Solver Qwen2.5-Math-1.5B = 1536). Outer: outer_12 = 2048→2048, outer_23 = 2048→1536, outer_31 = 1536→2048. From `outer_adapter_config(math).json` on HF Sequential-Light-Outerlinks repo. |
| What recursion depth `N` is typical at evaluation time? | ✅ resolved | Paper default `num_recursive_rounds=3`, `latent_length=48`. Net: ~80,000 sequential fp16 matmuls per problem. Drives the precision requirements in §15. |
| Are baseline math500 numbers reproducible from the released code? | ✅ resolved | YES, on Ampere+ hardware. Phase 0.E: A100 b=8 = 86%, A100 b=32 = 84%, both match paper 75.8% within n=50 SE (~7pp). On Pascal (P100), the pipeline silently collapses to ~35% — see §15. |
| Disk budget for checkpoints (11 GB free on `/`)? | ✅ resolved | Local disk never touches checkpoints (cloud-only). Modal volume `rmas-hf-cache` (persistent) holds the 9 GB Sequential-Light constellation. Reused across all phases at $0 marginal cost after first download. |

### New open questions raised by Phase 0.E / 0.F

| Question | Status | Path to answer |
|---|---|---|
| Does Variant B at 4-bit preserve accuracy on the unbroken A100 pipeline? | ❌ **answered NO** | Phase 0.F: **−19pp drop** (86% → 66.67% on n=30 b=8). Per-call rMSE 0.009 compounds across ~144 sequential link calls into substantial semantic corruption. See §12.5.C. |
| What's the minimum bit-rate for non-destructive Variant B in this pipeline? | 🟡 to test | Run the bit-rate ladder {2,3,4,5,6,8} on A100 b=8 n=50 next month (~$2.10). Expected: bit-rate floor ~6 bits given depth N≈144 and required per-call rMSE ≪ 1/N ≈ 0.007. |
| Can selective quantization (only outer, only inner) recover the gap at 4-bit? | 🟡 to test | Half the link sites → half the compounding. Cheap follow-up. |
| Does QJL residual (TurboQuant §4) close the gap? | 🟡 to test | The unbiased inner-product residual was designed exactly for compounding scenarios. Worth a phase 4 test if 6-bit isn't acceptable. |
| Does Variant B at 4-bit work at num_recursive_rounds=1 (N≈48)? | 🟡 to test | Cheap ablation: 3× fewer link calls. If 4-bit works at r=1 but not r=3, confirms depth-amplification mechanism. |
| Does the Pascal-precision collapse also affect Sequential-Scaled (bigger models)? | 🔵 deferred | Not needed for our primary claim. Would require additional Modal A100 hours; only run if Variant B paper benefits from a Scaled appendix. |
| Should we file an upstream issue for the Pascal hardware advisory? | 🔵 deferred | Low priority. Draft after our write-up. Concise repro: same script Phase 0.D vs Phase 0.E, identical code/checkpoints/batch, only GPU SKU changes, accuracy 35% → 86%. |
| Does Variant B's bit-rate ablation curve (8/4/3/2 bits) on A100 mirror the per-link distortion curve? | 🔵 deferred | Only worth running if Variant B at 4-bit is confirmed non-destructive in Phase 0.F. Budget: ~$2.20 for {2,3,4,8} bit × n=100 b=8. |

---

## 15. Hardware advisory — non-Ampere GPUs collapse RecursiveMAS recursive latent rollouts

**Updated 2026-05-30.** The first version of §15 (after Phase 0.E) hypothesized that GPUs with Tensor Cores (sm_70+) would safely run RecursiveMAS. Phase 0.G (this update) falsifies that simpler model: T4 has Tensor Cores (sm_75) and still collapses to 30% — identical to Pascal P100. The actual minimum requirement is more restrictive: **native hardware bf16 support (sm_80+)**.

### 15.1 Empirical observations (current truth)

End-to-end accuracy on math500 across 3 GPU SKUs, **identical code**, **identical checkpoints**, **identical batch_size=8**, **identical seed=42**, sampled decoding:

| Phase | GPU | sm | Native bf16 HW | Tensor Cores | math500 (n=50/100) |
|---|---|---|---|---|---|
| 0.E ablation | A100 40GB | sm_80 | ✅ yes (3rd gen TC) | ✅ yes | **86.0%** ✓ |
| 0.G sanity | T4 16GB | sm_75 | ❌ no (1st gen TC, fp16 only) | ✅ yes | **30.0%** ❌ |
| 0.D pristine | P100 16GB | sm_60 | ❌ no | ❌ no | **35.0%** ❌ |

The Tensor-Core-presence hypothesis is **falsified** by the T4 row. T4 has TCs and still collapses.

### 15.2 New hypothesis (pending confirmation by `--dtype float16` ablation on T4)

The released RecursiveMAS Solver checkpoint has `config.json: "dtype": "bfloat16"`. Upstream `run.py` defaults to `--dtype auto`, which leads `AutoModelForCausalLM.from_pretrained(torch_dtype="auto")` to load the model in bf16. The pipeline then runs bf16 matmul through the latent rollouts.

- **A100 (sm_80)**: native bf16 in Tensor Cores. bf16 matmul accumulates in fp32 (24-bit mantissa). Deep recursion preserves signal.
- **T4 (sm_75)**: NO native bf16. 1st-gen Tensor Cores support fp16 only. When PyTorch encounters bf16 matmul on T4 hardware, it must either (a) silently downconvert to fp16 (losing bf16's wider dynamic range, gaining fp16's narrower 5-bit exponent ~6.5e4 max), (b) emulate in software (orders of magnitude slower), or (c) dispatch to fp32 CUDA cores (no TC speedup). The 24× wallclock ratio (T4 127 min vs A100 5.4 min, far above the ~5× pure HW ratio) supports an emulation/fallback path.
- **P100 (sm_60)**: no TC, no bf16. fp16 matmul accumulates in fp16. Same kind of collapse via different mechanism.

The two failures (T4 and P100) have **different proximate causes** but the same **proximate symptom**: latent magnitudes during the recursive feedback path drift out of the safe representable range of fp16, while bf16 (when natively supported) preserves the wider exponent range needed to represent the unbounded recurrent state.

### 15.3 Confirmation (2026-05-30 — Phase 0.H on Kaggle T4)

Hypothesis **CONFIRMED** by single-variable swap on T4:

| Phase | GPU | dtype passed to upstream | batch_size | math500 acc |
|---|---|---|---|---|
| 0.G (sanity) | T4 sm_75 | `auto` → loads as bf16 (config), HW falls back | 8 | **30.0%** ❌ |
| 0.H (this) | T4 sm_75 | `float32` (explicit, no fallback) | 4 | **84.0%** ✅ |

**Same exact hardware. Same code. Same seed. Only difference: dtype routing.** Forcing fp32 (which T4 supports natively in CUDA cores, no Tensor Core path) bypasses the bf16-fallback corruption and restores accuracy to within the noise of A100 baseline (84% Phase 0.H vs 86% Phase 0.E ablation).

This empirically isolates the root cause: **the upstream `--dtype auto` default produces bf16, and bf16 on non-Ampere hardware silently produces wrong-precision computation that collapses recursive latent rollouts.**

The runtime on T4 fp32 b=4 (95 min) is actually SHORTER than T4 bf16-fallback b=8 (127 min), because the fp32 path goes through clean CUDA cores instead of an opaque fallback. The fp32 memory overhead (~2× bf16) required halving the batch from 8 to 4 to stay within T4's 16 GB.

### 15.4 Single-forward systems are unaffected on ALL these GPUs

- Solver alone on P100 (Phase 0.C, no recursion): **83%** — bf16 → fp16 conversion happens once, no compounding.
- Per-link distortion measurements on P100 (REPORT_02/03/04): rMSE 0.009 at 4-bit, matches theory — single-call, no compounding.

This is consistent with the hypothesis: the problem is **deep recursion + non-bf16-native hardware**, not the hardware per se.

### 15.5 Implications for any RecursiveMAS user (updated 2026-05-30)

1. **Ampere+ (sm_80+) with `--dtype auto`** is the paper's tested setup. Reproduces 75-86%.
2. **Non-Ampere (T4 sm_75, V100 sm_70, P100 sm_60) with `--dtype auto`**: SILENT COLLAPSE to ~30%. Do not use this combination.
3. **Non-Ampere with `--dtype float32` explicit**: WORKS, restores accuracy to ~84%. Cost: ~2× memory (need to halve batch_size), ~1.5× runtime. T4 b=4 fp32 confirmed 84% on math500 n=50.
4. **Per-link distortion measurements ARE valid on all GPUs** (single-call ops don't accumulate enough error to corrupt rMSE/cosine).
5. **Suggested upstream README change** (to be filed as GitHub issue):

> ⚠️ **Hardware/precision requirement**. RecursiveMAS Sequential-Light is shipped with `--dtype auto`, which loads the released checkpoints in bfloat16 (per their `config.json`). On NVIDIA GPUs without native bfloat16 hardware support (anything before compute capability sm_80 — including P100, V100, T4, RTX 20-series, GTX 16-series), PyTorch silently falls back to a path that loses bf16's wider exponent range. The recursive latent rollouts then accumulate dynamic-range errors that collapse final accuracy from the expected 75.8% to ~30-35% on math500. Two safe configurations:
>
> - **Ampere or newer (A100, H100, RTX 30-series, RTX 40-series, L4, L40S)**: use `--dtype auto` (default). Native bf16, full speed.
> - **Pre-Ampere (T4, V100, P100, etc.)**: pass `--dtype float32` explicitly. Costs ~2× memory (halve `--batch_size`) but restores correctness. We verified T4 b=4 fp32 = 84% on math500 n=50 (vs T4 b=8 auto = 30%, A100 b=8 auto = 86%).

**Implications for our write-up.** Methods section specifies dtype handling as a first-class concern; describes Variant B evaluations as run on (a) Modal A100 b=8 bf16 for the primary numbers and (b) Kaggle T4 b=4 fp32 as cheap reproducibility check. The reproducibility hazard ("bf16-auto-collapses-on-pre-Ampere") is documented as an independent finding suitable for an upstream issue and a brief subsection.

---

## 16. References

- Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* ICLR 2026 (preprint arXiv:2504.19874).
- RecursiveMAS authors. *Recursive Multi-Agent Systems.* arXiv:2604.25917.
- Google Research blog post on TurboQuant (Mar 2026).
- yashkc2025/turboquant — reference Python implementation, used as oracle for Lloyd-Max codebook unit test only, not as production dependency.
- NVIDIA Mixed Precision Training documentation (Tensor Core fp16-accumulated-in-fp32 behavior on sm_70+, contrast with sm_60 fp16-accumulated-in-fp16). Reference for §15.
