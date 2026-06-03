# CORE_DESIGN — design plan for the central research experiment

**Status:** design (not implemented)
**Last updated:** 2026-05-27
**Predecessors:** [RESEARCH.md](../RESEARCH.md) (problem statement), [REPORT.md](../reports/01_variant_a_synthetic.md), [REPORT_02.md](../reports/02_variant_b_synthetic.md), [REPORT_03.md](../reports/03_capture_replay_solver.md) (preceding findings)
**Successor:** Phase 0.B implementation + REPORT_04

This document is the architectural plan for the experiment that actually answers the central research question. It freezes the decisions that need to be frozen *before* code, and explicitly defers the ones that should stay open.

---

## 1. Where we are

| Established | Open |
|---|---|
| Variants A and B implemented + unit-tested, including 3 oracle tests vs `external/turboquant_ref`. | Whether **downstream multi-agent behavior** (math500 accuracy, logits KL, refusal/format breaks) survives quantization. |
| Variant B on synthetic Gaussian-on-sphere matches TurboQuant paper Table 1 to the third decimal. | Whether quantization error **compounds along the recursion** (multi-round) or stays bounded. |
| Variant B on REAL Solver inner-adapter output matches synthetic baseline to 2 %. | Whether **inner-link vs outer-link** behave differently (we've only tested inner). |
| Gate 0 (identity wrapper) passes cleanly: rMSE 2 × 10⁻⁹. | Whether the per-vector L2 normalization step interacts badly with the LayerNorm that consumes the quantized output downstream. |
| Kaggle pipeline functional: private kernels, push/poll/output, GPU+internet path verified. | Whether 3 bits is also passable — at the latent level it is (rMSE 0.034), but task accuracy unknown. |

Phase 0.A told us the **distortion is small and matches theory**. The central question — *does small latent distortion translate to preserved task behavior in a multi-agent loop* — is still open and is what this document is about.

---

## 2. Research question — refined into testable form

Original question (RESEARCH.md §1): *"Is the RecursiveLink latent compressible by TurboQuant-style quantization at 3-4 bits, without significant downstream accuracy loss?"*

Refined into a falsifiable claim, what we want to test:

> **Claim Q.** On RecursiveMAS Sequential-Light running math500, replacing the float32 output of every inner adapter and every outer adapter with a Variant B (Haar + Lloyd-Max-Gaussian) reconstruction at **b** bits per coordinate produces:
> 1. final-answer accuracy Δacc ≥ −1 pp (vs baseline) — for `b ≥ 4`;
> 2. median per-token logits KL ≤ 0.05 nat — for `b ≥ 4`;
> 3. no qualitative output collapse (e.g., garbage tokens, infinite loops) — for `b ≥ 3`.

If Q is true at b=4, the RecursiveLink is robustly compressible. If Q is false even at b=4, the channel is fragile in ways that synthetic+latent-only measurements failed to predict — that itself is an informative structural difference.

Three secondary questions we want to answer along the way:

- **Q.A** — does error compound along recursion rounds? Linear, sub-linear, or super-linear in round count?
- **Q.B** — do inner and outer links have different compressibility profiles? If yes, can we use mixed bit-rates (e.g., 4-bit inner, 3-bit outer)?
- **Q.C** — at what bit-rate does the failure mode appear (output collapse), and is the failure gradual or abrupt? An abrupt cliff at e.g. 3→2 bits is a sign of a phase transition that may inform a downstream theoretical contribution.

---

## 3. What "done" looks like

A single end-of-Phase-0.B summary table:

| bits | links quantized | Δacc (math500) | median logits KL | qualitative |
|---:|:---|---:|---:|:---|
| 8 | inner only | 0.0 pp | ~0 | clean |
| 4 | inner only | ? | ? | ? |
| 3 | inner only | ? | ? | ? |
| 2 | inner only | ? | ? | ? |
| 8 | outer only | … | | |
| 4 | outer only | | | |
| 3 | outer only | | | |
| 8 | both | | | |
| 4 | both | | | |
| 3 | both | | | |

With 95 % bootstrap CI on each Δacc cell over a math500 subsample. Sample size and the formal decision matrix in §5.5 and §8.

Done means: this table exists, every cell has data with CI, and we can state with statistical confidence whether Claim Q holds at b=4, at b=3, and what the boundary is. Anything beyond is bonus.

---

## 4. Architectural recap of Sequential-Light (the system under test)

Components (per `external/RecursiveMAS/system_loader.py` and `modeling.py`):

- **Planner** = Qwen3-1.7B (hidden 2048)
- **Critic** = Llama3.2-1B (hidden 2048)
- **Solver** = Qwen2.5-Math-1.5B (hidden 1536)
- **Inner adapter ×3** — one per agent, each maps hidden→hidden (same dim) via `proj1·GELU·proj2 + residual`, post-LN. Total trainable ≈ 9M params.
- **Outer adapter ×3** — `outer_12: Planner→Critic (2048→2048)`, `outer_23: Critic→Solver (2048→1536)`, `outer_31: Solver→Planner (1536→2048)`. Same shape as inner but with cross-dim projection. Hidden inside the MLP is `out_dim * 2`.
- **Pipeline per math prompt** (`inference_mas.py`): planner emits 48 latent tokens → outer_12 → critic uses them → critic emits 48 latent → outer_23 → solver uses them → solver emits 48 latent → outer_31 → planner uses for refinement. This is **one round**. The released `sequential_light` config does a fixed sequential pass (no iteration), so it's effectively **1 round = 6 RecursiveLink invocations** (3 inner + 3 outer).

This means error compounding **for the released sequential_light is bounded** — at most one application of each link in the path planner→critic→solver→planner. Multi-round compounding will become a concern only if we extend to a multi-round configuration. For Phase 0.B we accept the single-round design.

Disk/RAM/GPU budget for the full system:
- 3 models in fp16: ~6.5 GB
- Inner+outer adapters: <50 MB total
- Activations + KV cache for `latent_length=48` + chat prompts up to ~512 tokens: ~2 GB
- Fits comfortably on T4 16 GB.

---

## 5. Experimental design

### 5.1 Pipeline scope

We use the released `sequential_light` config with the released checkpoints, math500 as the task. No fine-tuning, no prompt changes, no rounds-count modifications. Two reasons:
1. Reproducibility against the paper's reported numbers.
2. We're testing the *channel*, not the *protocol*. Keep everything else fixed.

### 5.2 Quantization injection plan

Four cells of the experimental grid, multiplied by 4 bit-rates each = 16 runs (plus baseline = 17):

| Cell | Inner adapters | Outer adapters |
|---|---|---|
| baseline | float32 (released) | float32 (released) |
| inner-only | Variant B @ b bits | float32 |
| outer-only | float32 | Variant B @ b bits |
| both | Variant B @ b bits | Variant B @ b bits |

Bit-rates `b ∈ {8, 4, 3, 2}`. 8-bit cells serve as sanity (Δacc should be ≈ 0).

**Rationale for the 4-way split**: inner and outer links are the same module pattern but operate on very different conditional distributions (inner = post-LN output destined for the *same* model's input embedding space; outer = post-LN output destined for a *different* model's input embedding space, after dimension projection). Separating them lets us localize any failure.

### 5.3 Sample size

math500 has 500 problems. We use a **fixed deterministic subset of 100** problems (seed=42 on the 500-item index list). 100 problems gives:
- Standard error on accuracy ≈ √(0.5 · 0.5 / 100) = 5 pp at worst — adequate to distinguish Δacc = -1 pp from Δacc = -5 pp with confidence.
- ~30-60 minutes of inference per cell on T4 (5 LLM forwards per problem × 100 problems).
- Total Phase 0.B budget: 17 cells × ~45 min = ~13 GPU-hours. Comfortably within Kaggle's weekly 30h GPU quota.

If 100 is too noisy after Phase 0.B run we can extend to 200; the cost doubles but is still within budget.

### 5.4 Compute / Kaggle strategy

Single Kaggle script per cell (17 total) is wasteful — each one would have to re-download 9 GB of models. Two better designs:

**Design A — one big run, all cells.** Single notebook that loads everything once, sweeps all 17 cells in sequence, dumps a single results JSON. Wall-clock 13h, well within Kaggle's 12h soft limit per session — borderline. We'd need to split into 2-3 sessions.

**Design B — Kaggle Dataset cache + many small kernels.** Upload the captured baseline (per-problem prompts + tokenizer output + final generated tokens) as a private dataset once. Each cell kernel attaches that dataset, runs only the model with quantization, compares to cached baseline. Each cell ~40 min, runs independently, can be queued and parallelized.

**Decision: Design B**, with a twist:

- **First pass**: capture the BASELINE end-to-end (math500 100-problem subset, full pipeline, save all per-problem outputs + final-step logits) into a private Kaggle Dataset.
- **Per-cell pass**: load baseline dataset + models, run the quantized pipeline, compute Δacc / KL / per-link distortion, save results JSON.
- **Aggregator**: pull all 16 per-cell results locally, build the §3 table.

This isolates per-cell runs (failure recovery is cheap) and avoids redownloading the model corpus.

### 5.5 Metrics — the ladder

Cheap first, expensive last. Every cell records every level so we can correlate them in REPORT_04.

1. **Per-link latent distortion** (cheap, automatic from Variant B): rMSE, cosine, norm-ratio, ip-error.
2. **Per-agent logits KL**: KL(p_baseline ‖ p_quant) on the *last* token's logits before each agent's final emit. This is per-agent, per-prompt; aggregate as median + 95% bootstrap.
3. **Per-token KL on the planner's final output** (the answer): full sequence KL average. Captures whether the *generation* trajectory diverges, not just the entry logits.
4. **String-level answer match**: extract boxed answer (math500 convention), compare to ground truth. Binary correctness per problem.
5. **Δacc with bootstrap CI**: (cell_correct - baseline_correct) / N with 1000-sample bootstrap.
6. **Qualitative failure flagging**: tokens-per-answer outlier (truncated or runaway generation), format breakage (no boxed answer), repetition loops. Logged but not aggregated formally.

We also log Variant-B-internal stats per call (input vector norm, post-rotation per-coord max) to detect distributional drift from the assumed unit-sphere-like marginals.

---

## 6. Implementation design

### 6.1 `src/adapters/patch.py` — the patching utility

Goal: take a loaded `Adapter` or `CrossModelAdapter` and wrap its `forward` so the post-LN output passes through a quantizer **without modifying RecursiveMAS code**. Two requirements:

1. **Reversible** — un-patching restores the original `forward` exactly. Lets us A/B in the same Python process.
2. **Transparent to autograd** — Variant B already is. Just don't add anything that breaks gradients (we won't need them for inference but keeps optionality open for future QAT).

Proposed API:

```python
def patch_adapter(
    adapter: nn.Module,            # an instance of Adapter or CrossModelAdapter
    quantizer_factory: Callable[[int], nn.Module],
    *,
    role: str = "inner" | "outer",
) -> Callable[[], None]:
    """Wrap `adapter.forward` so its output is quantized in-place.

    Returns an `unpatch()` callable that restores the original forward.
    The quantizer_factory is called with the LAST output dim of the adapter
    (post-LN dim) so it can build a correctly-sized Haar matrix + codebook.

    Caches one quantizer instance per (d, role) pair — d-many Haar matrices
    cost O(d²) RAM each, so don't build duplicates.
    """
```

Usage in the experimental harness:

```python
from src.adapters.patch import patch_adapter
from src.quantizers.turboquant_honest import TurboQuantHonest

q_factory = lambda d, bits=4: TurboQuantHonest(d=d, bits=bits, seed=0).to(device)
patches = []
for agent in system.agents.values():
    patches.append(patch_adapter(agent.inner_adapter, lambda d: q_factory(d, bits=4), role="inner"))
for adapter in system.outer_adapters.values():
    patches.append(patch_adapter(adapter, lambda d: q_factory(d, bits=4), role="outer"))
try:
    run_one_problem(...)
finally:
    for un in patches: un()
```

Internal implementation sketch:

```python
def patch_adapter(adapter, quantizer_factory, role):
    original_forward = adapter.forward
    out_dim = adapter.proj2.out_features  # both Adapter and CrossModelAdapter expose this
    quantizer = quantizer_factory(out_dim).to(next(adapter.parameters()).device).eval()

    def patched_forward(x):
        out = original_forward(x)
        # Quantize in fp32 for numerical safety; cast back to original dtype.
        orig_dtype = out.dtype
        out_q = quantizer(out.float()).to(orig_dtype)
        return out_q

    adapter.forward = patched_forward
    def unpatch():
        adapter.forward = original_forward
    return unpatch
```

Open decisions on patch.py, deliberately left for implementation time:
- **Per-instance vs shared quantizer** — sharing per (d, role) reduces RAM but couples random Haar rotations across links. Probably want per-instance with seed = hash(role + adapter_id) so each link has its own rotation but the choice is deterministic across runs. Make this a parameter.
- **Stat collection** — should the patched forward optionally record per-call rMSE / cosine into a global accumulator? Yes — needed for metrics ladder level 1. Implement as a `record=True` flag.
- **fp16 vs fp32 quantization path** — current TurboQuantHonest is fp32-internal; on a 16 GB GPU running 3 fp16 models, allocating fp32 Haar matrices for 6 links of d=1536-2048 is ~250 MB. Acceptable but worth flagging.

### 6.2 Capture-then-replay vs in-line

For Phase 0.A (already done) we captured tensors and ran the quantizer offline — simple and isolated.

For Phase 0.B we **must** run the quantizer in-line in the model forward, because downstream behavior depends on what the *next agent* sees, which depends on the quantized output. Replay won't work: the receiver's forward depends on the post-quant tensor, which then triggers its own quantization, etc. The whole pipeline must run with quantization active.

**Decision**: in-line via `patch_adapter`. Capture per-call distortion stats as a side-effect for the metrics ladder.

### 6.3 Reproducibility plan

- Seed everything (torch + numpy + transformers generation seed) per cell run, same seed across cells.
- math500 subset is a deterministic slice indexed by seed=42 on the full 500.
- Haar rotation seeds: one fixed seed (0) per (d, role) pair, shared across cells so the same rotation is applied. This means cell-to-cell differences are purely from bit-rate, not from rotation noise.
- Tokenizer truncation/padding decisions match the released RecursiveMAS defaults — no custom prompt edits.
- Generation sampling: use `temperature=0.6, top_p=0.95, seed=42` per the released `RELEASE_RECOMMENDED_SETTINGS` for `(sequential_light, math500)`.
- Version pinning: Kaggle Python image varies; explicitly print `torch.__version__`, `transformers.__version__`, `numpy.__version__`, `scipy.__version__`, CUDA capability into the results JSON.

---

## 7. Stage gates for Phase 0.B

| Gate | Condition to pass | If fail |
|---|---|---|
| 0.B.0 — baseline reproduces | Δacc on the 100-problem subset matches the paper's reported number for sequential_light/math500 within ±2 pp (paper claims +8.3 % avg accuracy gain over single-agent baseline, not absolute accuracy, so we use our own baseline as reference, not theirs). | Investigate before any quant cell — broken baseline invalidates everything. |
| 0.B.1 — 8-bit inner | Δacc ≥ −0.5 pp, KL_median ≤ 0.01 nat. | Patching plumbing bug, debug. |
| 0.B.2 — 4-bit inner | Δacc ≥ −1 pp, KL_median ≤ 0.05 nat. | RecursiveLink fragile to inner-link quantization. Stop, write up. |
| 0.B.3 — 4-bit outer | Δacc ≥ −1 pp. | Outer link is the failure point. Useful localization. Continue with inner-only at 4-bit. |
| 0.B.4 — 4-bit both | Δacc ≥ −1 pp. | Combined error compounds; report and continue with separated 3-bit. |
| 0.B.5 — 3-bit inner | Δacc ≥ −2 pp. | Stop at 4-bit as the practical limit. |
| 0.B.6 — 3-bit both | Δacc ≥ −2 pp. | 3-bit feasible on inner only; report mixed-rate viability. |

The 2-bit cells are run for completeness but not gated; we expect collapse based on REPORT_02 cosine = 0.94 (a 6 % cosine drop typically destroys token-level next-step prediction).

---

## 8. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Baseline doesn't reproduce on Kaggle GPU within ±2 pp | medium | high — kills the whole experiment | Run baseline first as 0.B.0 gate; if off, debug seeds + sampling before any quant cell |
| LayerNorm input shift due to quant alters downstream distribution non-linearly | medium | medium | Per-link norm-ratio is already monitored; spot-check at multiple agents |
| 13 GPU-hour budget overruns Kaggle weekly quota | low | medium | Cells are independent; can run across multiple weeks if needed; or drop the 2-bit and "both"-mixed cells |
| Patching breaks when adapter is called inside `torch.no_grad()` or inside `compile`'d region | low | medium | Test patch with a unit test mocking the patched call inside `torch.no_grad()`; add `@torch.no_grad()` inside the patched forward if needed |
| Variant B fp32 Haar matrices push GPU OOM when combined with 3 fp16 models | low | medium | Build matrices on CPU and stream-rotate via .to(device); or downcast Haar to fp16 (small added quantization noise) |
| math500 sample of 100 too noisy to distinguish Δacc=-1 pp from -2 pp | medium | low | Pre-registered: if any gate's CI straddles its threshold, extend to 200 problems on that cell only |
| Kaggle Dataset upload fails or hits quota | low | low | Baseline JSON is small (<10 MB); dataset is overkill — can pass baseline via kernel_sources or inline base64 in the per-cell script |

---

## 9. Deliverables, in order

1. **`src/adapters/patch.py`** with the API in §6.1, plus 5-7 unit tests:
   - patch + unpatch is idempotent
   - patched forward produces same shape/dtype as original
   - patched output differs from baseline when bits < 16
   - stat collection (when enabled) records expected number of calls
   - works inside `torch.no_grad()`
2. **Baseline capture script** (_experiments/retracted_p100_inloop/ (archived; full retraction notice in docs/reports/04_kaggle_p100_RETRACTED.md)_): loads Sequential-Light, runs the 100-problem math500 subset with no quantization, saves per-problem outputs + final-token logits + correctness flags to `/kaggle/working/baseline.json` (or splits to multiple files if logits are too big). Pushes via `./bin/kaggle`.
3. **Baseline as Kaggle Dataset** (private). Manual one-time upload.
4. **Per-cell quantized run script** (_experiments/retracted_p100_inloop/ (archived; full retraction notice in docs/reports/04_kaggle_p100_RETRACTED.md)_): loads system, applies patches per the cell config (read from env vars or a metadata.json key), runs the same 100 problems, computes the metrics ladder against the baseline dataset, saves results JSON.
5. **Push 16 cells** (4 bits × 4 link-combinations). Use kernel-metadata.json variants or push the same script 16 times with different env. Each kernel ~40 min on T4.
6. **Local aggregator** (_experiments/retracted_p100_inloop/ (archived; full retraction notice in docs/reports/04_kaggle_p100_RETRACTED.md)_): pulls all 16 result JSONs, builds the §3 table, writes `REPORT_04.md`.
7. **REPORT_04.md** — the answer to Claim Q.

Optional (post-REPORT_04, depending on results):
8. Mixed-rate sweep (e.g., 4-bit inner + 3-bit outer or vice versa) if separation showed asymmetric sensitivity.
9. QAT on the patched system if 3-bit inference-only fell short (RESEARCH.md §7 Phase 3).
10. Packed low-bit representation and real-VRAM measurement (RESEARCH.md §7 Phase 5) — only if accuracy story holds.

---

## 10. What this design intentionally does NOT do

- **Does not extend to sequential_scaled or mixture or distillation or deliberation styles.** Those are 4× more compute and complicate the analysis. They're follow-on work, not core.
- **Does not introduce mixed-rate cells in the initial sweep.** Only uniform bit-rates. Mixed comes only if 6→8 cells reveal asymmetry.
- **Does not measure non-math500 benchmarks (gpqa, medqa, mbppplus).** Same logic — focus first, generalize later.
- **Does not change RecursiveMAS prompts, sampling temperature, or pipeline structure.** Anything we vary other than quantization weakens the causal claim.
- **Does not implement packed/Triton kernels.** Per RESEARCH.md §10 out-of-scope until accuracy story is solid.
- **Does not retrain or fine-tune the adapters.** Inference-only. QAT is Phase 3 in RESEARCH.md and conditional on this Phase failing.

---

## 11. What we'll know after Phase 0.B

If everything passes through 0.B.6:
> "Variant B is a drop-in, training-free, data-oblivious quantizer for RecursiveMAS-style latent communication channels, achieving a 4–5× memory reduction on adapter output with < 1 pp downstream accuracy delta at 4 bits and a soft failure mode at 3 bits."

That's a clean positive result with a clear practical implication. Worth a short write-up.

If we fail at 0.B.2 (4-bit inner):
> "Despite the latent-level distortion matching theoretical TurboQuant guarantees, the RecursiveLink channel of RecursiveMAS sequential_light cannot tolerate 4-bit quantization. The discrepancy between latent-level fidelity and downstream behavior indicates X (TBD diagnostic — likely related to per-channel sensitivity not captured by the rotation-invariant metrics, or to LayerNorm input distribution drift)."

That's a negative result that creates a useful research question: *what makes a latent channel "really" compressible* beyond rMSE? Worth a short write-up too, with a different framing.

Either outcome is informative. Design is therefore not biased toward either direction.
