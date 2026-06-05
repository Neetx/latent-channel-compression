# Roadmap

Where this research goes next. The current result is a **narrow but clean
measurement**: TurboQuant's MSE-optimal core compresses the inter-agent latent
channel of **RecursiveMAS `sequential_light` on `math500`** by 4×–16× with no
detected accuracy change under sampled decoding, while being *answer-preserving
but not trajectory-preserving* under greedy decoding.

The roadmap below turns that into a general, mechanistically-understood result. It
separates **work that needs no GPU** (doable now) from **work that needs compute**
(queued for when budget returns).

## Guiding question

The most novel thread is the *answer-preserving ≠ trajectory-preserving* finding.
The research question worth chasing:

> **Why** does the final answer survive while ~88% of trajectories change — and
> **when does that break?**

Working hypothesis: answer-preservation holds because `math500` has a low-dimensional,
redundant target (a boxed number). On tasks where the **trajectory is the output**
(code generation) or where intermediate **tool calls** matter, the same trajectory
drift may *not* be harmless. This is the test that separates a benchmark artifact
from a general property.

---

## Priority 1 — Generality across RecursiveMAS architectures × benchmarks

We measured one cell of RecursiveMAS's released matrix. The next step is to fill it.

**RecursiveMAS released styles × benchmarks** (from upstream
`RELEASE_RECOMMENDED_SETTINGS`):

| Style (tier) | math500 (math) | medqa (medicine) | gpqa (science) | mbppplus (code) |
|---|:---:|:---:|:---:|:---:|
| `sequential_light` (~1–2B agents) | ✅ done | ☐ | ☐ | ☐ |
| `sequential_scaled` (Gemma3-4B + Qwen3, larger) | ☐ | ☐ | ☐ | ☐ |

That is **8 cells; 1 done.** Run the same compression study (bit-rate ladder + the
greedy paired fidelity analysis) on every cell.

**Why each axis matters:**

- **`mbppplus` (code) is the key test of the guiding question.** For code, the
  generated tokens *are* the answer — answer-preservation and trajectory-preservation
  collapse into one. If compression hurts here but not on `math500`, that confirms
  the redundant-target hypothesis.
- **`medqa` / `gpqa`** check whether the result holds for non-math reasoning.
- **`sequential_scaled`** checks whether larger agents (with higher-dimensional
  channels) are more or less compressible than the light tier.

**Other topologies** RecursiveMAS implements — `mixture` (math/code/science +
summarizer), `distillation` (expert/learner), `deliberation` (reflector/toolcaller)
— are present in the upstream code but we did not find released checkpoints. Add
them **if/when weights are published**. `deliberation` is especially interesting:
its tool-caller is exactly a setting where trajectory drift could change behavior.

**Compute note:** `sequential_light` fits an A100-40GB (and a T4 in fp32).
`sequential_scaled` needs an A100-80GB or H100. Always force `--dtype float32` on
pre-Ampere GPUs (see REPORT_05).

---

## Now — progress without any GPU

These are doable today, from existing artifacts and code, and each one *unblocks or
sharpens* a queued experiment.

1. **Per-problem "flip churn" analysis** (existing greedy n=250 data). Net accuracy
   is −2.0 pp, but how many problems flip REF-correct→INT4-wrong vs the reverse? The
   net can hide a much larger churn. This directly quantifies *behavioral*
   perturbation beyond net accuracy — the core of the guiding question.
2. **Divergence vs. difficulty / length.** Does the divergence rate or matched-prefix
   length correlate with `math500` level or generated length? Tests whether
   harder/longer problems are more perturbed.
3. **Teacher-forced per-step fidelity (design + code + tests).** Our matched-prefix
   KL is confounded by trajectory divergence and is unstable. Forcing INT4 to decode
   REF's token sequence gives a position-aligned per-step KL. Build and unit-test the
   analyzer now (on synthetic captures); run it on real captures later. This is the
   cleanest fix for the depth-amplification and localization questions.
4. **Implement the QJL inner-product residual** in `src/` (+ tests vs the reference).
   We currently evaluate only the MSE core. QJL is the obvious reviewer question and
   may rescue the 2-bit rate; have it ready to ablate.
5. **Power / design analysis for the greedy equivalence.** Honest constraint:
   `math500` caps at n=500 and the observed Δ sits on the ±2 pp margin, so a
   single-benchmark ±2 pp equivalence test may stay inconclusive. Decide now whether
   to (a) reframe as effect estimation ("a small ~2 pp effect, quantified") or
   (b) pool seeds/benchmarks for n > 500.
6. **Strengthen Related Work** with verified KV-cache-quantization and latent-comms
   citations, and **archive the raw cloud artifacts** to a citable store (the
   reproducibility audit's top remaining gap).

---

## Next — cheap compute first (when budget returns)

7. **Seed robustness:** 3–5 seeds × 4-bit at n=250 on `math500`. Kills the
   single-seed limitation cheaply.
8. **Powered greedy equivalence / effect estimate:** full n=500 (+ pooling per the
   Now-#5 decision).
9. **Run the teacher-forced capture** and feed the Now-#3 analyzer → resolves
   depth-amplification and inner/outer localization with a clean metric.

---

## Later — higher cost / higher impact

10. **QJL ablation**, especially at 2-bit and on `mbppplus`.
11. **Localization with power**, or a **mixed bit-rate** scheme (high bits on the
    256 outer links, low on the inner) once localization is resolved.
12. **Wall-clock packed transport** in a distributed setup — turns the
    information-theoretic ~9.0 → ~1.1 MiB/problem figure into a measured saving.
13. **`sequential_scaled` tier** and, if checkpoints appear, the **mixture /
    distillation / deliberation** topologies.

---

## What "done" looks like

With Priority 1 (the 8-cell matrix) plus the Now/Next analyses, this becomes a solid
empirical study with a real scientific hook: *a data-oblivious quantizer compresses
the inter-agent latent channel of a recursive multi-agent system across architectures
and benchmarks, preserving answers but not trajectories — with the gap widening
exactly where the trajectory is the output.*
