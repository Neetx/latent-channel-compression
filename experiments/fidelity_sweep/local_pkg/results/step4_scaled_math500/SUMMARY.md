# Step 4 — sequential_scaled × math500 (closes the 2×2, and a key negative result)

**Provenance.** RTX 5070 Ti (16 GB, bf16), RecursiveMAS @ f95d512, seed 42, n=250, T=3,
Variant B at all links. Scaled batch recipe: ladder `batch=4`, capture `batch=1`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The cell took ~26 GPU-h (each scaled
ladder condition ~3 h; each greedy capture ~6.4 h at batch=1) and **survived a reboot
mid-run** via `run_cell.py --resume`, which reuses any condition that already has a valid
result JSON.
Reproduce: `python ../../run_cell.py --style sequential_scaled --dataset math500 --ladder-batch 4 --cap-batch 1 --n 250 --resume`.

## Purpose
This is the second **scaled** cell, completing the 2×2 {math500, mbppplus} × {light,
scaled}. Its specific job is to test whether the large trajectory-robustness gap observed
on mbppplus (light 92.8% → scaled 51.2% divergence within the capture window) is a
**general property of the scaled tier**, or **specific to the code task**.

## Phase 1 — sampled ladder (n=250, batch=4)
| bits | compression | accuracy |
|:---:|:---:|:---:|
| 0 (REF) | 1× | 82.8% |
| 8 | 4× | 85.6% |
| 4 | 8× | 84.0% |
| 2 | 16× | 84.8% |

→ **Flat / non-monotonic** at a high ~84% baseline: no accuracy cost from compression even
where the system is strong (scaled scores ~84% on math500 vs light's ~77%).

## Phase 2 — greedy paired fidelity + rigorous cross-cell

Greedy: REF **87.6%** vs INT4 **85.2%** → Δ=**−2.4 pp**; flip-churn **22/250 (8.8%)**,
split **14 losses / 8 gains**, McNemar **p=0.29**, loss-fraction CI **[0.41, 0.83]**
(straddles 0.50). From `analysis/compare_cells.py` (all five cells):

| cell | REF | INT4 | Δpp (95% CI) | b/c | churn% | McNemar p | loss-fraction (95% CI) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| math500 / light | 76.8 | 78.8 | +2.0 [−2.0,+6.0] | 10/15 | 10.0 | 0.42 | 0.40 [0.21, 0.61] |
| **math500 / scaled** | 87.6 | 85.2 | −2.4 [−6.0,+1.2] | 14/8 | 8.8 | 0.29 | 0.64 [0.41, 0.83] |
| mbppplus / light | 36.4 | 36.4 | +0.0 [−4.0,+4.0] | 12/12 | 9.6 | 1.00 | 0.50 [0.29, 0.71] |
| mbppplus / scaled | 74.4 | 72.4 | −2.0 [−4.8,+0.4] | 8/3 | 4.4 | 0.23 | 0.73 [0.39, 0.94] |
| medqa / light | 21.2 | 36.4 | +15.2 [+8.8,+21.6] | 19/57 | 30.4 | <0.001 | 0.25 [0.16, 0.36] |

Both scaled cells now lean slightly negative (−2.0, −2.4) but **neither resolves** — every
clean Δ 95% CI straddles 0 and every loss-fraction CI straddles 0.50. (medqa is the lone
significant cell and is confounded — see `../step3_light_medqa/SUMMARY.md`.)

## Tier-2 trajectory result — the key finding

From `analysis/tier2_logit_fidelity.py` (top-K=256, 128-position window, primary calls
only, retries excluded):

| cell | divergence /128 | mean prefix /128 | matched-prefix KL (nats) | channel cosine |
|---|:---:|:---:|:---:|:---:|
| math500 / light | 86.4% | 53.7 | 0.079 | 0.9952 |
| **math500 / scaled** | **80.4%** | **54.6** | **0.147** | **0.9953** |
| mbppplus / light | 92.8% | 35.8 | 0.113 | 0.9953 |
| mbppplus / scaled | 51.2% | 65.7 | 0.059 | 0.9953 |

**The scaled trajectory-robustness effect does NOT generalise across tasks.** On mbppplus,
light→scaled nearly halved divergence (92.8% → 51.2%) and raised the common prefix sharply
(35.8 → 65.7). On math500 the *same tier change barely moves divergence* (86.4% → 80.4%),
the prefix is essentially unchanged (53.7 → 54.6), and the matched-prefix KL actually
**rises** (0.079 → 0.147). So the dramatic mbppplus gap is **task-specific**, not a general
property of the larger constellation.

This is direct evidence **against** a simple parameter-count / capacity law and **for** the
honest framing already in the roadmap: the mbppplus light/scaled contrast is a
task-and-tier-conditioned association, not a mechanism that transfers across tasks. Channel
cosine is ~0.9953 in all four math/code cells — the data-oblivious 4-bit quantizer injects
identical distortion everywhere; what differs is how each system *absorbs* it, and that
difference is not monotone in tier once the task changes.

## Result

1. **The 2×2 is closed with no statistically resolved accuracy cost in any clean cell.**
   Every sampled ladder is flat; every clean greedy Δ 95% CI straddles 0.
2. **Aggregate answer robustness with per-problem churn** holds again (Δ≈−2.4 pp, 8.8%
   of individual answers flip).
3. **The headline mbppplus trajectory-robustness gap is task-specific** — it does not
   reproduce on math500. The "scaled is more trajectory-robust" statement must stay scoped
   to the task where it was observed until the multi-seed, length/margin-adjusted analysis
   in `ROADMAP.md` §1 either explains the mbppplus effect or shows it is a mediator (output
   length, REF token margin) rather than tier per se.

Raw Tier-2 logits, per-token generations, and machine logs are not committed (they carry
prompts / personal paths); regenerate them with the canonical local workflow above.
