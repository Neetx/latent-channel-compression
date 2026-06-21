# Step 2 — sequential_scaled × mbppplus (the high-baseline test)

**Provenance.** RTX 5070 Ti (16 GB, bf16), RecursiveMAS @ f95d512, seed 42, n=250, T=3,
Variant B at all links. **Scaled batch recipe (hard-won):** ladder `batch=4`, capture
`batch=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. (`batch=8` hit the 16 GB
ceiling on long 4000-token generations and **thrashed the CUDA allocator → ~50× slowdown
/ effective hang**; batch=4 peaks ~11 GB and runs clean.) Full run ~9.5 h.
Reproduce with `run_cell.py --style sequential_scaled --dataset mbppplus
--ladder-batch 4 --cap-batch 1 --n 250`; see the root `REPRODUCIBILITY.md`.

## Purpose
Step 1 (mbppplus / light) showed aggregate robustness like math500, but the light tier has a
**floor effect** (baseline ~33%, 147/250 wrong under both) → little room for compression
to *break* a correct answer. Scaled has a **high baseline (~74%)** → this is the powered
test of the guiding hypothesis ("compression breaks on code when the system is capable").

## Phase 1 — sampled ladder (n=250, batch=4)
| bits | compression | accuracy |
|:---:|:---:|:---:|
| 0 (REF) | 1× | 70.8% |
| 8 | 4× | 73.6% |
| 4 | 8× | 72.0% |
| 2 | 16× | 71.2% |

→ **Flat** (slightly up) at a ~71% baseline. No accuracy cost from compression even when
the system is good at the code task.

## Phase 2 — greedy paired fidelity + rigorous cross-cell analysis

Scaled greedy: REF **74.4%** vs INT4 **72.4%** → Δ=**−2.0 pp**; flip-churn **11/250 (4.4%)**,
split **8 losses / 3 gains**. From `analysis/compare_cells.py` (all three cells):

| cell | REF | INT4 | Δpp (95% CI) | b/c | churn% | McNemar p | loss-fraction (95% CI) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| math500 / light | 76.8 | 78.8 | +2.0 [−2.0,+6.0] | 10/15 | 10.0 | 0.42 | 0.40 [0.21, 0.61] |
| mbppplus / light | 36.4 | 36.4 | +0.0 [−4.0,+4.0] | 12/12 | 9.6 | 1.00 | 0.50 [0.29, 0.71] |
| mbppplus / scaled | 74.4 | 72.4 | −2.0 [−4.8,+0.4] | 8/3 | 4.4 | 0.23 | 0.73 [0.39, 0.94] |

## Result (honest, rigorous)

1. **No statistically resolved accuracy cost in the clean cells.** Every sampled
   ladder is non-monotonic; every clean greedy delta 95% CI straddles 0. Effects of a
   few percentage points remain compatible with the data.
2. **No significant churn asymmetry, anywhere.** The scaled 8:3 split *looks* like
   "compression breaks more than it fixes", but its loss-fraction CI is **[0.39, 0.94]**
   — it **straddles 0.50**, so it is **not distinguishable from balanced**. With single
   seeds and 11–25 discordant pairs, the apparent per-cell directions (+2 / 0 / −2,
   balanced / balanced / 8:3) are **within sampling noise** — a re-run with another seed
   could easily flip them.
3. **The high-baseline test does not rescue the hypothesis.** At ~74% baseline (floor
   effect removed) compression still shows no significant break on code. The faint
   scaled Δ=−2pp is the most suggestive single number but is **underpowered** (p=0.23).
4. **Power ceiling.** mbppplus caps at 378 problems and the effect is ≤±2pp, so even the
   full benchmark gives few discordants. These results are best reported as **effect
   estimates** ("the accuracy effect is at most a few points, ≈0") rather than
   significance tests.

**Calibration:** REF sampled 70.8% / greedy 74.4% — the high baseline we needed (vs
light's ~33%), confirming scaled is the capable-system regime.

Raw Tier-2 logits are not committed; regenerate them with the canonical local workflow.
