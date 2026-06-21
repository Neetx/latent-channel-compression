# Step 3 — sequential_light × medqa (and a greedy-decoding confound it exposed)

**Provenance.** RTX 5070 Ti (16 GB, bf16), RecursiveMAS @ f95d512, seed 42, n=250, T=3,
Variant B at all links. Light recipe: ladder `batch=16`, capture `batch=2`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Full run ~11 h.
Reproduce: `python ../../run_cell.py --style sequential_light --dataset medqa --ladder-batch 16 --cap-batch 2 --n 250`.

**Calibration.** REF sampled 34.4% ≈ the paper's light × medqa (Table 2, ~30.3%) — low
baseline (the 1.5B system is weak on medical QA), same floor-effect regime as mbppplus/light.

## Phase 1 — sampled ladder (n=250, batch=16)
| bits | compression | accuracy |
|:---:|:---:|:---:|
| 0 (REF) | 1× | 34.4% |
| 8 | 4× | 26.4% |
| 4 | 8× | 32.8% |
| 2 | 16× | 30.0% |

→ Scattered within ~±3 pp sampling noise, **no monotonic trend** with compression (the
*least*-compressed 8-bit is the lowest, the *most*-compressed 2-bit is mid) → **no
accuracy effect**, like every other cell's ladder.

## Phase 2 — greedy paired fidelity: an anomaly, then its cause

Greedy: REF **21.2%** vs INT4 **36.4%** → **Δ=+15.2 pp**, flip-churn **76/250 (30.4%)**,
19 losses / 57 gains, McNemar **p<0.001**, TOST **DIFFERENT**. This is the **only** cell
with a significant greedy effect — and it looks like "compression helps by 15 pp", which is
not credible. Investigating the per-problem predictions:

```
REF  parsed answers:  A=116  B=57  C=43  D=34   ← picks 'A' 46% of the time
INT4 parsed answers:  A=106  B=61  D=44  C=39   ← far more balanced
```

Both produce **clean A/B/C/D letters** (no parsing bug). The unquantized **REF, under
greedy, develops a strong 'A'-bias** (~46% vs the ~25% expected for a 4-way MCQ) → it
answers A too often → 21.2%. The quantizer's perturbation acts as a **dither that breaks
the bias** → balanced → 36.4%. Confirmed by the sampled REF being normal (34.4%) — sampling
already breaks the bias; only **greedy** REF degenerates.

## Result + a methodological finding

- **The +15 pp is a greedy-decoding pathology of the REF, not a channel-fidelity result.**
  medqa's greedy paired comparison is **confounded**; read this cell via its (flat)
  **sampled ladder**.
- **Design caveat (the real contribution here):** the greedy paired design — central to the
  fidelity/flip-churn analysis — assumes the *REF greedy run is well-behaved*. A weak model
  on a multiple-choice task violates this (degenerate first-option bias under greedy). medqa
  is the case that **exposes when the paired-greedy probe is uninformative**, and it should
  be reported as such.

## Cross-cell picture (4 cells; `analysis/compare_cells.py`)

| cell | Δpp (95% CI) | McNemar p | loss-fraction (95% CI) | read |
|---|:---:|:---:|:---:|---|
| math500 / light | +2.0 [−2.0,+6.0] | 0.42 | 0.40 [0.21,0.61] | ≈0, clean |
| mbppplus / light | +0.0 [−4.0,+4.0] | 1.00 | 0.50 [0.29,0.71] | ≈0, clean |
| mbppplus / scaled | −2.0 [−4.8,+0.4] | 0.23 | 0.73 [0.39,0.94] | ≈0, clean |
| medqa / light | +15.2 [+8.8,+21.6] | <0.001 | 0.25 [0.16,0.36] | **confounded (REF A-bias)** |

**Sampled ladders are flat in every cell.** The greedy paired probe is clean and ≈0 where
the REF greedy is well-behaved (math/code), and medqa is the cell that shows when it is not.

Raw Tier-2 logits not committed (~25 MB/run) — in `~/lcc/fid_out/medqa_vb{0,4}_T3_n250_b2_auto/`.
