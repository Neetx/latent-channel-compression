# Step 0 — local bf16 replication of math500 / `sequential_light`

**Provenance.** RTX 5070 Ti (16 GB, Blackwell sm_120), WSL2 + torch 2.9.0+cu128,
`--dtype auto` (bf16). Upstream RecursiveMAS @ **f95d512**. Seed **42**. Dataset
math500 (test, n=250). Quantizer = Variant B (TurboQuant MSE core, no QJL), injected
at **all** links (inner + outer). Runs: 2026-06-06 → 2026-06-19.

Reproduce: `bash ../../run_step0.sh` then `python ../../analysis/flip_churn_tost.py`.

## Phase 1 — sampled bit-rate ladder (n=250, batch=16, `--no-capture`)

| bits/coord | compression | local bf16 acc | cloud fp32 (REPORT_06) |
|:---:|:---:|:---:|:---:|
| 0 (REF) | 1× | **77.6%** | 75.2% |
| 8 | 4× | 76.0% | 78.4% |
| 4 | 8× | 78.8% | 76.8% |
| 2 | 16× | 78.0% | 75.2% |

→ Curve **flat** across 4×–16× (all within ~3 pp). Reproduces the headline: no detected
accuracy cost from compressing the inter-agent latent channel under sampled decoding.
Source: [`ladder/`](ladder/).

## Phase 2 — greedy paired fidelity (n=250, batch=2, Tier-2 capture)

| run | accuracy | per-problem | logit batches |
|---|:---:|:---:|:---:|
| REF (bits=0) | **76.80%** | 250/250 | 136 |
| INT4 (bits=4) | **78.80%** | 250/250 | 135 |

- **Calibration:** local greedy REF = 76.80% = **exact match** to the cloud REF (76.8%).
- **Δacc = +2.0 pp** (INT4 − REF). Cloud measured −2.0 pp → same magnitude, **opposite
  sign** ⇒ the effect is ≈ 0 / not robustly signed.

Source: [`fidelity/`](fidelity/). Raw Tier-2 logits (`fidelity_logits.npz`, ~25 MB/run)
are not committed; regenerate via `run_step0.sh`.

## Flip-churn + TOST (roadmap "Now #1", done)

From [`flip_churn_tost.txt`](flip_churn_tost.txt) (`analysis/flip_churn_tost.py`):

```
2x2:  both correct 182 | REF✓/INT4✗ 10 (loss) | REF✗/INT4✓ 15 (gain) | both wrong 43
flip-churn: 25/250 (10%) of answers flip; net +5 (+2.0 pp) → churn = 5× the net
paired bootstrap Δ=+2.0 pp, 90% CI [−1.2,+5.2] → TOST @±2pp INCONCLUSIVE
McNemar exact: discordant=25, two-sided p=0.424 (net not distinguishable from 0)
```

**Reading.** The flat net accuracy hides a 5×-larger behavioural churn: **10% of final
answers change** under 4-bit compression (beyond the 88% trajectory change reported in
REPORT_07), yet aggregate accuracy is unmoved. *Answer-preserving in aggregate, with
real per-problem churn underneath.*

> Caveat surfaced while inspecting examples: the upstream math500 grader can mark
> multi-value answers correct loosely (e.g. accepted `−1, 2` for gold `1, −2`), so a few
> individual flips include grader noise. Functional benchmarks (mbppplus) avoid this.
