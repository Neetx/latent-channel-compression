# Step 1 — mbppplus / `sequential_light` (the "trajectory IS the output" cell)

**Provenance.** RTX 5070 Ti (16 GB, bf16), WSL2 + torch 2.9.0+cu128, RecursiveMAS @
**f95d512**, seed **42**, n=250, T=3. Quantizer = Variant B (TurboQuant MSE core, no
QJL) injected at **all** links (inner+outer, task = `code`). Identical protocol to the
math500 Step 0 — only the dataset changed. Reproduce: `bash ../../run_step1_mbppplus.sh`.

mbppplus settings (RecursiveMAS recommended): `latent_length=16`, `temperature=0.2`,
`max_new_tokens=4000`, code eval `timeout=10s`, `num_prompt_tests=3`. Grading is
**functional** (the generated code is executed against tests) — no loose-match noise.

## Calibration vs the RecursiveMAS paper

Our REF baseline **32.8%** (sampled) ≈ the paper's **35.1%** (Table 2, Sequential-Light
on MBPP+) — within ~2.3 pp (paper is at r=1, we use r=3; n=250 subset; sampling noise).
The pipeline faithfully reproduces the upstream baseline, as it did on math500.

## Phase 1 — sampled bit-rate ladder (n=250, batch=16)

| bits/coord | compression | accuracy |
|:---:|:---:|:---:|
| 0 (REF) | 1× | 32.8% |
| 8 | 4× | 33.6% |
| 4 | 8× | 38.8% |
| 2 | 16× | 34.8% |

→ Flat (all within ~±3 pp sampling noise). No degradation from compression.

## Phase 2 — greedy paired fidelity + flip-churn (n=250, batch=2)

| | accuracy | per-problem |
|---|:---:|:---:|
| REF (bits=0) | **36.4%** | 250/250 |
| INT4 (bits=4) | **36.4%** | 250/250 |

```
2x2:  both correct 79 | REF✓/INT4✗ 12 (loss) | REF✗/INT4✓ 12 (gain) | both wrong 147
Δ = 0.0 pp   flip-churn 24/250 (9.6%)   TOST @±2pp INCONCLUSIVE   McNemar p=1.000
```

## Result — the guiding hypothesis is NOT confirmed

Prediction: compression should **break on code**, because there the trajectory *is* the
answer. It does not. mbppplus behaves **like math500**: answer-preserving in aggregate
(Δ≈0, even cleaner than math500's +2.0), not trajectory-preserving (~10% of answers
flip — essentially the same 9.6% vs 10%). The robustness is therefore **more general**
than the "redundant low-dimensional target" hypothesis predicted.

| | math500 | mbppplus |
|---|:---:|:---:|
| Δ greedy (INT4−REF) | +2.0 pp | 0.0 pp |
| flip-churn | 25/250 (10%) | 24/250 (9.6%) |
| TOST @±2pp | inconclusive | inconclusive |

## Honest caveats

- **Floor effect.** Baseline is low (~33–36%); **147/250 are wrong under both** REF and
  INT4. With most problems already failing, compression has little room to *break* more,
  so the flatness is partly a low-baseline artifact. A higher-capability code system
  (e.g. `mixture` code agent, or `sequential_scaled`) would be a cleaner test.
- **Faint echo of the hypothesis.** Among problems the REF got *right*, compression broke
  **13% (12/91)** on code vs **5% (10/192)** on math — ~2.5× more fragile per correct
  answer. But it is offset by equal gains (net 0), the counts are tiny, and McNemar
  p=1.00, so it is far too weak to conclude.
- No REF-vs-REF control was run for mbppplus; the math500 control established the
  pipeline is deterministic, so the 24 flips are attributed to the quantizer.
- Single seed, n=250 (of 378), one system.

Raw Tier-2 logits (`fidelity_logits.npz`, ~25 MB/run) not committed; in
`~/lcc/fid_out/mbppplus_vb{0,4}_T3_n250_b2_auto/`. Per-run timing in the `*.log` files.
