# Documentation index

## Read order for newcomers

1. **[reports/08_local_cross_cell_generalization.md](reports/08_local_cross_cell_generalization.md)** — primary five-cell RTX 5070 Ti result
2. **[RESEARCH.md](RESEARCH.md)** — current claims, method, limits, and next experiments
3. **[REPRODUCIBILITY.md](../REPRODUCIBILITY.md)** — clean-clone local reproduction
4. **[reports/05_hardware_root_cause.md](reports/05_hardware_root_cause.md)** — why pre-Ampere GPUs collapse
5. (Optional) **[reports/02_variant_b_synthetic.md](reports/02_variant_b_synthetic.md)** + **[reports/03_capture_replay_solver.md](reports/03_capture_replay_solver.md)** — per-link distortion validation

## Reports — chronological

| File | Phase | Status | Summary |
|---|---|---|---|
| [01_variant_a_synthetic.md](reports/01_variant_a_synthetic.md) | pre-Phase 0 | superseded by Variant B | Hadamard + uniform quantizer (baseline screening) |
| [02_variant_b_synthetic.md](reports/02_variant_b_synthetic.md) | pre-Phase 0 | ✅ active | Variant B (Haar + Lloyd-Max) on synthetic — matches TurboQuant Table 1 |
| [03_capture_replay_solver.md](reports/03_capture_replay_solver.md) | Phase 0.A | ✅ active | Variant B on real Solver inner adapter — rMSE 0.0093 @ 4-bit |
| [04_kaggle_p100_RETRACTED.md](reports/04_kaggle_p100_RETRACTED.md) | Phase 0.B | ⚠️ retracted | Original in-loop accuracy claims (P100 bf16 fallback collapse) |
| [05_hardware_root_cause.md](reports/05_hardware_root_cause.md) | Phase 0.C/0.D/0.E/0.G/0.H | ✅ active | The hardware/dtype investigation that explains the Phase 0.B retraction |
| [06_variant_b_in_loop_HEADLINE.md](reports/06_variant_b_in_loop_HEADLINE.md) | Phase 0.I/0.J | historical cloud validation | Original Math500 T4 ladder |
| [07_fidelity_sweep_modal.md](reports/07_fidelity_sweep_modal.md) | Tier 2 cloud | historical; legacy KL estimator | A100 controls, depth sweep, and fidelity history |
| [08_local_cross_cell_generalization.md](reports/08_local_cross_cell_generalization.md) | Primary local study | ✅ canonical | RTX 5070 Ti, math/code/medicine + light/scaled, corrected trajectory analysis |

## Design documents

| File | Purpose |
|---|---|
| [design/architecture.md](design/architecture.md) | Phase 0.B architectural plan (patching infrastructure) |
| [design/phase0c_investigation.md](design/phase0c_investigation.md) | Investigation design for the 40pp accuracy gap (resolved by REPORT_05) |

## Operations

| File | Purpose |
|---|---|
| [operations/experiments_log.md](operations/experiments_log.md) | Full inventory of every experiment ever run |
| [operations/external_reproducibility_audit.md](operations/external_reproducibility_audit.md) | External reproduction audit + remaining gaps |

## Figures (matplotlib PNG)

| File | Used by |
|---|---|
| [figures/bit_rate_ladder_n250.png](figures/bit_rate_ladder_n250.png) | REPORT_06 §2.2, README.md |
| [figures/distortion_vs_bits.png](figures/distortion_vs_bits.png) | REPORT_02 §3, REPORT_03 §2 |
| [figures/hardware_dtype_collapse.png](figures/hardware_dtype_collapse.png) | REPORT_05 §15 |
| [figures/sample_variance_n50_vs_n250.png](figures/sample_variance_n50_vs_n250.png) | REPORT_06 §2.3 |

Regenerate with: `.venv/bin/python docs/figures/_generate_figures.py` (matplotlib + numpy required).
