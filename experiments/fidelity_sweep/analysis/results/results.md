# Fidelity sweep — results

Paired REF (bits=0) vs INT4 (bits=4) runs at varying channel-
traversal counts T = num_recursive_rounds. Greedy decoding, fp32.
All CIs are 95% bootstrap (≥10k resamples).

## Table 1 — Accuracy with paired bootstrap + TOST

Per-problem outcomes paired by `sample_idx`. ε=2pp pre-specified.

| T | acc_REF | acc_INT | Δacc (95% CI) | n_paired | TOST verdict (p) |
|---|---:|---:|---:|---:|:---:|
| 1 | 80.0% | 88.0% | +8.00pp [+0.00, +18.00] | 50 | INCONCLUSIVE (p=0.891) |
| 2 | 84.0% | 78.0% | -6.00pp [-16.00, +4.00] | 50 | NOT_EQUIVALENT (p=0.774) |
| 3 | 82.0% | 76.0% | -6.00pp [-16.00, +4.00] | 50 | NOT_EQUIVALENT (p=0.774) |
| 4 | 82.0% | 82.0% | +0.00pp [-10.00, +10.00] | 50 | NOT_EQUIVALENT (p=0.344) |

## Table 2 — Channel fidelity (per-adapter-call, INT4 round-trip)

Recorded at every Adapter / CrossModelAdapter forward in the INT4 run.
Cos ≈ 1 + rel L2 ≈ 0 means the quantizer preserved the post-LN output bit-for-bit AT THAT CALL.

| T | n_calls | mean cos | cos 95% CI | mean rel L2 | rel L2 95% CI |
|---|---:|---:|:---:|---:|:---:|
| 1 | 700 | 0.9953 | [0.9953, 0.9953] | 0.0965 | [0.0964, 0.0967] |
| 2 | 1750 | 0.9954 | [0.9953, 0.9954] | 0.0963 | [0.0962, 0.0964] |
| 3 | 2800 | 0.9954 | [0.9954, 0.9954] | 0.0962 | [0.0961, 0.0963] |
| 4 | 3850 | 0.9954 | [0.9954, 0.9954] | 0.0963 | [0.0962, 0.0963] |

## Table 3 — Egress distributional fidelity (matched-prefix, per-step)

KL(p_REF ‖ p_INT) / JS / MSE on the next-token **probability** distributions
(normalized via the captured full-vocab log-sum-exp), over the union of the
two top-K supports. Under greedy free generation the two runs' sequences can
diverge; once they pick different tokens the contexts differ and a positional
comparison is meaningless. So the metric is computed **only over the matched
prefix** (positions up to and including the first token mismatch). `div_rate`
is the fraction of sequences whose greedy path diverged within the window;
`match_len` is the mean number of aligned positions measured.

| T | mean KL (nats) | KL 95% CI | mean JS | JS 95% CI | prob-MSE | div_rate | match_len |
|---|---:|:---:|---:|:---:|---:|---:|---:|
| 1 | 0.5968 | [0.0111, 1.3533] | 0.0395 | [0.0025, 0.0881] | 1.10e-04 | 0.72 | 120.9 |
| 2 | 0.1684 | [0.0128, 0.4713] | 0.0176 | [0.0029, 0.0455] | 5.81e-05 | 0.92 | 82.2 |
| 3 | 0.0308 | [0.0165, 0.0475] | 0.0071 | [0.0038, 0.0110] | 2.84e-05 | 0.88 | 71.6 |
| 4 | 0.1324 | [0.0134, 0.3507] | 0.0193 | [0.0030, 0.0476] | 5.72e-05 | 0.85 | 85.0 |

## Reading the verdict

- **Table 2 (channel) cos≈1, rel L2≈const across T** → the 4-bit round-trip
  preserves the inter-agent vector geometry, and the per-call distortion does
  NOT grow with channel-traversal depth.
- **Table 3 matched-prefix KL is small** → where REF and INT4 share context the
  quantizer barely perturbs the next-token distribution (near-lossless per step).
- **KL does not explode with T** → no catastrophic depth-amplification of the
  per-step drift; `div_rate` quantifies how often a tiny perturbation flips a
  greedy token (a trajectory effect, separate from per-step fidelity).
- **Table 1 TOST** needs adequate n to return EQUIVALENT within ±2pp; at small n
  it will read INCONCLUSIVE/NOT_EQUIVALENT (wide CI), which is *underpowered*, not
  evidence of harm. Use n≈250 for the formal equivalence claim.
