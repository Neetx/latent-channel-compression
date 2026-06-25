# MBPP+ inner/outer link ablation — is the trajectory drift localized to one link type?

**Provenance.** RTX 5070 Ti (16 GB, bf16), RecursiveMAS @ f95d512, MBPP+, n=250, generation
seed 42, quantizer seed 42, T=3, Variant B INT4. In RecursiveMAS Sequential each tier exposes
**8 inner + 8 outer** adapters (inner = same-model recursion adapter; outer = CrossModelAdapter);
the orchestrator [`run_links_ablation.py`](../run_links_ablation.py) quantizes **exactly one link
type at a time** (inner-only ⇒ `_li`, outer-only ⇒ `_lo`). The link-independent greedy REF
(bits=0) and the all-links INT4 are the existing seed-42 captures, **reused** — only the two
single-link INT4 captures per tier were produced here. 4 INT4 runs took ~6.9 h (light inner 70.0,
light outer 65.3, scaled inner 136.6, scaled outer 146.6 min), resumable + detached. Reproduce:
`python ../analysis/links_ablation_analysis.py`. Math500 was **not** ablated (its all-links tier
gap is ~0, so there is nothing to localize there).

## Trajectory divergence (greedy REF vs INT4), by link set

`within 128` is the divergence rate inside the 128-position capture window (first top-1
mismatch); the cumulative `1−S` columns are Kaplan–Meier incidence at positions 10/25/50/100.
The `all` rows reproduce the committed tier-effect numbers exactly (92.8% / 51.2%), so the
single-link rows are directly comparable.

| tier | links | accuracy | divergence within 128 | Δ vs all | 1−S @ 10/25/50/100 |
|---|---|:---:|:---:|:---:|:---:|
| light | REF (fp) | 36.4% | — | — | — |
| light | **all** | 36.4% | 92.8% | — | .248/.500/.748/.912 |
| light | **inner** | 36.4% | 88.8% | −4.0 pp | .236/.472/.700/.852 |
| light | **outer** | 35.6% | 94.4% | +1.6 pp | .216/.468/.700/.904 |
| scaled | REF (fp) | 74.4% | — | — | — |
| scaled | **all** | 72.4% | 51.2% | — | .020/.154/.321/.554 |
| scaled | **inner** | 74.8% | 50.4% | −0.8 pp | .032/.172/.347/.535 |
| scaled | **outer** | 72.8% | 52.0% | +0.8 pp | .032/.168/.339/.545 |

## Findings

1. **The trajectory drift is NOT localized to one link type.** On *both* tiers, inner-only and
   outer-only each independently reproduce essentially the entire all-links divergence (light:
   inner 88.8%, outer 94.4% vs all 92.8%; scaled: inner 50.4%, outer 52.0% vs all 51.2% — every
   single-link value is within ≤4 pp of the all-links value, and the cumulative hazard curves
   overlap). Quantizing *either* channel alone is sufficient to reroute the greedy path.
2. **This refutes the pre-registered "outer link is the failure point" hypothesis.** Neither link
   type is the unique culprit; the effect saturates with a single channel, so the two are
   redundant rather than separable at INT4.
3. **The tier robustness gap is intrinsic to model scale and present within each channel.** Scaled
   is ~40 pp more trajectory-robust than light for inner-only (50.4 vs 88.8) *and* for outer-only
   (52.0 vs 94.4) — mirroring the all-links gap. The light↔scaled effect is not carried by one
   link type.
4. **The accuracy/trajectory dissociation holds per-link, with one asymmetry.** Quantizing a
   single channel leaves the *answer* almost untouched while the *trajectory* moves wholesale.
   Inner-only is the most answer-preserving (light 36.4 = REF; scaled 74.8 ≈ REF 74.4), whereas
   the small all-links accuracy cost on scaled (−2.0 pp) is carried mainly by the **outer** links
   (outer-only −1.6 pp, inner-only +0.4 pp). So if anything the outer link marginally perturbs the
   answer — but both channels saturate the trajectory equally.

## Limitations

- Single generation seed (42), single quantizer rotation (42), single problem subset/order, INT4
  only. "Either channel is sufficient" is a **saturation** statement at INT4; a finer bit rate
  (e.g. 6/8-bit) might begin to separate the two channels' contributions.
- MBPP+ only (the tier-effect cell). Math500 was not ablated.
- Same caveat as the headline result: this is a behavioural divergence count, **not** a
  teacher-forced, position-aligned mechanism test.

Raw NPZ captures and logs are not committed; regenerate with `run_links_ablation.py` (inner/outer
INT4) — the REF and all-links captures come from the original fidelity runs.
