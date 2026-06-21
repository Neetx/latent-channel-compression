# fidelity_sweep — paired REF vs INT4 fidelity instrumentation

This experiment **adds continuous fidelity metrics alongside the headline accuracy**
measurement and **sweeps the channel-traversal count T = num_recursive_rounds** to
detect distributional drift that the boxed-answer accuracy metric hides.

It is **measurement only** — the model and the Variant B quantizer are untouched
(AGENTS.md invariant 5). The kernel injects instrumentation around the existing
pipeline; the quantizer math is the same `src/quantizers/turboquant_honest.py` used
by the headline experiment.

## What it measures

**Tier 1 — channel fidelity** (per Adapter / CrossModelAdapter forward, INT4 run):
- cosine similarity of (REF-would-receive, actually-received-after-quant) per call
- relative L2 ‖Δ‖₂ / ‖x‖₂ and norm ratio
- recorded via `src/adapters/patch.py:QuantStats` with `record=True`

**Tier 2 — logit-level distributional fidelity at egress** (paired REF vs INT4):
- per-decode-position **MSE / KL(p_REF ‖ p_INT) / JS** over the union of the captured
  top-K supports, with a residual-tail correction that does not double-count union tokens
- **per-problem correctness** for a paired design (upstream `--result_jsonl`)
- **paired bootstrap Δacc + TOST equivalence test** (ε=2pp pre-specified, α=0.05) →
  a formal EQUIVALENT / NOT_EQUIVALENT / INCONCLUSIVE verdict per T

All distortion/divergence math runs in fp32 regardless of pipeline dtype.

## How Tier 2 capture works (so the next maintainer doesn't have to reverse-engineer it)

1. The kernel monkey-patches `transformers.generation.utils.GenerationMixin.generate`
   to force `output_scores=True` on every call site, captures the top-K logits +
   indices per generated step (capped at `MAX_LOGIT_POSITIONS`), and returns the raw
   `sequences` tensor so all upstream call sites stay backward-compatible.
2. Both REF (bits=0) and INT4 (bits=4) use **greedy decoding + identical seed +
   identical data + identical batching**. Only the fixed primary calls, one per input
   batch, are paired positionally. Conditional answer-retry calls are explicitly
   excluded because their presence can differ after the trajectories diverge.
   `analyze.py` truncates each retained pair to `min(T_REF, T_INT)` positions and
   reports the number of excluded calls.
3. Per-problem correctness comes from upstream's `--result_jsonl`. **Sequential-Light
   runs with `num_rollouts=1`, which produces a FLAT record** (`correct` at the top
   level, no `rollouts` key). The parser handles both that and the nested
   `num_rollouts>1` schema, and skips the trailing `type=="summary"` row.

> ⚠️ Two non-obvious correctness requirements, both unit-tested
> (`tests/test_fidelity_kernel.py`):
> - The driver **propagates `VARIANT_B_BITS` / `CAPTURE_MODE` / `TOPK_LOGITS` /
>   `MAX_LOGIT_POSITIONS` into the subprocess env**. The injected head reads them
>   from `os.environ` in the child — without propagation the quantizer injection and
>   logit capture silently no-op (REF and INT4 would be identical).
> - The JSONL parser must read the FLAT schema; a nested-only parser would yield
>   `correct=None` for every problem → a false EQUIVALENT verdict.

## Sweep design

| Axis | Values | Mapping |
|---|---|---|
| **VARIANT_B_BITS** (paired condition) | `{0, 4}` | 0 = REF (BF16/FP32 channel, no quant). 4 = "INT4 analog" = 4-bit Lloyd-Max-Gaussian per coordinate after Haar rotation. |
| **T** (channel-traversal count) | `{1, 2, 3, 4}` | upstream `--num_recursive_rounds` (default 3). Patched as the argparse default per push. |
| Decoding | greedy (`CAPTURE_MODE=1` removes `--do_sample`) | makes the two runs exactly paired per problem. |
| Seed | 42 | deterministic. |
| n_samples | 50 (smoke) → 250 (final) | start small to validate end-to-end. |
| batch_size | 4 (drop to 2 if `output_scores` OOMs — see note) | fp32 on T4 16GB. |
| dtype | float32 | T4 lacks native bf16; auto dtype collapses the pipeline (REPORT 05 §15). |

Cost: 4 T values × 2 conditions = **8 paired runs** at ~95 min each on T4 fp32 b=4 =
~13h Kaggle quota for n=50. Free tier allows 2 concurrent GPU sessions → ~8h wallclock.

**Memory note.** Forcing `output_scores=True` makes `generate()` retain per-step
logits on the GPU until it returns (≈ `max_new_tokens × V × B × 4` bytes, transient).
On a 16 GB T4 with the fp32 constellation this is borderline at long generations. If a
kernel OOMs, push with `batch_size=2`; `MAX_LOGIT_POSITIONS` only bounds the CPU-side
dump, not the transient GPU retention.

## How to run

There are **three backends** sharing the tested instrumentation functions in
`kernel_pkg/fidelity_kernel.py`:

- **Local RTX 5070 Ti bf16** (`local_pkg/`) — the primary four-cell study and
  reproduction path. It treats RecursiveMAS as read-only and patches a disposable
  source copy. Start with [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md) and
  [`local_pkg/README.md`](local_pkg/README.md).
- **Modal A100 fp32** (`modal_pkg/`) — historical independent controls/depth sweep.
- **Kaggle T4 fp32** (`kernel_pkg/`) — historical independent T4 path.

### Historical backend — Modal A100 fp32

Driver: `modal_pkg/fidelity_modal.py`. A100-40GB runs fp32 natively (~15× faster than
the T4) and avoids both the bf16 collapse and the Phase 0.F cast artifact. The 9 GB
checkpoint cache (`rmas-hf-cache` volume) is reused — no re-download.

**Always launch long runs with `--detach`.** A plain `modal run` ties the app to the
local client; on disconnect Modal stops the containers. `--detach` keeps the run
alive server-side, and the function commits its outputs to the `rmas-fidelity-out`
volume at the end — so you recover results by reading the volume, even if the client
disconnected. Detached mode only keeps the *last triggered* function alive, so launch
REF and INT4 as **two separate single-function `::main` runs** (not the `::sweep`
fan-out) for a disconnect-safe paired run.

```bash
# tiny validation (~$0.2): INT4 exercises the full path incl. the quantizer
modal run experiments/fidelity_sweep/modal_pkg/fidelity_modal.py::main \
    --bits 4 --t 3 --n-samples 8 --batch-size 4 --topk 256 --maxpos 128

# a disconnect-safe paired comparison at depth T=3 (each ~75-90 min on A100, n=250)
modal run --detach experiments/fidelity_sweep/modal_pkg/fidelity_modal.py::main --bits 0 --t 3 --n-samples 250 --batch-size 8
modal run --detach experiments/fidelity_sweep/modal_pkg/fidelity_modal.py::main --bits 4 --t 3 --n-samples 250 --batch-size 8

# recover from the volume (per-config subdirs keep same-named NPZs from colliding) + analyze
modal volume get rmas-fidelity-out vb0_T3_n250 /tmp/fid_outputs/
modal volume get rmas-fidelity-out vb4_T3_n250 /tmp/fid_outputs/
.venv/bin/python bin/verify_artifacts.py /tmp/fid_outputs \
    --manifest /tmp/fid_outputs/SHA256SUMS.json
.venv/bin/python experiments/fidelity_sweep/analysis/analyze.py \
    --inputs /tmp/fid_outputs \
    --logit-dir /tmp/fid_outputs \
    --out experiments/fidelity_sweep/analysis/results
```

The `::sweep` entrypoint fans out all `(T × {REF,INT4})` at once — convenient when you
will stay connected, but not disconnect-safe under `--detach`.

Budget pacing: at ~$1/day, one paired (REF+INT4) comparison per day. Runs are
deterministic (seed=42, greedy), so REF today and INT4 tomorrow pair exactly by
`sample_idx`. Extend across T ∈ {1,2,4} and to n=50/250 as budget allows.

### Backend B — Kaggle T4 fp32 (free, when weekly quota is available)

### Prerequisites

1. `lqc-src-bundle` Kaggle dataset already exists (from the headline experiment). It
   is unchanged for this sweep — no new `src/` files were added.
2. Replace `<YOUR_KAGGLE_USERNAME>` in any generated push slug (the push helper writes
   it into the per-kernel `kernel-metadata.json`).

### Validate before spending quota (recommended)

The kernel's risky logic (the 9 surgical patches + the JSONL parser) is unit-tested
against the **real cloned upstream** with zero GPU:

```bash
git clone https://github.com/RecursiveMAS/RecursiveMAS.git external/RecursiveMAS  # if absent
git -C external/RecursiveMAS checkout f95d512017fb713e9ac519248fbfd3d270dafd68
.venv/bin/python -m pytest tests/test_fidelity_kernel.py tests/test_fidelity_analyze.py -v
```

Then optionally do a **1-kernel dry-run** (REF, T=1, n=10, b=2) to confirm the Kaggle
path end-to-end (~30 min) before committing the full 8-kernel budget:

```bash
./bin/push_fidelity_kernel.sh 0 1 10 2
```

### Push the full sweep

```bash
for T in 1 2 3 4; do
    ./bin/push_fidelity_kernel.sh 0 $T 50 4   # REF
    ./bin/push_fidelity_kernel.sh 4 $T 50 4   # INT4
done
```

### Download results — ONE subdir per kernel

Each kernel emits `fidelity_logits.npz` and `fidelity_call_stats.json` with the SAME
filename, so they must land in separate dirs. Use the `vb{bits}_T{T}/` convention that
`analyze.py` expects for the logit NPZs:

```bash
mkdir -p /tmp/fid_outputs
for T in 1 2 3 4; do
    ./bin/kaggle kernels output "<YOUR_KAGGLE_USERNAME>/rmas-fid-ref-T${T}-n50" -p /tmp/fid_outputs/vb0_T${T}/
    ./bin/kaggle kernels output "<YOUR_KAGGLE_USERNAME>/rmas-fid-4-T${T}-n50"   -p /tmp/fid_outputs/vb4_T${T}/
done
```

### Analyze

Verify artifacts first:

```bash
.venv/bin/python bin/verify_artifacts.py /tmp/fid_outputs \
    --manifest /tmp/fid_outputs/SHA256SUMS.json
```

This is not ceremonial: interrupted `modal volume get` downloads can leave a
same-named `fidelity_logits.npz` on disk with bad CRC. The verifier parses JSON,
tests NPZ/ZIP CRCs, and records SHA256 for every artifact.

`--inputs` is searched recursively for the per-kernel JSONs; `--logit-dir` points at
the same parent and finds `vb{bits}_T{T}/fidelity_logits.npz`:

```bash
.venv/bin/python experiments/fidelity_sweep/analysis/analyze.py \
    --inputs /tmp/fid_outputs \
    --logit-dir /tmp/fid_outputs \
    --eps-pp 2.0 \
    --out experiments/fidelity_sweep/analysis/results
```

Produces `results/results.md` (3 tables + verdict guide), `results/figures/*.png`
(6 plots), and `results/raw.npz`. Omit `--logit-dir` to skip Tier 2 logit metrics
(Table 3) and produce only accuracy + channel fidelity.

Analyze one comparison layout at a time. The analysis key is `(bits, T)`, so a
single input directory must not contain multiple runs with the same `(bits, T)`
such as `vb4_T3_n50`, `vb4_T3_n250`, `vb4_T3_n250_inner`, and
`vb4_T3_n250_outer`. The script fails loudly on such duplicates; create a clean
parent directory for each comparison or pass `--allow-duplicate-overwrite` only
for deliberate debugging.

## Determinism guarantees

Every public function in `src/metrics/{channel_fidelity,logit_metrics,bootstrap}.py`
and the analysis bootstraps take an explicit `seed`. The upstream pipeline uses
`seed=42`; the greedy switch removes the only sampling-time RNG. Reproducing means
re-running the same `push_fidelity_kernel.sh` calls and the same `analyze.py` command.

## Files in this experiment

| Path | Purpose |
|---|---|
| `kernel_pkg/fidelity_kernel.py` | Kaggle kernel. Pure, testable functions (`patch_run_py`, `patch_inference_mas`, `parse_per_problem_jsonl`) + a `main()` holding all side effects. Applies 6 run.py patches + 3 inference_mas.py patches, runs `python run.py`, dumps stats/logits/per-problem JSON. |
| `kernel_pkg/kernel-metadata.json` | Metadata template (the push helper overwrites the slug + dataset_sources per run). |
| `analysis/analyze.py` | Post-hoc; consumes downloaded JSON + NPZ, pairs primary calls only, expands the top-K union with corrected residual-tail accounting, and builds `results.md` + plots + raw.npz. |
| `analysis/results/` | Generated outputs (gitignored if large). |
| `../../tests/test_fidelity_kernel.py` | Validates the patches against real upstream + the JSONL parser (both schemas). |
| `../../tests/test_fidelity_analyze.py` | Validates correctness alignment, union expansion, logit metrics from synthetic NPZ, and end-to-end aggregation. |
