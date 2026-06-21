# External reproducibility audit

Date: 2026-06-03

Question audited: can an external researcher start from this workspace, rerun the
experiments, fetch the outputs, analyze them, and recover the reported paper
results without relying on chat history?

Short answer after remediation: **the local replication workflow is now much
closer to external-researcher standard, but a full external package still needs
public raw artifacts or a fresh cloud rerun**.

> **2026-06-21 update.** A portable local backend, compact four-cell correctness
> artifacts, REPORT_08, and a corrected primary-call/top-K analyzer are now public.
> Raw logit NPZs are still intentionally absent, so answer analyses are reproducible
> from the repository while Tier-2 numeric reproduction requires regenerated or
> separately archived captures. The historical test count below is a dated audit
> observation, not the current suite size.

The repository has strong components: pinned local analysis dependencies,
unit-tested patching/analysis code, Kaggle and Modal runners, and detailed
reports. However, the end-to-end replication path is not yet packaged as a
single auditable workflow. The largest gaps are the lack of a raw-artifact
manifest/checksums, no parser for the headline Kaggle ladder JSONs, hardcoded
publication figures, and incomplete instructions for fetching/analyzing all
Modal result variants without mixing runs.

## What is reproducible now

### Local code and analysis tests

The core local checks pass:

```bash
.venv/bin/python -m pytest \
    tests/test_fidelity_kernel.py \
    tests/test_fidelity_analyze.py \
    tests/test_fidelity_metrics.py \
    tests/test_turboquant_honest.py \
    tests/test_patch.py -q
```

Observed result:

```text
70 passed, 1 skipped, 1 warning in 46.31s
```

This validates the risky regex patching, JSONL parsing, paired analysis helpers,
quantizer behavior, and adapter patching. It does not prove the cloud reruns will
finish or match, but it means the local machinery is not obviously broken.

### Fidelity sweep rerun path

`experiments/fidelity_sweep/README.md` is the strongest reproducibility document.
It describes:

- Modal A100 and Kaggle T4 backends;
- run commands for REF/INT4;
- volume/output retrieval;
- separate output directories per run;
- `analyze.py` invocation;
- unit tests to run before spending GPU quota.

The committed `experiments/fidelity_sweep/analysis/results/` currently contains
the n=50 T-sweep output:

- `results.md`
- `raw.npz`
- six figures

`raw.npz` contains T=1..4, REF/INT4, n=50 values only. It does not contain the
powered n=250 all-link result or selective inner/outer runs.

### Modal artifacts fetched during this audit

Clean Modal retry artifacts exist locally at:

```text
/private/tmp/modal_fetch_lscr_retry
```

CRC status for the retry NPZs:

```text
vb0_T3_n250: OK
vb0_T3_n250_rep2: OK
vb4_T3_n250: OK
vb4_T3_n250_inner: OK
vb4_T3_n250_outer: OK
```

Useful retry analysis outputs:

```text
/private/tmp/lscr_analysis_n250_retry/results.md
/private/tmp/lscr_analysis_sweep_n50/results.md
```

Important: these files are in `/private/tmp`, not part of the repository and not
available to an external researcher unless archived.

## Remediation implemented after this audit

The following gaps identified below have been addressed in the workspace:

- Added root-level `REPRODUCIBILITY.md` as the single run/fetch/verify/analyze
  entrypoint.
- Added `bin/verify_artifacts.py` for JSON parsing, NPZ CRC checks, and SHA256
  manifests.
- Added `experiments/variant_b_ladder_t4_kaggle/analysis/analyze_ladder.py` to
  regenerate the sampled headline ladder table and figure from downloaded Kaggle
  JSON artifacts.
- Updated `docs/figures/_generate_figures.py` to read the generated ladder
  `summary.json` when present.
- Hardened `experiments/fidelity_sweep/analysis/analyze.py` against duplicate
  `(bits, T)` inputs and corrupt NPZ files.
- Added `fidelity_sweep` to `experiments/README.md` and documented artifact
  verification before analysis.
- Pinned RecursiveMAS to commit
  `f95d512017fb713e9ac519248fbfd3d270dafd68` in the cloud runners.
- Regenerated `experiments/variant_b_ladder_t4_kaggle/dataset_pkg/lqc_src.tar.gz`
  from the current `src/` snapshot and wrote
  `experiments/variant_b_ladder_t4_kaggle/dataset_pkg/SHA256SUMS.json`.

## What is not externally reproducible yet

### 1. Public raw artifacts are still absent

The repo now has a single replication entrypoint, but raw Modal/Kaggle artifacts
are still not published in a citable archive. An external researcher can rerun the
cloud jobs, but cannot fetch the original private Modal/Kaggle outputs.

### 2. Instructions remain distributed for background context

Detailed rationale is still spread across:

- `README.md`
- `experiments/README.md`
- `experiments/fidelity_sweep/README.md`
- `docs/reports/05_*`
- `docs/reports/06_*`
- `docs/reports/07_*`
- `AGENTS.md`

`REPRODUCIBILITY.md` is now the operational entrypoint; the files above remain
background evidence and rationale.

### 3. Headline Kaggle ladder post-hoc analysis now exists

The headline runner writes JSON files:

```text
phase0g_vb{bits}_n{N}_b{B}.json
```

`analyze_ladder.py` now reads these artifacts and emits `results.md`,
`summary.csv`, `summary.json`, and `figures/bit_rate_ladder_n250.png`.

### 4. Publication figures still have fallback values

`docs/figures/_generate_figures.py` hardcodes the key headline values:

```python
accs = [75.2, 78.4, 76.8, 75.2]
correct = [188, 196, 192, 188]
```

It now reads the generated ladder `summary.json` when present. The hardcoded
values remain only as a fallback to keep historical docs buildable without raw
artifacts.

### 5. Raw cloud artifacts are absent from the repo

The README explicitly says raw kernel output logs are not in the repo. That is
fine for repo size, but then an external replication package needs either:

- a public artifact archive with checksums; or
- complete commands to rerun all cloud jobs and fetch outputs.

For Modal, the current volume names (`rmas-fidelity-out`, `rmas-hf-cache`) are
account-local. External researchers cannot fetch existing outputs unless the
artifacts are exported separately.

### 6. Checksum/integrity workflow now exists

This audit found that interrupted `modal volume get` downloads can leave
CRC-corrupted `.npz` files while filenames still exist.

Use `bin/verify_artifacts.py` to parse JSON, test NPZ CRCs, and write SHA256
manifests before analysis.

### 7. `analyze.py` no longer silently overwrites same-key runs

`experiments/fidelity_sweep/analysis/analyze.py` keys runs by `(bits, T)`.

If a directory contains, for example:

- `vb4_T3_n50`
- `vb4_T3_n250`
- `vb4_T3_n250_inner`
- `vb4_T3_n250_outer`

then later recursive loads can overwrite earlier entries for `(4,3)`.

The script now fails loudly on duplicate `(bits,T)` keys unless
`--allow-duplicate-overwrite` is explicitly passed.

### 7. `analyze.py` produced partial outputs during this audit

When run on fetched n=250 artifacts, `analyze.py` generated `results.md` but exited
before writing figures and `raw.npz` in this local environment. Setting
`MPLCONFIGDIR=/private/tmp/mplconfig_lscr` removed the Matplotlib home-cache warning
but did not change the partial-output behavior.

The n=50 committed results show the script can produce all outputs in some prior
environment. Still, an external package should make this robust and test it with
the real downloaded artifact layout.

### 9. `experiments/README.md` now lists `fidelity_sweep`

The active-experiment table lists:

- `distortion_validation`
- `solver_diagnostic`
- `baseline_a100_modal`
- `variant_b_ladder_t4_kaggle`

but not `fidelity_sweep`, even though it supports the greedy TOST, trajectory
divergence, REF-vs-REF control, depth sweep, and inner/outer localization claims.

### 10. Kaggle source bundle has been regenerated

`experiments/variant_b_ladder_t4_kaggle/dataset_pkg/lqc_src.tar.gz` differs from
the current `src/adapters/patch.py`. The difference adds global stats registry
helpers:

- `register_stats`
- `collected_stats`
- `reset_stats`

The bundle has been regenerated from the current `src/` snapshot and checksumed.

### 11. Cloud dependency pinning is split

Local dependencies are pinned in `requirements.txt`, but cloud images pin their own
dependencies inside the Modal/Kaggle scripts. That is workable, but the external
workflow should explicitly state that:

- local analysis uses `requirements.txt`;
- Modal uses the image in `experiments/fidelity_sweep/modal_pkg/fidelity_modal.py`;
- Kaggle installs dependencies inside each kernel script;
- upstream RecursiveMAS is cloned from GitHub at HEAD unless a commit SHA is pinned.

The cloud runners now checkout the pinned RecursiveMAS commit listed above.

## Remaining fixes before claiming full external reproducibility

1. Export raw Modal/Kaggle artifacts to a public or citable archive, or require a
   fresh cloud rerun.
2. Add expected SHA256 manifests for any archived artifacts.
3. Add a final smoke command that starts from downloaded artifacts and produces:
    paper tables, figures, and a machine-readable summary.

## Current verdict

The project now has a concrete external workflow for rerunning and analyzing the
experiments. The remaining blocker for a fully self-contained external
replication package is artifact availability: the original raw cloud outputs are
still private/local unless exported.
