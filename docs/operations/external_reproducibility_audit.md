# External reproducibility audit — local RTX 5070 Ti study

**Updated:** 2026-06-21

**Primary target:** the four-cell local experiment in REPORT_08

**Historical secondary targets:** Kaggle T4 and Modal A100 runs in REPORT_06/07

## Verdict

The repository now has a coherent clean-clone path from environment setup to all
four local cells and both published analysis tables. The answer-level results are
recomputable immediately from compact committed JSONLs. A fresh researcher with a
compatible 16 GB CUDA GPU can regenerate the complete study with the commands in
`REPRODUCIBILITY.md`.

Two limitations prevent calling the package fully independently reproduced today:

1. the new clean-clone workflow has been unit/dry-run tested in the original
   workspace, but has not yet been executed end-to-end on a second machine/account;
2. raw local logit NPZs are not published, so exact Tier-2 numeric reproduction
   requires roughly 36 GPU-hours of reruns or a separate artifact archive.

Cloud credentials and private cloud volumes are no longer part of the primary
reproduction story.

## Reproducibility matrix

| component | status | evidence / remaining gap |
|---|---|---|
| source revision | reproducible | this repo plus RecursiveMAS pinned to `f95d512` |
| upstream ownership isolation | reproducible | local runner verifies and copies read-only upstream; tests prevent in-place patching |
| Python/CUDA environment | specified | Python 3.12, torch 2.9.0+cu128, exact Python pins |
| checkpoint identities | specified | all HF repo IDs explicit in setup scripts; model file hashes not yet exported |
| GPU smoke test | reproducible | `verify_gpu.py` plus four-sample full-path INT4 capture |
| four experiment cells | reproducible | one canonical `run_cell.py`, fixed layout, fail-fast result validation, manifests |
| answer-level table | reproducible without GPU | committed minimized JSONLs and `compare_cells.py` |
| Tier-2 table | code reproducible | current analyzer is tested; raw NPZ rerun/archive required |
| paper build | reproducible | `main.tex`, bibliography, figures, and Tectonic command |
| independent external rerun | pending | repeat on a second compatible machine |
| citable raw-artifact bundle | pending | publish NPZ/results/manifests outside git |

## Canonical clean-clone workflow

The single operational entrypoint is the root `REPRODUCIBILITY.md`. It specifies:

- exact hardware/software used by the published RTX 5070 Ti run;
- read-only cloning and commit verification of third-party RecursiveMAS;
- exact PyTorch cu128 and Python dependency installation;
- persistent HF cache and run-root configuration;
- checkpoint download and adapter-manifest validation;
- cheap full-path smoke test before committing GPU-hours;
- four `run_cell.py` invocations with measured durations and safe batches;
- deterministic output layout and `cell_manifest.json` contracts;
- answer and Tier-2 analysis commands;
- explicit scientific/reproducibility boundaries.

The local driver does not edit RecursiveMAS. Instrumentation is applied to a
disposable source copy under the condition output and deleted afterward.

## Artifact policy and authority

Committed artifacts contain only configuration/result JSON, minimized correctness
records, summaries, and analysis code. Prompts, generations, traces, cache paths,
machine logs, and NPZ captures are excluded from git.

Evidence authority is:

1. raw run outputs and manifests;
2. compact public result artifacts;
3. REPORT_08;
4. `docs/RESEARCH.md` and the paper;
5. historical cloud reports.

For publication-grade auditability, export the raw local condition directories to a
versioned archive and generate SHA256 manifests with `bin/verify_artifacts.py`. The
archive should include environment metadata (`nvidia-smi`, Python/pip freeze, git
SHAs) but must scrub tokens and personal paths.

## What can be checked from a fresh checkout without a GPU

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python experiments/fidelity_sweep/local_pkg/analysis/compare_cells.py
cd writeup && tectonic main.tex
```

This validates the quantizer/patch/analysis logic, the upstream-copy isolation,
published paired answer statistics, and paper build. It does not recreate sampled
generation or the raw Tier-2 captures.

## Remaining work before a strong external-reproducibility claim

1. Execute `REPRODUCIBILITY.md` from a genuinely clean clone on a second RTX 5070 Ti
   or comparable native-bf16 GPU.
2. Publish a checksumed raw local artifact bundle, especially the eight REF/INT4 NPZ
   captures and their call statistics.
3. Record model snapshot commit IDs or file hashes, not only HF repository names.
4. Extend the generated cell metadata with driver and package-freeze details (GPU,
   torch/CUDA, Python, and repository SHA are already recorded).
5. Regenerate every reported table directly into a machine-readable analysis output
   rather than manually copying values into Markdown/LaTeX.

Until those steps are complete, describe the project as **rerunnable from a documented
local workflow with partially published artifacts**, not as independently reproduced.
