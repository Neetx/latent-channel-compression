# Write-up

`main.tex` is the current technical paper and `main.pdf` is its compiled form.
The author is Antonio Pastorelli.

## Build

```bash
cd writeup
tectonic main.tex
```

With a full TeX Live installation:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Evidence map

| paper result | canonical source |
|---|---|
| synthetic and real-channel distortion | [REPORT_02](../docs/reports/02_variant_b_synthetic.md), [REPORT_03](../docs/reports/03_capture_replay_solver.md) |
| primary four-cell local answer and corrected Tier-2 results | [REPORT_08](../docs/reports/08_local_cross_cell_generalization.md) |
| independent historical cloud Math500 ladder | [REPORT_06](../docs/reports/06_variant_b_in_loop_HEADLINE.md) |
| cloud controls and historical fidelity | [REPORT_07](../docs/reports/07_fidelity_sweep_modal.md) |
| hardware/dtype failure analysis | [REPORT_05](../docs/reports/05_hardware_root_cause.md) |

The local Tier-2 table supersedes the first local calculation that paired retry calls
and double-counted some residual tail mass. Historical cloud KL values remain in
REPORT_07 for auditability and require reanalysis from the raw NPZs before direct
comparison with corrected local KL values.

See [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for run and analysis commands.
Figures are PNG and can be replaced by vector versions for final submission.
