"""Analyze downloaded Kaggle headline-ladder JSON artifacts.

The Kaggle runner writes one JSON per condition:

    phase0g_vb{bits}_n{n_samples}_b{batch_size}.json

This script turns those downloaded artifacts into the machine-readable summary,
paper table, and bit-rate figure for the sampled-decoding headline result. It is
intentionally post-hoc only: no GPU work happens here.

Example:
    .venv/bin/python experiments/variant_b_ladder_t4_kaggle/analysis/analyze_ladder.py \
        --inputs /tmp/rmas_ladder_outputs \
        --n-samples 250 \
        --batch-size 4 \
        --out experiments/variant_b_ladder_t4_kaggle/analysis/results
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "lscr_mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


JSON_RE = re.compile(r"phase0g_vb(?P<bits>\d+)_n(?P<n>\d+)_b(?P<batch>\d+)\.json$")
Z_975 = 1.959963984540054


@dataclass(frozen=True)
class LadderRun:
    bits: int
    n_samples: int
    batch_size: int
    accuracy_pct: float
    correct: int
    return_code: int | None
    patches_logged: int | None
    run_seconds: float | None
    json_path: str

    @property
    def compression_ratio(self) -> float:
        if self.bits == 0:
            return 1.0
        return 32.0 / float(self.bits)


@dataclass(frozen=True)
class LadderRow:
    bits: int
    compression_ratio: float
    accuracy_pct: float
    correct: int
    n_samples: int
    delta_pp: float | None
    wilson_low_pct: float
    wilson_high_pct: float
    z_score: float | None
    p_value: float | None
    patches_logged: int | None
    return_code: int | None
    run_seconds: float | None
    json_path: str


def _json_bits_from_name(path: Path) -> "tuple[int, int, int] | None":
    match = JSON_RE.search(path.name)
    if match is None:
        return None
    return (
        int(match.group("bits")),
        int(match.group("n")),
        int(match.group("batch")),
    )


def _correct_from_accuracy(acc_pct: float, n_samples: int) -> int:
    return int(round((acc_pct / 100.0) * n_samples))


def load_ladder_runs(input_dir: Path) -> list[LadderRun]:
    """Load all phase0g JSON artifacts under ``input_dir``."""
    runs: list[LadderRun] = []
    for path in sorted(input_dir.rglob("phase0g_vb*_n*_b*.json")):
        parsed_name = _json_bits_from_name(path)
        if parsed_name is None:
            continue
        name_bits, name_n, name_batch = parsed_name
        data = json.loads(path.read_text())
        cfg = data.get("config", {})
        bits = int(cfg.get("variant_b_bits", name_bits))
        n_samples = int(cfg.get("n_samples", name_n))
        batch_size = int(cfg.get("batch_size", name_batch))
        acc = data.get("final_accuracy")
        if acc is None:
            raise ValueError(f"{path} has no final_accuracy")
        acc_f = float(acc)
        runs.append(
            LadderRun(
                bits=bits,
                n_samples=n_samples,
                batch_size=batch_size,
                accuracy_pct=acc_f,
                correct=_correct_from_accuracy(acc_f, n_samples),
                return_code=data.get("return_code"),
                patches_logged=data.get("n_patches_logged_from_file"),
                run_seconds=data.get("run_seconds"),
                json_path=str(path),
            )
        )
    return runs


def filter_runs(
    runs: Iterable[LadderRun],
    *,
    n_samples: int | None,
    batch_size: int | None,
    bits: set[int] | None,
) -> list[LadderRun]:
    """Apply the requested n/bits/batch filters and reject duplicate bits."""
    selected = []
    for run in runs:
        if n_samples is not None and run.n_samples != n_samples:
            continue
        if batch_size is not None and run.batch_size != batch_size:
            continue
        if bits is not None and run.bits not in bits:
            continue
        selected.append(run)
    by_bits: dict[int, LadderRun] = {}
    for run in selected:
        if run.bits in by_bits:
            raise ValueError(
                f"duplicate ladder run for bits={run.bits}: "
                f"{by_bits[run.bits].json_path} and {run.json_path}"
            )
        by_bits[run.bits] = run
    return sorted(by_bits.values(), key=lambda r: (-1 if r.bits == 0 else 0, -r.bits))


def wilson_interval(correct: int, n_samples: int, z: float = Z_975) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion, returned in percent."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    p = correct / n_samples
    z2 = z * z
    denom = 1.0 + z2 / n_samples
    center = (p + z2 / (2.0 * n_samples)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n_samples)) / n_samples) / denom
    return (100.0 * (center - half), 100.0 * (center + half))


def two_proportion_z(
    baseline_correct: int,
    baseline_n: int,
    treat_correct: int,
    treat_n: int,
) -> tuple[float, float]:
    """Two-sided pooled two-proportion z-test for binomial outcomes."""
    p0 = baseline_correct / baseline_n
    p1 = treat_correct / treat_n
    pooled = (baseline_correct + treat_correct) / (baseline_n + treat_n)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / baseline_n + 1.0 / treat_n))
    if se == 0.0:
        return 0.0, 1.0
    z = (p1 - p0) / se
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p


def build_rows(runs: list[LadderRun]) -> list[LadderRow]:
    """Build baseline-relative table rows from loaded runs."""
    baseline = next((r for r in runs if r.bits == 0), None)
    if baseline is None:
        raise ValueError("no baseline run found (bits=0)")
    rows = []
    for run in runs:
        lo, hi = wilson_interval(run.correct, run.n_samples)
        if run.bits == 0:
            delta = None
            z = None
            p = None
        else:
            delta = run.accuracy_pct - baseline.accuracy_pct
            z, p = two_proportion_z(
                baseline.correct, baseline.n_samples, run.correct, run.n_samples
            )
        rows.append(
            LadderRow(
                bits=run.bits,
                compression_ratio=run.compression_ratio,
                accuracy_pct=run.accuracy_pct,
                correct=run.correct,
                n_samples=run.n_samples,
                delta_pp=delta,
                wilson_low_pct=lo,
                wilson_high_pct=hi,
                z_score=z,
                p_value=p,
                patches_logged=run.patches_logged,
                return_code=run.return_code,
                run_seconds=run.run_seconds,
                json_path=run.json_path,
            )
        )
    return rows


def _bits_label(row: LadderRow) -> str:
    if row.bits == 0:
        return "baseline\n(fp32)"
    return f"{row.bits}\n({row.compression_ratio:.0f}x)"


def write_summary(rows: list[LadderRow], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    with (out_dir / "summary.json").open("w") as f:
        json.dump([asdict(row) for row in rows], f, indent=2)

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    lines = [
        "# Variant B sampled bit-rate ladder",
        "",
        "Generated from downloaded Kaggle `phase0g_vb*_n*_b*.json` artifacts.",
        "",
        "| bits | compression | accuracy | correct/n | delta vs baseline | Wilson 95% CI | z | p | patches | rc |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        bits = "baseline" if row.bits == 0 else str(row.bits)
        delta = "--" if row.delta_pp is None else f"{row.delta_pp:+.1f} pp"
        z_cell = "--" if row.z_score is None else f"{row.z_score:.3f}"
        p_cell = "--" if row.p_value is None else f"{row.p_value:.3f}"
        patches = "--" if row.patches_logged is None else str(row.patches_logged)
        rc = "--" if row.return_code is None else str(row.return_code)
        lines.append(
            f"| {bits} | {row.compression_ratio:.0f}x | "
            f"{row.accuracy_pct:.1f}% | {row.correct}/{row.n_samples} | "
            f"{delta} | [{row.wilson_low_pct:.1f}, {row.wilson_high_pct:.1f}] | "
            f"{z_cell} | {p_cell} | {patches} | {rc} |"
        )
    lines.append("")
    lines.append("Interpret non-significant p-values as failure to detect degradation, not as proof of equality.")
    (out_dir / "results.md").write_text("\n".join(lines))


def plot_ladder(rows: list[LadderRow], out_path: Path) -> None:
    labels = [_bits_label(row) for row in rows]
    accs = np.array([row.accuracy_pct for row in rows])
    lows = np.array([row.wilson_low_pct for row in rows])
    highs = np.array([row.wilson_high_pct for row in rows])
    yerr = np.vstack([accs - lows, highs - accs])
    x = np.arange(len(rows))

    colors = ["#2c3e50" if row.bits == 0 else "#3498db" for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, accs, yerr=yerr, capsize=6, color=colors,
                  edgecolor="black", linewidth=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("bits per coordinate (compression ratio vs fp32)")
    ax.set_ylabel("math500 accuracy (%)")
    ax.set_title("Sampled bit-rate ladder from downloaded Kaggle artifacts")
    ax.set_ylim(0, 100)
    baseline = next(row for row in rows if row.bits == 0)
    ax.axhline(baseline.accuracy_pct, color="#27ae60", linestyle="--",
               linewidth=1, alpha=0.7, label=f"baseline {baseline.accuracy_pct:.1f}%")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for bar, row in zip(bars, rows):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 3.5,
                f"{row.accuracy_pct:.1f}%\n({row.correct}/{row.n_samples})",
                ha="center", va="bottom", fontsize=9)
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True,
                        help="directory containing downloaded phase0g JSON artifacts")
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/variant_b_ladder_t4_kaggle/analysis/results"))
    parser.add_argument("--n-samples", type=int, default=None,
                        help="only analyze runs with this n_samples")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="only analyze runs with this batch_size")
    parser.add_argument("--bits", type=str, default="0,8,4,2",
                        help="comma-separated bit-rates to include; include 0 for baseline")
    args = parser.parse_args()

    bits = {int(x) for x in args.bits.split(",") if x.strip()} if args.bits else None
    runs = filter_runs(
        load_ladder_runs(args.inputs),
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        bits=bits,
    )
    if not runs:
        raise SystemExit(f"no matching ladder JSONs found under {args.inputs}")
    rows = build_rows(runs)
    write_summary(rows, args.out)
    plot_ladder(rows, args.out / "figures" / "bit_rate_ladder_n250.png")
    print(f"wrote {args.out}/results.md")
    print(f"wrote {args.out}/summary.json")
    print(f"wrote {args.out}/summary.csv")
    print(f"wrote {args.out}/figures/bit_rate_ladder_n250.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
