#!/usr/bin/env python3
"""Corrected Tier-2 trajectory analysis for locally generated NPZ captures.

Pass the same ``LCC_RUN_ROOT`` used by ``run_cell.py``. Only fixed primary batches
are paired; conditional answer-retry calls are excluded. The reported divergence is
right-censored at the capture window and KL/JS are top-K matched-prefix estimates.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
ANALYZE = REPO / "experiments" / "fidelity_sweep" / "analysis" / "analyze.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("fidelity_analyze", str(ANALYZE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_dir(root: Path, style: str, dataset: str, bits: int, batch: int) -> Path:
    """Resolve the canonical run_cell layout, with a legacy flat-root fallback."""
    tag = f"{dataset}_vb{bits}_T3_n250_b{batch}_auto"
    canonical = root / f"sequential_{style}_{dataset}" / tag
    if canonical.exists():
        return canonical
    flat = root / tag
    if flat.exists():
        return flat
    if dataset == "math500":
        legacy = root / f"vb{bits}_T3_n250_b{batch}_auto"
        if legacy.exists():
            return legacy
    return canonical


def channel_cosine(callstats: Path) -> float | None:
    try:
        data = json.loads(callstats.read_text())
        cosines = [
            stat["cosine"]["mean"] for stat in data.get("per_adapter", [])
            if isinstance(stat.get("cosine"), dict) and "mean" in stat["cosine"]
        ]
        return float(np.mean(cosines)) if cosines else None
    except Exception:
        return None


def ci95_mean(values, boot: int = 10_000, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(boot, values.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default=os.environ.get("LCC_RUN_ROOT", str(Path.home() / "lcc" / "runs")),
        help="root populated by run_cell.py (default: LCC_RUN_ROOT or ~/lcc/runs)",
    )
    args = parser.parse_args()
    root = Path(args.run_root)
    analyzer = load_analyzer()

    cells = [
        ("math500 / light", "light", "math500", 2),
        ("math500 / scaled", "scaled", "math500", 1),
        ("mbppplus / light", "light", "mbppplus", 2),
        ("mbppplus / scaled", "scaled", "mbppplus", 1),
        ("medqa / light", "light", "medqa", 2),
    ]

    print(f"run root: {root}")
    print(f"{'cell':18} {'div.rate':>8} {'prefix':>8} {'KL nats (95% CI)':>22} "
          f"{'JS':>7} {'chan_cos':>9} {'n':>4} {'window':>6}")
    print("-" * 102)
    missing = 0
    for label, style, dataset, batch_size in cells:
        refd = run_dir(root, style, dataset, 0, batch_size)
        intd = run_dir(root, style, dataset, 4, batch_size)
        rnpz, inpz = refd / "fidelity_logits.npz", intd / "fidelity_logits.npz"
        if not (rnpz.is_file() and inpz.is_file()):
            missing += 1
            print(f"{label:18} MISSING ({rnpz} | {inpz})")
            continue

        primary_batches = (250 + batch_size - 1) // batch_size
        metrics = analyzer.compute_logit_metrics_pair(
            rnpz, inpz, max_batches=primary_batches
        )
        kl = metrics["per_problem_kl"]
        kl_mean = float(kl.mean()) if kl.size else float("nan")
        lo, hi = ci95_mean(kl)
        js = metrics["per_problem_js"]
        js_mean = float(js.mean()) if js.size else float("nan")
        cosine = channel_cosine(intd / "fidelity_call_stats.json")
        cosine_text = f"{cosine:.4f}" if cosine is not None else "n/a"
        print(
            f"{label:18} {metrics['divergence_rate']*100:7.1f}% "
            f"{metrics['mean_prefix_len']:8.1f} {kl_mean:7.3f} [{lo:.3f},{hi:.3f}] "
            f"{js_mean:7.3f} {cosine_text:>9} {metrics['n_items']:4d} "
            f"{metrics['max_positions_seen']:6d}"
        )
        print(
            f"  primary batches={metrics['n_batches_paired']}; excluded retries "
            f"REF={metrics['n_retry_batches_ref_excluded']} "
            f"INT4={metrics['n_retry_batches_int_excluded']}"
        )

    if missing:
        print(f"\n{missing} cell(s) missing raw NPZ captures. Run those cells first.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
