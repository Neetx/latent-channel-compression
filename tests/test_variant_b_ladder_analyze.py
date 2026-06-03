"""Tests for the sampled bit-rate ladder post-hoc analyzer."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYZE_PATH = (
    ROOT / "experiments" / "variant_b_ladder_t4_kaggle" / "analysis" / "analyze_ladder.py"
)


def _load_analyze_ladder():
    spec = importlib.util.spec_from_file_location("ladder_analyze", ANALYZE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def az():
    return _load_analyze_ladder()


def _write_run(root: Path, bits: int, acc: float, *, n: int = 250, batch: int = 4) -> Path:
    path = root / f"phase0g_vb{bits}_n{n}_b{batch}.json"
    path.write_text(json.dumps({
        "config": {
            "variant_b_bits": bits,
            "n_samples": n,
            "batch_size": batch,
        },
        "final_accuracy": acc,
        "return_code": 0,
        "n_patches_logged_from_file": 0 if bits == 0 else 16,
        "run_seconds": 10.0,
    }))
    return path


def test_ladder_summary_from_json_artifacts(az, tmp_path):
    _write_run(tmp_path, 0, 75.2)
    _write_run(tmp_path, 8, 78.4)
    _write_run(tmp_path, 4, 76.8)
    _write_run(tmp_path, 2, 75.2)

    runs = az.filter_runs(
        az.load_ladder_runs(tmp_path),
        n_samples=250,
        batch_size=4,
        bits={0, 8, 4, 2},
    )
    rows = az.build_rows(runs)
    assert [r.bits for r in rows] == [0, 8, 4, 2]
    assert [r.correct for r in rows] == [188, 196, 192, 188]
    assert rows[1].delta_pp == pytest.approx(3.2)
    assert rows[1].p_value > 0.3

    out = tmp_path / "analysis"
    az.write_summary(rows, out)
    az.plot_ladder(rows, out / "figures" / "bit_rate_ladder_n250.png")
    assert (out / "results.md").exists()
    assert (out / "summary.csv").exists()
    assert (out / "summary.json").exists()
    assert (out / "figures" / "bit_rate_ladder_n250.png").exists()


def test_ladder_rejects_duplicate_bits(az, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_run(a, 4, 76.8)
    _write_run(b, 4, 76.4)
    with pytest.raises(ValueError, match="duplicate ladder run"):
        az.filter_runs(az.load_ladder_runs(tmp_path), n_samples=250, batch_size=4, bits={4})
