"""Tests for the rotation-matrix problem-clustered bootstrap.

Validates the only new statistical logic (the divergence extraction is delegated to the
already-tested divergence_hazard helpers).

Run: .venv/bin/python -m pytest tests/test_rotation_matrix_analysis.py -v
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments" / "fidelity_sweep" / "local_pkg" / "analysis" / "rotation_matrix_analysis.py"


def _load():
    spec = importlib.util.spec_from_file_location("rma", ANALYSIS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_variance_gives_point_ci():
    rma = _load()
    light = np.full(250, 0.9)
    scaled = np.full(250, 0.5)
    lo, hi = rma.clustered_bootstrap(light, scaled, boot=2000, seed=42)
    assert lo == pytest.approx(0.4) and hi == pytest.approx(0.4)


def test_ci_brackets_true_delta_and_is_deterministic():
    rma = _load()
    rng = np.random.default_rng(0)
    light = rng.uniform(0.80, 1.00, 250)
    scaled = rng.uniform(0.40, 0.60, 250)
    lo1, hi1 = rma.clustered_bootstrap(light, scaled, boot=3000, seed=7)
    lo2, hi2 = rma.clustered_bootstrap(light, scaled, boot=3000, seed=7)
    assert (lo1, hi1) == (lo2, hi2)                         # seeded -> deterministic
    assert lo1 < (light.mean() - scaled.mean()) < hi1       # brackets the true contrast
    assert lo1 > 0                                          # a real positive gap stays positive


def test_zero_gap_ci_straddles_zero():
    rma = _load()
    rng = np.random.default_rng(1)
    x = rng.uniform(0.4, 0.6, 250)
    y = rng.uniform(0.4, 0.6, 250)
    lo, hi = rma.clustered_bootstrap(x, y, boot=3000, seed=3)
    assert lo < 0 < hi
