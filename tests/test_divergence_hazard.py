"""Unit tests for the length-censored first-divergence Kaplan-Meier estimator.

The survival math is the part that must be exactly right (the NPZ extraction is
validated separately by reproducing tier2_logit_fidelity's divergence rates). These
check the discrete-time KM against hand-computed values, including correct handling of
right-censored (identical-sequence) observations.

Run: .venv/bin/python -m pytest tests/test_divergence_hazard.py -v
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HAZARD = ROOT / "experiments" / "fidelity_sweep" / "local_pkg" / "analysis" / "divergence_hazard.py"


def _load():
    spec = importlib.util.spec_from_file_location("divergence_hazard", HAZARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_all_diverge_at_first_position():
    m = _load()
    S, h, at_risk = m.kaplan_meier(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), horizon=3)
    assert h[1] == pytest.approx(1.0)
    assert S[1] == pytest.approx(0.0)
    assert at_risk[1] == 4 and at_risk[2] == 0


def test_hand_computed_with_censoring():
    # times/events: diverge@1, diverge@2, diverge@2, censored@3 (identical sequence)
    m = _load()
    time = np.array([1, 2, 2, 3])
    event = np.array([1, 1, 1, 0])
    S, h, at_risk = m.kaplan_meier(time, event, horizon=3)
    # t=1: at risk 4, 1 event -> h=1/4, S=0.75
    assert at_risk[1] == 4 and h[1] == pytest.approx(0.25) and S[1] == pytest.approx(0.75)
    # t=2: at risk 3 (the two @2 + the censored @3), 2 events -> h=2/3, S=0.75*(1/3)=0.25
    assert at_risk[2] == 3 and h[2] == pytest.approx(2 / 3) and S[2] == pytest.approx(0.25)
    # t=3: at risk 1 (the censored), 0 events -> h=0, S unchanged
    assert at_risk[3] == 1 and h[3] == pytest.approx(0.0) and S[3] == pytest.approx(0.25)


def test_censored_observations_never_count_as_events():
    # everything censored -> survival stays 1, no hazard anywhere
    m = _load()
    S, h, at_risk = m.kaplan_meier(np.array([2, 3, 3]), np.array([0, 0, 0]), horizon=3)
    assert np.allclose(S, 1.0)
    assert np.allclose(h, 0.0)


def test_survival_is_monotone_nonincreasing():
    m = _load()
    rng = np.random.default_rng(0)
    time = rng.integers(1, 50, size=400)
    event = rng.integers(0, 2, size=400)
    S, _, _ = m.kaplan_meier(time, event, horizon=49)
    assert np.all(np.diff(S[1:]) <= 1e-12)


def test_valid_len_counts_nonzero_lse():
    m = _load()
    # 5 real positions then zero-padding
    col = np.array([12.0, 9.0, 31.0, 7.0, 22.0, 0.0, 0.0, 0.0])
    assert m._valid_len(col) == 5
