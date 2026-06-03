"""Tests for the adapter patching utility."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Make RecursiveMAS modeling importable for end-to-end tests against the real
# Adapter / CrossModelAdapter classes.
sys.path.insert(0, str(ROOT / "external" / "RecursiveMAS"))

from src.adapters.patch import (  # noqa: E402
    QuantStats,
    n_active_patches,
    patch_adapter,
)
from src.quantizers.turboquant_honest import TurboQuantHonest  # noqa: E402

# Import the *real* RecursiveMAS Adapter classes.
import modeling as rmas_modeling  # noqa: E402


def _make_inner(d=64):
    return rmas_modeling.Adapter(hidden_size=d, adapter_type="ln_res_adapter")


def _make_outer(in_d=64, out_d=48):
    return rmas_modeling.CrossModelAdapter(in_dim=in_d, out_dim=out_d, adapter_type="outer_ln_res_adapter")


def _q_factory(bits):
    def factory(d):
        return TurboQuantHonest(d=d, bits=bits, normalize=True, seed=0)
    return factory


# ---- Basic patch/unpatch lifecycle ----

def test_patch_returns_unpatch_callable():
    a = _make_inner()
    un = patch_adapter(a, _q_factory(16))
    assert callable(un)
    assert n_active_patches() == 1
    un()
    assert n_active_patches() == 0


def test_unpatch_is_idempotent():
    a = _make_inner()
    un = patch_adapter(a, _q_factory(16))
    un()
    un()  # should not raise or under-flow
    assert n_active_patches() == 0


def test_double_patch_raises():
    a = _make_inner()
    un = patch_adapter(a, _q_factory(16))
    try:
        with pytest.raises(RuntimeError):
            patch_adapter(a, _q_factory(8))
    finally:
        un()


def test_patch_preserves_shape_and_dtype():
    a = _make_inner(d=64).eval()
    x = torch.randn(2, 5, 64)
    baseline = a(x)
    un = patch_adapter(a, _q_factory(16))
    try:
        out = a(x)
        assert out.shape == baseline.shape
        assert out.dtype == baseline.dtype
    finally:
        un()


def test_unpatch_restores_exact_forward():
    a = _make_inner(d=64).eval()
    x = torch.randn(2, 5, 64)
    before = a(x).clone()
    un = patch_adapter(a, _q_factory(4))  # something that visibly changes output
    mid = a(x).clone()
    assert not torch.allclose(before, mid), "4-bit patched output should differ"
    un()
    after = a(x).clone()
    assert torch.allclose(before, after, atol=1e-6), "unpatch must restore exact forward"


# ---- Effect of patching at different bit-rates ----

def test_16bit_patch_is_near_identity():
    a = _make_inner(d=256).eval()
    x = torch.randn(2, 5, 256)
    baseline = a(x)
    un = patch_adapter(a, _q_factory(16))
    try:
        out = a(x)
        rel = (out - baseline).norm() / baseline.norm()
        assert rel < 1e-3, f"16-bit relative deviation should be tiny, got {rel}"
    finally:
        un()


def test_lower_bits_cause_larger_drift():
    a = _make_inner(d=256).eval()
    x = torch.randn(2, 5, 256)
    baseline = a(x)
    drifts = {}
    for bits in [8, 4, 3, 2]:
        un = patch_adapter(a, _q_factory(bits))
        try:
            drifts[bits] = float(((a(x) - baseline).norm() / baseline.norm()))
        finally:
            un()
    assert drifts[8] < drifts[4] < drifts[3] < drifts[2], f"non-monotonic drift: {drifts}"


# ---- Works with CrossModelAdapter (different output dim) ----

def test_patch_cross_model_adapter():
    a = _make_outer(in_d=64, out_d=48).eval()
    x = torch.randn(2, 5, 64)
    baseline = a(x)
    assert baseline.shape == (2, 5, 48)
    un = patch_adapter(a, _q_factory(4))
    try:
        out = a(x)
        assert out.shape == (2, 5, 48)
        assert out.dtype == baseline.dtype
        assert not torch.allclose(out, baseline), "patching at 4-bit should change output"
    finally:
        un()


# ---- Compatibility with torch.no_grad ----

def test_patched_forward_in_no_grad():
    a = _make_inner(d=128).eval()
    x = torch.randn(2, 5, 128)
    un = patch_adapter(a, _q_factory(4))
    try:
        with torch.no_grad():
            out = a(x)
        assert out.requires_grad is False
        assert out.shape == (2, 5, 128)
    finally:
        un()


# ---- Stat collection ----

def test_stat_recording_counts_calls():
    a = _make_inner(d=128).eval()
    stats = QuantStats(label="test")
    un = patch_adapter(a, _q_factory(4), stats=stats, record=True)
    try:
        x = torch.randn(2, 5, 128)
        _ = a(x)
        _ = a(x)
        _ = a(x)
    finally:
        un()
    assert stats.n_calls == 3
    assert len(stats.rmse_means) == 3
    assert stats.n_tokens_total == 3 * 2 * 5
    summ = stats.summary()
    assert summ["n_calls"] == 3
    assert 0.0 < summ["rmse"]["mean"] < 1.0
    assert 0.5 < summ["cosine"]["mean"] <= 1.0


def test_stat_recording_disabled_when_record_false():
    a = _make_inner(d=128).eval()
    un = patch_adapter(a, _q_factory(4), record=False)
    try:
        x = torch.randn(2, 5, 128)
        _ = a(x)
    finally:
        un()
    # Without record=True, no stats attribute should be attached.
    assert not hasattr(a, "_quant_stats")


# ---- Factory output dimension is correct ----

def test_factory_receives_correct_out_dim_for_outer():
    captured = []

    def factory(d):
        captured.append(d)
        return TurboQuantHonest(d=d, bits=16, normalize=True, seed=0)

    a = _make_outer(in_d=64, out_d=48).eval()
    un = patch_adapter(a, factory)
    un()
    assert captured == [48], f"factory should receive out_dim=48, got {captured}"


def test_factory_receives_correct_out_dim_for_inner():
    captured = []

    def factory(d):
        captured.append(d)
        return TurboQuantHonest(d=d, bits=16, normalize=True, seed=0)

    a = _make_inner(d=128).eval()
    un = patch_adapter(a, factory)
    un()
    assert captured == [128]


# ---- Smoke: real Variant B with real Adapter end-to-end ----

def test_end_to_end_real_variant_b():
    torch.manual_seed(0)
    a = _make_inner(d=256).eval()
    x = torch.randn(4, 10, 256)
    baseline = a(x)
    un = patch_adapter(a, _q_factory(4))
    try:
        out = a(x)
    finally:
        un()
    cos = torch.nn.functional.cosine_similarity(baseline, out, dim=-1)
    # Variant B at 4 bits on unit-sphere data gives cos > 0.99; the Adapter
    # output is not unit-norm but the post-LN is well-behaved.
    assert cos.mean() > 0.95, f"unexpectedly low cosine: {cos.mean()}"
