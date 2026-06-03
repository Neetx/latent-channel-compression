"""Tests for Variant A — Hadamard + uniform quantizer.

Run from project root with:
    .venv/bin/python -m pytest tests/test_hadamard_uniform.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# Make ``src`` importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quantizers.hadamard_uniform import (  # noqa: E402
    HadamardUniformQuantizer,
    RandomizedHadamard,
    UniformSymmetricQuantizer,
    fwht,
)
from src.metrics.distortion import (  # noqa: E402
    cosine,
    inner_product_error,
    norm_ratio,
    relative_mse,
)


# ----- FWHT primitives -----

@pytest.mark.parametrize("d", [2, 4, 8, 16, 64, 256, 1024])
def test_fwht_is_involutive(d):
    torch.manual_seed(0)
    x = torch.randn(5, d)
    y = fwht(x)
    x2 = fwht(y)
    assert torch.allclose(x, x2, atol=1e-5), f"d={d}: FWHT^2 != I"


@pytest.mark.parametrize("d", [4, 16, 128, 1024])
def test_fwht_preserves_norm(d):
    torch.manual_seed(0)
    x = torch.randn(10, d)
    y = fwht(x)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)


def test_fwht_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        fwht(torch.randn(3, 5))


# ----- Randomized Hadamard rotation -----

@pytest.mark.parametrize("d", [2, 4, 8, 16, 17, 100, 256, 1024])
def test_rotation_inverse(d):
    rot = RandomizedHadamard(d, seed=42)
    torch.manual_seed(0)
    x = torch.randn(7, d)
    y = rot(x)
    x_back = rot.inverse(y)
    assert torch.allclose(x, x_back, atol=1e-5), f"d={d}: rotation inverse failed"


@pytest.mark.parametrize("d", [4, 17, 256, 1024])
def test_rotation_preserves_norm_on_real_dims(d):
    """Rotation acts on padded space but real-dim norm is unchanged after inverse."""
    rot = RandomizedHadamard(d, seed=1)
    torch.manual_seed(0)
    x = torch.randn(20, d)
    y = rot(x)
    # Rotation in padded space is norm-preserving, but on the padded vector.
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_rotation_seed_determinism():
    a = RandomizedHadamard(64, seed=123)
    b = RandomizedHadamard(64, seed=123)
    assert torch.equal(a.signs, b.signs)
    c = RandomizedHadamard(64, seed=124)
    assert not torch.equal(a.signs, c.signs)


# ----- Uniform quantizer -----

@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_quantizer_levels_match_qmax(bits):
    q = UniformSymmetricQuantizer(bits)
    torch.manual_seed(0)
    y = torch.randn(1000)
    yq = q(y)
    # Number of distinct values should be ≤ 2*qmax + 1
    n_levels = yq.unique().numel()
    assert n_levels <= 2 * q.qmax + 1
    # Should use roughly all levels at high bits, but the 0 may or may not appear.
    if bits >= 3:
        assert n_levels >= q.qmax + 1


def test_quantizer_rejects_one_bit():
    with pytest.raises(ValueError):
        UniformSymmetricQuantizer(1)


# ----- End-to-end Variant A -----

def test_high_bits_near_identity():
    """At 16 bits with per-tensor scaling, distortion should be tiny."""
    torch.manual_seed(0)
    q = HadamardUniformQuantizer(d=128, bits=16)
    x = torch.randn(50, 128)
    x_q = q(x)
    rel = relative_mse(x, x_q)
    assert rel.mean().item() < 1e-3, f"16-bit rMSE={rel.mean().item()}"


def test_degradation_monotonic_in_bits():
    """Distortion should grow as bits decrease."""
    torch.manual_seed(0)
    x = torch.randn(200, 128)
    x = x / x.norm(dim=-1, keepdim=True)  # unit sphere

    errs = {}
    for b in [8, 4, 3, 2]:
        q = HadamardUniformQuantizer(d=128, bits=b, seed=0)
        x_q = q(x)
        errs[b] = relative_mse(x, x_q).mean().item()
    # Strict monotonic: lower bits => larger error.
    assert errs[8] < errs[4] < errs[3] < errs[2], f"Non-monotonic: {errs}"


def test_cosine_stays_high_at_8_bit():
    torch.manual_seed(0)
    x = torch.randn(200, 256)
    x = x / x.norm(dim=-1, keepdim=True)
    q = HadamardUniformQuantizer(d=256, bits=8, seed=0)
    x_q = q(x)
    c = cosine(x, x_q)
    assert c.mean().item() > 0.99


def test_norm_ratio_close_to_one():
    """Per-tensor scaling shouldn't introduce systematic scale bias at high bits."""
    torch.manual_seed(0)
    x = torch.randn(200, 256)
    q = HadamardUniformQuantizer(d=256, bits=8, seed=0)
    x_q = q(x)
    nr = norm_ratio(x, x_q)
    assert 0.95 < nr.mean().item() < 1.05


def test_inner_product_error_decreases_with_bits():
    torch.manual_seed(0)
    N, d = 100, 256
    x = torch.randn(N, d)
    y = torch.randn(N, d)

    errs = {}
    for b in [8, 4, 2]:
        q = HadamardUniformQuantizer(d=d, bits=b, seed=0)
        xq, yq = q(x), q(y)
        errs[b] = inner_product_error(x, y, xq, yq).mean().item()
    assert errs[8] < errs[4] < errs[2], f"IP-error non-monotonic: {errs}"


def test_works_with_non_power_of_two_dim():
    """RecursiveMAS hidden dims like 1536 are not powers of two."""
    for d in [1536, 2048, 3584]:
        torch.manual_seed(0)
        x = torch.randn(10, d)
        q = HadamardUniformQuantizer(d=d, bits=4, seed=0)
        x_q = q(x)
        assert x_q.shape == x.shape
        # Sanity: not completely destroying the signal
        assert cosine(x, x_q).mean().item() > 0.5
