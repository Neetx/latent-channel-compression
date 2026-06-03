"""Tests for Variant B — Haar + Lloyd-Max-for-Gaussian quantizer.

Includes oracle tests against the reference numpy implementation in
``external/turboquant_ref`` to make sure our codebook and rotation match
bit-for-bit (within float tolerance).

Run from project root:
    .venv/bin/python -m pytest tests/test_turboquant_honest.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "turboquant_ref"))

from src.quantizers.turboquant_honest import HaarRotation, TurboQuantHonest  # noqa: E402
from src.utils.lloyd_max import lloyd_max_gaussian  # noqa: E402
from src.metrics.distortion import cosine, norm_ratio, relative_mse  # noqa: E402

# Reference implementation (external/turboquant_ref; not in tree by default).
# Skip the entire module cleanly when the reference package is unavailable.
try:
    from turboquant.main.lloyd_max import lloyd_max as ref_lloyd_max  # noqa: E402
    from turboquant.main.rotation import random_rotation as ref_random_rotation  # noqa: E402
    from turboquant.main.mse import TurboQuantMSE as RefTQMSE  # noqa: E402
except ImportError as _ref_exc:
    pytest.skip(
        f"reference package not installed (run `git clone "
        f"https://github.com/yashkc2025/turboquant external/turboquant_ref` "
        f"to enable): {_ref_exc}",
        allow_module_level=True,
    )


# ----- Lloyd-Max oracle -----

@pytest.mark.parametrize("d,bits", [(64, 2), (256, 3), (256, 4), (1024, 4)])
def test_lloyd_max_matches_reference(d, bits):
    """Our analytical Lloyd-Max codebook should match the reference's
    numerically to within ~1e-4 (they use the same algorithm)."""
    ours = lloyd_max_gaussian(d, bits)
    ref = ref_lloyd_max(d, bits)
    assert ours.shape == ref.shape
    # The two run independent scipy.integrate.quad calls so tiny diffs are OK.
    np.testing.assert_allclose(ours, ref, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 6, 8])
def test_codebook_has_correct_n_levels(bits):
    cb = lloyd_max_gaussian(d=256, bits=bits)
    assert cb.shape == (2 ** bits,)
    # Sorted, symmetric around 0 (Gaussian is even).
    assert np.all(np.diff(cb) > 0)
    assert abs(cb.mean()) < 1e-3, f"codebook should be ~zero-mean, got mean={cb.mean()}"


# ----- Haar rotation oracle -----

@pytest.mark.parametrize("d,seed", [(64, 42), (256, 7), (1024, 1)])
def test_haar_rotation_matches_reference(d, seed):
    ours = HaarRotation(d, seed=seed).R.numpy()
    ref = ref_random_rotation(d, seed=seed)
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_haar_is_orthogonal():
    rot = HaarRotation(256, seed=0)
    R = rot.R
    eye = R @ R.T
    assert torch.allclose(eye, torch.eye(256), atol=1e-5)


@pytest.mark.parametrize("d", [64, 257, 1024, 1536])  # include non-power-of-2
def test_haar_rotation_inverse(d):
    rot = HaarRotation(d, seed=3)
    torch.manual_seed(0)
    x = torch.randn(5, d)
    y = rot(x)
    x_back = rot.inverse(y)
    assert torch.allclose(x, x_back, atol=1e-4), f"d={d}: rotation inverse failed"


# ----- End-to-end Variant B -----

def test_high_bits_near_identity():
    torch.manual_seed(0)
    q = TurboQuantHonest(d=256, bits=8, seed=0)
    x = torch.randn(50, 256)
    x = x / x.norm(dim=-1, keepdim=True)
    x_q = q(x)
    rel = relative_mse(x, x_q).mean().item()
    assert rel < 0.01, f"8-bit rMSE={rel}"


def test_degradation_monotonic_in_bits():
    torch.manual_seed(0)
    x = torch.randn(200, 256); x = x / x.norm(dim=-1, keepdim=True)
    errs = {}
    for b in [8, 4, 3, 2]:
        q = TurboQuantHonest(d=256, bits=b, seed=0)
        errs[b] = relative_mse(x, q(x)).mean().item()
    assert errs[8] < errs[4] < errs[3] < errs[2], f"Non-monotonic: {errs}"


def test_normalize_makes_pipeline_scale_invariant():
    """With normalize=True, the quantizer must be linear in input scale:
    q(c·x) == c·q(x) for every scalar c. This is what the pre-normalize +
    post-rescale step buys us — NOT exact norm preservation (Lloyd-Max
    centroids are biased toward zero, so ‖q(x)‖/‖x‖ ≈ 1 with a few % noise).
    """
    torch.manual_seed(0)
    x = torch.randn(50, 256)
    x = x / x.norm(dim=-1, keepdim=True)  # start unit-norm
    q = TurboQuantHonest(d=256, bits=4, normalize=True, seed=0)
    base = q(x)
    for c in [0.1, 2.5, 17.0, 1e-3]:
        scaled = q(c * x)
        # The output should scale exactly with c.
        assert torch.allclose(scaled, c * base, atol=1e-4 * max(abs(c), 1.0)), \
            f"scale invariance broken at c={c}"


def test_norm_ratio_close_to_one():
    """Sanity check: at 4 bits the post-quant norm drifts by at most a few percent."""
    torch.manual_seed(0)
    x = torch.randn(200, 256); x = x / x.norm(dim=-1, keepdim=True)
    q = TurboQuantHonest(d=256, bits=4, normalize=True, seed=0)
    nr = norm_ratio(x, q(x))
    assert 0.95 < nr.mean().item() < 1.05, f"norm-ratio mean drift: {nr.mean()}"
    assert nr.std().item() < 0.05, f"norm-ratio noise too high: {nr.std()}"


def test_variant_b_beats_uniform_at_4_bit():
    """Variant B should produce strictly lower rMSE than Variant A on
    unit-sphere Gaussian inputs at every bit-rate ≥ 2."""
    from src.quantizers.hadamard_uniform import HadamardUniformQuantizer

    torch.manual_seed(0)
    d = 1024
    x = torch.randn(500, d); x = x / x.norm(dim=-1, keepdim=True)
    for bits in [4, 3, 2]:
        qA = HadamardUniformQuantizer(d=d, bits=bits, per_channel=False, seed=0)
        qB = TurboQuantHonest(d=d, bits=bits, seed=0)
        errA = relative_mse(x, qA(x)).mean().item()
        errB = relative_mse(x, qB(x)).mean().item()
        assert errB < errA, f"bits={bits}: VariantB ({errB:.4f}) should beat VariantA ({errA:.4f})"


def test_matches_reference_mse_end_to_end():
    """Full pipeline reconstruction should match the reference TurboQuantMSE
    output to within ~1e-4 on unit-sphere inputs at moderate dim."""
    d, bits = 256, 4
    seed = 42
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((10, d))
    x_np /= np.linalg.norm(x_np, axis=-1, keepdims=True)

    ref = RefTQMSE(dim=d, bits=bits, seed=seed)
    idx = ref.quantize(x_np)
    x_ref = ref.dequantize(idx)

    ours = TurboQuantHonest(d=d, bits=bits, normalize=False, seed=seed)
    x_ours = ours(torch.from_numpy(x_np).float()).numpy()

    # The reference returns float64; we use float32 for the pipeline. Loosen tol.
    np.testing.assert_allclose(x_ours, x_ref, atol=5e-4, rtol=1e-2)


def test_works_with_non_power_of_two_dim():
    """RecursiveMAS hidden dims like 1536 should work natively (no padding
    unlike Hadamard)."""
    for d in [1536, 2048, 3584]:
        torch.manual_seed(0)
        x = torch.randn(10, d); x = x / x.norm(dim=-1, keepdim=True)
        q = TurboQuantHonest(d=d, bits=4, seed=0)
        x_q = q(x)
        assert x_q.shape == x.shape
        assert cosine(x, x_q).mean().item() > 0.9
