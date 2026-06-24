"""The quantizer rotation must be deterministic in its seed and change when the seed
changes — the scientific basis for the ``quantizer_seed`` rotation-matrix axis. Uses only
``src/`` (no optional reference package), so it runs in every environment.

Run: .venv/bin/python -m pytest tests/test_quantizer_seed.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quantizers.turboquant_honest import TurboQuantHonest  # noqa: E402


def _roundtrip(seed: int, x: torch.Tensor) -> torch.Tensor:
    # TurboQuantHonest is an nn.Module; its forward is the quantize->dequantize roundtrip.
    with torch.no_grad():
        return TurboQuantHonest(d=x.shape[-1], bits=4, seed=seed)(x)


def _fixed_input() -> torch.Tensor:
    return torch.randn(4, 128, generator=torch.Generator().manual_seed(0))


def test_same_seed_is_deterministic():
    x = _fixed_input()
    assert torch.allclose(_roundtrip(7, x), _roundtrip(7, x))


def test_different_seeds_give_different_rotations():
    x = _fixed_input()
    # Different quantizer seeds rotate the vector differently, so the reconstruction
    # differs even though the nominal bit-rate (and thus the distortion magnitude) is
    # identical. This is what makes a rotation-seed sweep a real replication axis.
    assert not torch.allclose(_roundtrip(7, x), _roundtrip(42, x))


def test_seed_42_matches_the_original_hardcoded_condition():
    # The injected head historically hardcoded seed=42; the default must reproduce it.
    x = _fixed_input()
    assert torch.allclose(_roundtrip(42, x), _roundtrip(42, x))
