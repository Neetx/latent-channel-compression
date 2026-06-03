"""Tests for src/metrics/{channel_fidelity,logit_metrics,bootstrap}.

All tests are deterministic via explicit seeds. Run:
    .venv/bin/python -m pytest tests/test_fidelity_metrics.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.metrics.bootstrap import (
    TOST_EQUIVALENT,
    TOST_INCONCLUSIVE,
    TOST_NOT_EQUIVALENT,
    bootstrap_ci_mean,
    paired_bootstrap_delta,
    paired_tost_binary,
)
from src.metrics.channel_fidelity import (
    FidelityRun,
    codebook_extreme_rate,
    effective_rank,
    relative_l2,
)
from src.metrics.logit_metrics import (
    PairedLogitStats,
    per_position_js,
    per_position_kl,
    per_position_mse,
    summarize_pair,
)


# ---------------------------------------------------------------------------
# channel_fidelity
# ---------------------------------------------------------------------------


class TestRelativeL2:
    def test_zero_when_equal(self):
        x = torch.randn(4, 8, generator=torch.Generator().manual_seed(0))
        assert torch.allclose(
            relative_l2(x, x), torch.zeros(4), atol=1e-6
        )

    def test_unit_when_orthogonal_with_same_norm(self):
        # x = e1, x_recv = -e1 → ||Δ||/||x|| = 2
        x = torch.tensor([[1.0, 0.0, 0.0]])
        x_recv = torch.tensor([[-1.0, 0.0, 0.0]])
        assert torch.allclose(relative_l2(x, x_recv), torch.tensor([2.0]))

    def test_matches_definition(self):
        x = torch.tensor([[3.0, 4.0]])  # norm 5
        x_recv = torch.tensor([[0.0, 0.0]])
        # ||Δ|| = 5, ||x|| = 5 → 1.0
        assert torch.allclose(relative_l2(x, x_recv), torch.tensor([1.0]))


class TestEffectiveRank:
    def test_rank_one_collapsed(self):
        # All vectors identical → centred matrix has rank 0 → degenerate
        # Use 2 identical rows: post-centring all zero → s = 0 → returns 1.0
        x = torch.ones(2, 16)
        assert effective_rank(x) == pytest.approx(1.0, abs=1e-3)

    def test_full_rank_gaussian(self):
        # iid Gaussian rows → spectrum is roughly flat → effective rank ≈ d (capped at N)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(64, 16, generator=g)
        er = effective_rank(x)
        assert 8.0 < er <= 16.0  # not collapsed, not all-equal

    def test_low_rank_construction(self):
        # Stack 100 copies of a length-3 base vector → effective rank should be ≈ 1
        base = torch.tensor([1.0, 2.0, 3.0])
        coefs = torch.linspace(0.1, 10.0, 100).unsqueeze(-1)
        x = coefs * base  # rank-1 outer product
        er = effective_rank(x)
        assert er < 1.2


class TestCodebookExtremeRate:
    def test_centred_distribution_low_rate(self):
        g = torch.Generator().manual_seed(0)
        codebook = torch.tensor([-3.0, -1.0, 1.0, 3.0])
        # Std-normal in [-3,3] → outermost half (idx 0 or 3) → rate ~ 0.5
        y = torch.randn(10_000, generator=g)
        rate = codebook_extreme_rate(y, codebook, tail_levels=1)
        # idx 0 = below -1 (50%), idx 3 = above 1 (50%) → rate ≈ 1.0 actually
        # with codebook [-3,-1,1,3] midpoints are [-2, 0, 2], so the "outer levels"
        # are coords mapped to idx 0 or 3 which is below -2 or above 2 → ~2.3%×2 = ~5%
        assert 0.0 <= rate <= 0.10

    def test_saturated_distribution(self):
        codebook = torch.tensor([-3.0, -1.0, 1.0, 3.0])
        # All values way outside the centre → most map to outer levels
        y = torch.cat([torch.full((100,), -5.0), torch.full((100,), 5.0)])
        rate = codebook_extreme_rate(y, codebook, tail_levels=1)
        assert rate == pytest.approx(1.0)

    def test_rejects_bad_codebook(self):
        with pytest.raises(ValueError):
            codebook_extreme_rate(torch.zeros(10), torch.tensor([1.0]), tail_levels=1)


class TestFidelityRun:
    def test_perfect_channel_summary(self):
        run = FidelityRun(label="test")
        g = torch.Generator().manual_seed(0)
        for _ in range(5):
            x = torch.randn(8, 32, generator=g)
            run.record(x, x.clone())
        s = run.summary()
        assert s["n_calls"] == 5
        assert s["cosine"]["mean"] == pytest.approx(1.0, abs=1e-5)
        assert s["rel_l2"]["mean"] == pytest.approx(0.0, abs=1e-5)

    def test_channel_stash(self):
        run = FidelityRun(label="test", max_samples_per_call=4)
        g = torch.Generator().manual_seed(0)
        for _ in range(3):
            x = torch.randn(20, 16, generator=g)
            run.record(x, x + 0.01 * torch.randn_like(x))
        s = run.summary()
        assert s["channel_matrix_shape"] == [12, 16]  # 3 calls × 4 stash
        assert "effective_rank" in s


# ---------------------------------------------------------------------------
# logit_metrics
# ---------------------------------------------------------------------------


class TestLogitMetrics:
    def test_zero_when_identical(self):
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(8, 100, generator=g)
        assert torch.allclose(per_position_mse(logits, logits), torch.zeros(8))
        kl = per_position_kl(logits, logits)
        assert torch.allclose(kl, torch.zeros(8), atol=1e-6)
        js = per_position_js(logits, logits)
        assert torch.allclose(js, torch.zeros(8), atol=1e-6)

    def test_kl_asymmetric_js_symmetric(self):
        g = torch.Generator().manual_seed(0)
        a = torch.randn(4, 50, generator=g)
        b = torch.randn(4, 50, generator=g)
        kl_ab = per_position_kl(a, b)
        kl_ba = per_position_kl(b, a)
        # Asymmetry expected for random pairs
        assert not torch.allclose(kl_ab, kl_ba, atol=1e-3)
        js_ab = per_position_js(a, b)
        js_ba = per_position_js(b, a)
        # JS is symmetric by construction
        assert torch.allclose(js_ab, js_ba, atol=1e-6)

    def test_kl_nonnegative(self):
        g = torch.Generator().manual_seed(0)
        a = torch.randn(8, 200, generator=g)
        b = torch.randn(8, 200, generator=g)
        assert (per_position_kl(a, b) >= -1e-6).all()

    def test_summarize_pair_truncates(self):
        g = torch.Generator().manual_seed(0)
        ref = torch.randn(10, 32, generator=g)
        intq = torch.randn(7, 32, generator=g)
        stats = summarize_pair(ref, intq)
        assert stats.n_positions == 7
        assert stats.n_positions_clamped == 3

    def test_summarize_pair_vocab_mismatch(self):
        with pytest.raises(ValueError):
            summarize_pair(torch.randn(4, 10), torch.randn(4, 20))


# ---------------------------------------------------------------------------
# bootstrap + TOST
# ---------------------------------------------------------------------------


class TestPairedBootstrap:
    def test_zero_when_identical(self):
        outcomes = np.array([1, 0, 1, 0, 1, 1, 0, 1])
        delta_obs, lo, hi, deltas = paired_bootstrap_delta(
            outcomes, outcomes, n_resamples=2000, seed=0
        )
        assert delta_obs == 0.0
        assert lo == 0.0 and hi == 0.0

    def test_positive_delta(self):
        # ref = all 0, treat = all 1 → Δ = +100 pct points
        n = 50
        delta_obs, lo, hi, _ = paired_bootstrap_delta(
            np.zeros(n, dtype=int), np.ones(n, dtype=int),
            n_resamples=2000, seed=0,
        )
        assert delta_obs == pytest.approx(100.0)
        # No variance possible → CI is point estimate
        assert lo == pytest.approx(100.0)
        assert hi == pytest.approx(100.0)

    def test_reproducible(self):
        rng = np.random.default_rng(0)
        ref = (rng.random(40) < 0.6).astype(int)
        treat = (rng.random(40) < 0.65).astype(int)
        d1, lo1, hi1, _ = paired_bootstrap_delta(ref, treat, n_resamples=1000, seed=7)
        d2, lo2, hi2, _ = paired_bootstrap_delta(ref, treat, n_resamples=1000, seed=7)
        assert (d1, lo1, hi1) == (d2, lo2, hi2)


class TestTOST:
    def test_equivalent_when_treatment_matches_within_eps(self):
        # treat slightly above ref, but well within eps=5
        rng = np.random.default_rng(0)
        n = 200
        ref = (rng.random(n) < 0.75).astype(int)
        # induce ~1 pct diff
        treat = ref.copy()
        flip = rng.choice(n, size=int(0.01 * n), replace=False)
        treat[flip] = 1 - treat[flip]
        r = paired_tost_binary(ref, treat, eps=5.0)
        assert r.verdict == TOST_EQUIVALENT

    def test_not_equivalent_when_far(self):
        rng = np.random.default_rng(0)
        n = 200
        ref = (rng.random(n) < 0.5).astype(int)
        treat = (rng.random(n) < 0.9).astype(int)  # ~+40pp Δ
        r = paired_tost_binary(ref, treat, eps=2.0)
        assert r.verdict in (TOST_NOT_EQUIVALENT, TOST_INCONCLUSIVE)
        # In this regime, NOT_EQUIVALENT should always hold
        assert r.delta_obs > 25.0

    def test_inconclusive_at_small_n(self):
        # Tiny n with mediocre signal → can't reject either tail
        rng = np.random.default_rng(0)
        n = 5
        ref = np.array([1, 0, 1, 1, 0])
        treat = np.array([1, 0, 0, 1, 0])
        r = paired_tost_binary(ref, treat, eps=2.0)
        # Likely inconclusive or not-equivalent — assert it isn't equivalent
        assert r.verdict != TOST_EQUIVALENT

    def test_degenerate_no_variance(self):
        ref = np.zeros(10, dtype=int)
        treat = np.zeros(10, dtype=int)
        r = paired_tost_binary(ref, treat, eps=2.0)
        assert r.verdict == TOST_EQUIVALENT


class TestBootstrapCIMean:
    def test_reproducible(self):
        vals = np.linspace(0.0, 1.0, 50)
        m1, lo1, hi1 = bootstrap_ci_mean(vals, n_resamples=1000, seed=7)
        m2, lo2, hi2 = bootstrap_ci_mean(vals, n_resamples=1000, seed=7)
        assert (m1, lo1, hi1) == (m2, lo2, hi2)

    def test_brackets_mean(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(loc=10.0, scale=2.0, size=200)
        mean, lo, hi = bootstrap_ci_mean(vals, n_resamples=2000, seed=0)
        assert lo < mean < hi
        # CI should contain the true mean (10.0) with high probability
        assert lo < 10.5 and hi > 9.5
