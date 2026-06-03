"""Paired bootstrap CI + TOST equivalence test for paired-design experiments.

Designed for the REF vs INT4 paired setup, where each problem produces one
binary outcome (correct/incorrect) for both conditions on identical input.

Determinism: every public function accepts ``seed`` and uses
``numpy.random.default_rng(seed)`` so a single seed reproduces the entire
bootstrap distribution bit-for-bit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paired bootstrap for binary outcomes (accuracy)
# ---------------------------------------------------------------------------


def paired_bootstrap_delta(
    ref: Sequence[int] | np.ndarray,
    treat: Sequence[int] | np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float, float, np.ndarray]:
    """Paired bootstrap on Δacc = mean(treat) − mean(ref) (treat in pct points).

    Both inputs are 1-D arrays of length n_problems, each value 0 or 1.

    Returns
    -------
    delta_obs : observed Δacc in percentage points
    ci_lo, ci_hi : 95% CI bounds in percentage points
    deltas : the n_resamples bootstrap replicates (pct points)
    """
    ref_a = np.asarray(ref, dtype=np.float64)
    treat_a = np.asarray(treat, dtype=np.float64)
    if ref_a.shape != treat_a.shape:
        raise ValueError("ref and treat must have the same shape")
    if ref_a.ndim != 1:
        raise ValueError("ref and treat must be 1-D")
    n = ref_a.shape[0]
    if n == 0:
        raise ValueError("empty inputs")

    rng = np.random.default_rng(int(seed))
    diff = (treat_a - ref_a) * 100.0  # pct points
    deltas = np.empty(n_resamples, dtype=np.float64)
    # Sample paired indices (preserves correlation between REF and INT4).
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = diff[idx].mean()

    delta_obs = float(diff.mean())
    ci_lo = float(np.quantile(deltas, 0.025))
    ci_hi = float(np.quantile(deltas, 0.975))
    return delta_obs, ci_lo, ci_hi, deltas


# ---------------------------------------------------------------------------
# TOST equivalence test (two one-sided t-tests, paired)
# ---------------------------------------------------------------------------

# Verdict labels are stable strings (no spaces) for easy table emission.
TOST_EQUIVALENT = "EQUIVALENT"
TOST_NOT_EQUIVALENT = "NOT_EQUIVALENT"
TOST_INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class TOSTResult:
    """Result of a paired TOST equivalence test for binary outcomes.

    delta_obs : observed Δ = mean(treat) - mean(ref), pct points
    se_delta : standard error of paired difference, pct points
    eps : pre-registered equivalence margin, pct points
    t_lower : t-statistic for H0: Δ ≤ -eps (we reject if t_lower > t_crit)
    t_upper : t-statistic for H0: Δ ≥ +eps (we reject if t_upper < -t_crit)
    p_lower, p_upper : the corresponding one-sided p-values
    p_tost : max(p_lower, p_upper) — the TOST p-value
    n : sample size
    verdict : one of {EQUIVALENT, NOT_EQUIVALENT, INCONCLUSIVE}
    """

    delta_obs: float
    se_delta: float
    eps: float
    t_lower: float
    t_upper: float
    p_lower: float
    p_upper: float
    p_tost: float
    n: int
    verdict: str


def paired_tost_binary(
    ref: Sequence[int] | np.ndarray,
    treat: Sequence[int] | np.ndarray,
    *,
    eps: float = 2.0,
    alpha: float = 0.05,
) -> TOSTResult:
    """Paired TOST for binary outcomes with pre-registered ``eps`` (pct points).

    H0_lower: Δacc ≤ -eps   (treatment is worse by more than eps)
    H0_upper: Δacc ≥ +eps   (treatment is better by more than eps)
    Both rejected at the same alpha → declare EQUIVALENT within ±eps.

    If neither rejected at alpha → declare NOT_EQUIVALENT.
    If exactly one rejected → INCONCLUSIVE.

    Approximates the binomial paired distribution with the t-test on per-problem
    differences (each in {-1, 0, +1}). At n ≥ 30 this is accurate; at smaller n
    the verdict label is conservative.
    """
    try:
        from scipy import stats  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for paired_tost_binary") from exc

    ref_a = np.asarray(ref, dtype=np.float64)
    treat_a = np.asarray(treat, dtype=np.float64)
    if ref_a.shape != treat_a.shape:
        raise ValueError("ref and treat must have the same shape")
    n = ref_a.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 paired observations, got n={n}")

    diff = (treat_a - ref_a) * 100.0  # pct points
    delta_obs = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        # All differences identical → degenerate; use exact verdict by inspection
        if abs(delta_obs) < eps:
            return TOSTResult(
                delta_obs=delta_obs, se_delta=0.0, eps=eps,
                t_lower=float("inf"), t_upper=float("-inf"),
                p_lower=0.0, p_upper=0.0, p_tost=0.0,
                n=n, verdict=TOST_EQUIVALENT,
            )
        else:
            return TOSTResult(
                delta_obs=delta_obs, se_delta=0.0, eps=eps,
                t_lower=float("-inf"), t_upper=float("inf"),
                p_lower=1.0, p_upper=1.0, p_tost=1.0,
                n=n, verdict=TOST_NOT_EQUIVALENT,
            )

    # H0_lower: Δ ≤ -eps. Reject if t = (Δ - (-eps)) / se > t_crit.
    # H0_upper: Δ ≥ +eps. Reject if t = (Δ - eps) / se < -t_crit.
    t_lower = (delta_obs - (-eps)) / se
    t_upper = (delta_obs - eps) / se
    df = n - 1
    # one-sided p-values
    p_lower = float(1.0 - stats.t.cdf(t_lower, df=df))
    p_upper = float(stats.t.cdf(t_upper, df=df))
    p_tost = max(p_lower, p_upper)

    reject_lower = p_lower < alpha
    reject_upper = p_upper < alpha
    if reject_lower and reject_upper:
        verdict = TOST_EQUIVALENT
    elif not reject_lower and not reject_upper:
        verdict = TOST_NOT_EQUIVALENT
    else:
        verdict = TOST_INCONCLUSIVE

    return TOSTResult(
        delta_obs=delta_obs, se_delta=se, eps=eps,
        t_lower=t_lower, t_upper=t_upper,
        p_lower=p_lower, p_upper=p_upper, p_tost=p_tost,
        n=n, verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Bootstrap CI for continuous metrics (one mean, across problems)
# ---------------------------------------------------------------------------


def bootstrap_ci_mean(
    values: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the mean of a 1-D array.

    Returns
    -------
    mean_obs, ci_lo, ci_hi
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim != 1:
        raise ValueError("values must be 1-D")
    n = vals.shape[0]
    if n == 0:
        raise ValueError("empty values")
    rng = np.random.default_rng(int(seed))
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        means[i] = vals[rng.integers(0, n, size=n)].mean()
    alpha = 1.0 - confidence
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1.0 - alpha / 2))
    return float(vals.mean()), lo, hi


__all__ = [
    "paired_bootstrap_delta",
    "paired_tost_binary",
    "bootstrap_ci_mean",
    "TOSTResult",
    "TOST_EQUIVALENT",
    "TOST_NOT_EQUIVALENT",
    "TOST_INCONCLUSIVE",
]
