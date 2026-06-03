"""Tests for the fidelity_sweep analysis pipeline (experiments/.../analyze.py).

These exercise the Tier 2 glue that turns downloaded kernel outputs into the
results table, WITHOUT a GPU or any real kernel run:

  * paired_correctness        — alignment + the "drop undetermined" safety prop
  * _expand_to_union          — top-K union support construction
  * compute_logit_metrics_pair— synthetic NPZ dumps in the kernel's exact format
  * aggregate_per_T           — end-to-end (EQUIVALENT / NOT_EQUIVALENT / NO_DATA)
  * write_results_md          — table emission

Run:
    .venv/bin/python -m pytest tests/test_fidelity_analyze.py -v
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

# Headless matplotlib before analyze.py imports pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
ANALYZE_PATH = ROOT / "experiments" / "fidelity_sweep" / "analysis" / "analyze.py"


def _load_analyze():
    spec = importlib.util.spec_from_file_location("fid_analyze", ANALYZE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def az():
    return _load_analyze()


# ---------------------------------------------------------------------------
# correctness alignment
# ---------------------------------------------------------------------------


def _run_with_problems(acc, problems):
    return {"final_accuracy": acc, "per_problem": problems}


class TestPairedCorrectness:
    def test_flat_alignment(self, az):
        ref = _run_with_problems(75.0, [
            {"sample_idx": 0, "correct": True},
            {"sample_idx": 1, "correct": False},
            {"sample_idx": 2, "correct": True},
        ])
        intq = _run_with_problems(75.0, [
            {"sample_idx": 0, "correct": True},
            {"sample_idx": 1, "correct": True},
            {"sample_idx": 2, "correct": True},
        ])
        r, i = az.paired_correctness(ref, intq)
        assert list(r) == [1, 0, 1]
        assert list(i) == [1, 1, 1]

    def test_aligns_by_sample_idx_not_order(self, az):
        ref = _run_with_problems(0, [
            {"sample_idx": 5, "correct": True},
            {"sample_idx": 2, "correct": False},
        ])
        intq = _run_with_problems(0, [
            {"sample_idx": 2, "correct": False},
            {"sample_idx": 5, "correct": True},
        ])
        r, i = az.paired_correctness(ref, intq)
        # common sorted = [2, 5] -> [False, True] both sides
        assert list(r) == [0, 1]
        assert list(i) == [0, 1]

    def test_only_common_indices(self, az):
        ref = _run_with_problems(0, [{"sample_idx": 0, "correct": True},
                                     {"sample_idx": 1, "correct": True}])
        intq = _run_with_problems(0, [{"sample_idx": 1, "correct": False},
                                      {"sample_idx": 2, "correct": False}])
        r, i = az.paired_correctness(ref, intq)
        assert len(r) == 1 and len(i) == 1  # only idx 1 shared

    def test_drops_undetermined_correctness(self, az):
        # SAFETY: a None correctness must be dropped, not coerced to 0.
        ref = _run_with_problems(0, [{"sample_idx": 0, "correct": None},
                                     {"sample_idx": 1, "correct": True}])
        intq = _run_with_problems(0, [{"sample_idx": 0, "correct": True},
                                      {"sample_idx": 1, "correct": True}])
        r, i = az.paired_correctness(ref, intq)
        assert list(r) == [1] and list(i) == [1]  # idx 0 dropped

    def test_all_none_yields_empty(self, az):
        ref = _run_with_problems(0, [{"sample_idx": i, "correct": None} for i in range(5)])
        intq = _run_with_problems(0, [{"sample_idx": i, "correct": None} for i in range(5)])
        r, i = az.paired_correctness(ref, intq)
        assert r.size == 0 and i.size == 0

    def test_correct_flag_handles_nested(self, az):
        assert az._correct_flag({"correct": True}) is True
        assert az._correct_flag({"rollouts": [{"correct": False}]}) is False
        assert az._correct_flag({"pass_at_k_correct": True}) is True
        assert az._correct_flag({"gold": "x"}) is None


# ---------------------------------------------------------------------------
# union-support construction
# ---------------------------------------------------------------------------


class TestDistOverUnion:
    def test_sums_to_one_and_matches_softmax(self, az):
        vals = np.array([2.0, 1.0, 0.0])
        idxs = np.array([10, 20, 30])
        union = np.array([10, 20, 30])
        full_lse = float(np.log(np.exp(vals).sum()))  # top-K captures all mass
        p = az.dist_over_union(vals, idxs, full_lse, union)
        assert p.shape == (4,)  # 3 union + 1 tail bucket
        assert p.sum() == pytest.approx(1.0)
        sm = np.exp(vals) / np.exp(vals).sum()
        np.testing.assert_allclose(p[:3], sm, atol=1e-6)
        assert p[-1] == pytest.approx(0.0, abs=1e-6)  # negligible tail

    def test_missing_token_gets_finite_boundary_prob(self, az):
        # token 3 is in the union but outside this run's top-K
        vals = np.array([2.0, 1.0]); idxs = np.array([1, 2])
        union = np.array([1, 2, 3])
        full_lse = float(np.log(np.exp(np.array([2.0, 1.0, 1.0])).sum()))
        p = az.dist_over_union(vals, idxs, full_lse, union)
        assert np.all(np.isfinite(p))
        assert p.sum() == pytest.approx(1.0)
        assert p[2] > 0.0  # boundary prob, NOT zero (avoids the -inf blow-up)


class TestKLJSProbs:
    def test_zero_when_identical(self, az):
        p = np.array([0.5, 0.3, 0.2])
        assert az.kl_probs(p, p) == pytest.approx(0.0, abs=1e-9)
        assert az.js_probs(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_kl_positive_js_bounded(self, az):
        p = np.array([0.9, 0.05, 0.05]); q = np.array([0.05, 0.9, 0.05])
        assert az.kl_probs(p, q) > 0
        assert 0 < az.js_probs(p, q) <= np.log(2) + 1e-9  # JS bounded by ln2

    def test_js_symmetric(self, az):
        p = np.array([0.7, 0.2, 0.1]); q = np.array([0.2, 0.3, 0.5])
        assert az.js_probs(p, q) == pytest.approx(az.js_probs(q, p))

    def test_zero_prob_entry_no_nan(self, az):
        p = np.array([0.0, 0.6, 0.4]); q = np.array([0.1, 0.5, 0.4])
        assert np.isfinite(az.kl_probs(p, q))  # 0*log0 skipped, q floored


# ---------------------------------------------------------------------------
# logit metrics from synthetic NPZ in the kernel's dump format
# ---------------------------------------------------------------------------


def _save_logit_npz(path, vals, idxs, full_lse, tail_log):
    """One-batch NPZ matching the kernel's _vb_dump_logits format.

    vals/idxs: (T, B, K); full_lse/tail_log: (T, B).
    """
    np.savez_compressed(
        path,
        batch0_vals=vals.astype(np.float32),
        batch0_idxs=idxs.astype(np.int32),
        batch0_full_lse=full_lse.astype(np.float64),
        batch0_tail_log=tail_log.astype(np.float64),
        n_batches=np.array(1),
        bits=np.array(0),
        topk=np.array(vals.shape[-1]),
    )


class TestComputeLogitMetricsPair:
    def test_identical_dumps_zero_divergence(self, az, tmp_path):
        T, B, K = 4, 2, 3
        rng = np.random.default_rng(0)
        vals = -np.sort(-rng.normal(size=(T, B, K)), axis=-1)  # descending like topk
        idxs = np.tile(np.array([10, 20, 30]), (T, B, 1))
        full_lse = np.full((T, B), 4.0)
        tail_log = np.full((T, B), -5.0)
        ref, intq = tmp_path / "ref.npz", tmp_path / "int.npz"
        _save_logit_npz(ref, vals, idxs, full_lse, tail_log)
        _save_logit_npz(intq, vals, idxs, full_lse, tail_log)
        out = az.compute_logit_metrics_pair(ref, intq)
        assert out["per_problem_kl"].size == B == 2
        np.testing.assert_allclose(out["per_problem_kl"], 0.0, atol=1e-6)
        np.testing.assert_allclose(out["per_problem_js"], 0.0, atol=1e-6)
        assert out["divergence_rate"] == 0.0            # identical -> never diverges
        assert out["mean_matched_len"] == pytest.approx(T)  # all positions matched

    def test_divergence_detected_and_bounded(self, az, tmp_path):
        # Flipped argmax at every position -> the greedy paths diverge at t=0.
        T, B, K = 3, 1, 3
        ref_vals = np.tile(np.array([5.0, 0.0, -5.0]), (T, B, 1))
        ref_idxs = np.tile(np.array([10, 20, 30]), (T, B, 1))
        int_vals = np.tile(np.array([5.0, 0.0, -5.0]), (T, B, 1))
        int_idxs = np.tile(np.array([30, 20, 10]), (T, B, 1))  # top-1 differs
        full_lse = np.full((T, B), 5.1)
        tail_log = np.full((T, B), -6.0)
        ref, intq = tmp_path / "ref.npz", tmp_path / "int.npz"
        _save_logit_npz(ref, ref_vals, ref_idxs, full_lse, tail_log)
        _save_logit_npz(intq, int_vals, int_idxs, full_lse, tail_log)
        out = az.compute_logit_metrics_pair(ref, intq)
        assert out["divergence_rate"] == 1.0           # diverged
        assert out["mean_matched_len"] == 1.0          # only the t=0 mismatch counted
        kl = out["per_problem_kl"][0]
        assert kl > 0.5 and np.isfinite(kl)            # positive, finite
        assert kl < 50.0                                # NOT the old -inf/1e3 blow-up
        assert out["per_problem_mse"][0] > 0.0


# ---------------------------------------------------------------------------
# end-to-end aggregation + table
# ---------------------------------------------------------------------------


def _full_run(acc, correct_pattern, *, cos=0.999, rmse=1e-4, n_problems=None):
    n = n_problems if n_problems is not None else len(correct_pattern)
    per_problem = [{"sample_idx": i, "correct": correct_pattern[i]} for i in range(len(correct_pattern))]
    return {
        "final_accuracy": acc,
        "per_problem": per_problem,
        "fidelity_summary": {
            "per_call": [{
                "cosine_means": [cos] * 8,
                "rmse_means": [rmse] * 8,
                "n_calls": 8,
                "n_tokens_total": 800,
            }],
        },
    }


class TestAggregatePerT:
    def test_equivalent_when_identical(self, az):
        pattern = [True, True, False, True, False, True, True, False] * 4  # n=32
        runs = {
            (0, 1): _full_run(75.0, pattern),
            (4, 1): _full_run(75.0, list(pattern)),  # identical correctness
        }
        rows = az.aggregate_per_T(runs, eps_pp=2.0)
        assert len(rows) == 1
        row = rows[0]
        assert row["T"] == 1 and row["bits_int"] == 4
        assert row["delta_acc_obs"] == pytest.approx(0.0)
        assert row["tost_verdict"] == "EQUIVALENT"
        assert row["mean_cosine"] == pytest.approx(0.999, abs=1e-6)
        assert row["n_paired_problems"] == 32

    def test_not_equivalent_when_far(self, az):
        ref = _full_run(100.0, [True] * 40)
        intq = _full_run(10.0, [i < 4 for i in range(40)])  # mostly wrong
        rows = az.aggregate_per_T({(0, 2): ref, (4, 2): intq}, eps_pp=2.0)
        assert rows[0]["tost_verdict"] in ("NOT_EQUIVALENT", "INCONCLUSIVE")
        assert rows[0]["delta_acc_obs"] < -25.0

    def test_no_paired_data_when_correctness_missing(self, az):
        # REGRESSION: undetermined correctness must NOT masquerade as EQUIVALENT.
        ref = {"final_accuracy": 75.0,
               "per_problem": [{"sample_idx": i, "correct": None} for i in range(20)],
               "fidelity_summary": {"per_call": []}}
        intq = {"final_accuracy": 75.0,
                "per_problem": [{"sample_idx": i, "correct": None} for i in range(20)],
                "fidelity_summary": {"per_call": []}}
        rows = az.aggregate_per_T({(0, 1): ref, (4, 1): intq}, eps_pp=2.0)
        assert rows[0]["tost_verdict"] == "NO_PAIRED_DATA"
        assert rows[0]["delta_acc_obs"] is None

    def test_skips_T_without_ref_or_int(self, az):
        runs = {(4, 1): _full_run(75.0, [True, False] * 10)}  # no REF at T=1
        rows = az.aggregate_per_T(runs, eps_pp=2.0)
        assert rows == []

    def test_write_results_md(self, az, tmp_path):
        pattern = [True, False] * 16
        runs = {(0, 1): _full_run(75.0, pattern), (4, 1): _full_run(75.0, list(pattern))}
        rows = az.aggregate_per_T(runs, eps_pp=2.0)
        out = tmp_path / "results.md"
        az.write_results_md(rows, out)
        text = out.read_text()
        assert "Table 1" in text and "TOST" in text
        assert "Table 2" in text and "Table 3" in text
        assert "EQUIVALENT" in text

    def test_load_runs_rejects_duplicate_bits_T(self, az, tmp_path):
        for name, acc in [("a", 75.0), ("b", 76.0)]:
            d = tmp_path / name
            d.mkdir()
            (d / "fidelity_vb4_T3_n50_b8_float32.json").write_text(json.dumps({
                "config": {"variant_b_bits": 4, "num_recursive_rounds": 3},
                "final_accuracy": acc,
                "fidelity_summary": {"per_call": []},
                "per_problem": [],
            }))
        with pytest.raises(ValueError, match="duplicate fidelity run"):
            az.load_runs(tmp_path)
