"""Tests for the fidelity_sweep Kaggle kernel's pure logic.

The kernel drives an expensive (~13h) Kaggle sweep, so its two highest-risk
pieces are validated here WITHOUT a GPU:

  1. The surgical regex patches to upstream run.py / inference_mas.py — applied
     against the REAL cloned upstream source (skipped if external/ is absent),
     asserting substitution counts AND that the patched source still compiles.
  2. The per-problem JSONL parser — exercised against BOTH upstream schemas
     (flat `correct` for num_rollouts==1, nested `rollouts[]` for >1) plus the
     trailing summary row.

Run:
    .venv/bin/python -m pytest tests/test_fidelity_kernel.py -v
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "experiments" / "fidelity_sweep" / "kernel_pkg" / "fidelity_kernel.py"
LOCAL_DRIVER_PATH = ROOT / "experiments" / "fidelity_sweep" / "local_pkg" / "fidelity_local.py"
RUN_CELL_PATH = ROOT / "experiments" / "fidelity_sweep" / "local_pkg" / "run_cell.py"
UPSTREAM = ROOT / "external" / "RecursiveMAS"


def _load_kernel():
    """Import the kernel module from file (side-effect-free: main() is guarded)."""
    spec = importlib.util.spec_from_file_location("fidelity_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def kernel():
    return _load_kernel()


class TestLocalBackendIsolation:
    def test_copy_upstream_uses_disposable_tree(self, tmp_path, monkeypatch):
        local = _load_path("fidelity_local_test", LOCAL_DRIVER_PATH)
        source = tmp_path / "source"
        destination = tmp_path / "run" / "_RecursiveMAS_work"
        (source / ".git").mkdir(parents=True)
        (source / "__pycache__").mkdir()
        (source / "inference_utils").mkdir()
        (source / "run.py").write_text("ORIGINAL = True\n")
        (source / "inference_utils" / "inference_mas.py").write_text("VALUE = 1\n")
        (source / ".git" / "config").write_text("private git metadata")
        (source / "__pycache__" / "x.pyc").write_bytes(b"cache")
        monkeypatch.setattr(local, "UPSTREAM_SOURCE", source)

        local.copy_upstream_source(destination)
        (destination / "run.py").write_text("PATCHED = True\n")

        assert (source / "run.py").read_text() == "ORIGINAL = True\n"
        assert not (destination / ".git").exists()
        assert not (destination / "__pycache__").exists()

    def test_local_driver_has_no_checkout_restore_path(self):
        source = LOCAL_DRIVER_PATH.read_text()
        assert "git_restore" not in source
        assert '["git", "-C", str(UPSTREAM_SOURCE), "checkout"' not in source

    def test_run_cell_validates_capture_contract(self, tmp_path):
        runner = _load_path("run_cell_test", RUN_CELL_PATH)
        tag = "math500_vb4_T3_n4_b2_auto"
        result_dir = tmp_path / tag
        result_dir.mkdir()
        result = {
            "return_code": 0,
            "final_accuracy": 50.0,
            "n_per_problem": 4,
            "n_logit_batches": 2,
            "call_stats_present": True,
        }
        (result_dir / f"fidelity_{tag}.json").write_text(json.dumps(result))
        valid, _ = runner.validate_result(tmp_path, "math500", 4, 4, 2, True)
        assert valid

        result["n_per_problem"] = 3
        (result_dir / f"fidelity_{tag}.json").write_text(json.dumps(result))
        valid, detail = runner.validate_result(tmp_path, "math500", 4, 4, 2, True)
        assert not valid
        assert "expected 4 paired records" in detail


_NEED_UPSTREAM = pytest.mark.skipif(
    not (UPSTREAM / "run.py").exists(),
    reason="upstream RecursiveMAS not cloned (git clone into external/ to enable)",
)


# ---------------------------------------------------------------------------
# run.py patches
# ---------------------------------------------------------------------------


class TestPatchRunPy:
    @_NEED_UPSTREAM
    def test_capture_mode_all_anchors(self, kernel):
        src = (UPSTREAM / "run.py").read_text()
        patched, counts = kernel.patch_run_py(
            src, n_samples=50, dtype="float32", num_recursive_rounds=2,
            capture_mode=True, jsonl_path="/kaggle/working/pp.jsonl",
        )
        assert counts == {
            "num_samples": 1, "dtype": 1, "outer_dtype": 1,
            "greedy": 1, "num_recursive_rounds": 1, "result_jsonl": 1,
        }
        # patched source must still be valid Python
        compile(patched, "run.py", "exec")
        assert '"--num_samples", "50"' in patched
        assert '"--dtype", "float32"' in patched
        assert '"--outer_dtype", "float32"' in patched
        assert '"--result_jsonl", "/kaggle/working/pp.jsonl"' in patched
        assert "default=2" in patched
        assert "# CAPTURE_MODE: do_sample removed" in patched

    @_NEED_UPSTREAM
    def test_no_capture_skips_greedy_and_jsonl(self, kernel):
        src = (UPSTREAM / "run.py").read_text()
        patched, counts = kernel.patch_run_py(
            src, n_samples=10, dtype="auto", num_recursive_rounds=3,
            capture_mode=False, jsonl_path="ignored",
        )
        assert "greedy" not in counts and "result_jsonl" not in counts
        assert counts["num_samples"] == 1 and counts["num_recursive_rounds"] == 1
        compile(patched, "run.py", "exec")
        # do_sample line is left intact when not capturing
        assert 'out.append("--do_sample")' in patched

    @_NEED_UPSTREAM
    def test_double_apply_raises(self, kernel):
        src = (UPSTREAM / "run.py").read_text()
        patched, _ = kernel.patch_run_py(
            src, n_samples=50, dtype="float32", num_recursive_rounds=2,
            capture_mode=True, jsonl_path="x",
        )
        with pytest.raises(RuntimeError):
            kernel.patch_run_py(
                patched, n_samples=50, dtype="float32", num_recursive_rounds=2,
                capture_mode=True, jsonl_path="x",
            )

    def test_missing_anchor_raises_on_synthetic(self, kernel):
        # A source missing every anchor must fail loudly, not silently no-op.
        with pytest.raises(RuntimeError):
            kernel.patch_run_py(
                "print('hello')\n", n_samples=1, dtype="float32",
                num_recursive_rounds=1, capture_mode=True, jsonl_path="x",
            )


# ---------------------------------------------------------------------------
# inference_mas.py patches
# ---------------------------------------------------------------------------


class TestPatchInferenceMas:
    @_NEED_UPSTREAM
    def test_all_anchors_and_compiles(self, kernel):
        src = (UPSTREAM / "inference_utils" / "inference_mas.py").read_text()
        patched, counts = kernel.patch_inference_mas(src, batch_size=4)
        assert counts == {"batch_size": 1, "head_injection": 1, "return_adapter": 2}
        compile(patched, "inference_mas.py", "exec")
        # dict pin preserves latent_length
        assert '"batch_size": 4, "latent_length": 48' in patched
        # head injected once, both loaders guarded
        assert "Variant B + Tier 2 fidelity capture" in patched
        assert patched.count("if _VB_BITS > 0:") >= 2
        # selective-quantization gate (VB_LINKS) injected in both adapter loaders
        assert patched.count('_vb_os.environ.get("VB_LINKS"') >= 2
        assert '_vb_links == "all" or _vb_links == _vb_label' in patched

    @_NEED_UPSTREAM
    def test_injected_head_is_valid_python(self, kernel):
        # The injected head string must itself be syntactically valid.
        compile(kernel.VARIANT_B_HEAD, "head", "exec")
        # and reads the env vars the driver propagates to the child, incl. the
        # FIDELITY_WORK_DIR / FIDELITY_SRC_ROOT knobs that make it portable to Modal
        for key in ("VARIANT_B_BITS", "CAPTURE_MODE", "TOPK_LOGITS",
                    "MAX_LOGIT_POSITIONS", "FIDELITY_WORK_DIR", "FIDELITY_SRC_ROOT"):
            assert key in kernel.VARIANT_B_HEAD

    def test_missing_anchor_raises_on_synthetic(self, kernel):
        with pytest.raises(RuntimeError):
            kernel.patch_inference_mas("x = 1\n", batch_size=4)


# ---------------------------------------------------------------------------
# per-problem JSONL parser (both upstream schemas)
# ---------------------------------------------------------------------------


class TestParsePerProblemJsonl:
    def _write(self, tmp_path, records):
        p = tmp_path / "pp.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return str(p)

    def test_flat_schema_num_rollouts_1(self, kernel, tmp_path):
        # This is the schema our actual sweep produces (num_rollouts==1).
        recs = [
            {"sample_idx": 0, "gold_answer_raw": "5", "correct": True, "rollout_idx": 0},
            {"sample_idx": 1, "gold_answer_raw": "7", "correct": False, "rollout_idx": 0},
            {"type": "summary", "accuracy": 50.0, "num_samples": 2},
        ]
        pp = kernel.parse_per_problem_jsonl(self._write(tmp_path, recs))
        assert [p["sample_idx"] for p in pp] == [0, 1]  # summary skipped
        assert [p["correct"] for p in pp] == [True, False]
        assert [p["gold"] for p in pp] == ["5", "7"]

    def test_nested_schema_num_rollouts_gt1(self, kernel, tmp_path):
        recs = [
            {"sample_idx": 0, "gold_answer_raw": "5", "pass_at_k_correct": True,
             "rollouts": [{"rollout_idx": 0, "correct": True},
                          {"rollout_idx": 1, "correct": False}]},
            {"sample_idx": 1, "gold_answer_raw": "7", "pass_at_k_correct": False,
             "rollouts": [{"rollout_idx": 0, "correct": False}]},
            {"type": "summary", "accuracy": 50.0},
        ]
        pp = kernel.parse_per_problem_jsonl(self._write(tmp_path, recs))
        assert [p["sample_idx"] for p in pp] == [0, 1]
        # uses rollouts[0].correct
        assert [p["correct"] for p in pp] == [True, False]

    def test_missing_correctness_is_none(self, kernel, tmp_path):
        recs = [{"sample_idx": 3, "gold_answer_raw": "1"}]
        pp = kernel.parse_per_problem_jsonl(self._write(tmp_path, recs))
        assert pp[0]["correct"] is None

    def test_missing_file_returns_empty(self, kernel):
        assert kernel.parse_per_problem_jsonl("/nonexistent/path.jsonl") == []

    def test_blank_and_malformed_lines_skipped(self, kernel, tmp_path):
        p = tmp_path / "pp.jsonl"
        p.write_text(
            '{"sample_idx": 0, "correct": true}\n'
            "\n"
            "not json at all\n"
            '{"sample_idx": 1, "correct": false}\n'
        )
        pp = kernel.parse_per_problem_jsonl(str(p))
        assert [p["sample_idx"] for p in pp] == [0, 1]
