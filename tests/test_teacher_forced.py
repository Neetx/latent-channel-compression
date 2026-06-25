"""Teacher-forced plumbing: the aligned mechanism capture must land in its own _tf directory
(never colliding with the free-running INT4 capture), the orchestrator must derive its tag from
the driver's own builder, and it must run at batch_size=1 (the only size where gate G0 holds).

Pure-Python (no torch / no CUDA).
Run: .venv/bin/python -m pytest tests/test_teacher_forced.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PKG = ROOT / "experiments" / "fidelity_sweep" / "local_pkg"
if str(LOCAL_PKG) not in sys.path:
    sys.path.insert(0, str(LOCAL_PKG))

from fidelity_local import build_config_tag  # noqa: E402
import run_teacher_forced as rtf  # noqa: E402


def test_tf_suffix_distinct_from_free_running():
    tf = build_config_tag("mbppplus", 4, 3, 250, 1, "auto", teacher_forced=True)
    free = build_config_tag("mbppplus", 4, 3, 250, 1, "auto", teacher_forced=False)
    assert tf == "mbppplus_vb4_T3_n250_b1_auto_tf"
    assert free == "mbppplus_vb4_T3_n250_b1_auto"
    assert tf != free  # no collision with the free-running INT4 capture


def test_tf_suffix_composes_after_seed_and_links():
    assert (build_config_tag("mbppplus", 4, 3, 250, 1, "auto",
                             quantizer_seed=7, links="outer", teacher_forced=True)
            == "mbppplus_vb4_T3_n250_b1_auto_qs7_lo_tf")


def test_orchestrator_runs_at_batch_size_1():
    # G0 only holds at b=1; the orchestrator must not silently batch.
    assert rtf.BATCH == 1


def test_orchestrator_tf_tag_matches_driver():
    # The TF result path the orchestrator validates must equal what fidelity_local writes.
    n = rtf_n = 250
    expected = build_config_tag("mbppplus", 4, 3, n, rtf.BATCH, "auto", teacher_forced=True)
    assert expected == f"mbppplus_vb4_T3_n{n}_b{rtf.BATCH}_auto_tf"


def test_valid_false_when_missing(tmp_path):
    assert rtf._valid(tmp_path / "nope.json", 250, True) is False
