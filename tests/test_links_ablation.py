"""The inner/outer link-ablation plumbing: a single-link run must land in its own output
directory (so inner-only, outer-only, and all-links captures never overwrite each other),
while the default all-links seed-42 tag stays byte-for-byte identical to the historical one.
The orchestrator derives its tag from the driver's own builder, so the two cannot drift.

Pure-Python (no torch / no CUDA): exercises only the tag/orchestration logic.
Run: .venv/bin/python -m pytest tests/test_links_ablation.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PKG = ROOT / "experiments" / "fidelity_sweep" / "local_pkg"
if str(LOCAL_PKG) not in sys.path:
    sys.path.insert(0, str(LOCAL_PKG))

from fidelity_local import build_config_tag  # noqa: E402
import run_links_ablation as rla  # noqa: E402


def test_all_links_seed42_is_the_original_tag():
    # Backward compatibility: the headline condition must keep resolving to the old path,
    # or every committed result and analyzer would silently stop matching.
    assert (build_config_tag("mbppplus", 4, 3, 250, 2, "auto")
            == "mbppplus_vb4_T3_n250_b2_auto")


def test_inner_and_outer_get_distinct_suffixes():
    inner = build_config_tag("mbppplus", 4, 3, 250, 2, "auto", links="inner")
    outer = build_config_tag("mbppplus", 4, 3, 250, 2, "auto", links="outer")
    assert inner == "mbppplus_vb4_T3_n250_b2_auto_li"
    assert outer == "mbppplus_vb4_T3_n250_b2_auto_lo"
    assert inner != outer  # the whole point: no shared-capture collision between links


def test_quantizer_seed_and_links_suffixes_compose_in_order():
    # _qs precedes _l so a rotated single-link run is still unambiguous.
    assert (build_config_tag("mbppplus", 4, 3, 250, 1, "auto", quantizer_seed=7, links="outer")
            == "mbppplus_vb4_T3_n250_b1_auto_qs7_lo")


def test_orchestrator_tag_matches_driver_tag():
    # rla.cond_tag must equal what fidelity_local will actually write, for every tier/link,
    # otherwise the resume-skip and the divergence analyzer look in the wrong directory.
    for batch in (1, 2):
        for links in ("inner", "outer"):
            assert (rla.cond_tag("mbppplus", batch, links, 250)
                    == build_config_tag("mbppplus", 4, 3, 250, batch, "auto", links=links))


def test_valid_result_false_when_missing(tmp_path):
    assert rla.valid_result(tmp_path, "mbppplus", 2, "inner", 250) is False
