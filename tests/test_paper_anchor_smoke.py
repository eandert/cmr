"""Smoke tests pinning expected AMOTAs for the paper's anchor + headline experiments.

These tests load each registered experiment's tracker output from disk, run the
clean evaluator's ``compute_stage`` for every stage (V2V / OG+FP / Ours /
Ours-merged), and assert the AMOTA matches the pinned expected value within
``TOL_AMOTA`` (1e-4 absolute on the 0..1 scale, ~0.01 percentage points).

Pinned values are the numbers in ``results/paper_tables.md`` at the moment of
freezing. Any drift — caused by:
  - a sneaky scorer change
  - an evaluator-state-mode regression
  - a GT integrity violation (the with_cav_merged manifest sha256 changed)
  - a tracker output dir being re-overwritten with different bytes
— will fail one or more of these tests immediately, with the experiment name
and stage in the diagnostic.

Tests are SLOW (each ~60–150s wall, totals ~30 min serial). Marked with
``pytest.mark.slow`` so the default ``pytest tests/`` invocation skips them.
Run explicitly with::

    pytest tests/test_paper_anchor_smoke.py -m slow -v

Or pin only the headline (fast subset)::

    pytest tests/test_paper_anchor_smoke.py::test_headline_amota -m slow -v

To re-pin after an intentional change, run ``python scripts/paper_tables.py``,
then update the EXPECTED_AMOTA dict below from the freshly-generated table.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "configs"))

AB3DMOT_ROOT = REPO / "third_party" / "AB3DMOT"
if AB3DMOT_ROOT.exists():
    sys.path.insert(0, str(AB3DMOT_ROOT))
    sys.path.insert(0, str(AB3DMOT_ROOT / "Xinshuo_PyToolbox"))

from eval_config import V2V4REAL_RESULTS_ROOT, SEQMAP_VAL  # noqa: E402
from gt_integrity import parse_seqmap  # noqa: E402
from paper_metrics import compute_stage, STAGES  # noqa: E402
from v2v4real_experiments import get as get_experiment  # noqa: E402


# Absolute tolerance on AMOTA (0..1 scale). 1e-4 = 0.01 percentage points.
# Our scorer is bit-identical (1.11e-16) to evaluate.py, so the only real
# source of drift at this tolerance would be a sneaky change to the scorer or
# the GT/tracker-output bytes.
TOL_AMOTA = 1e-4


# ── Pinned expected AMOTAs (frozen from results/paper_tables.md) ─────────────
# Keys are (experiment_name, stage_label). Values are AMOTA on the 0..1 scale.
# To extend, run paper_tables.py and add the new row(s) here verbatim.

_STAGE_V2V = "V2V (FP-ignore, base GT)"
_STAGE_OGFP = "OG+FP (FP-counted, base GT)"
_STAGE_OURS = "Ours (FP-counted, +ego GT)"
_STAGE_OURS_M = "Ours-merged (FP-counted, +ego GT, dup-merged)"

EXPECTED_AMOTA: Dict[Tuple[str, str], float] = {
    # V2V4Real-paper anchors (vanilla AB3DMOT, no GPEM, no CMR filter)
    ("LF_V2V4Real_anchor", _STAGE_V2V):     0.3954,
    ("LF_V2V4Real_anchor", _STAGE_OGFP):    0.3391,
    ("LF_V2V4Real_anchor", _STAGE_OURS):    0.2687,
    ("LF_V2V4Real_anchor", _STAGE_OURS_M):  0.3008,

    ("CoBEVT_V2V4Real_anchor", _STAGE_V2V):     0.3734,
    ("CoBEVT_V2V4Real_anchor", _STAGE_OGFP):    0.3444,
    ("CoBEVT_V2V4Real_anchor", _STAGE_OURS):    0.2702,
    ("CoBEVT_V2V4Real_anchor", _STAGE_OURS_M):  0.3057,

    # DMSTrack-paper anchors (per-source matching, no self-inject)
    ("DMSTrack_noinj_anchor", _STAGE_V2V):     0.4202,
    ("DMSTrack_noinj_anchor", _STAGE_OGFP):    0.3317,
    ("DMSTrack_noinj_anchor", _STAGE_OURS):    0.2645,
    ("DMSTrack_noinj_anchor", _STAGE_OURS_M):  0.2969,

    # Legacy DMSPerCavSweep* anchors (strong V2V baseline; exact flag provenance uncertain)
    ("DMSPerCavSweepEkf", _STAGE_V2V):     0.4678,
    ("DMSPerCavSweepEkf", _STAGE_OGFP):    0.3133,
    ("DMSPerCavSweepEkf", _STAGE_OURS):    0.2500,
    ("DMSPerCavSweepEkf", _STAGE_OURS_M):  0.2780,

    ("DMSPerCavSweepAkf", _STAGE_V2V):     0.4182,
    ("DMSPerCavSweepAkf", _STAGE_OGFP):    0.2664,
    ("DMSPerCavSweepAkf", _STAGE_OURS):    0.2151,
    ("DMSPerCavSweepAkf", _STAGE_OURS_M):  0.2341,

    ("DMSPerCavSweepCi", _STAGE_V2V):     0.4670,
    ("DMSPerCavSweepCi", _STAGE_OGFP):    0.3075,
    ("DMSPerCavSweepCi", _STAGE_OURS):    0.2442,
    ("DMSPerCavSweepCi", _STAGE_OURS_M):  0.2723,

    ("DMSPerCavSweepBici", _STAGE_V2V):     0.4670,
    ("DMSPerCavSweepBici", _STAGE_OGFP):    0.3075,
    ("DMSPerCavSweepBici", _STAGE_OURS):    0.2442,
    ("DMSPerCavSweepBici", _STAGE_OURS_M):  0.2723,

    ("DMSPerCavSweepSabre", _STAGE_V2V):     0.4433,
    ("DMSPerCavSweepSabre", _STAGE_OGFP):    0.3026,
    ("DMSPerCavSweepSabre", _STAGE_OURS):    0.2424,
    ("DMSPerCavSweepSabre", _STAGE_OURS_M):  0.2653,

    # Our paper's headline cooperative tracker
    ("DMS_S3NoNorm_sabre_quadratic_headline", _STAGE_V2V):     0.4197,
    ("DMS_S3NoNorm_sabre_quadratic_headline", _STAGE_OGFP):    0.3131,
    ("DMS_S3NoNorm_sabre_quadratic_headline", _STAGE_OURS):    0.2906,
    ("DMS_S3NoNorm_sabre_quadratic_headline", _STAGE_OURS_M):  0.3253,
}


# ── pytest plumbing ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def seqmap():
    return parse_seqmap(SEQMAP_VAL)


def _ids() -> List[str]:
    """Compact test id: ``<experiment_name>__<stage_short>``."""
    short = {
        _STAGE_V2V: "V2V",
        _STAGE_OGFP: "OGFP",
        _STAGE_OURS: "Ours",
        _STAGE_OURS_M: "OursMerged",
    }
    return [f"{n}__{short[s]}" for (n, s) in EXPECTED_AMOTA]


@pytest.mark.slow
@pytest.mark.parametrize(
    "experiment_name,stage_label,expected_amota",
    [(n, s, v) for (n, s), v in EXPECTED_AMOTA.items()],
    ids=_ids(),
)
def test_paper_anchor_amota_pinned(experiment_name: str, stage_label: str,
                                    expected_amota: float, seqmap) -> None:
    """Recompute AMOTA for one (experiment, stage) and assert no drift.

    SKIPs (rather than fails) when the experiment's tracker output isn't on
    disk yet — those experiments need to be run via v2v4real_runner.py first.
    """
    cfg = get_experiment(experiment_name)
    result_dir_name = cfg.result_dir_name()
    data_dir = V2V4REAL_RESULTS_ROOT / result_dir_name / "data_0"
    if not data_dir.is_dir() or not any(data_dir.glob("*.txt")):
        pytest.skip(f"tracker output for {experiment_name!r} not on disk "
                    f"({data_dir}). Run via:\n"
                    f"    python scripts/v2v4real_runner.py --experiment {experiment_name}")

    # Resolve the protocol + gt_subdir for this stage from paper_metrics.STAGES.
    matches = [s for s in STAGES if s[2] == stage_label]
    assert matches, f"stage_label {stage_label!r} not found in paper_metrics.STAGES"
    protocol, gt_subdir, _label = matches[0]

    result = compute_stage(result_dir_name, protocol, gt_subdir, seqmap)
    actual = float(result["amota"])

    diff = actual - expected_amota
    msg = (
        f"\n  experiment: {experiment_name}"
        f"\n  stage:      {stage_label}"
        f"\n  expected:   {expected_amota * 100:.4f}"
        f"\n  actual:     {actual * 100:.4f}"
        f"\n  delta:      {diff * 100:+.4f}"
        f"\n  tol:        {TOL_AMOTA * 100:.4f}"
        f"\n"
        f"\n  If this drift is intentional, re-pin by running"
        f"\n      python scripts/paper_tables.py"
        f"\n  and updating EXPECTED_AMOTA in this file."
    )
    assert abs(diff) <= TOL_AMOTA, msg
