"""log_odds pipeline — single-scenario wall-clock perf regression test.

Catches regressions in the per-frame fusion / matching hot path. The bottleneck
as of 2026-06-08 is ``geometry._clip_edge`` (Sutherland-Hodgman polygon clipping)
inside ``rotated_box_iou``; on the largest scenario one streams_keep=1 run
finishes well under the threshold below. Crossing the threshold means either:

  * an expensive operation regressed into per-frame code,
  * the streams_keep filter stopped suppressing the unused 29 streams,
  * the polygon clipper got slower (e.g. a Python fallback replaced a vectorised
    path).

The test deliberately uses ``streams_keep=('ci_gpem_quadratic',)`` so it isolates
the per-frame work to ONE filter+cov combo. With the full 30-stream sweep, this
test would have to set the threshold ~30× larger and would be much less precise
as a regression detector.

Run::

    pytest tests/test_log_odds_perf.py --runslow

Marked ``slow`` because it loads a real scenario + runs the tracker. Typical
runtime <30s on a modern workstation. CI must opt in via ``--runslow``.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PERF_THRESHOLD_S = float(os.environ.get("CMR_LOG_ODDS_PERF_THRESHOLD_S", "120.0"))
"""Wall-clock budget for ONE scenario through ONE fusion stream.

Default 120s gives ~3x headroom over the observed ~30-60s on a workstation
(scenario size varies). Override via env var if you're running on a slower
machine — but a regression that needs >120s on a single-stream single-scenario
run is almost certainly real."""


def _find_a_scenario() -> Path:
    """Locate one cmr_export scenario to feed the tracker.

    Prefers the pp_score0 bucket (the canonical headline input). Skips the
    test if no scenario is available — keeps the test file importable on
    a fresh CI clone that hasn't fetched buckets yet.
    """
    bucket_root = REPO / "data" / "v2v4real_inputs" / "ours" / "detectors" / "pp_score0"
    if not bucket_root.is_dir():
        pytest.skip("ours/detectors/pp_score0 bucket not on disk")
    # Build the merged view to find a scenario shaped like run_scenario expects.
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from input_bucket.merged_scenario_view import merged_view_from_bucket  # type: ignore
    except Exception as e:
        pytest.skip(f"merged_view_from_bucket import failed: {e}")
    return bucket_root, merged_view_from_bucket


@pytest.mark.slow
def test_single_scenario_single_stream_under_threshold():
    """One scenario × one stream MUST finish under PERF_THRESHOLD_S.

    Threshold rationale: the per-frame work is dominated by polygon clipping
    in rotated_box_iou. With streams_keep=1, only ~1/30 of that work runs
    per frame. Single-scenario tape ~600 frames; observed total ~30-60s.
    """
    bucket_root, merged_view_from_bucket = _find_a_scenario()
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "configs"))
    import v2v4real_replay  # type: ignore

    params = {
        "detector_name": "pointpillar_v2v4real_{vehicle}",
        "detector_max_range": 100.0,
        "score_threshold": 0.0,
        "lifecycle_mode": "log_odds",
        "p_tp_birth_gate": 0.30,
        "localizer_name": "rtk_v2v4real",
        "vehicle_classes": ["car", "truck", "bus", "construction_vehicle"],
        "use_static_matching": True,
        "self_report_egos": False,
        "record_tape_for": ["ci_gpem_quadratic"],
        "streams_keep": ["ci_gpem_quadratic"],
    }

    with merged_view_from_bucket(bucket_root) as merged_dir:
        scenarios = sorted(d for d in merged_dir.iterdir()
                           if d.is_dir() and d.name.startswith("test__"))
        if not scenarios:
            pytest.skip(f"no test__* scenarios in merged view {merged_dir}")
        # Use the first scenario for determinism.
        scen_dir = scenarios[0]

        t0 = time.time()
        result = v2v4real_replay.run_scenario(scen_dir, params)
        dt = time.time() - t0

    # Sanity check: the tracker actually ran something and the tape is present.
    tapes = result.get("match_tape_per_stream", {}) or {}
    assert "ci_gpem_quadratic" in tapes, (
        f"streams_keep filter dropped dump_stream; got tape keys: "
        f"{sorted(tapes.keys())!r}"
    )

    # Headline assertion: wall-clock under threshold.
    print(f"\n  [perf] scenario={scen_dir.name} wall_clock={dt:.2f}s "
          f"threshold={PERF_THRESHOLD_S}s")
    assert dt < PERF_THRESHOLD_S, (
        f"log_odds single-scenario single-stream run took {dt:.2f}s, "
        f"exceeds {PERF_THRESHOLD_S}s budget. Likely causes: "
        f"(1) streams_keep filter broke (29 unused streams now running per frame), "
        f"(2) per-frame work regressed in fusion / rotated_box_iou / Sutherland-Hodgman, "
        f"(3) calibration loader is being hit per-frame instead of once at startup. "
        f"Profile with cProfile + snakeviz on the same scenario to localise."
    )


@pytest.mark.slow
def test_streams_keep_actually_skips_unused_streams():
    """streams_keep=1 must be measurably faster than the full 30-stream sweep.

    Catches the case where streams_keep is accepted by the API but doesn't
    actually shortcut the per-frame loop (silent perf regression).
    """
    bucket_root, merged_view_from_bucket = _find_a_scenario()
    sys.path.insert(0, str(REPO / "src"))
    import v2v4real_replay  # type: ignore

    base_params = {
        "detector_name": "pointpillar_v2v4real_{vehicle}",
        "detector_max_range": 100.0,
        "score_threshold": 0.0,
        "lifecycle_mode": "log_odds",
        "p_tp_birth_gate": 0.30,
        "localizer_name": "rtk_v2v4real",
        "vehicle_classes": ["car", "truck", "bus", "construction_vehicle"],
        "use_static_matching": True,
        "self_report_egos": False,
        "record_tape_for": ["ci_gpem_quadratic"],
    }

    with merged_view_from_bucket(bucket_root) as merged_dir:
        scenarios = sorted(d for d in merged_dir.iterdir()
                           if d.is_dir() and d.name.startswith("test__"))
        if not scenarios:
            pytest.skip(f"no test__* scenarios in merged view {merged_dir}")
        # Use the smallest scenario so the all-30 sweep isn't punishing in CI.
        scen_dir = min(scenarios,
                       key=lambda d: sum(1 for _ in (d / "detections.csv").open()))

        # Full 30-stream sweep.
        t0 = time.time()
        v2v4real_replay.run_scenario(scen_dir, dict(base_params))
        dt_full = time.time() - t0

        # Single-stream sweep.
        t0 = time.time()
        v2v4real_replay.run_scenario(scen_dir, dict(base_params,
                                                    streams_keep=["ci_gpem_quadratic"]))
        dt_one = time.time() - t0

    print(f"\n  [perf] scen={scen_dir.name} 30-stream={dt_full:.1f}s  "
          f"1-stream={dt_one:.1f}s  ratio={dt_full/max(dt_one,1e-3):.1f}x")
    # The conservative bar: 1-stream must be at least 1.5x faster than 30-stream.
    # If streams_keep was broken (no-op), the ratio would be ~1.0.
    assert dt_one * 1.5 < dt_full, (
        f"streams_keep=1 ({dt_one:.1f}s) not measurably faster than full 30-stream "
        f"({dt_full:.1f}s). streams_keep filter may be silently ignored."
    )
