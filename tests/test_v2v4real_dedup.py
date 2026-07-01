"""
Unit tests for v2v4real_replay._dedup_gt_for_frame.

Covers the three behaviours that earlier had bugs:

1. ass_id=-1 rows are *not* collapsed into a single entry. They are
   spatially deduplicated against the same gate that positive ass_ids use.
2. Positive ass_id rows still take precedence in same-cluster decisions
   (the cross-vehicle association is more reliable annotation than a
   single-vehicle ass_id=-1 row).
3. GT rows whose position falls within ``ego_exclusion_gate_m`` of any
   ego pose are dropped (V2V4Real annotates the OTHER ego as a "car" GT
   in each ego's perspective; egos aren't tracking targets).

Run: PYTHONPATH=src python -m pytest tests/test_v2v4real_dedup.py -v
"""
import os
import sys
from pathlib import Path

# Make src/ importable when running directly without pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import sensor_fusion FIRST to break circular imports between sensor.py
# and sensor_fusion.py (same workaround used in v2v4real_replay.py).
import sensor_fusion  # noqa: F401  (imported for side effects)

import v2v4real_replay


def _row(x, y, ass_id, obj_id=1, vehicle="astuff", label="car"):
    """Helper: build the dict shape `_dedup_gt_for_frame` expects."""
    return {
        "ass_id": int(ass_id),
        "obj_id": int(obj_id),
        "vehicle_id": vehicle,
        "x": float(x),
        "y": float(y),
        "yaw": 0.0,
        "length": 4.5,
        "width": 2.0,
        "height": 1.5,
        "label": label,
    }


def test_empty_input_returns_empty_list():
    assert v2v4real_replay._dedup_gt_for_frame([]) == []


def test_single_row_kept():
    rows = [_row(10, 5, ass_id=3)]
    out = v2v4real_replay._dedup_gt_for_frame(rows)
    assert len(out) == 1
    assert out[0].centroid == [10.0, 5.0]


def test_multiple_minus_one_rows_NOT_collapsed():
    """The original bug: ass_id=-1 rows were keyed by ass_id and overwritten,
    keeping only one of N legitimate per-vehicle GT entries."""
    rows = [
        _row(10, 5,  ass_id=-1, vehicle="astuff", obj_id=1),
        _row(50, -3, ass_id=-1, vehicle="astuff", obj_id=2),
        _row(20, 30, ass_id=-1, vehicle="tesla",  obj_id=3),
    ]
    out = v2v4real_replay._dedup_gt_for_frame(rows)
    assert len(out) == 3, "All three spatially-distinct ass_id=-1 rows should survive"
    centroids = sorted([(o.centroid[0], o.centroid[1]) for o in out])
    assert centroids == [(10.0, 5.0), (20.0, 30.0), (50.0, -3.0)]


def test_spatial_dedup_within_gate():
    """Two rows of the same physical car (one from each ego) should collapse
    to one entry when they're within the spatial gate (2 m)."""
    rows = [
        _row(10.0, 5.0, ass_id=3,  vehicle="astuff"),
        _row(10.5, 5.3, ass_id=-1, vehicle="tesla"),  # same car, V2V4Real failed cross-association
    ]
    out = v2v4real_replay._dedup_gt_for_frame(rows)
    assert len(out) == 1


def test_spatial_dedup_outside_gate():
    """Two rows of physically different cars (>2 m apart) must NOT merge."""
    rows = [
        _row(10.0, 5.0, ass_id=3, vehicle="astuff"),
        _row(13.0, 8.0, ass_id=4, vehicle="astuff"),  # 4.24 m away
    ]
    out = v2v4real_replay._dedup_gt_for_frame(rows)
    assert len(out) == 2


def test_positive_ass_id_priority_in_cluster():
    """When positive-ass_id and ass_id=-1 rows fall in the same spatial
    cluster, the positive-ass_id row should be kept (more reliable annotation)."""
    rows = [
        _row(10.0, 5.0, ass_id=-1, vehicle="astuff", obj_id=99),  # listed first
        _row(10.4, 5.2, ass_id=7,  vehicle="tesla",  obj_id=1),   # but should win
    ]
    out = v2v4real_replay._dedup_gt_for_frame(rows)
    assert len(out) == 1
    assert out[0].vehicle_id == 7, "ass_id=7 row should be retained over ass_id=-1"


def test_ego_exclusion():
    """GT rows within ego_exclusion_gate_m of any ego pose are dropped
    (V2V4Real annotates the OTHER ego as a car in each ego's GT).

    Ego exclusion is OPT-IN: _dedup_gt_for_frame defaults ego_exclusion_gate_m=0.0
    (documented in its docstring — V2V4Real treats the other ego as a legitimate
    tracking target, so the production caller keeps them). To exercise the
    exclusion feature we must pass a positive gate; 2.0 m covers the ~0.3 m
    ego-proximity offsets in the fixture below."""
    ego_poses = [(0.0, 0.0), (50.0, 50.0)]
    rows = [
        _row(0.5, 0.5, ass_id=-1, vehicle="tesla"),    # inside ego_0 gate, drop
        _row(50.2, 49.8, ass_id=-1, vehicle="astuff"),  # inside ego_1 gate, drop
        _row(10.0, 5.0, ass_id=-1, vehicle="astuff"),  # legitimate, keep
        _row(30.0, 30.0, ass_id=-1, vehicle="tesla"),   # legitimate, keep
    ]
    out = v2v4real_replay._dedup_gt_for_frame(
        rows, ego_poses=ego_poses, ego_exclusion_gate_m=2.0)
    assert len(out) == 2
    centroids = sorted([(o.centroid[0], o.centroid[1]) for o in out])
    assert centroids == [(10.0, 5.0), (30.0, 30.0)]


def test_v2v4real_paper_realistic_frame():
    """Realistic test based on the actual frame 0 of test__Day19__...10-29-43_3.
    Before the fix this returned 4 GTs (ass_id keys: -1, 3, 5, 4); the fix
    should return ~10-11 (one per physical object after spatial dedup, with
    egos excluded)."""
    ego_poses = [(-2.27, -104.69), (0.58, -62.79)]  # astuff, tesla
    rows = [
        # Tesla's GT view (5 entries)
        _row(-2.29, -103.42, -1, vehicle="tesla", obj_id=1),    # = astuff ego, should be EXCLUDED
        _row(-15.34, -120.22, -1, vehicle="tesla", obj_id=3),   # unique car
        _row(22.50, -155.04, -1, vehicle="tesla", obj_id=4),    # unique car
        _row(-4.09, -141.94, -1, vehicle="tesla", obj_id=5),    # close to astuff,5,3 (same truck)
        _row(-13.81, -148.20, -1, vehicle="tesla", obj_id=6),   # unique-ish
        # Astuff's GT view (8 entries)
        _row(-3.98, -122.01, -1, vehicle="astuff", obj_id=1),
        _row(-15.35, -122.03, 3, vehicle="astuff", obj_id=2),   # cross-assoc with tesla,3
        _row(-5.01, -143.35, 5, vehicle="astuff", obj_id=3),    # cross-assoc with tesla,5
        _row(-12.53, -151.83, -1, vehicle="astuff", obj_id=4),
        _row(30.22, -145.66, -1, vehicle="astuff", obj_id=5),
        _row(24.17, -156.99, 4, vehicle="astuff", obj_id=6),
        _row(0.57, -62.74, -1, vehicle="astuff", obj_id=7),     # = tesla ego, should be EXCLUDED
        _row(-20.38, -17.16, -1, vehicle="astuff", obj_id=8),
    ]
    # Ego exclusion is opt-in (gate defaults to 0.0); pass a positive gate to
    # exercise it. 2.0 m covers the largest ego-proximity offset here (~1.27 m
    # for the astuff-ego row at (-2.29, -103.42) vs ego (-2.27, -104.69)).
    out = v2v4real_replay._dedup_gt_for_frame(
        rows, ego_poses=ego_poses, ego_exclusion_gate_m=2.0)
    # After ego exclusion (-2 entries) and spatial dedup of close pairs
    # (3 from tesla collapse with their astuff counterparts), expect ~8-9
    # unique physical cars / trucks.
    assert 7 <= len(out) <= 9, f"Expected ~8 dedup'd GTs after ego exclusion, got {len(out)}"


if __name__ == "__main__":
    # Allow plain `python tests/test_v2v4real_dedup.py` for quick smoke runs.
    tests = [
        test_empty_input_returns_empty_list,
        test_single_row_kept,
        test_multiple_minus_one_rows_NOT_collapsed,
        test_spatial_dedup_within_gate,
        test_spatial_dedup_outside_gate,
        test_positive_ass_id_priority_in_cluster,
        test_ego_exclusion,
        test_v2v4real_paper_realistic_frame,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")
            failures += 1
    if failures:
        print(f"\n{failures} test(s) failed.")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
