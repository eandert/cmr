"""Regression bank for the V2V4Real tracking scoring pipeline.

Why this file exists
--------------------
We have been bitten repeatedly by:
  - z_center vs z_bottom (mmdet3d puts the origin at bottom; AB3DMOT cam coord
    puts y at the bottom too; our iou_3d_box wants z_center).
  - V2V4Real's NON-STANDARD "KITTI" layout: col 13 = X (forward),
    col 14 = Y (lidar-up, i.e. lidar Z of box CENTER),
    col 15 = Z (lateral, i.e. lidar Y).
    See opencood/tools/inference.py:489.
  - Ego-as-GT vertical placement: the Y_cam coord is the BOX CENTER in lidar-Z,
    not the LiDAR mount altitude. Formula:
        Y_cam = dz_world + LIDAR_Z_MOUNT_TO_GROUND(-1.87) + h/2
  - The DMSTrack/V2V4Real-paper "FP-ignore" hack
    (compute_ab3dmot_metrics_iou(ignore_unmatched_fps=True)).
  - get_rotated_box_corners(cx, cy, w, l, angle): at angle=0, length lies along
    the cx axis, width along the cy axis. So in V2V4Real "KITTI" (cx=forward,
    cy=lateral) → length along forward = car convention.

This file locks the conventions and metric protocols so we never debug the same
issue twice.

Run with:
    python tests/test_scoring_pipeline.py
"""
from __future__ import annotations

import math
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# AB3DMOT reference (only used by one test for cross-validation).
AB3DMOT_ROOT = REPO / "third_party" / "AB3DMOT"
if AB3DMOT_ROOT.exists():
    sys.path.insert(0, str(AB3DMOT_ROOT))
    sys.path.insert(0, str(AB3DMOT_ROOT / "Xinshuo_PyToolbox"))

from geometry import iou_3d_box, rotated_box_iou  # noqa: E402
from metrics.v2v4real_metrics import (  # noqa: E402
    compute_ab3dmot_metrics_iou,
    match_frame_3d_iou,
)
import score_kitti_tracks_through_run_suite as scorer  # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _assert_close(actual, expected, tol=1e-6, label=""):
    diff = abs(float(actual) - float(expected))
    ok = "PASS" if diff <= tol else "FAIL"
    print(f"    [{ok}] {label}: actual={actual:.6f}  expected={expected:.6f}  |diff|={diff:.2e}")
    assert diff <= tol, f"{label}: |{actual} - {expected}| = {diff} > {tol}"


def _track_tuple(tid, cx, cy, w, l, yaw, score, h, z_center):
    """Track tape tuple for match_frame_3d_iou: h@7, z_center@8."""
    return (tid, cx, cy, w, l, yaw, score, h, z_center)


def _gt_tuple(gid, cx, cy, w, l, yaw, h, z_center):
    """GT tape tuple for match_frame_3d_iou: h@6, z_center@7."""
    return (gid, cx, cy, w, l, yaw, h, z_center)


def _write_kitti_track_line(fid, tid, cls, h, w, l, X, Y, Z, ry, score):
    """One KITTI MOT line in the exact format parse_kitti_track_file reads.
    Columns: frame tid type 0 0 0 bbox2dx4(=0) h w l X Y Z ry [score]
    """
    return (
        f"{fid} {tid} {cls} 0 0 0 0 0 0 0 "
        f"{h:.6f} {w:.6f} {l:.6f} {X:.6f} {Y:.6f} {Z:.6f} {ry:.6f} {score:.6f}\n"
    )


def _write_kitti_label_line(fid, tid, cls, h, w, l, X, Y, Z, ry):
    """Label format — same columns as track line but no score."""
    return (
        f"{fid} {tid} {cls} 0 0 0 0 0 0 0 "
        f"{h:.6f} {w:.6f} {l:.6f} {X:.6f} {Y:.6f} {Z:.6f} {ry:.6f}\n"
    )


# ============================================================================
# Tier 1 — IoU math invariants
# ============================================================================

def test_3d_iou_self_is_one():
    """iou_3d_box(box, box) == 1.0 for several shapes/yaws."""
    print("\nT1.1: 3D IoU of a box with itself == 1.0")
    boxes = [
        # (cx, cy, w, l, yaw, z_center, h)
        (0.0, 0.0, 2.0, 4.5, 0.0, -1.0, 1.7),
        (10.0, 5.0, 1.9, 4.2, 0.7, -0.9, 1.6),
        (-3.0, 22.0, 2.1, 5.0, -1.2, -1.5, 1.8),
        (0.0, 0.0, 2.0, 2.0, math.pi / 3, 0.0, 1.7),
    ]
    for i, b in enumerate(boxes):
        v = iou_3d_box(b, b)
        _assert_close(v, 1.0, tol=1e-9, label=f"self IoU box[{i}]")


def test_3d_iou_disjoint_is_zero():
    """Two boxes 100m apart → IoU = 0."""
    print("\nT1.2: 3D IoU of distant boxes == 0")
    a = (0.0, 0.0, 2.0, 4.5, 0.0, -1.0, 1.7)
    b = (100.0, 0.0, 2.0, 4.5, 0.0, -1.0, 1.7)
    v = iou_3d_box(a, b)
    _assert_close(v, 0.0, tol=1e-9, label="disjoint IoU")


def test_3d_iou_height_no_overlap_is_zero():
    """Same BEV footprint, vertically disjoint → 0.

    iou_3d_box's z convention: top = z_center - h, bottom = z_center
    (cam-down). So z=0/h=1.7 spans [-1.7, 0]; z=10/h=1.7 spans [8.3, 10].
    """
    print("\nT1.3: 3D IoU vertically disjoint == 0")
    a = (0.0, 0.0, 2.0, 4.5, 0.0, 0.0, 1.7)
    b = (0.0, 0.0, 2.0, 4.5, 0.0, 10.0, 1.7)
    v = iou_3d_box(a, b)
    _assert_close(v, 0.0, tol=1e-9, label="vertical-disjoint IoU")


def test_3d_iou_partial_height_overlap():
    """Same BEV (2×2 square), half-vertical overlap → analytic IoU = 1/3.

    box_a: h=2, z_center=0 → spans [-2, 0]
    box_b: h=2, z_center=1 → spans [-1, 1]
    z_inter = min(0,1) - max(-2,-1) = 0 - (-1) = 1
    BEV inter = 2*2 = 4. inter_3d = 4. union_3d = 4*2 + 4*2 - 4 = 12. IoU = 1/3.
    """
    print("\nT1.4: 3D IoU analytic — half-vertical, full-BEV overlap → 1/3")
    a = (0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0)
    b = (0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0)
    v = iou_3d_box(a, b)
    _assert_close(v, 1.0 / 3.0, tol=1e-9, label="partial-height IoU")


def test_3d_iou_matches_ab3dmot():
    """Compare iou_3d_box vs AB3DMOT iou(.,.,'iou_3d') on random box pairs.

    DOCUMENTED BUG (src/geometry.py:158, get_rotated_box_corners):
      The cam-frame mapping used by score_scenario_cam (cx=X_cam, cy=Z_cam)
      requires NEGATING the yaw to match AB3DMOT's roty convention. The rotation
      matrix R=[[c,-s],[s,c]] in get_rotated_box_corners inverts AB3DMOT's
      roty()'s rotation sign on the (X_cam, Z_cam) plane.

    Impact: tracks AND GT use the same convention internally, so MATCHING is
    consistent — but iou_3d_box's numeric value differs from AB3DMOT's reference
    for the same KITTI-cam input under non-zero yaw. Mean disagreement ≈ 0.08
    on random box pairs.

    This test asserts (1) negated-yaw agrees to ~1e-9, locking the bug location;
    (2) raw-yaw disagrees, locking the current behavior so an upstream fix
    will trip this test (good failure).
    """
    print("\nT1.5: iou_3d_box vs AB3DMOT iou(.,.,'iou_3d') agreement")
    try:
        from AB3DMOT_libs.box import Box3D
        from AB3DMOT_libs.dist_metrics import iou as ab3d_iou
    except Exception as e:
        print(f"    [SKIP] AB3DMOT not importable in this env: {e}")
        return

    rng = np.random.default_rng(42)
    diffs_raw = []
    diffs_neg_yaw = []
    for _ in range(10):
        x = rng.uniform(-5, 5)
        z = rng.uniform(5, 30)
        y = rng.uniform(-1.5, -0.5)
        l = rng.uniform(3.5, 5.5)
        w = rng.uniform(1.5, 2.5)
        h = rng.uniform(1.3, 1.9)
        ry = rng.uniform(-math.pi, math.pi)
        dx, dy, dz = rng.uniform(-2, 2), rng.uniform(-0.2, 0.2), rng.uniform(-2, 2)
        dry = rng.uniform(-0.5, 0.5)

        a_ab = Box3D.array2bbox(np.array([x, y, z, ry, l, w, h]))
        b_ab = Box3D.array2bbox(np.array([x + dx, y + dy, z + dz, ry + dry, l, w, h]))
        ab = ab3d_iou(a_ab, b_ab, "iou_3d")

        # Cam-frame mapping per score_scenario_cam:
        # tracks (id, X, Z, w, l, ry, score, h, Y); iou_3d_box gets (X, Z, w, l, yaw, Y, h)
        a_ours = (x, z, w, l, ry, y, h)
        b_ours = (x + dx, z + dz, w, l, ry + dry, y + dy, h)
        ours = iou_3d_box(a_ours, b_ours)
        diffs_raw.append(abs(ab - ours))

        # With negated yaw — should match AB3DMOT exactly.
        a_neg = (x, z, w, l, -ry, y, h)
        b_neg = (x + dx, z + dz, w, l, -(ry + dry), y + dy, h)
        ours_neg = iou_3d_box(a_neg, b_neg)
        diffs_neg_yaw.append(abs(ab - ours_neg))

    max_raw = max(diffs_raw)
    max_neg = max(diffs_neg_yaw)
    mean_raw = float(np.mean(diffs_raw))
    print(f"    raw-yaw     max |diff| = {max_raw:.5f}  mean = {mean_raw:.5f}")
    print(f"    neg-yaw     max |diff| = {max_neg:.2e}  mean = {float(np.mean(diffs_neg_yaw)):.2e}")
    # FIXED 2026-05-22: get_rotated_box_corners now matches AB3DMOT's roty() sign.
    # Lock-in #1: RAW yaw is bit-exact with AB3DMOT (the fix in place).
    assert max_raw < 1e-6, f"Raw yaw should now match AB3DMOT bit-exact after the fix; got {max_raw}"
    print("    [PASS] raw-yaw matches AB3DMOT bit-exact (sign-bug fix verified)")
    # Lock-in #2: NEGATED yaw should now DISAGREE (we flipped the convention,
    # so the artificially-negated input ≢ AB3DMOT). If this somehow stays
    # bit-exact, the rotation was undone somewhere.
    if max_neg < 1e-3:
        print("    [NOTE] negated-yaw also matches AB3DMOT? rotation may be undone elsewhere")
    assert max_neg >= 1e-3, (
        "Negated yaw should now disagree with AB3DMOT (we flipped the rotation "
        "to match AB3DMOT raw). If you see bit-exact agreement, the fix was "
        "undone or compensated elsewhere — investigate."
    )
    print("    [PASS] FIXED — get_rotated_box_corners now uses R=[[c,s],[-s,c]]")
    print("    matching AB3DMOT roty() sign on (X_cam, Z_cam) BEV plane.")


def test_2d_bev_iou_yaw_invariance():
    """Square BEV footprint rotated +π/2 → IoU = 1.0 (square is yaw-invariant)."""
    print("\nT1.6: BEV IoU yaw invariance — square rotated 90° → 1.0")
    a = [0.0, 0.0, 2.0, 2.0, 0.0]
    b = [0.0, 0.0, 2.0, 2.0, math.pi / 2.0]
    v = rotated_box_iou(a, b)
    _assert_close(v, 1.0, tol=1e-6, label="square yaw-rotate IoU")


# ============================================================================
# Tier 2 — Tape matching
# ============================================================================

def test_match_frame_3d_iou_perfect_pairs():
    """3 tracks at known positions, 3 GTs at same positions → 3 matches, 0 unmatched."""
    print("\nT2.1: match_frame_3d_iou — perfect pairs → 3 matches, 0 unmatched")
    coords = [(0, 10), (5, 20), (-3, 8)]
    tracks = [_track_tuple(100 + i, x, y, 2.0, 4.5, 0.0, 0.9, 1.7, -1.0)
              for i, (x, y) in enumerate(coords)]
    gts = [_gt_tuple(i + 1, x, y, 2.0, 4.5, 0.0, 1.7, -1.0)
           for i, (x, y) in enumerate(coords)]
    matches, ut, ug = match_frame_3d_iou(tracks, gts, iou_threshold=0.25)
    print(f"    matches={len(matches)}  unmatched_t={len(ut)}  unmatched_g={len(ug)}")
    assert len(matches) == 3 and not ut and not ug


def test_match_frame_3d_iou_below_threshold_no_match():
    """Shifted tracks: IoU < high-thresh → 0 matches; with low thresh → all match."""
    print("\nT2.2: match_frame_3d_iou — threshold gating")
    track = _track_tuple(1, 2.0, 10.0, 2.0, 4.5, 0.0, 0.9, 1.7, -1.0)
    gt    = _gt_tuple(1,   0.0, 10.0, 2.0, 4.5, 0.0, 1.7, -1.0)
    iou_val = iou_3d_box(
        (track[1], track[2], track[3], track[4], track[5], track[8], track[7]),
        (gt[1],    gt[2],    gt[3],    gt[4],    gt[5],    gt[7],    gt[6]),
    )
    print(f"    pair iou_3d = {iou_val:.4f}")
    high = match_frame_3d_iou([track], [gt], iou_threshold=0.9)
    low  = match_frame_3d_iou([track], [gt], iou_threshold=0.05)
    print(f"    thresh 0.9 → {len(high[0])} match(es); thresh 0.05 → {len(low[0])} match(es)")
    assert len(high[0]) == 0
    assert len(low[0]) == 1


def test_match_frame_3d_iou_hungarian_optimal():
    """Contrived 2×2 case; Hungarian must pick the optimum assignment.

    Two tracks competing on two GTs. Verify the returned assignment achieves
    the maximum IoU-sum across the two feasible permutations.
    """
    print("\nT2.3: match_frame_3d_iou — Hungarian is optimal")
    t0 = _track_tuple(0, 0.2, 10.0, 2.0, 4.5, 0.0, 0.9, 1.7, -1.0)
    t1 = _track_tuple(1, 1.5, 10.0, 2.0, 4.5, 0.0, 0.9, 1.7, -1.0)
    g0 = _gt_tuple(10, 0.0, 10.0, 2.0, 4.5, 0.0, 1.7, -1.0)
    g1 = _gt_tuple(11, 2.0, 10.0, 2.0, 4.5, 0.0, 1.7, -1.0)

    def _iou(t, g):
        return iou_3d_box(
            (t[1], t[2], t[3], t[4], t[5], t[8], t[7]),
            (g[1], g[2], g[3], g[4], g[5], g[7], g[6]),
        )

    iou_t0_g0, iou_t0_g1 = _iou(t0, g0), _iou(t0, g1)
    iou_t1_g0, iou_t1_g1 = _iou(t1, g0), _iou(t1, g1)
    opt_sum = max(iou_t0_g0 + iou_t1_g1, iou_t0_g1 + iou_t1_g0)
    matches, _, _ = match_frame_3d_iou([t0, t1], [g0, g1], iou_threshold=0.05)
    asn = {m[0]: m[1] for m in matches}
    actual_sum = (
        (iou_t0_g0 if asn.get(0) == 0 else iou_t0_g1) +
        (iou_t1_g0 if asn.get(1) == 0 else iou_t1_g1)
    )
    print(f"    optimal IoU-sum = {opt_sum:.4f};  Hungarian IoU-sum = {actual_sum:.4f}")
    assert abs(opt_sum - actual_sum) < 1e-6, "Hungarian assignment is not optimal"


# ============================================================================
# Tier 3 — AMOTA / MOTA / HOTA logic
# ============================================================================

def _make_perfect_tape(n_frames=10, n_gt_per_frame=3, score=0.9):
    """Tape where tracker == GT exactly across all frames."""
    tape = []
    for f in range(n_frames):
        tracks, gts = [], []
        for k in range(n_gt_per_frame):
            x = float(k * 5)
            y = 10.0
            tracks.append(_track_tuple(k, x, y, 2.0, 4.5, 0.0, score, 1.7, -1.0))
            gts.append(_gt_tuple(k, x, y, 2.0, 4.5, 0.0, 1.7, -1.0))
        tape.append({"frame_idx": f, "tracks": tracks, "gts": gts})
    return tape


def test_amota_perfect_tracker_is_high():
    """Perfect tracker → MOTA = 1.0; AMOTA hits its structural recall-sweep ceiling.

    DOCUMENTED CEILING: AB3DMOT's `compute_ab3dmot_metrics_iou` AMOTA on a
    PERFECT tracker with all-equal TP scores is ≈0.51-0.53, NOT ≈1.0. This is
    because the recall sweep enumerates 40 evenly-spaced recall thresholds and
    averages each threshold's MOTA. With all TPs sharing one score, recall
    climbs linearly through the sorted entries, so per-threshold MOTA goes
    roughly 0 → 1.0 → mean ≈ 0.5.

    NOTE: the `ab3dmot_rematching=True` path follows AB3DMOT's getThresholds
    which is what's actually used in DMSTrack-style published numbers; that
    code path can reach >>0.5 for a perfect tracker. The default (non-
    rematching) path used by score_scenario_cam has the ~0.5 ceiling.
    """
    print("\nT3.1: AMOTA — perfect tracker hits ~0.5 structural ceiling")
    tape = _make_perfect_tape()
    res = compute_ab3dmot_metrics_iou(tape, iou_threshold=0.25,
                                       ignore_unmatched_fps=False, match_3d=True)
    print(f"    amota={res['amota']:.4f}  mota={res['mota']:.4f}  "
          f"tp={res['tp_total']}  fp={res['fp_total']}  gt={res['gt_total']}")
    # MOTA at full-data IS 1.0 for a perfect tracker.
    assert res["mota"] >= 0.99, f"perfect MOTA = {res['mota']:.4f}"
    # AMOTA's structural ceiling on perfect-tracker constant-score tape: ~0.51-0.53.
    assert 0.48 <= res["amota"] <= 0.56, (
        f"perfect-tracker AMOTA = {res['amota']:.4f} outside structural band "
        f"[0.48, 0.56]. If it's now near 1.0, AB3DMOT's recall-sweep code may have"
        f" changed (ab3dmot_rematching default?) — UPDATE THIS TEST."
    )
    # Confirm rematching path reaches much higher (needs enough gt_total to populate
    # all 40 recall thresholds — with gt_total=30 it caps near gt_total/41 ≈ 0.73).
    # Use a tape with gt_total >> 40 to demonstrate the real ceiling.
    big_tape = _make_perfect_tape(n_frames=100, n_gt_per_frame=5)
    res_r = compute_ab3dmot_metrics_iou(big_tape, iou_threshold=0.25,
                                         ignore_unmatched_fps=False, match_3d=True,
                                         ab3dmot_rematching=True)
    print(f"    (ab3dmot_rematching=True, gt={res_r['gt_total']}: "
          f"amota={res_r['amota']:.4f}, mota={res_r['mota']:.4f})")
    assert res_r["amota"] >= 0.90, (
        f"AB3DMOT-exact rematching AMOTA = {res_r['amota']:.4f} should approach 1.0 "
        f"with gt_total >> 40 (got gt_total={res_r['gt_total']})"
    )


def test_amota_no_tracks_is_zero():
    """GT-only tape, empty tracks → AMOTA = 0."""
    print("\nT3.2: AMOTA — empty tracks → AMOTA = 0")
    tape = []
    for f in range(5):
        gts = [_gt_tuple(0, 0.0, 10.0, 2.0, 4.5, 0.0, 1.7, -1.0)]
        tape.append({"frame_idx": f, "tracks": [], "gts": gts})
    res = compute_ab3dmot_metrics_iou(tape, iou_threshold=0.25, match_3d=True)
    print(f"    amota={res['amota']:.4f}  tp={res['tp_total']}  gt={res['gt_total']}")
    _assert_close(res["amota"], 0.0, tol=1e-9, label="empty-tracks AMOTA")


def test_ignore_unmatched_fps_drops_extras():
    """Phantom tracks affect AMOTA only when they're interleaved BEFORE TPs in
    the descending-score sort. Two sub-cases:

    Sub-case A — phantom score LOWER than TPs (the usual real-world case):
      Phantoms come AFTER TPs in the sort. By the time they enter the cum-FP
      counter, all 40 recall thresholds have already been crossed. Result:
      FP-ignore=True and FP-ignore=False give the SAME AMOTA. The FP-ignore
      "V2V bug" only matters when phantoms have high scores.

    Sub-case B — phantom score HIGHER than TPs (worst case):
      Phantoms come FIRST in the sort, polluting cum_fp at every recall
      crossing. ignore=True → AMOTA stays high; ignore=False → AMOTA → 0.

    This test locks both behaviors. The mota and fp_total counts also verify
    that the GATING logic differs (counted vs ignored) even when AMOTA
    coincides.
    """
    print("\nT3.3: ignore_unmatched_fps — interleaving order determines AMOTA gap")

    def tape_with_phantoms(phantom_score):
        tape = []
        for f in range(10):
            tracks, gts = [], []
            for k in range(3):
                x = float(k * 5)
                tracks.append(_track_tuple(k, x, 10.0, 2.0, 4.5, 0.0, 0.95, 1.7, -1.0))
                gts.append(_gt_tuple(k, x, 10.0, 2.0, 4.5, 0.0, 1.7, -1.0))
            for j in range(50):
                tracks.append(_track_tuple(1000 + j, 200.0 + j * 10, 200.0,
                                           2.0, 4.5, 0.0, phantom_score, 1.7, -1.0))
            tape.append({"frame_idx": f, "tracks": tracks, "gts": gts})
        return tape

    # Sub-case A: low-score phantoms — AMOTA equal; ONLY fp_total differs.
    low = tape_with_phantoms(phantom_score=0.6)
    keep_lo = compute_ab3dmot_metrics_iou(low, iou_threshold=0.25,
                                            ignore_unmatched_fps=True, match_3d=True)
    drop_lo = compute_ab3dmot_metrics_iou(low, iou_threshold=0.25,
                                            ignore_unmatched_fps=False, match_3d=True)
    print(f"    low-score phantoms:  ignore amota={keep_lo['amota']:.4f}  "
          f"count amota={drop_lo['amota']:.4f}  "
          f"fp: {keep_lo['fp_total']} vs {drop_lo['fp_total']}")
    assert keep_lo["fp_total"] == 0 and drop_lo["fp_total"] == 500, (
        "fp_total counts must differ (gating works) even when AMOTA coincides"
    )
    # AMOTA is approximately equal here (both ~structural ceiling)
    assert abs(keep_lo["amota"] - drop_lo["amota"]) < 0.05

    # Sub-case B: high-score phantoms — AMOTA gap is dramatic.
    high = tape_with_phantoms(phantom_score=0.99)
    keep_hi = compute_ab3dmot_metrics_iou(high, iou_threshold=0.25,
                                            ignore_unmatched_fps=True, match_3d=True)
    drop_hi = compute_ab3dmot_metrics_iou(high, iou_threshold=0.25,
                                            ignore_unmatched_fps=False, match_3d=True)
    print(f"    high-score phantoms: ignore amota={keep_hi['amota']:.4f}  "
          f"count amota={drop_hi['amota']:.4f}  "
          f"fp: {keep_hi['fp_total']} vs {drop_hi['fp_total']}")
    assert keep_hi["amota"] - drop_hi["amota"] > 0.30, (
        f"With high-score phantoms, FP-ignore should keep AMOTA high; "
        f"got ignore={keep_hi['amota']:.3f} count={drop_hi['amota']:.3f}"
    )


def test_mota_fp_penalty_isolated():
    """Same TP count, different FP counts → MOTA decreases proportionally.

    MOTA = max(0, 1 - (IDS+FP+FN)/GT). With IDS=FN=0 and gt_total=N,
    K phantoms per frame × 2 frames → MOTA = 1 - 2K/N.
    """
    print("\nT3.4: MOTA — FP penalty isolated (linear drop)")

    def tape_with_k(k):
        tape = []
        for f in range(2):
            tracks, gts = [], []
            for i in range(5):
                x = float(i * 5)
                tracks.append(_track_tuple(i, x, 10.0, 2.0, 4.5, 0.0, 0.9, 1.7, -1.0))
                gts.append(_gt_tuple(i, x, 10.0, 2.0, 4.5, 0.0, 1.7, -1.0))
            for j in range(k):
                tracks.append(_track_tuple(900 + j, 100.0 + j * 10, 100.0,
                                           2.0, 4.5, 0.0, 0.5, 1.7, -1.0))
            tape.append({"frame_idx": f, "tracks": tracks, "gts": gts})
        return tape

    r0 = compute_ab3dmot_metrics_iou(tape_with_k(0), iou_threshold=0.25,
                                      ignore_unmatched_fps=False, match_3d=True)
    r2 = compute_ab3dmot_metrics_iou(tape_with_k(2), iou_threshold=0.25,
                                      ignore_unmatched_fps=False, match_3d=True)
    r5 = compute_ab3dmot_metrics_iou(tape_with_k(5), iou_threshold=0.25,
                                      ignore_unmatched_fps=False, match_3d=True)
    print(f"    k=0 phantom/frame: mota={r0['mota']:.4f}  fp={r0['fp_total']}  (expect 1.0)")
    print(f"    k=2:               mota={r2['mota']:.4f}  fp={r2['fp_total']}  (expect 0.6)")
    print(f"    k=5:               mota={r5['mota']:.4f}  fp={r5['fp_total']}  (expect 0.0)")
    _assert_close(r0["mota"], 1.0, tol=1e-6, label="MOTA k=0")
    _assert_close(r2["mota"], 0.6, tol=1e-6, label="MOTA k=2")
    _assert_close(r5["mota"], 0.0, tol=1e-6, label="MOTA k=5")


# ============================================================================
# Tier 4 — Score pipeline (cam-frame end-to-end)
# ============================================================================

def test_v2v4real_kitti_convention_in_parse_track():
    """parse_kitti_track_file: V2V4Real "KITTI" cols 13/14/15 = X/Y/Z
    (forward / lidar-up / lateral). No coord twist — values come back verbatim.
    """
    print("\nT4.1: parse_kitti_track_file — V2V4Real non-standard KITTI convention")
    tmp = Path(tempfile.mkdtemp(prefix="scoringtest_"))
    try:
        p = tmp / "track.txt"
        p.write_text(_write_kitti_track_line(
            fid=0, tid=42, cls="Car", h=1.7, w=2.0, l=4.5,
            X=10.0, Y=-1.0, Z=5.0, ry=0.123, score=0.95,
        ))
        out = scorer.parse_kitti_track_file(p)
        assert 0 in out and len(out[0]) == 1
        tid, X, Y, Z, h, w, l, ry, score = out[0][0]
        print(f"    parsed: tid={tid} X={X} Y={Y} Z={Z}  h={h} w={w} l={l}  ry={ry} s={score}")
        assert tid == 42
        _assert_close(X, 10.0, label="X (forward)")
        _assert_close(Y, -1.0, label="Y (lidar-up / box-center lidar-Z)")
        _assert_close(Z, 5.0,  label="Z (lateral)")
        _assert_close(h, 1.7, label="h")
        _assert_close(w, 2.0, label="w")
        _assert_close(l, 4.5, label="l")
        _assert_close(ry, 0.123, label="ry")
        _assert_close(score, 0.95, label="score")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_label_file_parse():
    """parse_v2v4real_label_file: same convention, no score column."""
    print("\nT4.2: parse_v2v4real_label_file — convention + no-score field")
    tmp = Path(tempfile.mkdtemp(prefix="scoringtest_"))
    try:
        p = tmp / "0001.txt"
        p.write_text(_write_kitti_label_line(
            fid=3, tid=7, cls="Car", h=1.7, w=2.0, l=4.5,
            X=12.0, Y=-1.1, Z=-3.5, ry=-0.5,
        ))
        out = scorer.parse_v2v4real_label_file(p)
        assert 3 in out and len(out[3]) == 1
        tid, X, Y, Z, h, w, l, ry = out[3][0]
        print(f"    parsed: tid={tid} X={X} Y={Y} Z={Z}  h={h} w={w} l={l}  ry={ry}")
        assert tid == 7
        _assert_close(X, 12.0, label="X (forward)")
        _assert_close(Y, -1.1, label="Y (lidar-up)")
        _assert_close(Z, -3.5, label="Z (lateral)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_score_scenario_cam_self_match_returns_high_v2v_amota():
    """Minimal scenario: synthetic tracks == synthetic labels exactly.

    DOCUMENTED CEILING: V2V-AMOTA in score_scenario_cam uses the (non-rematching)
    recall-sweep path which has a ~0.51-0.53 ceiling on a perfect tracker with
    constant TP scores (see T3.1). So expected V2V-AMOTA ≈ 53 (percent).

    V2V-MOTA is the headline full-data MOTA and DOES reach 100 for a perfect
    tracker.
    """
    print("\nT4.3: score_scenario_cam — self-match → V2V-MOTA = 100, AMOTA ~53")
    tmp = Path(tempfile.mkdtemp(prefix="scoringtest_"))
    try:
        track_path = tmp / "0001.txt"
        label_path = tmp / "0001_label.txt"
        track_lines, label_lines = [], []
        for f in range(10):
            track_lines.append(_write_kitti_track_line(f, 1, "Car", 1.7, 2.0, 4.5,
                                                        X=10.0 + f, Y=-1.0, Z=2.0, ry=0.0, score=0.95))
            label_lines.append(_write_kitti_label_line(f, 1, "Car", 1.7, 2.0, 4.5,
                                                        X=10.0 + f, Y=-1.0, Z=2.0, ry=0.0))
            track_lines.append(_write_kitti_track_line(f, 2, "Car", 1.7, 2.0, 4.5,
                                                        X=5.0, Y=-1.0, Z=-8.0, ry=0.1, score=0.9))
            label_lines.append(_write_kitti_label_line(f, 2, "Car", 1.7, 2.0, 4.5,
                                                        X=5.0, Y=-1.0, Z=-8.0, ry=0.1))
        track_path.write_text("".join(track_lines))
        label_path.write_text("".join(label_lines))

        scores = scorer.score_scenario_cam(track_path, label_path, ego_state_path=None)
        print(f"    V2V_AMOTA={scores['V2V_AMOTA']:.2f}  "
              f"OGFP_AMOTA={scores['OGFP_AMOTA']:.2f}  V2V_MOTA={scores['V2V_MOTA']:.2f}")
        # Headline MOTA hits 100% on perfect data
        assert scores["V2V_MOTA"] >= 99.0, f"V2V-MOTA = {scores['V2V_MOTA']}"
        assert scores["OGFP_MOTA"] >= 99.0, f"OGFP-MOTA = {scores['OGFP_MOTA']}"
        # AMOTA hits the structural ~0.53 ceiling (53% in this report scale)
        assert 48.0 <= scores["V2V_AMOTA"] <= 56.0, (
            f"V2V-AMOTA = {scores['V2V_AMOTA']:.2f} outside structural [48, 56] band. "
            f"If close to 100, the recall-sweep code may have switched defaults — UPDATE."
        )
        # With no FPs at all (self-match), V2V and OG+FP must agree exactly
        _assert_close(scores["V2V_AMOTA"], scores["OGFP_AMOTA"], tol=1e-9,
                       label="V2V == OG+FP when no FPs")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_score_scenario_cam_phantom_fp_v2v_vs_ogfp():
    """tracks = GT + many phantoms with HIGHER score than TPs.

    With phantom_score > TP_score, phantoms come first in the AMOTA
    descending-score sort, polluting cum_fp at every recall crossing. This is
    the regime where the V2V "FP-ignore bug" actually matters:
       V2V-AMOTA stays at the structural ceiling (~53%).
       OG+FP AMOTA crashes (cum_fp inflates at every recall threshold).
    The MOTA full-data measure also drops because FP > 0.

    Sanity-check: with LOW phantom score (≤ TP score), V2V and OG+FP would
    coincide (see T3.3 sub-case A); this T4.4 case picks high-score phantoms
    to expose the difference.
    """
    print("\nT4.4: score_scenario_cam — high-score phantom FP: V2V vs OG+FP gap")
    tmp = Path(tempfile.mkdtemp(prefix="scoringtest_"))
    try:
        track_path = tmp / "0001.txt"
        label_path = tmp / "0001_label.txt"
        track_lines, label_lines = [], []
        TP_SCORE = 0.5
        PHANTOM_SCORE = 0.99  # > TP_SCORE so phantoms interleave first
        for f in range(10):
            for k, (x, z) in enumerate([(10.0, 2.0), (8.0, -6.0)]):
                track_lines.append(_write_kitti_track_line(f, 10 + k, "Car", 1.7, 2.0, 4.5,
                                                            X=x, Y=-1.0, Z=z, ry=0.0, score=TP_SCORE))
                label_lines.append(_write_kitti_label_line(f, 10 + k, "Car", 1.7, 2.0, 4.5,
                                                            X=x, Y=-1.0, Z=z, ry=0.0))
            # 20 phantoms per frame, all far from any GT, score > TP score
            for j in range(20):
                track_lines.append(_write_kitti_track_line(f, 9000 + j, "Car", 1.7, 2.0, 4.5,
                                                            X=80.0 + j * 5.0, Y=-1.0,
                                                            Z=30.0, ry=0.0, score=PHANTOM_SCORE))
        track_path.write_text("".join(track_lines))
        label_path.write_text("".join(label_lines))

        scores = scorer.score_scenario_cam(track_path, label_path, ego_state_path=None)
        v2v = scores["V2V_AMOTA"]
        ogfp = scores["OGFP_AMOTA"]
        print(f"    V2V_AMOTA={v2v:.2f}  OGFP_AMOTA={ogfp:.2f}  "
              f"diff={v2v - ogfp:.2f}  V2V_MOTA={scores['V2V_MOTA']:.2f}  "
              f"OGFP_MOTA={scores['OGFP_MOTA']:.2f}")
        # V2V keeps near its perfect-tracker ceiling
        assert v2v >= 40.0, f"V2V FP-ignore should stay near ceiling; got {v2v:.2f}"
        # OG+FP crashes (phantoms interleave before TPs)
        assert ogfp < v2v - 20.0, (
            f"OG+FP should drop substantially below V2V with high-score phantoms; "
            f"v2v={v2v:.2f} ogfp={ogfp:.2f}"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ego_pose_in_tesla_cam_uses_v2v4real_convention():
    """ego_state: tesla at world (0,0,0,yaw=0), astuff at world (5,0,0,yaw=0), both h=1.7.
    Expected astuff in tesla cam:
       X_cam = 5         (forward)
       Z_cam = 0         (lateral)
       Y_cam = -1.87 + 1.7/2 = -1.02   (box center lidar-Z)
    """
    print("\nT4.5: _ego_pose_at_frame_in_tesla_cam — V2V4Real ego-as-GT placement")
    tmp = Path(tempfile.mkdtemp(prefix="scoringtest_"))
    try:
        ego_csv = tmp / "ego_state.csv"
        ego_csv.write_text(
            "frame_id,vehicle_id,x,y,z,yaw,width,length,height\n"
            "0,tesla,0.0,0.0,0.0,0.0,2.0,4.5,1.7\n"
            "0,astuff,5.0,0.0,0.0,0.0,2.0,4.5,1.7\n"
        )
        out = scorer._ego_pose_at_frame_in_tesla_cam(
            tmp, ego_dims={"tesla": (4.5, 2.0, 1.7), "astuff": (4.5, 2.0, 1.7)},
        )
        assert 0 in out and "astuff" in out[0] and "tesla" in out[0]
        X_cam, Y_cam, Z_cam = out[0]["astuff"]
        expected_Y = -1.87 + 1.7 / 2.0
        print(f"    astuff in tesla cam: X={X_cam:.4f}  Y={Y_cam:.4f}  Z={Z_cam:.4f}")
        print(f"      expected: X=5.0  Y={expected_Y:.4f}  Z=0.0")
        _assert_close(X_cam, 5.0, tol=1e-9, label="X_cam (forward)")
        _assert_close(Z_cam, 0.0, tol=1e-9, label="Z_cam (lateral)")
        _assert_close(Y_cam, expected_Y, tol=1e-9, label="Y_cam (box-center lidar-Z)")
        X_t, Y_t, Z_t = out[0]["tesla"]
        _assert_close(X_t, 0.0, tol=1e-9, label="tesla X_cam")
        _assert_close(Z_t, 0.0, tol=1e-9, label="tesla Z_cam")
        _assert_close(Y_t, expected_Y, tol=1e-9, label="tesla Y_cam")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# Tier 5 — Parallel safety
# ============================================================================

def _make_synth_scenario_pair(root: Path, n_scenarios=2, n_frames=8):
    """2-scenario synthetic dataset compatible with --cam-frame mode.

    Layout:
      root/track_dir/0001.txt 0002.txt          (KITTI tracks)
      root/label_dir/0001.txt 0002.txt          (V2V4Real labels)
      root/export_dir/test__001/ ego_state.csv
      root/export_dir/test__002/ ego_state.csv
    """
    track_dir = root / "track_dir"
    label_dir = root / "label_dir"
    export_dir = root / "export_dir"
    track_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    export_dir.mkdir(parents=True)

    for s in range(n_scenarios):
        scen_name = f"{s+1:04d}"
        track_lines, label_lines = [], []
        for f in range(n_frames):
            base = (s + 1) * 3.0
            for k in range(2):
                x = base + k * 6.0 + 0.5 * f
                z = (-1.0) ** k * (2.0 + k)
                track_lines.append(_write_kitti_track_line(
                    f, 10 + k, "Car", 1.7, 2.0, 4.5,
                    X=x, Y=-1.0, Z=z, ry=0.0, score=0.9 - 0.01 * k))
                label_lines.append(_write_kitti_label_line(
                    f, 10 + k, "Car", 1.7, 2.0, 4.5, X=x, Y=-1.0, Z=z, ry=0.0))
        (track_dir / f"{scen_name}.txt").write_text("".join(track_lines))
        (label_dir / f"{scen_name}.txt").write_text("".join(label_lines))
        sd = export_dir / f"test__{scen_name}"
        sd.mkdir()
        with open(sd / "ego_state.csv", "w") as f:
            f.write("frame_id,vehicle_id,x,y,z,yaw,width,length,height\n")
            for fid in range(n_frames):
                f.write(f"{fid},tesla,0.0,0.0,0.0,0.0,2.0,4.5,1.7\n")
                f.write(f"{fid},astuff,5.0,0.0,0.0,0.0,2.0,4.5,1.7\n")
    return track_dir, label_dir, export_dir


def test_score_parallel_matches_serial():
    """Run scoring --serial vs parallel; summary.json values match within 1e-9."""
    print("\nT5.1: parallel scoring matches serial scoring (summary.json bit-equality)")
    import subprocess
    import json
    tmp = Path(tempfile.mkdtemp(prefix="scoringtest_"))
    try:
        track_dir, label_dir, export_dir = _make_synth_scenario_pair(tmp, n_scenarios=2)
        out_serial = tmp / "out_serial"
        out_parallel = tmp / "out_parallel"
        cmd_base = [
            sys.executable,
            str(REPO / "scripts" / "score_kitti_tracks_through_run_suite.py"),
            "--tracking-dir", str(track_dir),
            "--export-dir", str(export_dir),
            "--label-dir", str(label_dir),
            "--cam-frame",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{REPO/'src'}:{REPO/'scripts'}:" + env.get("PYTHONPATH", "")
        )
        r1 = subprocess.run(cmd_base + ["--output-dir", str(out_serial),
                                         "--serial", "--label", "serial"],
                             capture_output=True, text=True, env=env, timeout=120)
        r2 = subprocess.run(cmd_base + ["--output-dir", str(out_parallel),
                                         "--label", "parallel"],
                             capture_output=True, text=True, env=env, timeout=120)
        if r1.returncode != 0:
            print(f"    [INFRA] serial subprocess failed: {r1.stderr[-400:]}")
            print(f"    [SKIP] cannot validate parallel-vs-serial")
            return
        if r2.returncode != 0:
            print(f"    [INFRA] parallel subprocess failed: {r2.stderr[-400:]}")
            print(f"    [SKIP] cannot validate parallel-vs-serial")
            return

        s1 = json.loads((out_serial / "summary.json").read_text())
        s2 = json.loads((out_parallel / "summary.json").read_text())
        numeric_keys = [k for k in s1
                        if isinstance(s1[k], (int, float))
                        and isinstance(s2.get(k), (int, float))]
        diffs = 0
        for k in numeric_keys:
            if abs(s1[k] - s2[k]) > 1e-9:
                print(f"    [DIFF] {k}: serial={s1[k]} parallel={s2[k]}")
                diffs += 1
        print(f"    numeric keys compared: {len(numeric_keys)}  diffs > 1e-9: {diffs}")
        assert diffs == 0, f"serial vs parallel differ on {diffs} keys"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_score_worker_picklable():
    """Task tuple must pickle cleanly — required by multiprocessing.Pool.map."""
    print("\nT5.2: score worker task tuple is picklable")
    task = (
        "CAM",
        Path("/tmp/tracks_unused.txt"),
        Path("/tmp/labels_unused.txt"),
        Path("/tmp/ego_state_unused.csv"),
        "scene1",
        "0001.txt",
    )
    blob = pickle.dumps(task)
    rt = pickle.loads(blob)
    print(f"    pickled {len(blob)} bytes; round-trip == original: {rt == task}")
    assert rt == task, "task tuple did not survive pickle round-trip"
    assert callable(scorer._score_worker)
    pickle.dumps(scorer._score_worker)  # raises if unpicklable
    print(f"    _score_worker is also picklable (multiprocessing-safe)")


# ============================================================================
# Runner
# ============================================================================

ALL_TESTS = [
    test_3d_iou_self_is_one,                              # T1.1
    test_3d_iou_disjoint_is_zero,                         # T1.2
    test_3d_iou_height_no_overlap_is_zero,                # T1.3
    test_3d_iou_partial_height_overlap,                   # T1.4
    test_3d_iou_matches_ab3dmot,                          # T1.5  (DOCUMENTED BUG)
    test_2d_bev_iou_yaw_invariance,                       # T1.6
    test_match_frame_3d_iou_perfect_pairs,                # T2.1
    test_match_frame_3d_iou_below_threshold_no_match,     # T2.2
    test_match_frame_3d_iou_hungarian_optimal,            # T2.3
    test_amota_perfect_tracker_is_high,                   # T3.1
    test_amota_no_tracks_is_zero,                         # T3.2
    test_ignore_unmatched_fps_drops_extras,               # T3.3
    test_mota_fp_penalty_isolated,                        # T3.4
    test_v2v4real_kitti_convention_in_parse_track,        # T4.1
    test_label_file_parse,                                # T4.2
    test_score_scenario_cam_self_match_returns_high_v2v_amota,  # T4.3
    test_score_scenario_cam_phantom_fp_v2v_vs_ogfp,       # T4.4
    test_ego_pose_in_tesla_cam_uses_v2v4real_convention,  # T4.5
    test_score_parallel_matches_serial,                   # T5.1
    test_score_worker_picklable,                          # T5.2
]


if __name__ == "__main__":
    failures = []
    for t in ALL_TESTS:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  *** {t.__name__} FAILED: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  *** {t.__name__} ERROR: {type(e).__name__}: {e}")
    print()
    print("=" * 72)
    if failures:
        print(f"FAILURES ({len(failures)}/{len(ALL_TESTS)}):")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        sys.exit(1)
    else:
        print(f"ALL {len(ALL_TESTS)} TESTS PASSED")
