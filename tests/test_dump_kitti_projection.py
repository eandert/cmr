"""Regression test for dump_tracks_to_kitti_mot's global->KITTI projection.

Bug (root-caused 2026-06-29, results/GT_MERGE_RTK/Z_6DOF_FIX_PLAN.md): the
cmr_export flattened V2V4Real's 6-DOF ``lidar_pose`` to (x, y, z, yaw), so the
dump's old global->KITTI projection rotated tracks with a yaw-only matrix and
pinned the vertical to a constant. The dropped pitch/roll (~1 deg per frame) put
a *range x tilt* error into the KITTI Y (vertical) that a constant can't fix and
a naive per-object z made worse. This collapsed 3D-IoU matching against the
official KITTI GT (CPft single-pass recall ~0.68 instead of ~0.85).

Fix (Fix A, dump-only): ``global_to_kitti`` now takes the raw LiDAR->world 3x3
rotation R and does the FULL 6-DOF inverse
``lid = R^T (p_world - p_ego)``; KITTI (X, Y, Z) = (lid_x, lid_z + h/2, lid_y),
using the track's real per-object z. The old yaw-only + constant-z path is kept
behind ``--legacy-constant-z`` (R is None) for bit-identical reproduction of the
pre-2026-06-29 number.

These tests are data-independent (pure-function, synthetic boxes): they pin the
projection math itself, so they run in CI without the raw V2V4Real drive. The
data-dependent end-to-end check lives in test_merged_gt_transform.py.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import dump_tracks_to_kitti_mot as dump  # noqa: E402
import build_merged_gt as bmg  # noqa: E402


def _rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """LiDAR->world 3x3 rotation R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _track(x, y, z, *, w=1.8, l=4.2, yaw=0.3, score=0.9, h=1.6, tid=7):
    """Build a full track tuple (id, x, y, w, l, yaw, score, height, z)."""
    return (tid, x, y, w, l, yaw, score, h, z)


def test_full_6dof_round_trip_recovers_lidar_box():
    """A box placed in the world via R round-trips back to its LiDAR coords,
    even with non-trivial roll/pitch/yaw. KITTI (X, Y, Z) = (lid_x, lid_z+h/2,
    lid_y)."""
    R = _rot(roll=0.04, pitch=-0.03, yaw=1.1)        # ~2 deg tilt + real yaw
    ego = (120.0, -55.0, 3.0, 1.1)                   # cx, cy, cz, cyaw=R's yaw
    lx, ly, lz, h = 2.5, 37.0, -1.4, 1.6             # known LiDAR-frame box

    p_world = R @ np.array([lx, ly, lz]) + np.array(ego[:3])
    trk = _track(p_world[0], p_world[1], p_world[2], yaw=ego[3] + 0.2, h=h)

    kx, ky, kz, kry, kw, kl, kh = dump.global_to_kitti(trk, ego, R)

    assert kx == pytest.approx(lx, abs=1e-9)
    assert kz == pytest.approx(ly, abs=1e-9)          # forward distance
    assert ky == pytest.approx(lz + h / 2, abs=1e-9)  # vertical (camera-down)
    assert kry == pytest.approx(0.2, abs=1e-9)        # yaw_g - cyaw, wrapped
    assert (kw, kl, kh) == (trk[3], trk[4], h)


def test_full_6dof_matches_build_merged_gt_reference():
    """The dump's full-6DOF projection must equal build_merged_gt.world_to_kitti_full
    for the same box/ego/R. The two paths share the GT vertical datum; pin them
    so they can never silently drift apart (the GT was fixed first, the dump
    second — a future edit to one must touch the other)."""
    R = _rot(roll=-0.02, pitch=0.05, yaw=-0.7)
    ego = (10.0, 20.0, 1.5, -0.7)
    x_g, y_g, z_g = 48.3, -12.1, 0.4
    w, l, h, yaw_g = 1.9, 4.5, 1.7, 0.9

    trk = _track(x_g, y_g, z_g, w=w, l=l, yaw=yaw_g, h=h)
    kx, ky, kz, kry, *_ = dump.global_to_kitti(trk, ego, R)

    # world_to_kitti_full target = (x, y, z, yaw, l, w, h); ego = (x, y, z, yaw, ...).
    X, Y, Z, ry, *_ = bmg.world_to_kitti_full(
        (x_g, y_g, z_g, yaw_g, l, w, h), (*ego, l, w, h), R)

    assert (kx, ky, kz) == pytest.approx((X, Y, Z), abs=1e-12)
    assert kry == pytest.approx(ry, abs=1e-12)


def test_yaw_only_vertical_error_grows_with_range():
    """The bug guard: a yaw-only projection (the old behaviour) puts a range x
    tilt error into the vertical, whereas the 6-DOF projection is range-invariant
    in Y for a box at fixed LiDAR height. If global_to_kitti ever reverts to
    yaw-only, far boxes drift in Y and this fails."""
    theta = math.radians(1.5)                        # ~1.5 deg forward tilt
    R = _rot(roll=theta, pitch=0.0, yaw=0.0)          # couples forward -> vertical
    ego = (0.0, 0.0, 0.0, 0.0)
    lz, h = -1.5, 1.6                                 # fixed LiDAR height

    def y_err(fwd: float) -> tuple[float, float]:
        # Box straight ahead at forward range `fwd`, fixed LiDAR height lz.
        p_world = R @ np.array([0.0, fwd, lz])
        trk = _track(p_world[0], p_world[1], p_world[2], h=h)
        _, ky_6dof, _, *_ = dump.global_to_kitti(trk, ego, R)
        # What a yaw-only projection (ego yaw = 0 -> identity) would have used:
        ky_yaw_only = p_world[2] + h / 2
        true_y = lz + h / 2
        return abs(ky_6dof - true_y), abs(ky_yaw_only - true_y)

    near_6dof, near_yaw = y_err(10.0)
    far_6dof, far_yaw = y_err(90.0)

    # 6-DOF is correct (range-invariant) at both ranges.
    assert near_6dof < 1e-9 and far_6dof < 1e-9
    # Yaw-only error is non-trivial and grows ~linearly with range.
    assert far_yaw > near_yaw + 1.0                  # ~80 m * sin(1.5 deg) ~ 2.1 m
    assert far_yaw == pytest.approx(90.0 * math.sin(theta), abs=0.1)


def test_legacy_constant_z_reproduces_old_vertical():
    """--legacy-constant-z (R is None) reproduces the pre-2026-06-29 behaviour:
    yaw-only 2D rotation + a constant vertical DEFAULT_LIDAR_Z + h/2, and yaw
    is NOT wrapped (matches the legacy code path exactly)."""
    ego = (5.0, 7.0, 2.0, math.pi / 6)               # cyaw = 30 deg
    x_g, y_g, h, yaw_g = 25.0, -3.0, 1.6, 0.5
    trk = _track(x_g, y_g, 99.0, h=h, yaw=yaw_g)      # world z is ignored in legacy

    kx, ky, kz, kry, _, _, kh = dump.global_to_kitti(trk, ego, R=None)

    cy, sy = math.cos(ego[3]), math.sin(ego[3])
    dx, dy = x_g - ego[0], y_g - ego[1]
    assert kx == pytest.approx(cy * dx + sy * dy, abs=1e-12)
    assert kz == pytest.approx(-sy * dx + cy * dy, abs=1e-12)
    assert ky == pytest.approx(dump.DEFAULT_LIDAR_Z + h / 2, abs=1e-12)
    assert kry == pytest.approx(yaw_g - ego[3], abs=1e-12)  # unwrapped in legacy
    assert kh == h


def test_legacy_vertical_is_independent_of_world_z():
    """Legacy mode pins every box to the same constant vertical regardless of its
    world z — that is exactly the limitation the 6-DOF fix removes."""
    ego = (0.0, 0.0, 0.0, 0.0)
    low = dump.global_to_kitti(_track(30.0, 0.0, -5.0, h=1.6), ego, R=None)
    high = dump.global_to_kitti(_track(30.0, 0.0, 5.0, h=1.6), ego, R=None)
    assert low[1] == pytest.approx(high[1], abs=1e-12)
    assert low[1] == pytest.approx(dump.DEFAULT_LIDAR_Z + 1.6 / 2, abs=1e-12)


def test_missing_vertical_fields_fall_back_to_constants():
    """A track that never matched a 3D detection has no height/z; the projection
    must fall back to DEFAULT_H and the constant vertical even in full-6DOF mode
    (it can't invent a real z, so it can't claim a tilt-corrected vertical)."""
    R = _rot(roll=0.03, pitch=0.0, yaw=0.0)
    ego = (0.0, 0.0, 0.0, 0.0)

    # No vertical fields at all (len-7 tuple): height -> DEFAULT_H, z -> None.
    short = (3, 40.0, 0.0, 1.8, 4.2, 0.0, 0.9)
    _, ky, _, *_ = dump.global_to_kitti(short, ego, R)
    assert ky == pytest.approx(dump.DEFAULT_LIDAR_Z + dump.DEFAULT_H / 2, abs=1e-12)

    # Height present but z is None -> still the constant vertical.
    no_z = (3, 40.0, 0.0, 1.8, 4.2, 0.0, 0.9, 1.7, None)
    _, ky2, _, _, _, _, kh2 = dump.global_to_kitti(no_z, ego, R)
    assert ky2 == pytest.approx(dump.DEFAULT_LIDAR_Z + 1.7 / 2, abs=1e-12)
    assert kh2 == 1.7
