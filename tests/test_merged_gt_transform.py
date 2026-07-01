"""Regression test for the build_merged_gt world->KITTI transform.

Bug (2026-06-21): build_merged_gt.py transformed world boxes into the tesla
KITTI frame with a 2D yaw-only rotation and a raw world-z for the vertical
(`kitti_Y = (world_z - ego_z) + h/2`). That drops the sensor pitch/roll carried
by the raw 4x4 ``lidar_pose`` (the cmr_export flattened poses to x,y,z,yaw), so
the KITTI Y error grew with range (near +0.8 m -> far +3.2 m) and destroyed
3D-IoU matching against DMSTrack's precomputed tracks (Tier-3 AMOTA collapsed to
~2.2). Fix: full 3D rotation ``lidar = R_ego^T (p_world - p_ego)`` from the raw
``lidar_pose``; KITTI (X,Y,Z) = (lidar_x, lidar_z + h/2, lidar_y).

This test asserts the fixed transform reproduces the official paper_gt for
tesla-observed boxes to <1e-2 m. It is data-dependent (needs the raw V2V4Real
drive + cmr_export + frozen paper_gt) and skips cleanly when any is absent.
"""
from __future__ import annotations

import csv
import glob
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "configs"))

from v2v4real_seq_map import SEQ_TO_SCENARIO  # noqa: E402
import build_merged_gt as bmg  # noqa: E402

_SCEN = SEQ_TO_SCENARIO[0]
_CMR = bmg.DEFAULT_EXPORT / _SCEN
_RAW = bmg.DEFAULT_RAW_BASE / _SCEN.split("__", 2)[2] / "0"
_OFFICIAL = (REPO / "data/v2v4real_inputs/baselines/paper_gt/kitti_labels"
             / "v2v4real_val_label" / "0000.txt")

_have_data = (_CMR / "gt_objects.csv").exists() and bool(
    glob.glob(str(_RAW / "*.yaml"))) and _OFFICIAL.exists()
pytestmark = pytest.mark.skipif(
    not _have_data, reason="raw V2V4Real / cmr_export / paper_gt not available")


def _load_official(path: Path) -> dict[int, list[tuple]]:
    out: dict[int, list[tuple]] = {}
    for ln in path.read_text().splitlines():
        t = ln.split()
        if len(t) < 17:
            continue
        out.setdefault(int(float(t[0])), []).append(
            (float(t[13]), float(t[14]), float(t[15])))  # X, Y, Z
    return out


def test_full_transform_reproduces_official_gt():
    """Tesla boxes transformed via world_to_kitti_full must land on paper_gt."""
    rot = bmg.load_tesla_rotation(_SCEN, bmg.DEFAULT_RAW_BASE)
    ego = bmg.load_tesla_ego(_CMR)
    official = _load_official(_OFFICIAL)

    tesla_boxes: dict[int, list[tuple]] = {}
    with open(_CMR / "gt_objects.csv") as f:
        for r in csv.DictReader(f):
            if r["vehicle_id"] != "tesla":
                continue
            tesla_boxes.setdefault(int(r["frame_id"]), []).append(
                (float(r["x"]), float(r["y"]), float(r["z"]), float(r["yaw"]),
                 float(r["length"]), float(r["width"]), float(r["height"])))

    n_matched = 0
    max_err = 0.0
    for fr, boxes in tesla_boxes.items():
        if fr not in rot or fr not in ego or fr not in official:
            continue
        for box in boxes:
            X, Y, Z, *_ = bmg.world_to_kitti_full(box, ego[fr], rot[fr])
            # nearest official box in the BEV (X-Z) plane
            best = min(official[fr],
                       key=lambda o: math.hypot(X - o[0], Z - o[2]))
            dxz = math.hypot(X - best[0], Z - best[2])
            if dxz < 0.5:                       # same physical object
                n_matched += 1
                max_err = max(max_err, abs(X - best[0]), abs(Y - best[1]),
                              abs(Z - best[2]))

    assert n_matched >= 50, f"too few matched boxes ({n_matched}) to trust the check"
    assert max_err < 1e-2, f"transform diverges from official GT by {max_err:.4f} m"
