#!/usr/bin/env python3
"""Augment V2V4Real per-CAV detection files and GT labels with CAV self-reports.

V2V4Real's released data has TWO gaps:
1. Each CAV's PointPillar detection set does NOT include the CAV itself
   (LiDAR is blind to its own host vehicle).
2. The GT label set does NOT include the CAVs as targets (only third-party
   cars are annotated).

This is wrong: CAVs are real cars in the scene and should be tracked. They
have known RTK-precise poses that they can self-report as detections.

This script generates augmented copies of both:
- `pp_per_cav_with_self_Car_val/{ego,1}/<seq>.txt` — per-CAV PP detections +
  each CAV's own self-report (in its own KITTI frame) and the OTHER CAV's
  cross-detection (the CAV's known pose, transformed into the observing
  CAV's frame, simulating "you also publish the other CAV's pose via V2V").
- `v2v4real_val_label_with_cav/<seq>.txt` — GT labels + tesla + astuff
  positions in tesla's KITTI frame (the common reference frame for GT).

The transformation between frames:
- KITTI X_cam = LiDAR X (lateral right)
- KITTI Z_cam = LiDAR Y (forward) — y/z swap convention from V2V4Real
- KITTI Y_cam = -LiDAR Z + h/2 (vertical, camera-down to LiDAR-up flip)
  Empirically GT mean Y_cam ≈ -0.94, h ≈ 1.85 → lidar_z ≈ -1.87 (LiDAR mount
  height above ground).

Tesla self-report → in tesla's frame: (X=0, Y=h/2-1.87, Z=0)
Astuff in tesla's frame → compute relative XYZ, apply tesla.yaw rotation,
                          set Y based on astuff.h.
"""
from __future__ import annotations
import csv
import math
import shutil
import sys
from pathlib import Path

# Resolve the cmr repo root and put src + configs on the import path so the
# constants below can come from a single, visible source rather than buried
# inline in this script.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "configs"))

from eval_config import (        # noqa: E402
    CMR_EXPORT,
    AB3DMOT_DETS_DIR as AB3DMOT_DETS,
    BASE_GT_LABEL_DIR as AB3DMOT_GT,
    WITH_CAV_GT_LABEL_DIR as OUT_GT,
    TESLA_GT_ID,
    ASTUFF_GT_ID,
    SELF_REPORT_SCORE_SENTINEL as SELF_REPORT_SCORE,
)
from v2v4real_vehicles import (   # noqa: E402
    LIDAR_MOUNT_HEIGHT_M,
    DEFAULT_DIMS,
    DIMS_ASSERTION_TOL_M,
)
from v2v4real_seq_map import (    # noqa: E402
    SCENARIO_TO_SEQ,
    SEQ_TO_NUM_FRAMES,
)

OUT_DETS = AB3DMOT_DETS / "pp_per_cav_with_self_Car_val"
# LF-input variant: augment the already-common-frame LF detection set.
# This is the CORRECT augmentation because LF is in tesla's KITTI frame
# (V2V4Real's data prep handles the per-CAV → common-frame transform).
OUT_LF_DETS = AB3DMOT_DETS / "late_fusion_with_self_Car_val"


def map_scenarios_to_seqs() -> dict[str, int]:
    """Return the explicit scenario→seq map, raising on any inconsistency.

    The mapping itself comes from ``configs/v2v4real_seq_map.py`` (the single
    visible source of truth). This function only VERIFIES that every declared
    scenario directory exists under CMR_EXPORT and that its frame count matches
    the declared seqmap row — both surfaces must agree.
    """
    out: dict[str, int] = {}
    for scen_name, seq in SCENARIO_TO_SEQ.items():
        d = CMR_EXPORT / scen_name
        if not d.is_dir():
            raise FileNotFoundError(
                f"scenario {scen_name!r} declared in v2v4real_seq_map.py is "
                f"missing under {CMR_EXPORT}")
        ego_csv = d / "ego_state.csv"
        if not ego_csv.exists():
            raise FileNotFoundError(f"{ego_csv} missing (required for seq {seq})")
        with open(ego_csv) as f:
            tesla_frames = [int(r["frame_id"]) for r in csv.DictReader(f)
                            if r["vehicle_id"] == "tesla"]
        expected_n = SEQ_TO_NUM_FRAMES[seq]
        max_f = max(tesla_frames) if tesla_frames else -1
        if max_f + 1 != expected_n:
            raise ValueError(
                f"seq {seq:04d} ({scen_name}): tesla frames go 0..{max_f} "
                f"({max_f + 1} total) but seqmap declares {expected_n}; "
                f"edit configs/v2v4real_seq_map.py if V2V4Real changed.")
        out[scen_name] = seq
    return out


def read_poses(ego_csv: Path) -> dict[tuple[str, int], tuple[float, ...]]:
    """Returns {(vehicle_id, frame): (x, y, z, yaw, l, w, h)}.

    Duplicate rows must AGREE on every field; mismatches raise (silent dedup
    was masking real data corruption in earlier runs).
    """
    out: dict[tuple[str, int], tuple[float, ...]] = {}
    with open(ego_csv) as f:
        for r in csv.DictReader(f):
            v = r["vehicle_id"]
            fid = int(r["frame_id"])
            key = (v, fid)
            row = (float(r["x"]), float(r["y"]), float(r["z"]), float(r["yaw"]),
                   float(r["length"]), float(r["width"]), float(r["height"]))
            if key in out:
                if out[key] != row:
                    raise ValueError(
                        f"{ego_csv}: duplicate {key} disagrees: "
                        f"have {out[key]} vs {row} (fix the CSV before "
                        f"regenerating with_cav GT)")
                continue
            out[key] = row
    return out


def _emit_ego_self(vehicle: str, pose: tuple) -> tuple:
    """Build the (X, Y, Z, ry, l, w, h) self-report pose for a CAV in its own
    KITTI cam frame: at origin, Y = per-vehicle mount + h/2, no yaw.
    Asserts dims match the declared per-vehicle defaults (catches drift loudly).
    """
    _, _, _, _, l, w, h = pose
    expect_l, expect_w, expect_h = DEFAULT_DIMS[vehicle]
    drift = max(abs(l - expect_l), abs(w - expect_w), abs(h - expect_h))
    if drift > DIMS_ASSERTION_TOL_M:
        raise ValueError(
            f"{vehicle} ego_state dims drift {drift:.3f} m > tol "
            f"{DIMS_ASSERTION_TOL_M}: csv ({l}, {w}, {h}) vs default "
            f"{DEFAULT_DIMS[vehicle]}. Edit configs/v2v4real_vehicles.py "
            f"if the platform changed.")
    return (0.0, LIDAR_MOUNT_HEIGHT_M[vehicle] + h / 2.0, 0.0, 0.0, l, w, h)


def world_to_local_kitti(target_world, ego_world, ego_vehicle: str):
    """Transform target's world pose into ego's KITTI camera frame.

    target_world, ego_world: (x, y, z, yaw, l, w, h)
    ego_vehicle: 'tesla' or 'astuff' — selects the per-vehicle LiDAR mount
                 height from configs/v2v4real_vehicles.py.
    Returns (kitti_X, kitti_Y, kitti_Z, kitti_ry, l, w, h).
    """
    tx, ty, tz, tyaw, tl, tw, th = target_world
    ex, ey, ez, eyaw, _el, _ew, _eh = ego_world

    # Translate: world → ego-translated
    dx = tx - ex
    dy = ty - ey
    dz = tz - ez

    # Rotate by -eyaw to get ego-local LiDAR coords
    # (V2V4Real LiDAR convention: X=ego-forward, Y=ego-lateral after rotation)
    cos_e = math.cos(-eyaw); sin_e = math.sin(-eyaw)
    lidar_x = cos_e * dx - sin_e * dy
    lidar_y = sin_e * dx + cos_e * dy

    # KITTI camera frame (V2V4Real convention):
    #   X_cam = LiDAR X
    #   Z_cam = LiDAR Y (y/z swap)
    #   Y_cam = ego's static LiDAR mount  +  th/2
    #
    # V2V4Real / DMSTrack Y convention (verified 2026-06-08 by matching
    # bucket Y values across DMS PP per-CAV and the published GT):
    #
    #     kitti_Y = dz + th / 2
    #
    # where dz = target_world_z - ego_world_z. That is: Y is the box CENTER's
    # vertical offset from the ego lidar, sign-preserved (negative = below
    # lidar). No separate LIDAR_MOUNT_HEIGHT_M term — the offset to ground is
    # already baked into dz (target ground-level objects have target_z roughly
    # 1.87m below ego_z, so dz ≈ -1.87 and Y ≈ -1.87 + 0.85 ≈ -1.0, matching
    # GT empirically).
    #
    # Two prior misdiagnoses were ruled out: (1) the +dz term was removed
    # earlier because mixing per-CAV-tree centroids gave a phantom 2m delta;
    # fixed in import_ab3dmot_format.py by rehydrating to shared world.
    # (2) the LIDAR_MOUNT term was added back later thinking the bare dz was
    # off by a constant; verified against DMS/GT to be SPURIOUS — it doubles
    # the mount-height correction.
    #
    # AB3DMOT's matching uses giou_3d which depends on Y; a flat fudged Y
    # silently breaks 3D-IoU associations between same-XZ detections at
    # different elevations. Verified against DMSPerCavSweepEkf to recover the
    # 46.78 V2V AMOTA baseline.
    kitti_X = lidar_x
    kitti_Z = lidar_y
    kitti_Y = dz + th / 2.0

    # Yaw delta, wrapped to [-pi, pi]
    kitti_ry = (tyaw - eyaw + math.pi) % (2 * math.pi) - math.pi

    return (kitti_X, kitti_Y, kitti_Z, kitti_ry, tl, tw, th)


def emit_kitti_det_row(frame: int, kitti_pose: tuple, score: float = SELF_REPORT_SCORE) -> str:
    """V2V4Real KITTI MOT detection row, 15 fields comma-separated.
    Format: frame, type=2, ?,?,?,?, score, h, w, l, X, Y, Z, ry, 0
    """
    X, Y, Z, ry, l, w, h = kitti_pose
    return (f"{frame},2,0,0,0,0,{score:.6f},{h:.6f},{w:.6f},{l:.6f},"
            f"{X:.6f},{Y:.6f},{Z:.6f},{ry:.6f},0.000000")


def emit_kitti_gt_row(frame: int, tid: int, kitti_pose: tuple) -> str:
    """V2V4Real KITTI MOT GT row, 17 fields space-separated.
    Format: frame tid type 0 0 0 0 0 0 0 h w l X Y Z ry
    """
    X, Y, Z, ry, l, w, h = kitti_pose
    return (f"{frame} {tid} Car 0 0 0 0 0 0 0 "
            f"{h} {w} {l} {X} {Y} {Z} {ry}")


def main() -> None:
    scen_to_seq = map_scenarios_to_seqs()
    print(f"Mapped {len(scen_to_seq)} scenarios to seqs")

    OUT_DETS.mkdir(parents=True, exist_ok=True)
    (OUT_DETS / "ego").mkdir(parents=True, exist_ok=True)
    (OUT_DETS / "1").mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    OUT_LF_DETS.mkdir(parents=True, exist_ok=True)
    LF_SRC = AB3DMOT_DETS / "late_fusion_Car_val"

    # Existing per-CAV detection input
    SRC_EGO = AB3DMOT_DETS / "multi_sensor_differentiable_kalman_filter_Car_val" / "ego"
    SRC_1   = AB3DMOT_DETS / "multi_sensor_differentiable_kalman_filter_Car_val" / "1"

    n_self_dets_total = 0
    n_gt_added_total = 0
    for scen, seq in scen_to_seq.items():
        ego_csv = CMR_EXPORT / scen / "ego_state.csv"
        poses = read_poses(ego_csv)
        # Frames to emit: declared by the seqmap, NOT inferred from poses.
        expected_n = SEQ_TO_NUM_FRAMES[seq]
        frames = list(range(expected_n))

        # Frame-completeness: every declared frame must carry BOTH tesla and
        # astuff poses; missing entries used to be silently skipped, which
        # left GT/det rows holes that lowered ego-augmented AMOTA.
        missing = [(v, fid) for v in ("tesla", "astuff") for fid in frames
                   if (v, fid) not in poses]
        if missing:
            raise ValueError(
                f"seq {seq:04d} ({scen}): {len(missing)} ego_state.csv rows "
                f"missing (e.g. {missing[:3]}); cannot emit complete CAV "
                f"augmentation. Fix the CSV and re-run.")

        # ─── Augment per-CAV detections ────────────────────────────────────
        seq_str = f"{seq:04d}"
        src_ego = SRC_EGO / f"{seq_str}.txt"
        src_1 = SRC_1 / f"{seq_str}.txt"
        dst_ego = OUT_DETS / "ego" / f"{seq_str}.txt"
        dst_1 = OUT_DETS / "1" / f"{seq_str}.txt"

        with open(dst_ego, "w") as fout:
            shutil.copyfileobj(open(src_ego), fout)
            for fid in frames:
                tesla_p = poses[("tesla", fid)]
                astuff_p = poses[("astuff", fid)]
                fout.write(emit_kitti_det_row(fid, _emit_ego_self("tesla", tesla_p)) + "\n")
                fout.write(emit_kitti_det_row(fid, world_to_local_kitti(astuff_p, tesla_p, "tesla")) + "\n")
                n_self_dets_total += 2

        with open(dst_1, "w") as fout:
            shutil.copyfileobj(open(src_1), fout)
            for fid in frames:
                tesla_p = poses[("tesla", fid)]
                astuff_p = poses[("astuff", fid)]
                fout.write(emit_kitti_det_row(fid, _emit_ego_self("astuff", astuff_p)) + "\n")
                fout.write(emit_kitti_det_row(fid, world_to_local_kitti(tesla_p, astuff_p, "astuff")) + "\n")
                n_self_dets_total += 2

        # ─── Augment LF detection set (single common frame = tesla's) ─────
        src_lf = LF_SRC / f"{seq_str}.txt"
        dst_lf = OUT_LF_DETS / f"{seq_str}.txt"
        with open(dst_lf, "w") as fout:
            shutil.copyfileobj(open(src_lf), fout)
            for fid in frames:
                tesla_p = poses[("tesla", fid)]
                astuff_p = poses[("astuff", fid)]
                fout.write(emit_kitti_det_row(fid, _emit_ego_self("tesla", tesla_p)) + "\n")
                fout.write(emit_kitti_det_row(fid, world_to_local_kitti(astuff_p, tesla_p, "tesla")) + "\n")

        # ─── Augment GT (in tesla's KITTI frame) ──────────────────────────
        src_gt = AB3DMOT_GT / f"{seq_str}.txt"
        dst_gt = OUT_GT / f"{seq_str}.txt"
        with open(dst_gt, "w") as fout:
            shutil.copyfileobj(open(src_gt), fout)
            for fid in frames:
                tesla_p = poses[("tesla", fid)]
                astuff_p = poses[("astuff", fid)]
                fout.write(emit_kitti_gt_row(fid, TESLA_GT_ID, _emit_ego_self("tesla", tesla_p)) + "\n")
                fout.write(emit_kitti_gt_row(fid, ASTUFF_GT_ID,
                                             world_to_local_kitti(astuff_p, tesla_p, "tesla")) + "\n")
                n_gt_added_total += 2

        print(f"  {scen[:50]:50s} → seq {seq_str}: dets ✓, gt ✓")

    print(f"\nAdded {n_self_dets_total} self/cross CAV detection rows")
    print(f"Added {n_gt_added_total} CAV GT rows")
    print(f"\nOutputs:")
    print(f"  detections: {OUT_DETS}")
    print(f"  gt:         {OUT_GT}")


if __name__ == "__main__":
    main()
