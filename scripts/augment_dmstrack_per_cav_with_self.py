"""Augment DMSTrack-bundled per-CAV detection files with ego self-reports.

The DMSTrack per-CAV data (multi_sensor_differentiable_kalman_filter_Car_val/{ego,1}/)
has cross-CAV detections only — tesla sees astuff as a car, astuff sees tesla as a
car, no self-detections. Those cross-CAV ego detections come out at moderate
PointPillar scores (0.48-0.71), which costs us across the AMOTA recall sweep.

This script writes a new det_name `pp_dmstrack_per_cav_self` where each ego's
stream is augmented with a self-report row at the ego's own position with
score=0.9 (high but in PointPillar's natural [0,1] range — NOT the score=2.0
magic value we tried before, which was too aggressive for NMS).

The self-report position is the ego's body center in tesla's V2V4Real-KITTI
frame:
    X = lidar X (forward),  Y = lidar Z (up = LIDAR_Z_MOUNT_TO_GROUND + h/2),
    Z = lidar Y (lateral).

For tesla, that's at (0, -1.87+h_tesla/2, 0) in its own frame.
For astuff, that's astuff's pose transformed into tesla's frame (since the
DMSTrack per-CAV files are all expressed in tesla's frame).

Output: data/v2v4real/detection/pp_dmstrack_per_cav_self_Car_val/{seq}_{src}.txt
"""
from __future__ import annotations
import csv
import math
import os
import shutil
from pathlib import Path
import sys

# Reuse the scenario→seq mapping helper.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment_v2v4real_with_cav_self_reports import map_scenarios_to_seqs  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

SELF_SCORE = 0.9
LIDAR_Z_MOUNT_TO_GROUND = -1.87

DEFAULT_DIMS = {
    "tesla":  {"l": 4.97, "w": 1.96, "h": 1.44},
    "astuff": {"l": 5.18, "w": 2.03, "h": 1.77},
}

CMR_EXPORT = REPO / "data" / "v2v4real_inputs" / "ours" / "pp_score0" / "cmr_export"
AB3DMOT_DETS = REPO / "third_party" / "AB3DMOT" / "data" / "v2v4real" / "detection"
SRC_DET_DIR = AB3DMOT_DETS / "pp_dmstrack_per_cav_Car_val"
OUT_DET_DIR = AB3DMOT_DETS / "pp_dmstrack_per_cav_self_Car_val"


def load_ego_poses_per_frame(scen_dir: Path):
    """Returns dict (vehicle_id, frame_id) -> (wx, wy, wz, yaw)."""
    out = {}
    with open(scen_dir / "ego_state.csv") as f:
        for r in csv.DictReader(f):
            out[(r["vehicle_id"], int(r["frame_id"]))] = (
                float(r["x"]), float(r["y"]), float(r["z"]), float(r["yaw"])
            )
    return out


def ego_pose_in_tesla_kitti(ego_pose, tesla_pose, ego_h):
    """Convert ego's world pose into tesla's V2V4Real-KITTI frame box center.
    Returns (X_kitti, Y_kitti, Z_kitti, ry_kitti)  — V2V4Real convention:
        X = lidar X (forward),  Y = lidar Z (up),  Z = lidar Y (lateral).
    """
    wx, wy, wz, wyaw = ego_pose
    tx, ty, tz, tyaw = tesla_pose
    dx, dy, dz = wx - tx, wy - ty, wz - tz
    cos_e = math.cos(-tyaw); sin_e = math.sin(-tyaw)
    lidar_x = cos_e * dx - sin_e * dy  # forward in tesla LiDAR
    lidar_y = sin_e * dx + cos_e * dy  # lateral in tesla LiDAR
    lidar_z = dz                        # vertical mount-to-mount
    body_center_lidar_z = lidar_z + LIDAR_Z_MOUNT_TO_GROUND + ego_h / 2.0
    X_k = lidar_x
    Y_k = body_center_lidar_z
    Z_k = lidar_y
    ry_k = wyaw - tyaw
    # wrap to [-pi, pi]
    ry_k = (ry_k + math.pi) % (2 * math.pi) - math.pi
    return X_k, Y_k, Z_k, ry_k


def emit_det_row(frame, X, Y, Z, ry, l, w, h, score):
    """AB3DMOT KITTI MOT det format: frame,type,...,h,w,l,X,Y,Z,ry,score (then trailing 0 for orientation flag)."""
    return (
        f"{frame},2,0,0,0,0,{score:.6f},{h:.6f},{w:.6f},{l:.6f},"
        f"{X:.6f},{Y:.6f},{Z:.6f},{ry:.6f},0.000000\n"
    )


def main():
    OUT_DET_DIR.mkdir(parents=True, exist_ok=True)
    scen_to_seq = map_scenarios_to_seqs()
    n_self_rows = 0
    for scen, seq in sorted(scen_to_seq.items(), key=lambda x: x[1]):
        seq_str = f"{int(seq):04d}"
        scen_dir = CMR_EXPORT / scen
        if not scen_dir.is_dir():
            print(f"SKIP seq {seq_str}: missing scen dir {scen}")
            continue
        poses = load_ego_poses_per_frame(scen_dir)
        for src in ("tesla", "astuff"):
            src_file = SRC_DET_DIR / f"{seq_str}_{src}.txt"
            dst_file = OUT_DET_DIR / f"{seq_str}_{src}.txt"
            if not src_file.exists():
                print(f"SKIP {seq_str}_{src}: src file missing ({src_file})")
                continue
            # Copy the original detections first
            shutil.copy(src_file, dst_file)
            # Append a self-report row per frame
            dims = DEFAULT_DIMS[src]
            with open(dst_file, "a") as f:
                # Find all frames in the original file
                frames = set()
                with open(src_file) as srcf:
                    for line in srcf:
                        parts = line.strip().split(",")
                        if parts:
                            frames.add(int(parts[0]))
                for fid in sorted(frames):
                    tesla_pose = poses.get(("tesla", fid))
                    src_pose   = poses.get((src, fid))
                    if tesla_pose is None or src_pose is None:
                        continue
                    X, Y, Z, ry = ego_pose_in_tesla_kitti(src_pose, tesla_pose, dims["h"])
                    f.write(emit_det_row(fid, X, Y, Z, ry, dims["l"], dims["w"], dims["h"], SELF_SCORE))
                    n_self_rows += 1
        print(f"seq {seq_str} ({scen[:40]:40s}): wrote {src} self-rows for {len(frames)} frames")
    print(f"\nWrote {n_self_rows} self-report rows total into {OUT_DET_DIR}")


if __name__ == "__main__":
    main()
