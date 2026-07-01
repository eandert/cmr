#!/usr/bin/env python3
"""Append a perfect tesla + astuff ego-self-report to every frame of a tracker output.

The tracker (pp_dmstrack with --inject_self_reports) ALREADY injects ego
self-reports as detections at score=0.9, but two issues:

  1. **Coverage is incomplete.** Some sequences only carry one of the two
     ego rows per frame (verified empirically: seqs 0002-0006, 0008 are
     missing astuff cross-broadcast rows; total ego rows ≈ half of
     2 × n_frames).
  2. **EKF drift.** Even when injected, the box that lands in data_0/*.txt
     is the EKF-filtered output — dims drift ~5 cm, position drifts ~5 cm,
     yaw drifts a few degrees. That's enough to break 3D-IoU @ 0.25
     against the ego GT.

This script does a POST-PROCESSING augmentation: for every frame in every
seq of the input tracker output, append two NEW rows with track ids
99001 (tesla self) and 99002 (astuff cross-broadcast in tesla's cam
frame), at score = SELF_REPORT_INJECTION_SCORE. The positions are computed
from ego_state.csv (raw, not EKF-filtered) using the same per-vehicle
LiDAR mount + world-z elevation logic that the with_cav GT augmenter uses
— so each appended row matches its ego-GT counterpart exactly. Existing
self-report rows in the input file are preserved (they get tested against
the same GT and won't double-count for the same GT id under Hungarian
assignment).

Output: a new tracker SHA dir, leaving the input untouched.

Usage:
  python scripts/inject_ego_into_tracker_output.py \\
      --in-sha pp_dmstrack_per_cav_Car_val_DMS_S2_EKF_linear_H1 \\
      --out-sha pp_dmstrack_per_cav_Car_val_DMS_S2_EKF_linear_H1_egofull
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "configs"))

from eval_config import (   # noqa: E402
    CMR_EXPORT,
    V2V4REAL_RESULTS_ROOT,
    SELF_REPORT_INJECTION_SCORE,
)
from v2v4real_seq_map import SCENARIO_TO_SEQ, SEQ_TO_NUM_FRAMES   # noqa: E402
# Reuse the augmenter's pose-reading + world→local-cam math so the appended
# rows are byte-identical to what the with_cav GT augmenter would emit
# (modulo the trailing score column for the tracker format).
from augment_v2v4real_with_cav_self_reports import (   # noqa: E402
    read_poses,
    world_to_local_kitti,
    _emit_ego_self,
)


# Track ids reserved for the post-injected ego rows. Chosen distinct from
# the with_cav GT ids (90001/90002) and well above any natural tracker id.
EGO_INJECT_TID = {"tesla": 99001, "astuff": 99002}


def _format_tracker_row(frame: int, tid: int, kitti_pose: tuple,
                        score: float) -> str:
    """V2V4Real KITTI tracker row format (18 fields, space-separated, 2D-bbox zeros).

    Mirrors the existing tracker output produced by AB3DMOT main.py: each
    field formatted with %.6f, score at the end. 2D bbox cols (5..9) all
    zero — this is required by evaluate.py's V2V4Real zero-2D-box assertion.
    """
    X, Y, Z, ry, l, w, h = kitti_pose
    # Fields: frame tid type trunc occl alpha x1 y1 x2 y2 h w l X Y Z ry score
    return (f"{frame} {tid} Car 0 0 0.000000 0.000000 0.000000 0.000000 0.000000 "
            f"{h:.6f} {w:.6f} {l:.6f} {X:.6f} {Y:.6f} {Z:.6f} {ry:.6f} {score:.6f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-sha", required=True, help="source result_sha")
    ap.add_argument("--out-sha", required=True, help="destination result_sha (will be created)")
    ap.add_argument("--num-hypo", type=int, default=1)
    ap.add_argument("--score", type=float, default=SELF_REPORT_INJECTION_SCORE)
    args = ap.parse_args()

    src_root = V2V4REAL_RESULTS_ROOT / args.in_sha / f"data_{args.num_hypo - 1}"
    dst_root = V2V4REAL_RESULTS_ROOT / args.out_sha / f"data_{args.num_hypo - 1}"
    if not src_root.is_dir():
        raise SystemExit(f"source not found: {src_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    n_appended = 0
    for scen_name, seq in SCENARIO_TO_SEQ.items():
        seq_str = f"{seq:04d}"
        src_file = src_root / f"{seq_str}.txt"
        dst_file = dst_root / f"{seq_str}.txt"
        if not src_file.is_file():
            print(f"  WARN: {src_file} missing; skipping")
            continue

        ego_csv = CMR_EXPORT / scen_name / "ego_state.csv"
        poses = read_poses(ego_csv)
        n_frames = SEQ_TO_NUM_FRAMES[seq]

        with open(dst_file, "w") as fout:
            # Copy the existing tracker dump verbatim — keep its rows so
            # any *real* (non-ego) detections remain. The Hungarian matcher
            # in the evaluator handles overlap with our injected ego rows
            # naturally (1-1 assignment per frame).
            with open(src_file) as fin:
                for line in fin:
                    fout.write(line if line.endswith("\n") else line + "\n")
            # Append the two perfect ego rows for every frame.
            for fid in range(n_frames):
                tesla_p = poses[("tesla", fid)]
                astuff_p = poses[("astuff", fid)]
                tesla_pose  = _emit_ego_self("tesla", tesla_p)
                astuff_pose = world_to_local_kitti(astuff_p, tesla_p, "tesla")
                fout.write(_format_tracker_row(fid, EGO_INJECT_TID["tesla"],
                                               tesla_pose, args.score) + "\n")
                fout.write(_format_tracker_row(fid, EGO_INJECT_TID["astuff"],
                                               astuff_pose, args.score) + "\n")
                n_appended += 2
        print(f"  seq {seq_str}: {n_frames} frames × 2 ego = {2*n_frames} appended")

    print(f"\nAppended {n_appended} ego rows total")
    print(f"Output: {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
