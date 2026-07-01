"""Per-CAV local-frame detection augmentation for V2V4Real.

Produces detection files where each CAV's detections live in THAT CAV's own
LiDAR/KITTI frame (no LF pre-collapse). Each scenario gets:

    pp_per_cav_local_Car_val/
        {seq}_astuff.txt           KITTI-MOT-format detections (TESLA's frame —
                                   coords are transformed to a common frame so
                                   the per-source AB3DMOT loop can match them
                                   without a per-source transform at runtime)
        {seq}_tesla.txt            same, src=tesla
        {seq}_poses.csv            per-(frame, source) RTK poses
        {seq}_{src}_ranges.csv     side-car: per-row source-local range
                                   (Euclidean distance from src_pose to
                                   target_world). Row order matches the
                                   corresponding {seq}_{src}.txt one-to-one.
                                   Tracker uses this for GPEM R lookup so it
                                   keys into the right per-source bin even
                                   though the matching frame is tesla's.

Each per-source .txt file also contains the CAV's self-report at the origin in
its own frame (score = SELF_REPORT_SCORE sentinel for the RTK flag). The
corresponding row in {src}_ranges.csv has range=0 for the self-report.

This is the foundation for the per-source sequential AB3DMOT loop:
    for frame:
        predict_all_tracks()
        for src in [tesla, astuff]:
            dets_local = read {seq}_{src}.txt filtered to this frame
            ranges    = read {seq}_{src}_ranges.csv filtered to this frame
            pose_local = read {seq}_poses.csv filtered to (frame, src)
            for each det: R_loc = GPEM_lookup(range=ranges[i]); ...
            match + update with source_id=src
        age out tracks unmatched across ALL sources
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

CMR_EXPORT = REPO / "data" / "v2v4real_inputs" / "ours" / "pp_score0" / "cmr_export"
AB3DMOT_DETS = REPO / "third_party" / "AB3DMOT" / "data" / "v2v4real" / "detection"
OUT_DETS = AB3DMOT_DETS / "pp_per_cav_local_Car_val"

# Drop detections whose source-local Euclidean range exceeds this. 142m is
# the diagonal of a 100m × 100m point-cloud grid (sqrt(2)·100 ≈ 141.42), i.e.
# the maximum source-local range any of our detectors (PP 80×40, CP-FT 54m,
# CP-100m) could physically produce. Anything past 142m is a known artifact
# — specifically cmr_export_cobevt*, which duplicates each fused cobevt
# detection across BOTH vehicles at the same world position, so detections
# near vehicle A get an inflated apparent range from vehicle B (= inter-cav
# distance + true range from A). See the audit in DMSTRACK_COBEVT_NOTES.md
# or git log around this commit.
MAX_SOURCE_LOCAL_RANGE_M = 142.0

# Re-use helpers from the LF-augmentation script (same file structure).
import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment_v2v4real_with_cav_self_reports import (
    LIDAR_Z_MOUNT_TO_GROUND, SELF_REPORT_SCORE,
    map_scenarios_to_seqs, read_poses, world_to_local_kitti,
    emit_kitti_det_row,
)

SOURCES = ("astuff", "tesla")


def emit_pose_row(frame: int, src: str, pose: tuple) -> str:
    """Row in {seq}_poses.csv. pose = (x, y, z, yaw, l, w, h)."""
    x, y, z, yaw, l, w, h = pose
    return f"{frame},{src},{x:.6f},{y:.6f},{z:.6f},{yaw:.6f},{l:.6f},{w:.6f},{h:.6f}"


def read_detections(det_csv: Path) -> dict[tuple[str, int], list[tuple]]:
    """Return {(vehicle_id, frame): [(x, y, z, yaw, l, w, h, score), ...]}
    parsed from cmr_export's detections.csv. Detections are in WORLD frame.

    Upstream detections.csv has duplicate rows (each detection appears 2x with
    byte-identical values — verified in test__Day19__... frame 0). We dedup by
    (vehicle_id, frame_id, det_idx) so each detection only counts once.
    """
    seen_keys: set[tuple[str, int, int]] = set()
    by_src_frame: dict[tuple[str, int], list[tuple]] = {}
    with open(det_csv) as f:
        for r in csv.DictReader(f):
            v = r["vehicle_id"]
            fid = int(r["frame_id"])
            did = int(r.get("det_idx", "0"))
            dedup_key = (v, fid, did)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            by_src_frame.setdefault((v, fid), []).append((
                float(r["x"]), float(r["y"]), float(r["z"]),
                float(r["yaw"]),
                float(r["length"]), float(r["width"]), float(r["height"]),
                float(r["score"]),
            ))
    return by_src_frame


def main() -> None:
    scen_to_seq = map_scenarios_to_seqs()
    print(f"Mapped {len(scen_to_seq)} scenarios to seqs")
    OUT_DETS.mkdir(parents=True, exist_ok=True)

    total_dets = 0
    total_self = 0
    total_pose_rows = 0
    for scen, seq in sorted(scen_to_seq.items(), key=lambda x: x[1]):
        seq_str = f"{seq:04d}"
        scen_dir = CMR_EXPORT / scen
        ego_csv = scen_dir / "ego_state.csv"
        det_csv = scen_dir / "detections.csv"
        if not ego_csv.exists() or not det_csv.exists():
            print(f"  SKIP {scen}: missing ego/detections csv")
            continue

        poses = read_poses(ego_csv)  # {(src, frame): pose7}
        dets = read_detections(det_csv)  # {(src, frame): [world-frame dets]}

        # Which frames exist for which source? Use poses (the canonical source
        # of "this CAV was present in this frame").
        frames_by_src: dict[str, list[int]] = {}
        for (src, fid) in poses.keys():
            frames_by_src.setdefault(src, []).append(fid)
        for src in frames_by_src:
            frames_by_src[src].sort()

        # ----- write per-source detection files in TESLA's common KITTI frame -----
        # We attribute the source (file naming) but normalize coordinates so the
        # per-source AB3DMOT loop can match against a consistent frame.
        # Side-car {src}_ranges.csv records the source-local Euclidean distance
        # (src_pose → target_world) for each row in lockstep, so GPEM R can be
        # looked up using each source's actual sensing geometry even though
        # the matching coordinates are in tesla's frame. AB3DMOT's loader
        # cannot accept extra columns on the KITTI line (it slices [:, -1]
        # for orientation) — hence a separate file.
        for src in SOURCES:
            if src not in frames_by_src:
                continue
            dst = OUT_DETS / f"{seq_str}_{src}.txt"
            dst_ranges = OUT_DETS / f"{seq_str}_{src}_ranges.csv"
            with open(dst, "w") as fout, open(dst_ranges, "w") as frng:
                frng.write("frame,line_idx_in_frame,source_local_range_m,kind\n")
                for fid in frames_by_src[src]:
                    src_pose = poses.get((src, fid))
                    tesla_pose = poses.get(("tesla", fid))
                    if src_pose is None or tesla_pose is None:
                        continue
                    line_idx = 0  # resets each frame

                    # 1. self-report: src's actual position transformed into tesla's frame.
                    # For src=tesla this is (0, ground_off, 0); for src=astuff it's the
                    # astuff-in-tesla position. Source-local range = 0 (it's the source itself).
                    self_kitti = world_to_local_kitti(src_pose, tesla_pose)
                    fout.write(emit_kitti_det_row(fid, self_kitti) + "\n")
                    frng.write(f"{fid},{line_idx},0.000000,self\n")
                    line_idx += 1
                    total_self += 1

                    # 2. PP detections by this src (world frame) → tesla's common frame.
                    # Source-local range = Euclidean(target_world_xyz, src_pose_xyz).
                    # Drop dets with src-local range > MAX_SOURCE_LOCAL_RANGE_M
                    # (handles cobevt-export duplication where shared fused dets
                    # get inflated ranges via inter-cav offsets).
                    src_dets = dets.get((src, fid), [])
                    sx, sy, sz = src_pose[0], src_pose[1], src_pose[2]
                    for d in src_dets:
                        target_world = d[:7]  # x, y, z, yaw, l, w, h
                        score = d[7]
                        tx, ty, tz = target_world[0], target_world[1], target_world[2]
                        src_range = ((tx - sx) ** 2 + (ty - sy) ** 2 + (tz - sz) ** 2) ** 0.5
                        if src_range > MAX_SOURCE_LOCAL_RANGE_M:
                            continue
                        kitti = world_to_local_kitti(target_world, tesla_pose)
                        fout.write(emit_kitti_det_row(fid, kitti, score=score) + "\n")
                        frng.write(f"{fid},{line_idx},{src_range:.6f},det\n")
                        line_idx += 1
                        total_dets += 1

        # ----- write pose file (one row per (frame, source)) -----
        pose_dst = OUT_DETS / f"{seq_str}_poses.csv"
        with open(pose_dst, "w") as f:
            f.write("frame,source,x,y,z,yaw,length,width,height\n")
            # Frame range = union of all sources' frames
            all_frames = sorted({fid for fl in frames_by_src.values() for fid in fl})
            for fid in all_frames:
                for src in SOURCES:
                    pose = poses.get((src, fid))
                    if pose is None:
                        continue
                    f.write(emit_pose_row(fid, src, pose) + "\n")
                    total_pose_rows += 1

        print(f"  scen seq {seq_str} ({scen[:40]:40s}): "
              f"frames={len(all_frames)}  sources={sorted(frames_by_src)}")

    print()
    print(f"Wrote {total_dets} PP det rows, {total_self} self-report rows, "
          f"{total_pose_rows} pose rows into {OUT_DETS}")


if __name__ == "__main__":
    main()
