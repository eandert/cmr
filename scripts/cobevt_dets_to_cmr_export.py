#!/usr/bin/env python3
"""Convert CoBEVT KITTI detection files to CMR export format.

CoBEVT outputs cooperative detections in cav-0's LiDAR frame, encoded in
AB3DMOT-style KITTI format. CMR's detections.csv expects global (UTM) coords.

Two-step transform per detection:

  Step 1: AB3DMOT KITTI → mmdet3d LiDAR (cav-0 frame)
    (matches `ab3dmot_to_lidar()` in mmdet3d/tools/dump_dmstrack_predictions.py)
      lidar_x = KITTI_col10
      lidar_y = KITTI_col12     (y/z swap)
      lidar_z = KITTI_col11 − h/2  (bottom-Z = center-Z − h/2)
      yaw     = KITTI_col13

  Step 2: cav-0 LiDAR → global UTM
    Default V2V4Real val has tesla as cav-0 (no --swap_ego).
    Apply tesla's per-frame pose from ego_state.csv:
      global_x = tesla_x + cos(tesla_yaw)·lidar_x − sin(tesla_yaw)·lidar_y
      global_y = tesla_y + sin(tesla_yaw)·lidar_x + cos(tesla_yaw)·lidar_y
      global_z = tesla_z + lidar_z
      global_yaw = tesla_yaw + yaw

Sequence mapping: sorted CMR test__* dirs ↔ sorted KITTI *.txt files.
Frame counts verified to match 1-to-1 for all 9 val sequences.

Each detection is written twice — once per vehicle (astuff, tesla) — since
CoBEVT cooperative fusion gives both vehicles the same fused detection set.

Usage:
    python scripts/cobevt_dets_to_cmr_export.py \\
        --cobevt-det-dir .../detection/cobevt_Car_val \\
        --source-export-dir .../cmr_export \\
        --output-dir .../cmr_export_cobevt
"""
import argparse
import csv
import math
import shutil
from pathlib import Path

VEHICLES_BOTH = ("astuff", "tesla")
LABEL = "car"

DET_HEADER = [
    "timestamp", "vehicle_id", "frame_id", "det_idx",
    "x", "y", "z", "yaw", "vx", "vy",
    "length", "width", "height", "score", "label",
]


def read_kitti_dets(path: Path) -> list[dict]:
    """Read CoBEVT KITTI detection file → list of dicts in cav-0 LiDAR frame.

    Applies the AB3DMOT y/z swap + bottom-Z conversion (matches
    `ab3dmot_to_lidar()` in mmdet3d/tools/dump_dmstrack_predictions.py).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",") if "," in line else line.split()
            if len(parts) < 14:
                continue
            try:
                frame_id = int(float(parts[0]))
                score    = float(parts[6])
                h        = float(parts[7])
                w        = float(parts[8])
                l        = float(parts[9])
                ab_x     = float(parts[10])
                ab_y     = float(parts[11])
                ab_z     = float(parts[12])
                yaw      = float(parts[13])
            except (ValueError, IndexError):
                continue
            # AB3DMOT KITTI → mmdet3d LiDAR (cav-0 frame)
            lidar_x = ab_x
            lidar_y = ab_z          # y/z swap
            lidar_z = ab_y - h / 2  # camera-y → z-center
            rows.append({
                "frame_id": frame_id,
                "score":    score,
                "height":   h,
                "width":    w,
                "length":   l,
                "x":        lidar_x,
                "y":        lidar_y,
                "z":        lidar_z,
                "yaw":      yaw,
            })
    return rows


def load_cav0_poses(ego_state_path: Path, cav0_vehicle_id: str) -> dict[int, tuple]:
    """Read ego_state.csv → {frame_id: (x, y, z, yaw)} for cav-0."""
    poses = {}
    with open(ego_state_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["vehicle_id"] != cav0_vehicle_id:
                continue
            frame = int(r["frame_id"])
            poses[frame] = (float(r["x"]), float(r["y"]), float(r["z"]), float(r["yaw"]))
    return poses


def lidar_to_global(d: dict, cav0_pose: tuple) -> dict:
    """Transform one detection from cav-0 LiDAR → global UTM."""
    cx, cy, cz, cyaw = cav0_pose
    cos_y, sin_y = math.cos(cyaw), math.sin(cyaw)
    return {
        "frame_id": d["frame_id"],
        "score":    d["score"],
        "height":   d["height"],
        "width":    d["width"],
        "length":   d["length"],
        "x":        cx + cos_y * d["x"] - sin_y * d["y"],
        "y":        cy + sin_y * d["x"] + cos_y * d["y"],
        "z":        cz + d["z"],
        "yaw":      cyaw + d["yaw"],
    }


def _filter_csv_to_vehicle(in_path: Path, out_path: Path, vehicle_id: str) -> None:
    """Copy a CSV (ego_state.csv or gt_objects.csv) keeping only rows
    whose vehicle_id matches the given vehicle."""
    with open(in_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if r.get("vehicle_id") == vehicle_id]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convert_scenario(kitti_path: Path, src_dir: Path, out_dir: Path,
                     cav0_vehicle_id: str = "tesla",
                     single_ego: bool = False) -> None:
    """Convert one KITTI sequence → one CMR scenario dir.

    cav0_vehicle_id: which vehicle's pose is cav-0 (default "tesla" — matches
    V2V4Real default val without --swap_ego).

    single_ego: If True, write export with ONLY cav-0 (tesla by default) — strip
    other vehicle from ego_state.csv and gt_objects.csv, attribute all dets to
    cav-0 only. Matches CoBEVT+AB3DMOT's official eval setup (single tracker,
    single GT). If False, duplicate dets to both vehicles (per-ego eval setup).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if single_ego:
        # Filter ego_state.csv and gt_objects.csv to cav-0 only
        _filter_csv_to_vehicle(src_dir / "ego_state.csv",
                               out_dir / "ego_state.csv", cav0_vehicle_id)
        _filter_csv_to_vehicle(src_dir / "gt_objects.csv",
                               out_dir / "gt_objects.csv", cav0_vehicle_id)
        if (src_dir / "meta.yaml").exists():
            shutil.copy2(src_dir / "meta.yaml", out_dir / "meta.yaml")
        out_vehicles = (cav0_vehicle_id,)
    else:
        for fname in ("ego_state.csv", "gt_objects.csv", "meta.yaml"):
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)
        out_vehicles = VEHICLES_BOTH

    cav0_poses = load_cav0_poses(src_dir / "ego_state.csv", cav0_vehicle_id)
    if not cav0_poses:
        raise SystemExit(f"No cav-0 poses for {cav0_vehicle_id} in {src_dir.name}")

    dets_lidar = read_kitti_dets(kitti_path)

    frame_dets: dict[int, list] = {}
    for d in dets_lidar:
        if d["frame_id"] not in cav0_poses:
            continue  # skip if no pose for this frame
        d_global = lidar_to_global(d, cav0_poses[d["frame_id"]])
        frame_dets.setdefault(d_global["frame_id"], []).append(d_global)

    rows = []
    for frame_id in sorted(frame_dets):
        ts = frame_id * 0.1
        for vehicle_id in out_vehicles:
            for det_idx, d in enumerate(frame_dets[frame_id]):
                rows.append({
                    "timestamp":  f"{ts:.6f}",
                    "vehicle_id": vehicle_id,
                    "frame_id":   frame_id,
                    "det_idx":    det_idx,
                    "x":          f"{d['x']:.6f}",
                    "y":          f"{d['y']:.6f}",
                    "z":          f"{d['z']:.6f}",
                    "yaw":        f"{d['yaw']:.6f}",
                    "vx":         "0.000000",
                    "vy":         "0.000000",
                    "length":     f"{d['length']:.6f}",
                    "width":      f"{d['width']:.6f}",
                    "height":     f"{d['height']:.6f}",
                    "score":      f"{d['score']:.6f}",
                    "label":      LABEL,
                })

    with open(out_dir / "detections.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DET_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {out_dir.name}: {len(frame_dets)} frames, "
          f"{len(dets_lidar)} dets × {len(out_vehicles)} = {len(rows)} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cobevt-det-dir", type=Path, required=True,
                    help="Dir with CoBEVT KITTI detection txts (0000.txt … 0008.txt)")
    ap.add_argument("--source-export-dir", type=Path, required=True,
                    help="CMR export dir containing test__* scenario subdirs")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Destination dir (will be created if needed)")
    ap.add_argument("--cav0-vehicle", default="tesla",
                    help="Which vehicle is cav-0 in CoBEVT's frame (default: tesla)")
    ap.add_argument("--single-ego", action="store_true",
                    help="Match CoBEVT+AB3DMOT eval: only cav-0 in export "
                         "(strip other vehicle from ego_state, gt_objects, dets)")
    ap.add_argument("--split-prefix", default="test__",
                    help="CMR scenario dir prefix to match (default: test__; "
                         "use train__ for train split)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only convert the first N sequences (for smoke testing)")
    args = ap.parse_args()

    kitti_files = sorted(args.cobevt_det_dir.glob("*.txt"))
    cmr_scenarios = sorted(d for d in args.source_export_dir.iterdir()
                           if d.is_dir() and d.name.startswith(args.split_prefix))

    # Match by max-frame-count (robust when KITTI count != CMR scenario count)
    def _kitti_max_frame(p: Path) -> int:
        m = -1
        with open(p) as f:
            for line in f:
                if not line.strip(): continue
                parts = line.split(",") if "," in line else line.split()
                try: m = max(m, int(float(parts[0])))
                except (ValueError, IndexError): pass
        return m

    def _cmr_max_frame(d: Path) -> int:
        m = -1
        with open(d / "ego_state.csv") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try: m = max(m, int(r["frame_id"]))
                except (ValueError, KeyError): pass
        return m

    cmr_by_frame: dict[int, Path] = {}
    for d in cmr_scenarios:
        cmr_by_frame.setdefault(_cmr_max_frame(d), d)

    pairs: list[tuple[Path, Path]] = []
    skipped: list[Path] = []
    for kf in kitti_files:
        mf = _kitti_max_frame(kf)
        if mf in cmr_by_frame:
            pairs.append((kf, cmr_by_frame[mf]))
        else:
            skipped.append(kf)

    if args.limit:
        pairs = pairs[:args.limit]

    print(f"Mapping {len(pairs)} sequences (cav-0 = {args.cav0_vehicle}, "
          f"prefix = {args.split_prefix}):")
    for kf, ts in pairs:
        print(f"  {kf.name} (max_frame={_kitti_max_frame(kf)}) → {ts.name}")
    if skipped:
        print(f"\nSkipped {len(skipped)} KITTI files (no matching CMR scenario):")
        for kf in skipped:
            print(f"  {kf.name} (max_frame={_kitti_max_frame(kf)})")
    print()

    for kf, src in pairs:
        convert_scenario(kf, src, args.output_dir / src.name,
                         args.cav0_vehicle, args.single_ego)

    print(f"\nDone. {len(pairs)} scenarios → {args.output_dir}")


if __name__ == "__main__":
    main()
