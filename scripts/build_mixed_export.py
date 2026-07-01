#!/usr/bin/env python3
"""Build a cross-detector mixed CMR export from existing per-vehicle detections.

Takes detections.csv from two source export dirs, takes astuff rows from one
and tesla rows from the other, writes the merged detections.csv to a new dir.
ego_state.csv, gt_objects.csv, meta.yaml are copied unchanged (identical
across detector exports — same ground truth, same ego poses).

NO DETECTOR RE-RUN. This is a pure CSV row-filter + concat operation.

Usage:
    python scripts/build_mixed_export.py \\
        --src-astuff <your-mmdet3d>/work_dirs/cmr_export_pointpillar_score02 \\
        --src-tesla  <your-mmdet3d>/work_dirs/cmr_export \\
        --output-dir <your-mmdet3d>/work_dirs/cmr_export_mix_pp_cpft \\
        [--split-prefix test__]
"""
import argparse
import csv
import shutil
from pathlib import Path

DET_HEADER = [
    "timestamp", "vehicle_id", "frame_id", "det_idx",
    "x", "y", "z", "yaw", "vx", "vy",
    "length", "width", "height", "score", "label",
]


def filter_dets(src_csv: Path, vehicle_id: str) -> list[dict]:
    """Return rows from detections.csv matching vehicle_id."""
    if not src_csv.exists():
        return []
    with open(src_csv, newline="") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r.get("vehicle_id") == vehicle_id]


def merge_scenario(astuff_src: Path, tesla_src: Path, out_dir: Path) -> tuple[int, int]:
    """Build one merged scenario dir. Returns (n_astuff_dets, n_tesla_dets)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy ego_state, gt_objects, meta.yaml from astuff source (identical across exports)
    for fname in ("ego_state.csv", "gt_objects.csv", "meta.yaml"):
        src = astuff_src / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    astuff_rows = filter_dets(astuff_src / "detections.csv", "astuff")
    tesla_rows  = filter_dets(tesla_src  / "detections.csv", "tesla")

    # Sort by (frame_id, vehicle_id, det_idx) for stable output
    all_rows = astuff_rows + tesla_rows
    all_rows.sort(key=lambda r: (int(r["frame_id"]), r["vehicle_id"], int(r["det_idx"])))

    with open(out_dir / "detections.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DET_HEADER)
        writer.writeheader()
        writer.writerows(all_rows)

    return len(astuff_rows), len(tesla_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-astuff", type=Path, required=True,
                    help="Source export dir for astuff's detections")
    ap.add_argument("--src-tesla", type=Path, required=True,
                    help="Source export dir for tesla's detections")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Destination dir for merged export")
    ap.add_argument("--split-prefix", default="test__",
                    help="Scenario dir prefix to merge (default: test__)")
    args = ap.parse_args()

    # Find scenarios that exist in BOTH sources (intersection by name)
    astuff_scenarios = {d.name: d for d in args.src_astuff.iterdir()
                        if d.is_dir() and d.name.startswith(args.split_prefix)}
    tesla_scenarios  = {d.name: d for d in args.src_tesla.iterdir()
                        if d.is_dir() and d.name.startswith(args.split_prefix)}
    common = sorted(set(astuff_scenarios) & set(tesla_scenarios))

    if not common:
        raise SystemExit(f"No common {args.split_prefix} scenarios between sources")

    print(f"Merging {len(common)} scenarios:")
    print(f"  astuff source: {args.src_astuff}")
    print(f"  tesla  source: {args.src_tesla}")
    print(f"  output:        {args.output_dir}")
    print()

    total_a, total_t = 0, 0
    for name in common:
        out = args.output_dir / name
        n_a, n_t = merge_scenario(astuff_scenarios[name], tesla_scenarios[name], out)
        total_a += n_a; total_t += n_t
        print(f"  {name}: astuff={n_a}, tesla={n_t}")

    print(f"\nDone. Total: astuff={total_a}, tesla={total_t} rows.")


if __name__ == "__main__":
    main()
