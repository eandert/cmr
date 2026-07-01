#!/usr/bin/env python3
"""Enrich polar_calibration CSVs with per-bin FP shape stats.

Reads each detector's fp_records.csv (range, angle_deg, score, width, length,
box_height, yaw, class_label), bins records into the existing polar grid, and
appends shape-distribution columns to the polar_calibration.csv:

  fp_width_mean,  fp_width_std
  fp_length_mean, fp_length_std
  fp_height_mean, fp_height_std
  fp_yaw_std

The class-count columns (n_fp_car etc.) and rate columns (fp_per_frame etc.)
are already present in the mmdetection3d-generated polar_calibration files;
those are copied as-is.

Output: src/data/sensor_models/<detector>_polar_calibration.csv

Usage:
    python scripts/enrich_fp_calibration.py          # all detectors
    python scripts/enrich_fp_calibration.py --dry-run
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

# ── local_paths: developer-set external repos (see paths.local.yaml) ──
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import local_paths as _LOCAL_PATHS  # noqa: E402


REPO    = Path(__file__).resolve().parent.parent
SRC_DIR = REPO / "src/data/sensor_models"
MM_DIR  = _LOCAL_PATHS.get("mmdet3d_root") / "src/data/sensor_models"

FP_CLASSES = ["car", "truck", "bus", "construction_vehicle"]

NEW_COLS = [
    "fp_width_mean",  "fp_width_std",
    "fp_length_mean", "fp_length_std",
    "fp_height_mean", "fp_height_std",
    "fp_yaw_std",
]

# (source polar_calibration, source fp_records, dest in cmr)
TARGETS = [
    ("centerpoint_polar_calibration.csv",
     "centerpoint_fp_records.csv",
     "centerpoint_polar_calibration.csv"),
    ("bevfusion_polar_calibration.csv",
     "bevfusion_fp_records.csv",
     "bev_fusion_polar_calibration.csv"),
    ("detr3d_polar_calibration.csv",
     "detr3d_fp_records.csv",
     "detr3d_polar_calibration.csv"),
    ("centerpoint_finetune_astuff_polar_calibration.csv",
     "centerpoint_finetune_astuff_fp_records.csv",
     "centerpoint_54m_v2v4real_finetune_astuff_polar_calibration.csv"),
    ("centerpoint_finetune_tesla_polar_calibration.csv",
     "centerpoint_finetune_tesla_fp_records.csv",
     "centerpoint_54m_v2v4real_finetune_tesla_polar_calibration.csv"),
]


def _in_bin(r: float, a: float, row: dict) -> bool:
    return (float(row["range_lo"]) <= r < float(row["range_hi"]) and
            float(row["angle_lo"]) <= a < float(row["angle_hi"]))


def _build_bin_key(r: float, a: float, bins: list[tuple]) -> tuple | None:
    for (rlo, rhi, alo, ahi) in bins:
        if rlo <= r < rhi and alo <= a < ahi:
            return (rlo, rhi, alo, ahi)
    return None


def _load_fp_shape_stats(fp_path: Path, bins: list[tuple]) -> dict:
    """Stream fp_records.csv and accumulate per-bin shape stats."""
    widths  = defaultdict(list)
    lengths = defaultdict(list)
    heights = defaultdict(list)
    yaws    = defaultdict(list)

    print(f"  Reading {fp_path.name} ({fp_path.stat().st_size // 1_000_000}M) ...")
    n_read = 0
    with open(fp_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_read += 1
            try:
                r = float(row["range"])
                a = float(row["angle_deg"])
                w = float(row["width"])
                l = float(row["length"])
                h = float(row["box_height"])
                y = float(row["yaw"])
            except (ValueError, KeyError):
                continue
            key = _build_bin_key(r, a, bins)
            if key is None:
                continue
            widths[key].append(w)
            lengths[key].append(l)
            heights[key].append(h)
            yaws[key].append(y)

    print(f"  Processed {n_read:,} FP records")

    stats = {}
    for key in bins:
        ws, ls, hs, ys = widths[key], lengths[key], heights[key], yaws[key]
        stats[key] = {
            "fp_width_mean":  mean(ws) if ws else 0.0,
            "fp_width_std":   stdev(ws) if len(ws) > 1 else 0.0,
            "fp_length_mean": mean(ls) if ls else 0.0,
            "fp_length_std":  stdev(ls) if len(ls) > 1 else 0.0,
            "fp_height_mean": mean(hs) if hs else 0.0,
            "fp_height_std":  stdev(hs) if len(hs) > 1 else 0.0,
            "fp_yaw_std":     stdev(ys) if len(ys) > 1 else math.pi / 4,
        }
    return stats


def enrich(src_polar: Path, fp_records: Path, dest: Path, dry_run: bool) -> None:
    if not src_polar.exists():
        print(f"  [skip] not found: {src_polar}")
        return
    if not fp_records.exists():
        print(f"  [skip] not found: {fp_records}")
        return

    with open(src_polar, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"  [skip] empty: {src_polar}")
        return

    if NEW_COLS[0] in rows[0]:
        print(f"  [skip] already enriched: {dest.name}")
        return

    bins = sorted({
        (float(r["range_lo"]), float(r["range_hi"]),
         float(r["angle_lo"]), float(r["angle_hi"]))
        for r in rows
    })

    stats = _load_fp_shape_stats(fp_records, bins)

    if dry_run:
        print(f"  [dry-run] would write {dest} ({len(rows)} rows, +{len(NEW_COLS)} cols)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + NEW_COLS
    with open(dest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            key = (float(row["range_lo"]), float(row["range_hi"]),
                   float(row["angle_lo"]), float(row["angle_hi"]))
            for col in NEW_COLS:
                row[col] = stats.get(key, {}).get(col, 0.0)
            writer.writerow(row)

    print(f"  wrote {dest}  ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for src_cal, src_fp, dest_name in TARGETS:
        print(f"\n=== {dest_name} ===")
        enrich(
            src_polar   = MM_DIR / src_cal,
            fp_records  = MM_DIR / src_fp,
            dest        = SRC_DIR / dest_name,
            dry_run     = args.dry_run,
        )


if __name__ == "__main__":
    main()
