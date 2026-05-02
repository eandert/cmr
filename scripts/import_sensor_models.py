#!/usr/bin/env python3
"""Import sensor model CSVs from the mmdetection3d repo into cmr.

This script is the canonical data import step between the two repos.
Run it whenever new detector characterizations are produced.

Steps performed:
  1. Copy polar_calibration CSVs from mmdetection3d → cmr (these carry the
     FP rate, TP error stats, and class-count columns already written by
     the mmdetection3d evaluate_errors pipeline).
  2. Enrich each polar_calibration CSV with FP shape-distribution columns
     (fp_width_mean/std, fp_length_mean/std, fp_height_mean/std, fp_yaw_std)
     by streaming the corresponding fp_records.csv through the polar grid.
  3. Copy the distributions CSV (quadratic/linear GPEM coefficients) if present.

Source repo layout (mmdetection3d):
  src/data/sensor_models/
    <detector>_polar_calibration.csv   — per-bin FP rate + TP stats + class counts
    <detector>_fp_records.csv          — one row per FP (range, angle_deg, score,
                                         width, length, box_height, yaw, class_label)
    <detector>_distributions.csv       — GPEM polynomial coefficients (optional)

Destination (cmr):
  src/data/sensor_models/
    <detector>_polar_calibration.csv   — enriched with FP shape stats
    <detector>_distributions.csv       — copied as-is

Detector name mapping (mmdetection3d → cmr):
  centerpoint              → centerpoint
  bevfusion                → bev_fusion
  detr3d                   → detr3d
  centerpoint_finetune_astuff → centerpoint_54m_v2v4real_finetune_astuff
  centerpoint_finetune_tesla  → centerpoint_54m_v2v4real_finetune_tesla

Usage:
    # Full import (copy + enrich):
    python scripts/import_sensor_models.py

    # Dry-run (no files written):
    python scripts/import_sensor_models.py --dry-run

    # Custom mmdetection3d root:
    python scripts/import_sensor_models.py --mm-root /path/to/mmdetection3d
"""

import argparse
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

REPO    = Path(__file__).resolve().parent.parent
SRC_DIR = REPO / "src/data/sensor_models"

DEFAULT_MM_ROOT = Path("/home/rave/test/mmdetection3d")

# (mm_detector_stem, cmr_detector_stem)
DETECTOR_MAP = [
    ("centerpoint",              "centerpoint"),
    ("bevfusion",                "bev_fusion"),
    ("detr3d",                   "detr3d"),
    ("centerpoint_finetune_astuff", "centerpoint_54m_v2v4real_finetune_astuff"),
    ("centerpoint_finetune_tesla",  "centerpoint_54m_v2v4real_finetune_tesla"),
    ("cobevt_tracker_astuff",       "cobevt_tracker_astuff"),
    ("cobevt_tracker_tesla",        "cobevt_tracker_tesla"),
    ("dmstrack_astuff",             "dmstrack_astuff"),
    ("dmstrack_tesla",              "dmstrack_tesla"),
]

FP_SHAPE_COLS = [
    "fp_width_mean",  "fp_width_std",
    "fp_length_mean", "fp_length_std",
    "fp_height_mean", "fp_height_std",
    "fp_yaw_std",
]


# ---------------------------------------------------------------------------
# FP shape enrichment
# ---------------------------------------------------------------------------

def _build_bins(rows: list[dict]) -> list[tuple]:
    return sorted({
        (float(r["range_lo"]), float(r["range_hi"]),
         float(r["angle_lo"]), float(r["angle_hi"]))
        for r in rows
    })


def _load_shape_stats(fp_path: Path, bins: list[tuple]) -> dict:
    widths  = defaultdict(list)
    lengths = defaultdict(list)
    heights = defaultdict(list)
    yaws    = defaultdict(list)

    mb = fp_path.stat().st_size // 1_000_000
    print(f"    streaming {fp_path.name} ({mb}M) ...")

    bin_arr = list(bins)

    with open(fp_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                r = float(row["range"])
                a = float(row["angle_deg"])
                w = float(row["width"])
                l = float(row["length"])
                h = float(row["box_height"])
                y = float(row["yaw"])
            except (ValueError, KeyError):
                continue
            for (rlo, rhi, alo, ahi) in bin_arr:
                if rlo <= r < rhi and alo <= a < ahi:
                    key = (rlo, rhi, alo, ahi)
                    widths[key].append(w)
                    lengths[key].append(l)
                    heights[key].append(h)
                    yaws[key].append(y)
                    break

    stats = {}
    for key in bin_arr:
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


def _enrich_and_write(src_polar: Path, fp_records: Path,
                      dest: Path, dry_run: bool) -> bool:
    with open(src_polar, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"    [skip] empty: {src_polar.name}")
        return False

    already_enriched = FP_SHAPE_COLS[0] in rows[0]

    if already_enriched and dest.exists():
        print(f"    [skip] already enriched: {dest.name}")
        return False

    fieldnames = list(rows[0].keys())
    if not already_enriched:
        if not fp_records.exists():
            print(f"    [warn] fp_records not found — copying without shape stats: {fp_records.name}")
            stats = {}
        else:
            bins  = _build_bins(rows)
            stats = _load_shape_stats(fp_records, bins)
        fieldnames = fieldnames + FP_SHAPE_COLS
    else:
        stats = {}

    if dry_run:
        extra = f" +{len(FP_SHAPE_COLS)} shape cols" if not already_enriched else ""
        print(f"    [dry-run] {dest.name}  ({len(rows)} rows{extra})")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not already_enriched:
                key = (float(row["range_lo"]), float(row["range_hi"]),
                       float(row["angle_lo"]), float(row["angle_hi"]))
                for col in FP_SHAPE_COLS:
                    row[col] = stats.get(key, {}).get(col, 0.0)
            writer.writerow(row)

    print(f"    wrote {dest.name}  ({len(rows)} rows)")
    return True


# ---------------------------------------------------------------------------
# Main import logic
# ---------------------------------------------------------------------------

def run_import(mm_root: Path, dry_run: bool) -> None:
    mm_models = mm_root / "src/data/sensor_models"
    if not mm_models.exists():
        raise SystemExit(f"mmdetection3d sensor_models dir not found: {mm_models}")

    for mm_stem, cmr_stem in DETECTOR_MAP:
        print(f"\n{'='*60}")
        print(f"  {mm_stem} → {cmr_stem}")
        print(f"{'='*60}")

        polar_src  = mm_models / f"{mm_stem}_polar_calibration.csv"
        fp_src     = mm_models / f"{mm_stem}_fp_records.csv"
        reg_src    = mm_models / f"{mm_stem}.csv"
        dist_src   = mm_models / f"{mm_stem}_distributions.csv"
        polar_dest = SRC_DIR / f"{cmr_stem}_polar_calibration.csv"
        reg_dest   = SRC_DIR / f"{cmr_stem}.csv"
        dist_dest  = SRC_DIR / f"{cmr_stem}_distributions.csv"

        # -- main regression CSV (polynomial coefficients for GPEM linear/quad) --
        if reg_src.exists():
            if dry_run:
                print(f"  [dry-run] copy {reg_src.name} → {reg_dest.name}")
            else:
                shutil.copy2(reg_src, reg_dest)
                print(f"  copied {reg_dest.name}")

        # -- polar calibration (copy + enrich with shape stats) --
        if not polar_src.exists():
            print(f"  [skip] not found: {polar_src.name}")
        else:
            _enrich_and_write(polar_src, fp_src, polar_dest, dry_run)

        # -- distributions CSV (binned normal distributions for GPEM polar/binned) --
        if dist_src.exists():
            if dry_run:
                print(f"  [dry-run] copy {dist_src.name} → {dist_dest.name}")
            else:
                shutil.copy2(dist_src, dist_dest)
                print(f"  copied {dist_dest.name}")

    print(f"\n{'='*60}")
    print("  Import complete." if not dry_run else "  Dry-run complete — no files written.")
    print(f"{'='*60}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mm-root", type=Path, default=DEFAULT_MM_ROOT,
                    help=f"Root of mmdetection3d repo (default: {DEFAULT_MM_ROOT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be done without writing any files")
    args = ap.parse_args()
    run_import(args.mm_root, args.dry_run)


if __name__ == "__main__":
    main()
