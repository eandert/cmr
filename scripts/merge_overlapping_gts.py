#!/usr/bin/env python3
"""Merge overlapping GT rows per frame ("average duplicate hits").

Operates on a V2V4Real KITTI MOT label directory. For each frame, any pair of
GT rows whose 3D-IoU exceeds --threshold is treated as the same physical
object and merged into one row with averaged position, dims, and yaw (circular
mean). The smaller of the two track_ids survives.

Why: the with_cav GT augmentation appends ego rows (tesla=90001 / astuff=90002)
that occasionally overlap a real-world third-party GT car (e.g., when astuff
happens to be near another annotated vehicle). Without de-dup, the evaluator
sees two GT rows for the same object — inflating GT_total and creating an
unmatchable FN even if the tracker correctly reports one box.

Usage:
    python scripts/merge_overlapping_gts.py \\
        --in-dir  data/v2v4real_inputs/augmented_gt/with_cav \\
        --out-dir data/v2v4real_inputs/augmented_gt/with_cav_merged \\
        --threshold 0.5

The default in-dir / out-dir use ``eval_config.AUGMENTED_GT_WITH_CAV`` and
``AUGMENTED_GT_WITH_CAV_MERGED`` if not specified. After regenerating, the
bucket's MANIFEST.json sha256s will drift — rebuild it::

    python scripts/build_v2v4real_input_bucket.py --bucket augmented_gt \\
        --rebuild manifest readme --apply
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "configs"))

from eval_config import (   # noqa: E402
    BBOX2D_COLS, DIM_COLS, POS_COLS, YAW_COL, N_LABEL_COLS,
)
from geometry import iou_3d_box  # noqa: E402


def _parse_label_row(line: str) -> List:
    """Parse one KITTI MOT label row → list of typed fields.

    Format (17 cols): frame tid type truncation occlusion alpha
                      x1 y1 x2 y2 h w l X Y Z ry
    """
    f = line.strip().split()
    if len(f) != N_LABEL_COLS:
        raise ValueError(f"expected {N_LABEL_COLS} cols, got {len(f)}: {line!r}")
    return [
        int(float(f[0])),   # frame
        int(float(f[1])),   # tid
        f[2],               # type
        int(float(f[3])),   # truncation
        int(float(f[4])),   # occlusion
        float(f[5]),        # alpha
        float(f[6]), float(f[7]), float(f[8]), float(f[9]),   # 2D bbox
        float(f[10]), float(f[11]), float(f[12]),  # h, w, l
        float(f[13]), float(f[14]), float(f[15]),  # X, Y, Z
        float(f[16]),       # ry
    ]


def _format_label_row(row: List) -> str:
    """Inverse of _parse_label_row — pretty-print one row in GT format."""
    return (f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]} "
            f"{row[6]} {row[7]} {row[8]} {row[9]} "
            f"{row[10]} {row[11]} {row[12]} {row[13]} {row[14]} {row[15]} "
            f"{row[16]}")


def _row_to_iou_box(row: List) -> Tuple[float, ...]:
    """Convert a parsed row to the (cx, cy, w, l, yaw, z_center, h) tuple expected by iou_3d_box.

    Note: V2V4Real cam frame uses (X, Y, Z) where:
      - X = lateral, Z = forward in BEV plane
      - Y = vertical (down-positive)
    iou_3d_box expects (cx, cy, w, l, yaw, z_center, h) with cx/cy in the BEV
    plane and z_center = vertical. So we map X→cx, Z→cy, Y→z_center.
    """
    h, w, l = row[10], row[11], row[12]
    X, Y, Z, ry = row[13], row[14], row[15], row[16]
    return (X, Z, w, l, ry, Y, h)


def _circular_mean(angles: List[float]) -> float:
    """Mean of angles via the standard sin/cos trick (handles wraparound at ±π)."""
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return math.atan2(s, c)


def _merge_rows(rows: List[List]) -> List:
    """Merge N rows into one by averaging numeric fields. Categorical fields
    take from the row with the smallest track_id (the "survivor").
    """
    survivor = min(rows, key=lambda r: r[1])
    n = len(rows)
    out = list(survivor)  # copy categorical fields
    # average h, w, l
    for c in DIM_COLS:
        out[c] = sum(r[c] for r in rows) / n
    # average X, Y, Z
    for c in POS_COLS:
        out[c] = sum(r[c] for r in rows) / n
    # circular-mean yaw
    out[YAW_COL] = _circular_mean([r[YAW_COL] for r in rows])
    return out


def _greedy_merge_frame(rows: List[List], iou_threshold: float) -> Tuple[List[List], int]:
    """Greedy per-frame merge: repeatedly find the highest-IoU overlapping pair
    and fold them together until no pair exceeds the threshold.

    Returns (merged_rows, n_merges_performed). n_merges_performed counts the
    NUMBER OF ROWS REMOVED via merging, not the number of merge operations.
    """
    if len(rows) <= 1:
        return rows, 0

    work = [list(r) for r in rows]
    n_removed = 0

    while True:
        n = len(work)
        if n <= 1:
            break
        best_iou = 0.0
        best_pair = None
        boxes = [_row_to_iou_box(r) for r in work]
        for i in range(n):
            for j in range(i + 1, n):
                iou = iou_3d_box(boxes[i], boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_pair = (i, j)
        if best_pair is None or best_iou < iou_threshold:
            break
        i, j = best_pair
        merged = _merge_rows([work[i], work[j]])
        # remove i and j (j first since j > i), insert merged
        del work[j]
        del work[i]
        work.append(merged)
        n_removed += 1

    return work, n_removed


def process_dir(in_dir: Path, out_dir: Path, iou_threshold: float) -> Dict:
    """Process every label file in in_dir, write merged copies to out_dir.

    Returns a summary dict: {seq: {n_rows_in, n_rows_out, n_rows_merged}}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, int]] = {}

    for label_file in sorted(in_dir.glob("*.txt")):
        seq = label_file.stem
        n_in = 0
        n_out = 0
        n_merged = 0
        # Group rows by frame
        by_frame: Dict[int, List[List]] = defaultdict(list)
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            row = _parse_label_row(line)
            by_frame[row[0]].append(row)
            n_in += 1

        # Merge per frame; collect output rows
        out_rows: List[List] = []
        for frame in sorted(by_frame.keys()):
            merged_rows, removed = _greedy_merge_frame(by_frame[frame], iou_threshold)
            n_merged += removed
            out_rows.extend(sorted(merged_rows, key=lambda r: r[1]))
            n_out += len(merged_rows)

        (out_dir / label_file.name).write_text(
            "\n".join(_format_label_row(r) for r in out_rows) + "\n"
        )
        summary[seq] = {"n_rows_in": n_in, "n_rows_out": n_out, "n_rows_merged": n_merged}
        print(f"  {seq}: {n_in} in → {n_out} out (merged {n_merged} duplicate rows)")

    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True,
                    help="source GT label directory (KITTI MOT 17-col format)")
    ap.add_argument("--out-dir", required=True,
                    help="destination directory for merged GT")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="3D-IoU above which two rows are deemed the same object (default 0.5, "
                         "matching V2V4Real's NMS_thresh)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"in-dir not found: {in_dir}")
    if in_dir == out_dir:
        raise SystemExit("refusing to overwrite the source directory in place")

    print(f"merging overlapping GT rows: 3D-IoU threshold = {args.threshold}")
    print(f"  in:  {in_dir}")
    print(f"  out: {out_dir}")
    summary = process_dir(in_dir, out_dir, args.threshold)

    total_in = sum(s["n_rows_in"] for s in summary.values())
    total_out = sum(s["n_rows_out"] for s in summary.values())
    total_merged = sum(s["n_rows_merged"] for s in summary.values())
    print(f"\ntotal: {total_in} → {total_out} rows (merged {total_merged} duplicate rows)")
    print(f"de-dup ratio: {100 * total_merged / max(total_in, 1):.2f}% of rows merged away")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
