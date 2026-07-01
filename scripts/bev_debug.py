#!/usr/bin/env python3
"""BEV debug plot: bucket detections + tracker dump + GT, side-by-side.

Single-frame or multi-frame view, all in the same V2V4Real ego-lidar frame
(forward-X / lateral-Z, mount-derived Y) that GT and the bucket .txt files share.

Two panels per frame:
  Left   — raw per-CAV detections (tesla=orange, astuff=blue) + GT (green)
  Right  — tracker tracks (red, with track-ID labels) + GT (green)

When multiple frames are requested, the output is a vertical stack of (left, right)
pairs — handy for eyeballing how one car's track evolves across the scenario.

Examples:

    # One frame, log_odds dump:
    scripts/bev_debug.py 0 50 \\
        --bucket-dir data/v2v4real_inputs/ours/detectors/pp_score0/ab3dmot_detections \\
        --dump-dir results/V2V4Real_KITTI_DUMP_PP_Score0_log_odds_autotuned_fast/data_0 \\
        --gt-dir data/v2v4real_inputs/baselines/paper_gt/kitti_labels \\
        --out /tmp/bev_seq0_f50.png

    # Several frames stacked into one PNG (smoke-check coordinate fidelity over time):
    scripts/bev_debug.py 0 0 50 100 150 200 250 300 \\
        --bucket-dir data/v2v4real_inputs/ours/detectors/pp_score0/ab3dmot_detections \\
        --dump-dir third_party/AB3DMOT/results/v2v4real/pp_score0_Car_val_PP_Score0_S3_sabre_quadratic_H1/data_0 \\
        --out /tmp/bev_seq0_stack.png

Coordinate convention (all sources):
  X = ego-forward (m), Z = ego-lateral (m), Y = LIDAR-mount-derived (static here).
  Plot axes match: X-horizontal (forward), Z-vertical (lateral).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# --------------------------------------------------------------------------
# IO — parse the three artifact formats we touch.
# --------------------------------------------------------------------------

def load_dets_csv(path: Path, frame: int) -> List[Tuple[float, float, float, float, float, float]]:
    """Bucket detection .txt — comma-separated, KITTI-MOT-style.

    Returns list of (X, Z, l, w, ry, score).
    """
    out: List[Tuple[float, float, float, float, float, float]] = []
    if not path.is_file():
        return out
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if not parts or len(parts) < 15:
                continue
            try:
                if int(parts[0]) != frame:
                    continue
            except ValueError:
                continue
            score = float(parts[6])
            # KITTI MOT location columns 11/12/13 → x/y/z; rot col 14
            X, _Y, Z = float(parts[10]), float(parts[11]), float(parts[12])
            ry = float(parts[13])
            # dims: h=col 8, w=col 9, l=col 10 in KITTI cam order;
            # in our writer h/w/l are 8/9/10 in 0-indexed = parts[7]/[8]/[9].
            l, w = float(parts[9]), float(parts[8])
            out.append((X, Z, l, w, ry, score))
    return out


def load_tracks(path: Path, frame: int) -> List[Tuple[int, float, float, float, float, float]]:
    """KITTI MOT track dump — whitespace-separated (frame, tid, type, ...)."""
    out: List[Tuple[int, float, float, float, float, float]] = []
    if not path.is_file():
        return out
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or len(parts) < 17:
                continue
            try:
                if int(parts[0]) != frame:
                    continue
            except ValueError:
                continue
            tid = int(parts[1])
            # In the dump's KITTI MOT writer: cols 11/12/13 = h/w/l, cols 14/15/16 = x/y/z, col 17 = ry.
            l, w = float(parts[12]), float(parts[11])
            X, _Y, Z = float(parts[13]), float(parts[14]), float(parts[15])
            ry = float(parts[16])
            out.append((tid, X, Z, l, w, ry))
    return out


def load_gt(path: Path, frame: int) -> List[Tuple[int, float, float, float, float, float]]:
    """GT KITTI label — whitespace, with (frame, tid, type, ...) header. Cars only."""
    out: List[Tuple[int, float, float, float, float, float]] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or len(parts) < 17:
                continue
            try:
                if int(parts[0]) != frame or parts[2] != "Car":
                    continue
            except ValueError:
                continue
            tid = int(parts[1])
            l, w = float(parts[12]), float(parts[11])
            x, _y, z = float(parts[13]), float(parts[14]), float(parts[15])
            ry = float(parts[16])
            out.append((tid, x, z, l, w, ry))
    return out


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def draw_box(ax, X, Z, l, w, ry, color, alpha=0.8, lw=1.2, ls="-",
             label: Optional[str] = None, label_size: int = 6):
    cr, sr = np.cos(ry), np.sin(ry)
    corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    R = np.array([[cr, sr], [-sr, cr]])
    world = corners @ R.T + np.array([X, Z])
    ax.add_patch(patches.Polygon(world, closed=True, fill=False, edgecolor=color,
                                  linestyle=ls, linewidth=lw, alpha=alpha))
    # Heading tick — short line from center toward the front of the box.
    heading_end = np.array([X + (l/2) * cr, Z + (l/2) * (-sr)])
    ax.plot([X, heading_end[0]], [Z, heading_end[1]],
            color=color, alpha=alpha, linewidth=lw)
    if label:
        ax.text(X, Z + 1.5, label, color=color, fontsize=label_size,
                ha="center", alpha=alpha)


def _draw_one_frame(axes, frame: int, gt, tesla, astuff, tracks,
                    x_range, z_range, show_track_ids: bool,
                    det_score_floor: float = 0.0):
    ax_dets, ax_trk = axes

    # Apply the left-panel score floor (default 0 = no filter).
    if det_score_floor > 0.0:
        tesla = [d for d in tesla if d[5] >= det_score_floor]
        astuff = [d for d in astuff if d[5] >= det_score_floor]

    dets_title = (
        f"dets score≥{det_score_floor:.2f} (tesla=orange, astuff=blue) + GT"
        if det_score_floor > 0.0
        else "dets (tesla=orange, astuff=blue) + GT"
    )
    for ax, title in [(ax_dets, dets_title),
                      (ax_trk, "tracks (red) + GT")]:
        ax.set_title(f"frame {frame} — {title}", fontsize=10)
        ax.set_xlim(*x_range); ax.set_ylim(*z_range)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.set_xlabel("ego-forward X (m)"); ax.set_ylabel("ego-lateral Z (m)")
        # GT outlines on both panels (reference).
        for tid, x, z, l, w, ry in gt:
            draw_box(ax, x, z, l, w, ry, "green", alpha=0.7, lw=1.2,
                     label=f"gt#{tid}", label_size=7)

    # Dets panel
    for X, Z, l, w, ry, score in tesla:
        a = min(1.0, max(0.25, score / 1.5))
        draw_box(ax_dets, X, Z, l, w, ry, "tab:orange", alpha=a, lw=1.0,
                 label=f"{score:.2f}", label_size=5)
    for X, Z, l, w, ry, score in astuff:
        a = min(1.0, max(0.25, score / 1.5))
        draw_box(ax_dets, X, Z, l, w, ry, "tab:blue", alpha=a, lw=1.0,
                 label=f"{score:.2f}", label_size=5)

    # Tracks panel
    for tid, X, Z, l, w, ry in tracks:
        draw_box(ax_trk, X, Z, l, w, ry, "red", alpha=0.9, lw=1.5,
                 label=(f"#{tid}" if show_track_ids else None), label_size=7)

    # Counts
    ax_dets.text(0.02, 0.97,
                 f"tesla_n={len(tesla)}  astuff_n={len(astuff)}  GT={len(gt)}",
                 transform=ax_dets.transAxes, va="top", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    ax_trk.text(0.02, 0.97,
                f"tracks={len(tracks)}  GT={len(gt)}",
                transform=ax_trk.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seq", type=lambda s: f"{int(s):04d}",
                   help="sequence id (0..8, will be zero-padded)")
    p.add_argument("frames", type=int, nargs="+",
                   help="frame number(s) to render (one panel-pair per frame)")
    p.add_argument("--bucket-dir", type=Path, default=None,
                   help="ab3dmot_detections dir with {seq}_{astuff,tesla}.txt "
                        "(omit for tracker-only BEV — useful for cooperative-fused "
                        "buckets like late_fusion that have no per-CAV split)")
    p.add_argument("--dump-dir", type=Path, required=True,
                   help="tracker dump dir with {seq}.txt (KITTI MOT format)")
    p.add_argument("--gt-dir", type=Path,
                   default=Path("data/v2v4real_inputs/baselines/paper_gt/kitti_labels"),
                   help="GT labels dir with {seq}.txt")
    p.add_argument("--out", type=Path, required=True,
                   help="output PNG path")
    p.add_argument("--x-range", type=str, default="-100,100",
                   help="ego-forward X axis range (comma-sep, default '-100,100'; "
                        "V2V4Real annotates GT to ±100m forward)")
    p.add_argument("--z-range", type=str, default="-40,40",
                   help="ego-lateral Z axis range (comma-sep, default '-40,40'; "
                        "V2V4Real annotates GT to ±40m lateral)")
    p.add_argument("--no-track-ids", action="store_true",
                   help="suppress track-ID labels (cleaner for dense plots)")
    p.add_argument("--det-score-floor", type=float, default=0.20,
                   help="hide left-panel detections below this score "
                        "(default 0.20; pass 0 to show every raw det)")
    args = p.parse_args()

    x_range = tuple(float(v) for v in args.x_range.split(","))
    z_range = tuple(float(v) for v in args.z_range.split(","))

    gt_path = args.gt_dir / f"{args.seq}.txt"
    dump_path = args.dump_dir / f"{args.seq}.txt"
    tesla_path = (args.bucket_dir / f"{args.seq}_tesla.txt"
                  if args.bucket_dir is not None else None)
    astuff_path = (args.bucket_dir / f"{args.seq}_astuff.txt"
                   if args.bucket_dir is not None else None)

    required = [("gt", gt_path), ("tracker dump", dump_path)]
    if args.bucket_dir is not None:
        required += [("tesla bucket", tesla_path),
                     ("astuff bucket", astuff_path)]
    for label, p_ in required:
        if not p_.is_file():
            raise SystemExit(
                f"missing {label} file: {p_}\n"
                f"  Fix: confirm the paths exist and the seq id "
                f"{args.seq!r} matches a file in each dir."
            )

    n_frames = len(args.frames)
    fig, axes = plt.subplots(n_frames, 2, figsize=(16, 6 * n_frames),
                             squeeze=False)
    for i, frame in enumerate(args.frames):
        gt = load_gt(gt_path, frame)
        tesla = load_dets_csv(tesla_path, frame) if tesla_path else []
        astuff = load_dets_csv(astuff_path, frame) if astuff_path else []
        tracks = load_tracks(dump_path, frame)
        _draw_one_frame(axes[i], frame, gt, tesla, astuff, tracks,
                        x_range, z_range,
                        show_track_ids=not args.no_track_ids,
                        det_score_floor=args.det_score_floor)
        n_tesla_shown = sum(1 for d in tesla if d[5] >= args.det_score_floor)
        n_astuff_shown = sum(1 for d in astuff if d[5] >= args.det_score_floor)
        print(f"  frame {frame}: tesla={n_tesla_shown}/{len(tesla)} "
              f"astuff={n_astuff_shown}/{len(astuff)} "
              f"tracks={len(tracks)} gt={len(gt)}  "
              f"(shown/total dets at floor={args.det_score_floor:.2f})")

    bucket_label = args.bucket_dir.name if args.bucket_dir is not None else "(none)"
    fig.suptitle(f"seq {args.seq}  —  bucket={bucket_label}  "
                 f"dump={args.dump_dir.name}", fontsize=12)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
