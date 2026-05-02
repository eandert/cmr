#!/usr/bin/env python3
"""Evaluate pre-computed KITTI MOT tracking outputs with our V2V4Real metrics.

Reads 18-column KITTI tracking result files (as written by AB3DMOT / DMSTrack)
and scores them using our compute_ab3dmot_metrics_iou, giving the same two
views shown in dmstrack_cobevt_through_our_eval.py:
  - std_eval  : FPs fully counted
  - v2v4_proto: unmatched FPs in GT-free regions ignored

18-column tracking format per row:
    frame  track_id  type  trunc  occ  obs_angle
    x1  y1  x2  y2          (2D bbox, ignored)
    h  w  l                  (3D dims)
    X  Y  Z                  (3D centre)
    ry                        (yaw)
    score                     (confidence)

GT format (17 columns, no score):
    frame  gt_id  type  0...0  h  w  l  X  Y  Z  ry

Both files use the same coordinate convention (V2V4Real KITTI-like).
For DMSTrack's differentiable KF output (true KITTI camera frame):
  pass --camera-frame → bev_x = Z, bev_y = -X

Usage:
    python scripts/evaluate_precomputed_tracks.py \\
        --tracking-dir /path/to/tracking/data \\
        --gt-dir /home/rave/test/DMSTrack/AB3DMOT/scripts/KITTI/v2v4real_val_label \\
        --label "CoBEVT+AB3DMOT (official)"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dmstrack_cobevt_through_our_eval import build_per_frame_tape
from metrics.v2v4real_metrics import (
    compute_ab3dmot_metrics,
    compute_ab3dmot_metrics_iou,
    compute_hota_iou,
)

GT_DIR_DEFAULT = Path("/home/rave/test/DMSTrack/AB3DMOT/scripts/KITTI/v2v4real_val_label")


def parse_kitti_gt(path: Path) -> Dict[int, List[Tuple]]:
    """Parse V2V4Real GT → {frame: [(gt_id, bev_x, bev_y, w, l, yaw), ...]}.

    Uses KITTI-standard X-Z ground plane (bev_x=X col13, bev_y=Z col15)
    to match the official DMSTrack evaluator's BEV convention.
    """
    out: Dict[int, List[Tuple]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        frame = int(float(toks[0]))
        gid   = int(float(toks[1]))
        if toks[2].lower() not in ("car",):
            continue
        h = float(toks[10]); w = float(toks[11]); l = float(toks[12])
        X = float(toks[13]); Y = float(toks[14]); Z = float(toks[15])
        ry = float(toks[16])
        out.setdefault(frame, []).append((gid, X, Z, w, l, ry))  # X-Z BEV
    return out


def parse_kitti_tracking(
    path: Path,
    camera_frame: bool = False,
) -> Dict[int, List[Tuple]]:
    """Parse 18-col KITTI MOT tracking output.

    Returns {frame: [(track_id, bev_x, bev_y, w, l, yaw, score, h, z_center), ...]}.
    h and z_center are the box height and vertical centre for 3D IoU.

    camera_frame=True: input is KITTI camera frame (right X, down Y, forward Z).
      BEV: bev_x=Z, bev_y=-X.  Vertical: z_center = -(Y - h/2) [camera-down → world-up].
    camera_frame=False (V2V4Real lidar-like frame): bev_x=X, bev_y=Y.
      Vertical: z_center = Z (col 15, elevation relative to sensor).
    """
    out: Dict[int, List[Tuple]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) < 17:
            continue
        frame = int(float(toks[0]))
        track_id = int(float(toks[1]))
        typ = toks[2]
        if typ.lower() not in ("car",):
            continue
        h = float(toks[10])
        w = float(toks[11])
        l = float(toks[12])
        X = float(toks[13])
        Y = float(toks[14])
        Z = float(toks[15])
        ry = float(toks[16])
        score = float(toks[17]) if len(toks) > 17 else 1.0

        if camera_frame:
            # KITTI camera frame: right X, down Y, forward Z.
            # BEV ground plane = X-Z; lateral BEV axis bev_y = −X (left positive).
            bev_x = Z
            bev_y = -X
            # Height: camera Y is down-positive; box bottom at Y, top at Y−h.
            # z_center for 3D IoU (camera-frame convention, used with parse_kitti_gt_3d).
            z_center = Y  # pass raw camera-Y; iou_3d_kitti_camframe interprets [Y−h, Y]
        else:
            # V2V4Real KITTI-like: file columns are (X, Y, Z) = (right, down, forward).
            # BEV ground plane = X-Z (KITTI standard).
            bev_x = X
            bev_y = Z
            z_center = Y  # camera-down Y; box interval [Y−h, Y] for 3D IoU

        out.setdefault(frame, []).append((track_id, bev_x, bev_y, w, l, ry, score, h, z_center))
    return out


def parse_kitti_gt_3d(path: Path) -> Dict[int, List[Tuple]]:
    """Parse GT file returning 8-tuples including h and z_center for 3D IoU.

    V2V4Real GT uses KITTI camera-frame convention: X=right, Y=down, Z=forward.
    BEV ground plane = X-Z.  Returns {frame: [(gt_id, bev_x, bev_y, w, l, yaw, h, z_center), ...]}.
    z_center = Y (camera-down; interval [Y−h, Y] matches the track convention).
    """
    out: Dict[int, List[Tuple]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        frame = int(float(toks[0]))
        gid   = int(float(toks[1]))
        typ   = toks[2]
        if typ.lower() not in ("car",):
            continue
        h = float(toks[10]); w = float(toks[11]); l = float(toks[12])
        X = float(toks[13]); Y = float(toks[14]); Z = float(toks[15])
        ry = float(toks[16])
        out.setdefault(frame, []).append((gid, X, Z, w, l, ry, h, Y))  # bev=(X,Z), z_center=Y
    return out


def run_one(
    scen_id: str,
    tracking_dir: Path,
    gt_dir: Path,
    camera_frame: bool,
    iou_threshold: float,
    match_3d: bool = False,
    convex_hull: bool = False,
    use_track_avg_score: bool = False,
    ab3dmot_rematching: bool = False,
) -> Tuple[Dict, Dict, Dict, Dict, int]:
    tracking_path = tracking_dir / f"{scen_id}.txt"
    gt_path = gt_dir / f"{scen_id}.txt"

    if not tracking_path.exists():
        raise FileNotFoundError(f"Tracking file not found: {tracking_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    tracks = parse_kitti_tracking(tracking_path, camera_frame=camera_frame)

    if match_3d:
        gts = parse_kitti_gt_3d(gt_path)
    else:
        gts = parse_kitti_gt(gt_path)

    tape = build_per_frame_tape(tracks, gts)
    std_m = compute_ab3dmot_metrics(tape)
    v2v_m = compute_ab3dmot_metrics_iou(
        tape, iou_threshold=iou_threshold, ignore_unmatched_fps=True,
        match_3d=match_3d, convex_hull=convex_hull,
        use_track_avg_score=use_track_avg_score,
        ab3dmot_rematching=ab3dmot_rematching,
    )
    std_hota = compute_hota_iou(
        tape, iou_threshold=iou_threshold, ignore_unmatched_fps=False,
        match_3d=match_3d, convex_hull=convex_hull,
    )
    v2v_hota = compute_hota_iou(
        tape, iou_threshold=iou_threshold, ignore_unmatched_fps=True,
        match_3d=match_3d, convex_hull=convex_hull,
    )
    return std_m, v2v_m, std_hota, v2v_hota, sum(len(v) for v in gts.values())


def run_global(
    scen_ids: List[str],
    tracking_dir: Path,
    gt_dir: Path,
    camera_frame: bool,
    iou_threshold: float,
    match_3d: bool = False,
    convex_hull: bool = False,
    ab3dmot_rematching: bool = False,
) -> Tuple[Dict, int]:
    """Global AB3DMOT-exact evaluation: all sequences combined into one tape.

    Matches the official DMSTrack/AB3DMOT evaluator which accumulates TP/FP/FN
    across all sequences before the recall sweep.  With ab3dmot_rematching=True
    and match_3d=True this reproduces the paper's 37.16 AMOTA within ~0.1.
    """
    combined_tape = []
    gt_count = 0
    for seq_idx, seq_id in enumerate(scen_ids):
        tracks = parse_kitti_tracking(tracking_dir / f"{seq_id}.txt", camera_frame=camera_frame)
        if match_3d:
            gts = parse_kitti_gt_3d(gt_dir / f"{seq_id}.txt")
        else:
            gts = parse_kitti_gt(gt_dir / f"{seq_id}.txt")
        track_off = seq_idx * 100_000; gt_off = seq_idx * 100_000; frame_off = seq_idx * 100_000
        g_tracks = {f + frame_off: [(t[0]+track_off,) + t[1:] for t in tl] for f, tl in tracks.items()}
        g_gts    = {f + frame_off: [(g[0]+gt_off,)   + g[1:] for g in gl] for f, gl in gts.items()}
        combined_tape.extend(build_per_frame_tape(g_tracks, g_gts))
        gt_count += sum(len(v) for v in gts.values())

    v2v_m = compute_ab3dmot_metrics_iou(
        combined_tape, iou_threshold=iou_threshold, ignore_unmatched_fps=True,
        match_3d=match_3d, convex_hull=convex_hull,
        use_track_avg_score=True, ab3dmot_rematching=ab3dmot_rematching,
    )
    return v2v_m, gt_count


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--tracking-dir",
        required=True,
        type=Path,
        help="Directory containing {0000..0008}.txt KITTI MOT tracking files",
    )
    ap.add_argument(
        "--gt-dir",
        type=Path,
        default=GT_DIR_DEFAULT,
        help=f"Directory containing GT label files (default: {GT_DIR_DEFAULT})",
    )
    ap.add_argument(
        "--label",
        type=str,
        default="Tracker",
        help="Display label for the method (default: Tracker)",
    )
    ap.add_argument(
        "--camera-frame",
        action="store_true",
        default=False,
        help="Input boxes are in KITTI camera frame (right x, down y, forward z). "
             "Converts to BEV: bev_x=Z, bev_y=-X.",
    )
    ap.add_argument(
        "--iou-threshold",
        type=float,
        default=0.25,
        help="IoU threshold for TP matching (default: 0.25)",
    )
    ap.add_argument(
        "--sequences",
        type=str,
        default=None,
        help="Comma-separated sequence IDs to evaluate, e.g. '0000,0001' (default: 0000..0008)",
    )
    ap.add_argument(
        "--match-3d",
        action="store_true",
        default=False,
        help="Use full 3D IoU (BEV footprint × height interval) instead of BEV-only IoU. "
             "Requires h and z_center in both track and GT tuples.",
    )
    ap.add_argument(
        "--convex-hull",
        action="store_true",
        default=False,
        help="Use scipy.spatial.ConvexHull for intersection area (matches DMSTrack's evaluator exactly).",
    )
    ap.add_argument(
        "--track-avg-score",
        action="store_true",
        default=False,
        help="Replace per-detection scores with track-averaged scores before the recall sweep, "
             "matching AB3DMOT's evaluate.py behaviour.",
    )
    ap.add_argument(
        "--ab3dmot-rematching",
        action="store_true",
        default=False,
        help="Re-run Hungarian matching at each of 40 recall thresholds (AB3DMOT-exact). "
             "Automatically enables --track-avg-score. Slower (~40× per sequence).",
    )
    ap.add_argument(
        "--global-eval",
        action="store_true",
        default=False,
        help="Combine all sequences into a single global tape before evaluation, "
             "matching AB3DMOT's multi-sequence accumulation. Use with --ab3dmot-rematching "
             "and --match-3d to reproduce the paper's reported numbers.",
    )
    args = ap.parse_args()

    if args.sequences:
        scen_ids = [s.strip().zfill(4) for s in args.sequences.split(",")]
    else:
        scen_ids = [f"{i:04d}" for i in range(9)]

    match_mode = "3D IoU (BEV × height)" if args.match_3d else "BEV IoU"
    if args.convex_hull:
        match_mode += " + ConvexHull area"
    if args.track_avg_score or args.ab3dmot_rematching:
        match_mode += " + track-avg score"
    if args.ab3dmot_rematching:
        match_mode += " + AB3DMOT re-match"
    # ab3dmot_rematching implies track_avg_score
    use_track_avg = args.track_avg_score or args.ab3dmot_rematching
    print(f"\n=== {args.label} — pre-computed tracks through our evaluator ===")
    print(f"  Tracking dir : {args.tracking_dir}")
    print(f"  GT dir       : {args.gt_dir}")
    print(f"  Coord frame  : {'KITTI camera → BEV transform' if args.camera_frame else 'as-is'}")
    print(f"  IoU threshold: {args.iou_threshold}  ({match_mode})")
    if args.global_eval:
        match_mode += " + global-eval"
    print()

    if args.global_eval:
        print("=== Global evaluation (all sequences combined, AB3DMOT protocol) ===")
        v2v_global, total_gt = run_global(
            scen_ids, args.tracking_dir, args.gt_dir, args.camera_frame, args.iou_threshold,
            match_3d=args.match_3d, convex_hull=args.convex_hull,
            ab3dmot_rematching=args.ab3dmot_rematching,
        )
        print(f"  GT boxes     : {total_gt}")
        print()
        print(f"  {'Metric':<10s}  {'V2V4_protocol':>15s}")
        print(f"  {'AMOTA':<10s}  {v2v_global['amota']*100:>15.2f}")
        print(f"  {'sAMOTA':<10s}  {v2v_global['samota']*100:>15.2f}")
        print(f"  {'AMOTP':<10s}  {v2v_global['amotp']*100:>15.2f}")
        print(f"  {'MOTA':<10s}  {v2v_global['mota']*100:>15.2f}")
        print()
        print("Reference — DMSTrack paper (CoBEVT+AB3DMOT): AMOTA=37.16  sAMOTA=84.54  MOTA=84.14")
        return

    hdr = f"{'seq':<6s}{'std_AMOTA':>11s}{'V2V_AMOTA':>11s}{'std_MOTA':>10s}{'V2V_MOTA':>10s}{'gt_boxes':>10s}"
    print(hdr)
    print("-" * len(hdr))

    sums = {"std_a": 0.0, "v2v_a": 0.0, "std_m": 0.0, "v2v_m": 0.0,
            "std_p": 0.0, "v2v_p": 0.0, "std_sa": 0.0, "v2v_sa": 0.0}
    hota_sums = {"std_h": 0.0, "v2v_h": 0.0, "std_d": 0.0, "v2v_d": 0.0,
                 "std_as": 0.0, "v2v_as": 0.0, "std_l": 0.0, "v2v_l": 0.0}
    n_ok = 0
    seq_results = []

    for sid in scen_ids:
        try:
            s, v, sh, vh, ng = run_one(
                sid, args.tracking_dir, args.gt_dir, args.camera_frame, args.iou_threshold,
                match_3d=args.match_3d, convex_hull=args.convex_hull,
                use_track_avg_score=use_track_avg,
                ab3dmot_rematching=args.ab3dmot_rematching,
            )
        except FileNotFoundError as e:
            print(f"{sid:<6s}  (skipped — {e})")
            continue
        sums["std_a"]  += s["amota"]  * 100
        sums["v2v_a"]  += v["amota"]  * 100
        sums["std_m"]  += s["mota"]   * 100
        sums["v2v_m"]  += v["mota"]   * 100
        sums["std_sa"] += s["samota"] * 100
        sums["v2v_sa"] += v["samota"] * 100
        sums["std_p"]  += s["amotp"]  * 100
        sums["v2v_p"]  += v["amotp"]  * 100
        hota_sums["std_h"]  += sh["hota"] * 100
        hota_sums["v2v_h"]  += vh["hota"] * 100
        hota_sums["std_d"]  += sh["deta"] * 100
        hota_sums["v2v_d"]  += vh["deta"] * 100
        hota_sums["std_as"] += sh["assa"] * 100
        hota_sums["v2v_as"] += vh["assa"] * 100
        hota_sums["std_l"]  += sh["loca"] * 100
        hota_sums["v2v_l"]  += vh["loca"] * 100
        print(f"{sid:<6s}{s['amota']*100:>11.2f}{v['amota']*100:>11.2f}"
              f"{s['mota']*100:>10.2f}{v['mota']*100:>10.2f}{ng:>10d}")
        seq_results.append((sid, sh, vh))
        n_ok += 1

    if n_ok == 0:
        print("No sequences evaluated.")
        return

    print("-" * len(hdr))
    print(f"\n=== Mean across {n_ok} sequences ===")
    print(f"               std_eval (FPs counted)   v2v4_protocol (FPs ignored)")
    print(f"  AMOTA  :     {sums['std_a']/n_ok:>10.2f}              {sums['v2v_a']/n_ok:>10.2f}")
    print(f"  AMOTP  :     {sums['std_p']/n_ok:>10.2f}              {sums['v2v_p']/n_ok:>10.2f}")
    print(f"  sAMOTA :     {sums['std_sa']/n_ok:>10.2f}              {sums['v2v_sa']/n_ok:>10.2f}")
    print(f"  MOTA   :     {sums['std_m']/n_ok:>10.2f}              {sums['v2v_m']/n_ok:>10.2f}")
    print()
    print("Reference — DMSTrack paper (CoBEVT+AB3DMOT): AMOTA=37.16  sAMOTA=84.54  MOTA=84.14")

    # HOTA table
    print()
    hdr2 = f"{'seq':<6s}{'std_HOTA':>10s}{'V2V_HOTA':>10s}{'std_DetA':>10s}{'V2V_DetA':>10s}{'std_AssA':>10s}{'V2V_AssA':>10s}{'std_LocA':>10s}{'V2V_LocA':>10s}"
    print(f"--- HOTA (IoU≥{args.iou_threshold}) ---")
    print(hdr2)
    print("-" * len(hdr2))
    for sid, sh, vh in seq_results:
        print(f"{sid:<6s}{sh['hota']*100:>10.2f}{vh['hota']*100:>10.2f}"
              f"{sh['deta']*100:>10.2f}{vh['deta']*100:>10.2f}"
              f"{sh['assa']*100:>10.2f}{vh['assa']*100:>10.2f}"
              f"{sh['loca']*100:>10.2f}{vh['loca']*100:>10.2f}")
    print("-" * len(hdr2))
    print(f"{'mean':<6s}{hota_sums['std_h']/n_ok:>10.2f}{hota_sums['v2v_h']/n_ok:>10.2f}"
          f"{hota_sums['std_d']/n_ok:>10.2f}{hota_sums['v2v_d']/n_ok:>10.2f}"
          f"{hota_sums['std_as']/n_ok:>10.2f}{hota_sums['v2v_as']/n_ok:>10.2f}"
          f"{hota_sums['std_l']/n_ok:>10.2f}{hota_sums['v2v_l']/n_ok:>10.2f}")


if __name__ == "__main__":
    main()
