#!/usr/bin/env python3
"""Quick-and-dirty AB3DMOT-style replica for V2V4Real.

Goal: reproduce V2V4Real's ~29 AMOTA "Late Fusion" baseline using a
minimal outside-the-framework implementation. Once we hit the number,
we know the recipe is right and can re-apply our calibration on top.

Pipeline per frame:
  1. Concat both CAVs' detections (already in world frame from cmr_export).
  2. Filter by class + score≥0.20.
  3. BEV-NMS @ IoU 0.15 (V2V4Real reference).
  4. Predict all live tracks forward via vanilla CV Kalman.
  5. Hungarian match tracks ↔ detections by NEGATIVE BEV IoU (cost = -IoU,
     gated at IoU > 0.01).
  6. Update matched tracks; spawn unmatched detections as new tracks.
  7. Kill tracks with `misses_since_match >= 2` (AB3DMOT default).
  8. Emit tracks with `hits >= 3` to the AMOTA tape.

Score on tape entry = the latest matched det.score (no decay; matches
AB3DMOT's behaviour where coasting tracks keep their last detection's
confidence).

Evaluated via the same `compute_ab3dmot_metrics` we use everywhere else,
so the AMOTA number is apples-to-apples with our internal benchmarks.

Usage:
    python scripts/v2v4real_ab3dmot_replica.py [--splits test]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from scipy.optimize import linear_sum_assignment

import utils as cmr_utils
from v2v4real_replay import load_scenario, _in_ego_range  # reuse loaders
from metrics.v2v4real_metrics import compute_ab3dmot_metrics, compute_ab3dmot_metrics_iou


# ---------------------------------------------------------------------
# AB3DMOT-style CV Kalman (4-D state: [x, y, vx, vy])
# ---------------------------------------------------------------------

class CVKalman:
    """Constant-velocity Kalman with AB3DMOT's published Q/R/P_0 values.

    Hack #3 — score persistence: AB3DMOT's `output()` emits the FIRST
    detection's score throughout the track lifetime (never updated on
    match). This is the dominant driver of the published 0.41 max_recall
    in V2V4Real Table 4. We mirror that here via `self.score_initial`.
    """

    def __init__(self, x: float, y: float, time: float, score: float,
                 width: float, length: float, yaw: float, label: str,
                 track_id: int):
        self.id = track_id
        # Hack #3: persist the first-detection score for the whole track.
        # Do NOT update on match (see update()).
        self.score_initial = float(score)
        self.score = float(score)            # exposed as constant for clarity
        self.width = float(width)
        self.length = float(length)
        self.yaw = float(yaw)
        self.label = label
        self.last_time = float(time)

        # CV state
        self.mu = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.diag([10.0, 10.0, 1000.0, 1000.0])
        self.Q = np.diag([1.0, 1.0, 0.01, 0.01])
        self.R = np.diag([1.0, 1.0])
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

        # Lifecycle counters
        self.hits = 1
        self.misses_since_match = 0

    def _F(self, dt: float) -> np.ndarray:
        return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)

    def predict(self, time: float) -> Tuple[np.ndarray, np.ndarray]:
        dt = max(0.0, time - self.last_time)
        F = self._F(dt)
        mu_pred = F @ self.mu
        P_pred  = F @ self.P @ F.T + self.Q * dt
        return mu_pred, P_pred

    def predict_inplace(self, time: float):
        mu_pred, P_pred = self.predict(time)
        self.mu, self.P = mu_pred, P_pred
        self.last_time = float(time)
        # NOTE: hits/misses updated by caller after match

    def update(self, x: float, y: float, score: float,
               width: float, length: float, yaw: float, time: float):
        # Vanilla Kalman update
        z = np.array([x, y], float)
        innovation = z - self.H @ self.mu
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ innovation
        self.P  = (np.eye(4) - K @ self.H) @ self.P
        # Hack #3: score is intentionally NOT updated. trk.info[-1] in
        # AB3DMOT's output() carries the first-detection score forever.
        # self.score remains == self.score_initial.
        self.width = float(width)
        self.length = float(length)
        self.yaw = float(yaw)
        self.hits += 1
        self.misses_since_match = 0

    @property
    def x(self): return float(self.mu[0])
    @property
    def y(self): return float(self.mu[1])
    @property
    def bbox(self):
        return [self.x, self.y, self.width, self.length, self.yaw]


# ---------------------------------------------------------------------
# Per-frame helpers
# ---------------------------------------------------------------------

def _bev_nms(dets: List[Dict], iou_thresh: float) -> List[Dict]:
    """BEV-NMS over the union of detections from all CAVs."""
    if iou_thresh >= 1.0 or len(dets) <= 1:
        return dets
    ordered = sorted(dets, key=lambda d: -d["score"])
    kept: List[Dict] = []
    for d in ordered:
        d_box = [d["x"], d["y"], d["width"], d["length"], d["yaw"]]
        suppress = False
        for k in kept:
            k_box = [k["x"], k["y"], k["width"], k["length"], k["yaw"]]
            if cmr_utils.rotated_box_iou(d_box, k_box) >= iou_thresh:
                suppress = True
                break
        if not suppress:
            kept.append(d)
    return kept


def _hungarian_match(tracks: List[CVKalman], dets: List[Dict],
                      iou_gate: float) -> Tuple[List[Tuple[int, int]],
                                                 List[int], List[int]]:
    """Hungarian assignment with cost = (1 - IoU). IoU below gate → no match.

    Returns (matches, unmatched_track_idxs, unmatched_det_idxs).
    """
    n_t, n_d = len(tracks), len(dets)
    if n_t == 0:
        return [], [], list(range(n_d))
    if n_d == 0:
        return [], list(range(n_t)), []

    BIG = 1e9
    cost = np.full((n_t, n_d), BIG)
    iou_matrix = np.zeros((n_t, n_d))
    for i, t in enumerate(tracks):
        t_box = t.bbox
        for j, d in enumerate(dets):
            d_box = [d["x"], d["y"], d["width"], d["length"], d["yaw"]]
            iou = cmr_utils.rotated_box_iou(t_box, d_box)
            iou_matrix[i, j] = iou
            if iou >= iou_gate:
                cost[i, j] = 1.0 - iou

    row_ind, col_ind = linear_sum_assignment(cost)
    matches: List[Tuple[int, int]] = []
    matched_t = set()
    matched_d = set()
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < BIG:
            matches.append((int(i), int(j)))
            matched_t.add(int(i))
            matched_d.add(int(j))
    um_t = [i for i in range(n_t) if i not in matched_t]
    um_d = [j for j in range(n_d) if j not in matched_d]
    return matches, um_t, um_d


# ---------------------------------------------------------------------
# Per-scenario replay
# ---------------------------------------------------------------------

# Hack #1: V2V4Real's tracking benchmark evaluates car ONLY (DMSTrack
# evaluate_v2v4real branch, class_list=['car']). Using the 4-class union
# inflates GT denominator and depresses AMOTA.
VEHICLE_CLASSES_CAR_ONLY = {"car"}
VEHICLE_CLASSES_FOUR     = {"car", "truck", "bus", "construction_vehicle"}


def replay_scenario(scenario_dir: Path,
                    score_threshold: float = 0.20,
                    nms_iou: float = 0.15,
                    match_iou_gate: float = 0.01,
                    min_hits: int = 3,
                    max_age: int = 2,
                    lidar_range: Optional[List[float]] = None,
                    gt_range: Optional[List[float]] = None,
                    car_only: bool = True,
                    eval_mode: str = "center_dist",   # "center_dist" or "iou"
                    eval_iou_threshold: float = 0.25,
                    min_gt_lifetime_frames: int = 1,
                    max_eval_range: Optional[float] = None,
                    ignore_unmatched_fps: bool = False) -> Dict:
    """Run the AB3DMOT-style pipeline on one scenario; return AMOTA metrics."""
    scen = load_scenario(scenario_dir)

    tracks: List[CVKalman] = []
    next_id = 0
    paper_match_tape_merged: List[Dict] = []

    for fid in scen.frame_ids:
        frame = scen.frames[fid]
        timestamp = fid * 0.1

        # 1+2: collect detections from all CAVs, filter by class + score + range
        cls_filter = VEHICLE_CLASSES_CAR_ONLY if car_only else VEHICLE_CLASSES_FOUR
        all_dets: List[Dict] = []
        for vid, ego in frame["vehicles"].items():
            for d in frame["detections"].get(vid, []):
                if d["label"] not in cls_filter:
                    continue
                if d["score"] < score_threshold:
                    continue
                if lidar_range is not None and not _in_ego_range(
                    d["x"], d["y"], d.get("z", 0.0),
                    ego["x"], ego["y"], ego["yaw"], lidar_range,
                ):
                    continue
                all_dets.append(d)

        # 3: BEV-NMS over the union
        all_dets = _bev_nms(all_dets, nms_iou)

        # 4: predict all live tracks forward
        for t in tracks:
            t.predict_inplace(timestamp)
            t.misses_since_match += 1   # incremented; cleared on match below

        # 5: Hungarian match
        matches, um_t, um_d = _hungarian_match(tracks, all_dets, match_iou_gate)

        # 6: update matched tracks
        for t_idx, d_idx in matches:
            d = all_dets[d_idx]
            tracks[t_idx].update(d["x"], d["y"], d["score"],
                                 d["width"], d["length"], d["yaw"], timestamp)

        # 7: spawn unmatched detections as new tracks
        for d_idx in um_d:
            d = all_dets[d_idx]
            tracks.append(CVKalman(
                d["x"], d["y"], timestamp, d["score"],
                d["width"], d["length"], d["yaw"], d["label"],
                track_id=next_id,
            ))
            next_id += 1

        # 8: kill tracks with too many misses
        tracks = [t for t in tracks if t.misses_since_match < max_age]

        # 9: emit confirmed tracks (hits >= min_hits) to the merged tape.
        # Hack #3: t.score == t.score_initial — first-detection score
        # persisted (NOT updated on match). Same value across the whole
        # track lifetime. This produces the sparse-high-precision tape
        # signature with max_recall ≈ 0.41 in the published numbers.
        tracks_for_tape: List[Tuple] = []
        for t in tracks:
            if t.hits >= min_hits:
                # Tape entry: (id, x, y, w, l, yaw, score) — full bbox so the
                # KITTI-style IoU evaluator can match. The center-distance
                # evaluator just ignores the bbox fields.
                tracks_for_tape.append(
                    (t.id, t.x, t.y, t.width, t.length, t.yaw, t.score_initial)
                )

        # Build merged GT for this frame: dedup across CAVs by ass_id.
        # Hack #1 GT side: filter to car-only too (DMSTrack class_list=['car']).
        gts_seen: set = set()
        gt_records: List[Tuple[int, float, float]] = []
        for g in frame["gt"]:
            if car_only and g.get("type", g.get("label", "car")) != "car":
                continue
            ass = int(g.get("ass_id", -1))
            if ass < 0:
                gid = -int(g.get("obj_id", 0))
            else:
                gid = ass
            if gid in gts_seen:
                continue
            # Apply gt_range filter (relative to either ego)
            if gt_range is not None:
                in_any = False
                for ego_id, ego in frame["vehicles"].items():
                    if ego_id == g.get("vehicle_id") and _in_ego_range(
                        g["x"], g["y"], g.get("z", 0.0),
                        ego["x"], ego["y"], ego["yaw"], gt_range,
                    ):
                        in_any = True
                        break
                if not in_any:
                    continue
            gts_seen.add(gid)
            # GT tape entry: (id, x, y, w, l, yaw) — same bbox-aware shape
            # so IoU eval can use it; center-dist eval ignores extra fields.
            gt_records.append((gid, g["x"], g["y"],
                               g["width"], g["length"], g["yaw"]))

        paper_match_tape_merged.append({
            "frame_idx": fid,
            "tracks": tracks_for_tape,
            "gts": gt_records,
        })

    # 10: AMOTA via the chosen evaluator. For the range-cap proxy
    # (KITTI min_height v2) we also need ego positions per frame.
    ego_xy_by_fid: Dict = {}
    for fid in scen.frame_ids:
        egos = []
        for vid, ego in scen.frames[fid]["vehicles"].items():
            egos.append((ego["x"], ego["y"]))
        ego_xy_by_fid[fid] = egos

    if eval_mode == "iou":
        metrics = compute_ab3dmot_metrics_iou(
            paper_match_tape_merged,
            iou_threshold=eval_iou_threshold,
            min_gt_lifetime_frames=min_gt_lifetime_frames,
            max_eval_range_m=max_eval_range,
            ego_xy_per_frame=ego_xy_by_fid,
            ignore_unmatched_fps=ignore_unmatched_fps,
        )
    else:
        metrics = compute_ab3dmot_metrics(paper_match_tape_merged)
    return metrics


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", default="/home/rave/test/mmdetection3d/work_dirs/cmr_export")
    ap.add_argument("--splits", default="test")
    ap.add_argument("--score-threshold", type=float, default=0.20)
    ap.add_argument("--nms-iou", type=float, default=0.15,
                    help="BEV-NMS over the union of CAV detections. Set to "
                         "1.0 to disable (V2V4Real may not double-NMS — "
                         "OpenCOOD already NMSes per-CAV at 0.15).")
    ap.add_argument("--match-iou-gate", type=float, default=0.01,
                    help="Hungarian match gate on BEV-IoU. AB3DMOT's V2V4Real "
                         "config uses 3D-GIoU thres=-0.2 which is more "
                         "permissive than IoU; lowering to 0.0 (any positive "
                         "overlap) is the closest BEV approximation.")
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-age", type=int, default=2)
    ap.add_argument("--car-only", action="store_true",
                    help="Hack #1: filter detections AND GT to class==car "
                         "only, matching DMSTrack's `evaluate_v2v4real` "
                         "branch (class_list=['car']). Required to "
                         "reproduce the published 29.28 AMOTA.")
    ap.add_argument("--four-class", action="store_true",
                    help="Force 4-class eval (overrides --car-only).")
    ap.add_argument("--eval-mode", choices=["center_dist", "iou"],
                    default="center_dist",
                    help="AMOTA match gate. center_dist (default) = 2m "
                         "center-distance (AB3DMOT-nuScenes). iou = BEV-IoU "
                         "≥ --eval-iou-threshold (KITTI-V2V4Real style).")
    ap.add_argument("--eval-iou-threshold", type=float, default=0.25,
                    help="IoU gate when --eval-mode=iou. Default 0.25 = "
                         "DMSTrack's V2V4Real protocol.")
    ap.add_argument("--min-gt-lifetime-frames", type=int, default=1,
                    help="KITTI-min_height proxy v1: drop GTs appearing "
                         "in fewer than N frames. Default 1 (no filter).")
    ap.add_argument("--max-eval-range", type=float, default=None,
                    help="KITTI-min_height proxy v2: drop tracks AND GTs "
                         "beyond N metres of any ego at eval time. ~43m "
                         "approximates DMSTrack's evaluate_v2v4real range "
                         "cut. Default off.")
    ap.add_argument("--ignore-unmatched-fps", action="store_true",
                    help="DMSTrack/V2V4Real evaluator quirk: unmatched "
                         "tracker entries with 2D bbox height < min_height "
                         "are IGNORED rather than counted as FPs. Since "
                         "V2V4Real labels have 2D bbox=0,0,0,0 (LiDAR-only), "
                         "this flag effectively zeros out the FP count.")
    ap.add_argument("--max-scenarios", type=int, default=None,
                    help="Optional cap (debug); default = all in split.")
    args = ap.parse_args()

    car_only = (not args.four_class) and (args.car_only or True)  # default car-only
    splits = set(args.splits.split(","))
    export = Path(args.export_dir)
    scenarios = sorted([d for d in export.iterdir()
                         if d.is_dir() and d.name.split("__", 1)[0] in splits])
    if args.max_scenarios:
        scenarios = scenarios[:args.max_scenarios]

    print(f"=== AB3DMOT-style replica on {args.splits} split ({len(scenarios)} scenarios) ===")
    print(f"  score_threshold={args.score_threshold}  nms_iou={args.nms_iou}")
    print(f"  match_iou_gate={args.match_iou_gate}  min_hits={args.min_hits}  max_age={args.max_age}")
    print(f"  car_only={car_only}  (Hack #1: V2V4Real evaluates car-only)")
    print(f"  first-score-persistence: ON (Hack #3)")
    print()

    # V2V4Real-exact range gates (rectangular ego-frame)
    LIDAR_RANGE = [-70.4, -40, -5, 70.4, 40, 3]
    GT_RANGE    = [-100, -40, -5, 100, 40, 3]

    sum_amota = 0.0
    sum_amotp = 0.0
    sum_samota = 0.0
    sum_mota = 0.0
    n = 0
    for scen in scenarios:
        m = replay_scenario(scen,
                            score_threshold=args.score_threshold,
                            nms_iou=args.nms_iou,
                            match_iou_gate=args.match_iou_gate,
                            min_hits=args.min_hits,
                            max_age=args.max_age,
                            lidar_range=LIDAR_RANGE,
                            gt_range=GT_RANGE,
                            car_only=car_only,
                            eval_mode=args.eval_mode,
                            eval_iou_threshold=args.eval_iou_threshold,
                            min_gt_lifetime_frames=args.min_gt_lifetime_frames,
                            max_eval_range=args.max_eval_range,
                            ignore_unmatched_fps=args.ignore_unmatched_fps)
        amota = m["amota"] * 100
        amotp = m["amotp"] * 100
        sa    = m["samota"] * 100
        mota  = m["mota"] * 100
        sum_amota += amota; sum_amotp += amotp; sum_samota += sa; sum_mota += mota
        n += 1
        print(f"  {scen.name[:60]:<60s}  AMOTA={amota:>5.2f}  sAMOTA={sa:>5.2f}  MOTA={mota:>5.2f}", flush=True)

    if n > 0:
        print()
        print(f"=== Mean across {n} scenarios ===")
        print(f"  AMOTA   = {sum_amota/n:>5.2f}")
        print(f"  AMOTP   = {sum_amotp/n:>5.2f}")
        print(f"  sAMOTA  = {sum_samota/n:>5.2f}")
        print(f"  MOTA    = {sum_mota/n:>5.2f}")
        print()
        print(f"V2V4Real reported Late Fusion AMOTA = 29.28")
        print(f"V2V4Real reported sAMOTA            = 71.05")


if __name__ == "__main__":
    main()
