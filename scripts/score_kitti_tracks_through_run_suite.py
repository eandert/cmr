#!/usr/bin/env python3
"""Score precomputed KITTI MOT tracks through run_suite's evaluator.

Bridges the gap: AB3DMOT and DMSTrack tracks are in cav-0 (tesla) LiDAR /
KITTI-camera frame (single-instance against V2V4Real's official cav-0 GT),
but our merged-GT and per-(range, strategy) metrics live in run_suite which
expects tracker output in global UTM frame against our gt_objects.csv.

This script:
  1. Loads KITTI MOT track files (cav-0 frame)
  2. Inverts the AB3DMOT-KITTI → cav-0-LiDAR axis swap
  3. Transforms cav-0 LiDAR → global UTM via tesla's per-frame pose from
     ego_state.csv
  4. Builds the run_suite-style match tape per scenario
  5. Computes ALL 6 (range_mode × merge_strategy) variants of paper_amota,
     v2v4real_mg_amota, and HOTA — same metrics as cobevt_dets_cmr_se_tuned

Usage:
    python scripts/score_kitti_tracks_through_run_suite.py \\
        --tracking-dir /path/to/cobevt_Car_val_H1/data_0 \\
        --export-dir   <your-mmdet3d>/work_dirs/cmr_export_cobevt_se \\
        --output-dir   results/AB3DMOT_FULL_GRID \\
        --label        "CoBEVT+AB3DMOT (full grid)"
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from metrics.v2v4real_metrics import compute_ab3dmot_metrics, compute_ab3dmot_metrics_iou, compute_hota_iou  # noqa: E402
from v2v4real_replay import (  # noqa: E402
    _cluster_gt_rows, _pick_cluster_representative, _row_to_gt_object,
    _in_ego_range,
)


def parse_kitti_track_file(path: Path) -> Dict[int, List[Tuple]]:
    """Read AB3DMOT-style KITTI MOT tracking file.
    Format per line: frame, track_id, type, 0,0,0,0,0,0,0, h, w, l, X, Y, Z, ry, score
    Returns {frame_id: [(track_id, x_kitti, y_kitti, z_kitti, h, w, l, ry, score)]}
    """
    out: Dict[int, List[Tuple]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            try:
                frame = int(float(parts[0]))
                tid   = int(float(parts[1]))
                if parts[2].lower() not in ("car",):
                    continue
                h     = float(parts[10])
                w     = float(parts[11])
                l     = float(parts[12])
                X     = float(parts[13])
                Y     = float(parts[14])
                Z     = float(parts[15])
                ry    = float(parts[16])
                score = float(parts[17]) if len(parts) > 17 else 1.0
            except (ValueError, IndexError):
                continue
            out.setdefault(frame, []).append((tid, X, Y, Z, h, w, l, ry, score))
    return out


def kitti_to_global(track: Tuple, tesla_pose: Tuple[float, float, float, float]) -> Tuple:
    """Transform one KITTI-frame track to global UTM via tesla's pose.
    Returns (track_id, x_g, y_g, w, l, yaw_g, score, h, z_center_g) so 3D-IoU
    matching (which expects h at index 7 and z_center at index 8) works.
    """
    tid, kitti_x, kitti_y_cam, kitti_z_fwd, h, w, l, ry, score = track
    # KITTI cam → mmdet3d LiDAR (cav-0 frame): inverse of the V2V4Real y/z swap.
    # Both V2V4Real and our augment write the KITTI "Y" column as the LiDAR z-center
    # directly (not bottom-of-box). So the reverse is identity, NOT `kitti_Y - h/2`.
    lidar_x = kitti_x
    lidar_y = kitti_z_fwd          # KITTI z (forward) → LiDAR y
    lidar_z = kitti_y_cam          # KITTI Y already stores LiDAR z_center (ego-relative)
    # cav-0 LiDAR → global UTM via tesla pose
    cx, cy, cz, cyaw = tesla_pose
    cos_y, sin_y = math.cos(cyaw), math.sin(cyaw)
    x_g = cx + cos_y * lidar_x - sin_y * lidar_y
    y_g = cy + sin_y * lidar_x + cos_y * lidar_y
    yaw_g = ry + cyaw
    z_center_g = cz + lidar_z
    return (tid, x_g, y_g, w, l, yaw_g, score, h, z_center_g)


def load_tesla_poses(scenario_dir: Path) -> Dict[int, Tuple[float, float, float, float]]:
    poses: Dict[int, Tuple] = {}
    with open(scenario_dir / "ego_state.csv") as f:
        for r in csv.DictReader(f):
            if r["vehicle_id"] != "tesla":
                continue
            poses[int(r["frame_id"])] = (
                float(r["x"]), float(r["y"]), float(r["z"]), float(r["yaw"])
            )
    return poses


def load_per_ego_gt(scenario_dir: Path) -> Dict[int, List[Dict]]:
    """Per-frame list of GT row dicts (both vehicles' annotations)."""
    out: Dict[int, List[Dict]] = {}
    with open(scenario_dir / "gt_objects.csv") as f:
        for r in csv.DictReader(f):
            fid = int(r["frame_id"])
            row = {
                "vehicle_id": r["vehicle_id"],
                "obj_id":     r["obj_id"],
                "ass_id":     int(float(r["ass_id"])) if r["ass_id"] else -1,
                "x":          float(r["x"]),
                "y":          float(r["y"]),
                "z":          float(r["z"]),
                "yaw":        float(r["yaw"]),
                "width":      float(r["width"]),
                "length":     float(r["length"]),
                "height":     float(r["height"]),
                "label":      r.get("label", "car"),
            }
            out.setdefault(fid, []).append(row)
    return out


GT_RANGE     = [-100.0, -40.0, -5.0, 100.0, 40.0, 5.0]
WORLD_RANGE  = [-100.0, -40.0, -5.0, 100.0, 40.0, 5.0]
RANGE_MODES  = ("per_ego", "world")
MERGE_STRATEGIES = ("average",)  # NMS-cluster then average — the single locked strategy


def load_ego_as_gt_rows(scenario_dir: Path) -> Dict[int, List[Dict]]:
    """Build GT rows from the egos' OWN localization (ego_state.csv).

    V2V4Real's released gt_objects.csv contains other-vehicle annotations only;
    the ego CAVs themselves are not annotated. For 'Ours' protocol we add the
    egos as additional GT entries using their own pose so tracker outputs that
    pick up the ego (LF self-reports etc.) get credit / are correctly scored.

    Returns: dict frame_id -> list of GT row dicts (same schema as load_per_ego_gt).
    """
    out: Dict[int, List[Dict]] = {}
    with open(scenario_dir / "ego_state.csv") as f:
        for r in csv.DictReader(f):
            fid = int(r["frame_id"])
            veh = r["vehicle_id"]
            # Use a stable per-ego obj_id (negative to avoid colliding with V2V4Real ids)
            ego_obj_id = -(abs(hash(veh)) & 0xFFFF) - 1
            row = {
                "vehicle_id": veh,
                "obj_id":     str(ego_obj_id),
                "ass_id":     ego_obj_id,
                "x":          float(r["x"]),
                "y":          float(r["y"]),
                "z":          float(r["z"]),
                "yaw":        float(r["yaw"]),
                "width":      float(r.get("width", 2.0)),
                "length":     float(r.get("length", 4.5)),
                "height":     float(r.get("height", 1.7)),
                "label":      "car",
            }
            out.setdefault(fid, []).append(row)
    return out


def filter_gt_per_ego(gt_rows: List[Dict], frame_egos: Dict[str, Tuple]) -> List[Dict]:
    """Keep g where g.vehicle_id == ego AND within that ego's GT_RANGE."""
    out = []
    for g in gt_rows:
        ego = frame_egos.get(g["vehicle_id"])
        if ego is None:
            continue
        if _in_ego_range(g["x"], g["y"], g.get("z", 0.0),
                         ego[0], ego[1], ego[3], GT_RANGE):
            out.append(g)
    return out


def filter_gt_world(gt_rows: List[Dict], frame_egos: Dict[str, Tuple]) -> List[Dict]:
    """Keep g if within union of any ego's WORLD_RANGE."""
    egos = list(frame_egos.values())
    return [g for g in gt_rows
            if any(_in_ego_range(g["x"], g["y"], g.get("z", 0.0),
                                 e[0], e[1], e[3], WORLD_RANGE)
                   for e in egos)]


def _build_tape_from_pool(gt_pool_per_frame, kitti_tracks, tesla_poses, frame_egos,
                           range_filter_fn):
    """Build a metric-eval tape: NMS-cluster GT, average cluster reps, pair with
    transformed tracks. range_filter_fn is filter_gt_per_ego or filter_gt_world.

    Tracks carry (id, x, y, w, l, yaw, score, h, z_center) so 3D-IoU matching is
    supported. GT carry (id, x, y, w, l, yaw, h, z_center). DMSTrack publishes
    using 3D-IoU (AB3DMOT/scripts/KITTI/evaluate.py:587), so we follow.
    """
    tape = []
    for fid, gt_rows in gt_pool_per_frame.items():
        if fid not in tesla_poses:
            continue
        ego_dict = frame_egos.get(fid, {})
        tracks_g = [kitti_to_global(t, tesla_poses[fid])
                    for t in kitti_tracks.get(fid, [])]
        # Augment tracks with (h, z_center) for 3D-IoU matching.
        # parse_kitti_track_file emits (tid, x_cam, y_cam, z_cam, w, l, h, yaw, score) — see below.
        pool = range_filter_fn(gt_rows, ego_dict)
        clusters = _cluster_gt_rows(pool)
        gt_objs = []
        for cluster in clusters:
            rep = _pick_cluster_representative(cluster, "average")
            anchor = cluster[0]
            if anchor["ass_id"] >= 0:
                track_id = int(anchor["ass_id"])
            else:
                track_id = hash((anchor["vehicle_id"], int(anchor["obj_id"]))) & 0x7FFFFFFF
            gt_objs.append(_row_to_gt_object(rep, track_id))
        # GT tuple: (id, x, y, w, l, yaw, h, z_center).
        # gt_objects.csv z is mmdet3d-LiDAR z_bottom (mmdet3d box convention puts
        # the box origin at the BOTTOM center); convert to z_center by adding h/2.
        gt_records = []
        for g, cluster in zip(gt_objs, clusters):
            rep = _pick_cluster_representative(cluster, "average")
            h_rep = float(rep.get("height", 1.7))
            z_bottom = float(rep.get("z", 0.0))
            z_center = z_bottom + h_rep / 2.0
            gt_records.append((g.vehicle_id, g.centroid[0], g.centroid[1],
                               g.dimensions[0], g.dimensions[1], g.angle,
                               h_rep, z_center))
        tape.append({"frame_idx": fid, "tracks": tracks_g, "gts": gt_records})
    return tape


def score_scenario(scenario_dir: Path, kitti_tracks: Dict[int, List[Tuple]],
                   ego_pose_loader=None) -> Dict:
    """Score one scenario emitting the LOCKED metric set:
      V2V-{AMOTA, AMOTP, MOTA}    — V2V4Real protocol (IoU 0.25, FP-ignore) on OG GT
      Ours-{AMOTA, AMOTP, MOTA}  — same IoU 0.25 matching but counts FPs correctly,
                                     on GT augmented with ego CAVs (own-localization)
      Ours-HOTA                  — HOTA on ego-augmented GT
    Each metric is computed as the average between the per_ego and world range modes.
    """
    tesla_poses = load_tesla_poses(scenario_dir)
    all_gt_orig = load_per_ego_gt(scenario_dir)
    ego_gt_rows = load_ego_as_gt_rows(scenario_dir)

    # Augmented GT = orig + ego rows (kept separate; orig is untouched on disk).
    all_gt_with_egos: Dict[int, List[Dict]] = {}
    for fid in set(all_gt_orig.keys()) | set(ego_gt_rows.keys()):
        all_gt_with_egos[fid] = list(all_gt_orig.get(fid, [])) + list(ego_gt_rows.get(fid, []))

    # Load all egos' poses per frame for range filters
    frame_egos: Dict[int, Dict[str, Tuple]] = {}
    with open(scenario_dir / "ego_state.csv") as f:
        for r in csv.DictReader(f):
            fid = int(r["frame_id"])
            frame_egos.setdefault(fid, {})[r["vehicle_id"]] = (
                float(r["x"]), float(r["y"]), float(r["z"]), float(r["yaw"])
            )

    # Build tapes for both protocols:
    #   V2V protocol → OG GT, both range modes
    #   Ours protocol → ego-augmented GT, both range modes
    v2v_tape_per_ego  = _build_tape_from_pool(all_gt_orig,      kitti_tracks, tesla_poses, frame_egos, filter_gt_per_ego)
    v2v_tape_world    = _build_tape_from_pool(all_gt_orig,      kitti_tracks, tesla_poses, frame_egos, filter_gt_world)
    ours_tape_per_ego = _build_tape_from_pool(all_gt_with_egos, kitti_tracks, tesla_poses, frame_egos, filter_gt_per_ego)
    ours_tape_world   = _build_tape_from_pool(all_gt_with_egos, kitti_tracks, tesla_poses, frame_egos, filter_gt_world)

    # All matching is 3D IoU @ 0.25 to match DMSTrack's published protocol
    # (AB3DMOT/scripts/KITTI/evaluate.py:587 uses iou_3d).
    # V2V metric: 3D IoU 0.25, FP-ignore (the V2V4Real/DMSTrack-paper "bug")
    v2v_pe = compute_ab3dmot_metrics_iou(v2v_tape_per_ego, iou_threshold=0.25,
                                          ignore_unmatched_fps=True, match_3d=True)
    v2v_w  = compute_ab3dmot_metrics_iou(v2v_tape_world,   iou_threshold=0.25,
                                          ignore_unmatched_fps=True, match_3d=True)
    # Ours metric: 3D IoU 0.25 matching, FPs counted + ego-augmented GT
    ours_pe = compute_ab3dmot_metrics_iou(ours_tape_per_ego, iou_threshold=0.25,
                                           ignore_unmatched_fps=False, match_3d=True)
    ours_w  = compute_ab3dmot_metrics_iou(ours_tape_world,   iou_threshold=0.25,
                                           ignore_unmatched_fps=False, match_3d=True)
    ours_hota_pe = compute_hota_iou(ours_tape_per_ego, iou_threshold=0.25,
                                     ignore_unmatched_fps=False, match_3d=True)
    ours_hota_w  = compute_hota_iou(ours_tape_world,   iou_threshold=0.25,
                                     ignore_unmatched_fps=False, match_3d=True)

    def avg(a, b): return 0.5 * (float(a) + float(b))

    out = {
        # V2V protocol (FP-ignore, OG GT, NMS-clustered + averaged)
        "V2V_AMOTA": 100.0 * avg(v2v_pe.get("amota", 0.0), v2v_w.get("amota", 0.0)),
        "V2V_AMOTP": 100.0 * avg(v2v_pe.get("amotp", 0.0), v2v_w.get("amotp", 0.0)),
        "V2V_MOTA":  100.0 * avg(v2v_pe.get("mota",  0.0), v2v_w.get("mota",  0.0)),
        # Ours protocol (FPs counted, ego-augmented GT)
        "Ours_AMOTA": 100.0 * avg(ours_pe.get("amota", 0.0), ours_w.get("amota", 0.0)),
        "Ours_AMOTP": 100.0 * avg(ours_pe.get("amotp", 0.0), ours_w.get("amotp", 0.0)),
        "Ours_MOTA":  100.0 * avg(ours_pe.get("mota",  0.0), ours_w.get("mota",  0.0)),
        "Ours_HOTA":  100.0 * avg(ours_hota_pe.get("hota", 0.0), ours_hota_w.get("hota", 0.0)),
        # GT counts for sanity
        "v2v_gt_total":  int(v2v_pe.get("gt_total", 0)  + v2v_w.get("gt_total", 0)),
        "ours_gt_total": int(ours_pe.get("gt_total", 0) + ours_w.get("gt_total", 0)),
    }
    return out


def _score_worker(task):
    """Worker for multiprocessing.Pool — scores one scenario.
    task = (scenario_dir, kitti_tracks_dict, scenario_name, track_file_name) for world-UTM path,
        or ('CAM', track_file_path, kitti_label_path, ego_state_path_or_None, name, tf_name) for cam-frame path.
    """
    if task[0] == 'CAM':
        _, track_path, label_path, ego_state_path, name, tf_name = task
        scores = score_scenario_cam(track_path, label_path, ego_state_path=ego_state_path)
        return tf_name, name, scores
    scenario_dir, kitti_tracks, name, tf_name = task
    scores = score_scenario(scenario_dir, kitti_tracks)
    return tf_name, name, scores


def parse_v2v4real_label_file(path: Path) -> Dict[int, List[Tuple]]:
    """Read v2v4real_val_label/{seq}.txt in KITTI MOT label format.
    Format per line: frame track_id type 0 0 0 bbox2d×4 h w l X Y Z ry
    Returns {frame: [(track_id, X_cam, Y_cam, Z_cam, h, w, l, ry)]}
    """
    out: Dict[int, List[Tuple]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17 or parts[2].lower() != "car":
                continue
            frame = int(float(parts[0]))
            tid   = int(float(parts[1]))
            h, w, l = float(parts[10]), float(parts[11]), float(parts[12])
            X, Y, Z = float(parts[13]), float(parts[14]), float(parts[15])
            ry      = float(parts[16])
            out.setdefault(frame, []).append((tid, X, Y, Z, h, w, l, ry))
    return out


def _ego_pose_at_frame_in_tesla_cam(scenario_dir: Path,
                                     ego_dims: Optional[Dict[str, Tuple[float, float, float]]] = None
                                     ) -> Dict[int, Dict[str, Tuple[float, float, float]]]:
    """For ego-as-GT in cam frame: convert each ego's BODY CENTER into tesla's
    V2V4Real-KITTI cam frame.

    V2V4Real "KITTI" convention (NON-STANDARD — see opencood/tools/inference.py:489):
      X = lidar X (forward),  Y = lidar Z (up),  Z = lidar Y (lateral).

    Y is the lidar-up coordinate of the BOX CENTER, not the LiDAR mount. So we
    apply the +(LIDAR_Z_MOUNT_TO_GROUND + h/2) offset that the augment uses for
    car-on-ground GT centers. Without this, the ego-as-GT entries land at the
    LiDAR-mount altitude (~0 in lidar Z) instead of the car's body center
    (~-1.0 in lidar Z), missing every real track by ~1m vertically and tanking
    Ours-AMOTA.
    """
    LIDAR_Z_MOUNT_TO_GROUND = -1.87
    DEFAULT_DIMS = {"tesla": (4.97, 1.96, 1.44), "astuff": (5.18, 2.03, 1.77)}
    if ego_dims is None:
        ego_dims = DEFAULT_DIMS
    tesla = load_tesla_poses(scenario_dir)
    out: Dict[int, Dict[str, Tuple[float, float, float]]] = {}
    with open(scenario_dir / "ego_state.csv") as f:
        for r in csv.DictReader(f):
            fid = int(r["frame_id"])
            veh = r["vehicle_id"]
            if fid not in tesla:
                continue
            tx, ty, tz, tyaw = tesla[fid]
            wx, wy, wz = float(r["x"]), float(r["y"]), float(r["z"])
            dx, dy, dz = wx - tx, wy - ty, wz - tz
            cos_y, sin_y = math.cos(-tyaw), math.sin(-tyaw)
            lidar_x =  cos_y * dx - sin_y * dy   # forward in tesla LiDAR
            lidar_y =  sin_y * dx + cos_y * dy   # lateral in tesla LiDAR
            lidar_z = dz                          # vertical mount-to-mount
            # Compute the box CENTER y in lidar-z (NOT the LiDAR mount altitude)
            l_d, w_d, h_d = ego_dims.get(veh, (4.5, 2.0, 1.7))
            body_center_lidar_z = lidar_z + LIDAR_Z_MOUNT_TO_GROUND + h_d / 2.0
            X_cam = lidar_x          # forward
            Y_cam = body_center_lidar_z  # box center vertical, in lidar-z (V2V4Real "Y")
            Z_cam = lidar_y          # lateral
            out.setdefault(fid, {})[veh] = (X_cam, Y_cam, Z_cam)
    return out


def score_scenario_cam(track_path: Path, label_path: Path,
                       ego_state_path: Optional[Path] = None) -> Dict:
    """Score one scenario by matching DIRECTLY in tesla's KITTI cam frame.

    Tracks: parsed from AB3DMOT KITTI MOT output (already in tesla cam frame).
    GT: parsed from v2v4real_val_label/{seq}.txt (the same file AB3DMOT eval uses).
    Optionally augment GT with ego CAVs from ego_state.csv → tesla cam frame.

    Both metrics computed:
      V2V-AMOTA = IoU 0.25, FP-ignore, OG GT.
      Ours-AMOTA = IoU 0.25, FPs counted, ego-augmented GT.
    Each computed with 3D IoU matching to match DMSTrack's eval at iou_3d.
    """
    tracks_per_frame = parse_kitti_track_file(track_path)
    gt_per_frame = parse_v2v4real_label_file(label_path)
    egos_in_cam = _ego_pose_at_frame_in_tesla_cam(Path(ego_state_path).parent) \
        if ego_state_path is not None else {}

    # Build tracks tuple: (id, X_cam, Z_cam, w, l, ry, score, h, Y_cam)
    # NOTE: cam BEV is X-Z plane; Y is vertical (down). 3D IoU box: (cx,cy,w,l,yaw,z,h)
    #   cx = X_cam, cy = Z_cam, z_center = Y_cam (cam-down sign — both track and GT use same sign so IoU works).
    # Vehicle dims for self-report GT (fallback if not in ego_state.csv).
    DEFAULT_DIMS = {"tesla": (4.97, 1.96, 1.44), "astuff": (5.18, 2.03, 1.77)}
    EGO_OBJ_ID_BASE = -1000

    v2v_tape = []
    ours_tape = []
    for fid, gt_list in gt_per_frame.items():
        tk_raw = tracks_per_frame.get(fid, [])
        tk_tuples = []
        for (tid, X, Y, Z, h, w, l, ry, score) in tk_raw:
            tk_tuples.append((tid, X, Z, w, l, ry, score, h, Y))
        # OG GT in cam frame
        og_gt_tuples = []
        for (gid, X, Y, Z, h, w, l, ry) in gt_list:
            og_gt_tuples.append((gid, X, Z, w, l, ry, h, Y))
        # Ego-augmented GT
        ours_gt_tuples = list(og_gt_tuples)
        for ego_name, (X, Y, Z) in egos_in_cam.get(fid, {}).items():
            l_d, w_d, h_d = DEFAULT_DIMS.get(ego_name, (4.5, 2.0, 1.7))
            ego_id = EGO_OBJ_ID_BASE - (abs(hash(ego_name)) & 0xFFFF)
            ours_gt_tuples.append((ego_id, X, Z, w_d, l_d, 0.0, h_d, Y))
        v2v_tape.append({"frame_idx": fid, "tracks": tk_tuples, "gts": og_gt_tuples})
        ours_tape.append({"frame_idx": fid, "tracks": tk_tuples, "gts": ours_gt_tuples})

    # Run metrics — 3D IoU 0.25. Three stages to isolate each protocol change:
    #   V2V        = FP-ignore + OG GT      (DMSTrack-paper reproduction)
    #   OG+FP      = FP-counted + OG GT     (isolates the FP-counting penalty)
    #   Ours       = FP-counted + ego-augmented GT (adds ego-as-GT drop)
    v2v = compute_ab3dmot_metrics_iou(v2v_tape, iou_threshold=0.25,
                                       ignore_unmatched_fps=True, match_3d=True)
    ogfp = compute_ab3dmot_metrics_iou(v2v_tape, iou_threshold=0.25,
                                        ignore_unmatched_fps=False, match_3d=True)
    ours = compute_ab3dmot_metrics_iou(ours_tape, iou_threshold=0.25,
                                        ignore_unmatched_fps=False, match_3d=True)
    # HOTA on the same three GT scopes (each uses FP-counted, since HOTA doesn't have FP-ignore)
    v2v_hota = compute_hota_iou(v2v_tape, iou_threshold=0.25,
                                 ignore_unmatched_fps=False, match_3d=True)
    ours_hota = compute_hota_iou(ours_tape, iou_threshold=0.25,
                                  ignore_unmatched_fps=False, match_3d=True)
    return {
        # V2V: FP-ignore + OG GT (DMSTrack-paper reproduction)
        "V2V_AMOTA": 100.0 * float(v2v.get("amota", 0.0)),
        "V2V_AMOTP": 100.0 * float(v2v.get("amotp", 0.0)),
        "V2V_MOTA":  100.0 * float(v2v.get("mota",  0.0)),
        # OG+FP: FP-counted + OG GT (isolates FP-counting impact)
        "OGFP_AMOTA": 100.0 * float(ogfp.get("amota", 0.0)),
        "OGFP_AMOTP": 100.0 * float(ogfp.get("amotp", 0.0)),
        "OGFP_MOTA":  100.0 * float(ogfp.get("mota",  0.0)),
        "OGFP_HOTA":  100.0 * float(v2v_hota.get("hota", 0.0)),
        # Ours: FP-counted + ego-augmented GT
        "Ours_AMOTA": 100.0 * float(ours.get("amota", 0.0)),
        "Ours_AMOTP": 100.0 * float(ours.get("amotp", 0.0)),
        "Ours_MOTA":  100.0 * float(ours.get("mota",  0.0)),
        "Ours_HOTA":  100.0 * float(ours_hota.get("hota", 0.0)),
        "v2v_gt_total":  int(v2v.get("gt_total", 0)),
        "ours_gt_total": int(ours.get("gt_total", 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracking-dir", type=Path, required=True,
                    help="Dir of KITTI MOT track files (0000.txt … 0008.txt)")
    ap.add_argument("--export-dir", type=Path,
                    default=_LOCAL_PATHS.get("mmdet3d_root") / "work_dirs/cmr_export_cobevt_se",
                    help="CMR export with ego_state.csv + gt_objects.csv per scenario "
                         "(any single-ego cobevt export works since GT is from per-vehicle annotations)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--label", default="kitti tracks (full grid)")
    ap.add_argument("--serial", action="store_true",
                    help="Disable parallel scenario scoring (for validating parallel-safety).")
    ap.add_argument("--cam-frame", action="store_true",
                    help="Match in tesla's KITTI cam frame using v2v4real_val_label/ GT instead of world UTM. "
                         "Apples-to-apples with AB3DMOT's evaluate.py (DMSTrack-paper protocol).")
    ap.add_argument("--label-dir", type=Path,
                    default=REPO / "data" / "v2v4real_inputs" / "baselines" / "paper_gt" / "kitti_labels",
                    help="Directory containing {seq}.txt KITTI MOT label files (used by --cam-frame mode).")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Map sorted KITTI files to sorted test__* scenario dirs
    track_files = sorted(args.tracking_dir.glob("*.txt"))
    test_scenarios = sorted(d for d in args.export_dir.iterdir()
                            if d.is_dir() and d.name.startswith("test__"))
    if len(track_files) != len(test_scenarios):
        raise SystemExit(f"Mismatch: {len(track_files)} track files vs "
                         f"{len(test_scenarios)} scenarios")

    # Score each scenario in parallel. Each score_scenario() is a pure function
    # (takes scenario_dir + per-frame tracks, returns a metrics dict), no shared
    # mutable state. Final summary.json is written ONCE in this main process,
    # after aggregation — no per-scenario file writes. So parallel-safe unlike
    # AB3DMOT's scripts/KITTI/evaluate.py (which has shared-dir race conditions).
    import multiprocessing as _mp
    n_workers = min(8, len(track_files)) if not getattr(args, 'serial', False) else 1
    tasks = []
    if getattr(args, 'cam_frame', False):
        # Cam-frame mode: GT comes from v2v4real_val_label/{seq}.txt
        for tf, sd in zip(track_files, test_scenarios):
            seq_num = tf.stem  # e.g. "0008"
            label_path = args.label_dir / f"{seq_num}.txt"
            if not label_path.exists():
                raise SystemExit(f"--cam-frame requires {label_path}")
            ego_state = sd / "ego_state.csv" if (sd / "ego_state.csv").exists() else None
            tasks.append(('CAM', tf, label_path, ego_state, sd.name, tf.name))
    else:
        for tf, sd in zip(track_files, test_scenarios):
            kitti_tracks = parse_kitti_track_file(tf)
            tasks.append((sd, kitti_tracks, sd.name, tf.name))

    if n_workers > 1:
        print(f"[parallel] scoring {len(tasks)} scenarios with {n_workers} workers", flush=True)
        with _mp.Pool(n_workers) as pool:
            results = pool.map(_score_worker, tasks)
    else:
        results = [_score_worker(t) for t in tasks]

    per_scenario = []
    for tf_name, name, scores in results:
        print(f"  {tf_name} → {name}", flush=True)
        per_scenario.append({"scenario": name, "scores": scores})

    # Aggregate: GT-weighted across scenarios for each of the locked metrics.
    out: Dict = {"label": args.label, "n_scenarios": len(per_scenario)}
    metric_keys = list(per_scenario[0]["scores"].keys()) if per_scenario else []
    # Determine the GT-total weight key for each metric (V2V uses v2v_gt_total, Ours uses ours_gt_total)
    for k in metric_keys:
        vals = [r["scores"].get(k, 0.0) for r in per_scenario]
        if k.endswith("_gt_total"):
            out[k] = int(sum(vals))
            continue
        if k.startswith("V2V_") or k.startswith("OGFP_"):
            wf = "v2v_gt_total"
        elif k.startswith("Ours_"):
            wf = "ours_gt_total"
        else:
            wf = None
        if wf is not None:
            wts = [r["scores"].get(wf, 0) for r in per_scenario]
            tot = sum(wts)
            out[k + "_mean"] = (sum(v*w for v,w in zip(vals, wts)) / tot) if tot > 0 \
                               else (sum(vals)/len(vals) if vals else 0.0)
        else:
            out[k + "_mean"] = sum(vals) / len(vals) if vals else 0.0

    out_json = args.output_dir / "summary.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_json}")
    print()
    print(f"=== {args.label} ===")
    print(f"{'stage':<6}  {'AMOTA':>7} {'AMOTP':>7} {'MOTA':>7} {'HOTA':>7}")
    print("-" * 50)
    print(f"{'V2V':<6}  {out.get('V2V_AMOTA_mean',0):>7.2f} {out.get('V2V_AMOTP_mean',0):>7.2f} "
          f"{out.get('V2V_MOTA_mean',0):>7.2f}    -")
    print(f"{'OG+FP':<6}  {out.get('OGFP_AMOTA_mean',0):>7.2f} {out.get('OGFP_AMOTP_mean',0):>7.2f} "
          f"{out.get('OGFP_MOTA_mean',0):>7.2f} {out.get('OGFP_HOTA_mean',0):>7.2f}")
    print(f"{'Ours':<6}  {out.get('Ours_AMOTA_mean',0):>7.2f} {out.get('Ours_AMOTP_mean',0):>7.2f} "
          f"{out.get('Ours_MOTA_mean',0):>7.2f} {out.get('Ours_HOTA_mean',0):>7.2f}")


if __name__ == "__main__":
    main()
