"""
V2V4Real CSV replay → CMR/GPEM fusion pipeline.

Standalone replacement for `traci_interface.run_simulation()` when the input
data is the V2V4Real CSV export rather than a live SUMO/TraCI simulation.
Drives the same 30-stream gpem_triple_fusion pipeline (5 covariance modes ×
6 filter classes) and returns a metrics dict in the same shape so downstream
aggregation and plotting can reuse existing tooling.

Inputs (per scenario):
  <scenario_dir>/ego_state.csv      per-frame ego pose for each vehicle
  <scenario_dir>/detections.csv     per-frame CenterPoint detections per vehicle
  <scenario_dir>/gt_objects.csv     per-frame GT (one row per vehicle that saw it)
  <scenario_dir>/meta.yaml          scenario metadata

Outputs (returned dict mirrors traci_interface.run_simulation when
gpem_triple_fusion=True):
  triple_global_metrics             dict: 30 stream keys → {amota, amotp, ...}
  triple_global_amota_frames_count  dict: 30 stream keys → frame counts
  triple_global_amotp_frames_count  dict: 30 stream keys → frame counts
  triple_global_hota                dict: 30 stream keys → {hota, deta, assa}
  iterations / recorded_steps       int: total frames replayed
  total_cav_metrics / total_cis_metrics    placeholder dicts (V2V4Real has no
                                            non-CAV traffic — both vehicles are
                                            CAVs, so global metrics carry the
                                            information)
"""
import csv
import math
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# CMR-side imports — these are read-only library calls.
import sensor_fusion
import sensor as cmr_sensor
import ground_truth as cmr_ground_truth
import localizer as cmr_localizer
import utils as cmr_utils
from error import ErrorPackage, ErrorType
from metrics import amota
from metrics.hota import HotaAccumulator
from metrics.v2v4real_metrics import compute_ab3dmot_metrics, compute_ab3dmot_metrics_iou
from filters.covariance_intersection import CovarianceIntersectionFilter
from filters.adaptive_kalman import AdaptiveKalman
from filters.particle_filter import ParticleFilter
from filters.bici_filter import BICIFilter
from filters.sabre_filter import SABREFilter
from filters.ab3dmot_kalman import AB3DMOTKalman
from traci_interface import (
    _detection_with_covariance_from_model,
    _detection_with_flat_covariance,
    _get_detector_error_models,
    _compute_baseline_covariance,
)

# 30 stream keys, in the same order produced by traci_interface.run_simulation.
# Used for everything: fusion-stream init, metrics-dict layout, and the
# experiment_runner method_keys mapping at experiment_runner.py:1125-1130.
COV_MODES = ["baseline", "static", "gpem_linear", "gpem_quadratic", "gpem_polar"]
FILTER_CLASSES: List[Tuple[str, Optional[type]]] = [
    ("kalman", None),                          # default ResizableKalman in Fusion
    ("ci",     CovarianceIntersectionFilter),
    ("akf",    AdaptiveKalman),
    ("pf",     ParticleFilter),
    ("bici",   BICIFilter),
    ("sabre",  SABREFilter),
    ("ab3dmot", AB3DMOTKalman),                # AB3DMOT CV Kalman, fixed Q/R
]
# Names used as keys in the returned metrics dict and inside the per-config
# names produced by experiment_runner. EKF group has no prefix; other groups
# use their filter prefix (matches experiment_runner.py:1125-1130).
def _stream_key(filter_name: str, cov: str) -> str:
    if filter_name == "kalman":
        return cov if cov in ("baseline",) else (
            "static" if cov == "static" else cov  # "gpem_linear", "gpem_quadratic", "gpem_polar"
        )
    return f"{filter_name}_{cov}"


def _active_cov_modes(active_key_set: set) -> set:
    """Cov modes consumed by at least one active stream key.

    run_scenario builds per-detection observations only for these modes — both
    to save work and to avoid loading cov data a detector's CSV may not carry
    (e.g. an older GPEM CSV with no polar columns when 'gpem_polar' isn't in the
    kept streams). Each (filter, cov) maps to a stream key via _stream_key; a cov
    mode is kept iff that key is active.
    """
    return {
        cov for fn, _ in FILTER_CLASSES for cov in COV_MODES
        if _stream_key(fn, cov) in active_key_set
    }


# Same canonical 30 keys (in stable order) as traci_interface.run_simulation.
STREAM_KEYS: List[str] = [
    _stream_key(fname, cov) for fname, _ in FILTER_CLASSES for cov in COV_MODES
]

# Cleanup time (matches config_templates.py:70). Stale tracks > this many
# seconds without an update get dropped.
DEFAULT_CLEANUP_TIME = 0.5

# Stable per-vehicle int IDs (used for fusion's source_participant_id encoding
# and for tracking which CAV produced each measurement; trust scoring is off).
# The id only needs to be a small, stable, collision-free integer per source:
# Fusion encodes `vehicle_id = participant_id * max_id + idx` and recovers the
# source as `vehicle_id // max_id`, so the per-source lifecycle/gate maps key on
# these ids at runtime. V2V4Real (astuff/tesla) and DAIR-V2X-Seq (veh/inf) both
# register here; any other dataset's vehicles fall back to `hash & 0xFF` at the
# call site, which is fine for runs that don't use the per-source maps.
VEHICLE_PARTICIPANT_IDS: Dict[str, int] = {
    "astuff": 1,
    "tesla":  2,
    # DAIR-V2X-Seq (SPD) cooperative pair: vehicle ego + roadside infrastructure.
    "veh":    3,
    "inf":    4,
}


def _create_metrics_dict() -> Dict:
    """Mirrors traci_interface.create_metrics_dict (line 506)."""
    return {
        "total_mse_violations_gt": 0,
        "total_mse_violations_sensor": 0,
        "total_ttc_violations_gt": 0,
        "total_ttc_violations_sensor": 0,
        "true_positives_mse": 0,
        "false_positives_mse": 0,
        "false_negatives_mse": 0,
        "true_positives_ttc": 0,
        "false_positives_ttc": 0,
        "false_negatives_ttc": 0,
        "amota": 0.0,
        "amotp": 0.0,
        "total_mse_violations_gt_av": 0,
        "total_mse_violations_gt_non_av": 0,
        "total_mse_violations_sensor_av": 0,
        "total_mse_violations_sensor_non_av": 0,
        "total_ttc_violations_gt_av": 0,
        "total_ttc_violations_gt_non_av": 0,
        "total_ttc_violations_sensor_av": 0,
        "total_ttc_violations_sensor_non_av": 0,
    }


# --------------------------------------------------------------------------
# CSV loading
# --------------------------------------------------------------------------

class ScenarioData:
    """In-memory snapshot of one V2V4Real scenario."""

    def __init__(self, scenario_dir: Path):
        self.scenario_dir = Path(scenario_dir)
        self.meta: Dict = {}
        # frames[frame_id]: {"vehicles": {vehicle_id: ego_row},
        #                    "detections": {vehicle_id: [det_row, ...]},
        #                    "gt": [gt_row, ...]}
        self.frames: Dict[int, Dict] = {}
        self.frame_ids: List[int] = []

    def load(self):
        # --- meta.yaml ---
        with open(self.scenario_dir / "meta.yaml") as f:
            self.meta = yaml.safe_load(f)

        # --- ego_state.csv ---
        with open(self.scenario_dir / "ego_state.csv") as f:
            for row in csv.DictReader(f):
                fid = int(row["frame_id"])
                self.frames.setdefault(fid, {"vehicles": {}, "detections": {}, "gt": []})
                self.frames[fid]["vehicles"][row["vehicle_id"]] = {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row.get("z", 0.0) or 0.0),
                    "yaw": float(row["yaw"]),
                    "vx": float(row["vx"]),
                    "vy": float(row["vy"]),
                    "length": float(row["length"]),
                    "width": float(row["width"]),
                    "height": float(row["height"]),
                    "timestamp": float(row["timestamp"]),
                }

        # --- detections.csv ---
        with open(self.scenario_dir / "detections.csv") as f:
            for row in csv.DictReader(f):
                fid = int(row["frame_id"])
                if fid not in self.frames:
                    continue
                vid = row["vehicle_id"]
                self.frames[fid]["detections"].setdefault(vid, []).append({
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row.get("z", 0.0) or 0.0),
                    "yaw": float(row["yaw"]),
                    "vx": float(row["vx"]),
                    "vy": float(row["vy"]),
                    "length": float(row["length"]),
                    "width": float(row["width"]),
                    "height": float(row["height"]),
                    "score": float(row["score"]),
                    "label": row["label"],
                    "det_idx": int(row["det_idx"]),
                })

        # --- gt_objects.csv ---
        with open(self.scenario_dir / "gt_objects.csv") as f:
            for row in csv.DictReader(f):
                fid = int(row["frame_id"])
                if fid not in self.frames:
                    continue
                self.frames[fid]["gt"].append({
                    "ass_id": int(row["ass_id"]),
                    "obj_id": int(row["obj_id"]),
                    "vehicle_id": row["vehicle_id"],  # which vehicle observed it
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row.get("z", 0.0) or 0.0),
                    "yaw": float(row["yaw"]),
                    "length": float(row["length"]),
                    "width": float(row["width"]),
                    "height": float(row["height"]),
                    "label": row["label"],
                })

        self.frame_ids = sorted(self.frames.keys())


def load_scenario(scenario_dir) -> ScenarioData:
    s = ScenarioData(scenario_dir)
    s.load()
    return s


# --------------------------------------------------------------------------
# Conversion: CSV row → CMR objects
# --------------------------------------------------------------------------

def _in_ego_range(
    px: float, py: float, pz: float,
    ego_x: float, ego_y: float, ego_yaw: float,
    rng: List[float],  # [x_min, y_min, z_min, x_max, y_max, z_max]
) -> bool:
    """Test if a point is inside the rectangular range `rng` expressed in
    ego-frame coordinates (x = forward along heading, y = lateral).

    V2V4Real / OpenCOOD use this exact gate for both detector training
    (lidar_range) and eval (GT_RANGE). Note pz is checked even when our
    pipeline doesn't *consume* z — V2V4Real drops z-out-of-range entries
    from eval GT, so we match that for honest counts.
    """
    dx = px - ego_x
    dy = py - ego_y
    c, s = math.cos(-ego_yaw), math.sin(-ego_yaw)
    fwd = c * dx - s * dy   # forward along ego heading
    lat = s * dx + c * dy   # lateral perpendicular to heading
    return (rng[0] <= fwd <= rng[3] and
            rng[1] <= lat <= rng[4] and
            rng[2] <= pz  <= rng[5])


def _make_raw_detected_object(det_row: Dict) -> cmr_sensor.DetectedObject:
    """Build a raw DetectedObject from a detections.csv row, with no covariance
    (the per-cov-mode helpers below fill that in)."""
    return cmr_sensor.DetectedObject(
        vehicle_id=None,                  # set by Fusion.processDetectionFrame
        vehicle_type=det_row["label"],
        detected_bbox=None,
        centroid=[det_row["x"], det_row["y"]],
        width=det_row["width"],
        length=det_row["length"],
        angle=det_row["yaw"],
        expected_error_gaussian=None,
        velocity_vector=[det_row["vx"], det_row["vy"]],
        error_covariance=None,
        width_std=0.5,
        length_std=0.5,
        z=det_row.get("z"),
        height=det_row.get("height"),
    )


def _per_ego_nms(det_rows: List[Dict], iou_threshold: float) -> List[Dict]:
    """BEV-NMS within a single CAV's detections. Anchor-based detectors
    (PointPillar) emit multiple boxes per car (one per anchor that fires);
    upstream V2V4Real export does NOT apply NMS, so we get those raw
    duplicates. NMS keeps the highest-score box per overlap cluster.

    Cross-CAV NMS is intentionally NOT applied — the cross-CAV detection
    diversity is the cooperation signal our fusion engine consumes; killing
    it would erase the very thing we're trying to characterize.

    iou_threshold ≥ 1.0 disables NMS (no boxes survive the IoU check, so
    every box is kept). Default 0.15 matches V2V4Real's reference Late
    Fusion config.
    """
    if iou_threshold >= 1.0 or len(det_rows) <= 1:
        return det_rows
    # Sort by descending score so the highest-confidence box wins.
    ordered = sorted(det_rows, key=lambda d: -d["score"])
    kept: List[Dict] = []
    for d in ordered:
        d_box = [d["x"], d["y"], d["width"], d["length"], d["yaw"]]
        suppressed = False
        for k in kept:
            k_box = [k["x"], k["y"], k["width"], k["length"], k["yaw"]]
            if cmr_utils.rotated_box_iou(d_box, k_box) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(d)
    return kept


def _cluster_gt_rows(
    gt_rows: List[Dict],
    ego_poses: Optional[List[Tuple[float, float]]] = None,
    spatial_gate_m: float = 2.0,
    ego_exclusion_gate_m: float = 0.0,
) -> List[List[Dict]]:
    """Group per-ego GT rows into spatial clusters (2m gate) of correlated annotations.

    Returns a list of clusters; each cluster is a list of dict rows that are
    within ``spatial_gate_m`` of one another. Each cluster represents one
    physical object that may be annotated multiple times (once per vehicle).
    Cluster representative for downstream metrics is chosen by the caller via
    `_pick_cluster_representative` (cav0 / average / sort).
    """
    if not gt_rows:
        return []

    rows = list(gt_rows)

    # 1. Optional ego exclusion (drop "the other ego is in my GT" rows).
    if ego_poses:
        rows = [r for r in rows
                if not any((r["x"] - ex)**2 + (r["y"] - ey)**2 < ego_exclusion_gate_m**2
                           for ex, ey in ego_poses)]

    # 2. Greedy spatial clustering. Sort first so the cluster "anchor" is
    #    deterministic (positive ass_id wins, then alphabetical by vehicle_id,
    #    then obj_id). The anchor's position seeds the cluster; every row
    #    within `spatial_gate_m` of the anchor joins.
    rows_sorted = sorted(rows, key=lambda r: (r["ass_id"] < 0, r["ass_id"],
                                              r["vehicle_id"], int(r["obj_id"])))
    clusters: List[List[Dict]] = []
    anchor_xy: List[Tuple[float, float]] = []
    for r in rows_sorted:
        joined = False
        for i, (ax, ay) in enumerate(anchor_xy):
            if (r["x"] - ax)**2 + (r["y"] - ay)**2 < spatial_gate_m**2:
                clusters[i].append(r)
                joined = True
                break
        if not joined:
            clusters.append([r])
            anchor_xy.append((r["x"], r["y"]))
    return clusters


def _circular_mean(angles: List[float]) -> float:
    """Mean of angles handling ±π wraparound."""
    if not angles:
        return 0.0
    sin_sum = sum(math.sin(a) for a in angles)
    cos_sum = sum(math.cos(a) for a in angles)
    return math.atan2(sin_sum / len(angles), cos_sum / len(angles))


def _pick_cluster_representative(cluster: List[Dict], strategy: str) -> Dict:
    """Return a single dict representing the cluster, per the chosen strategy.

    strategy:
      "sort"    — return cluster[0] (current default; alphabetical cav order
                  → astuff wins, since cluster is already sorted by anchor).
      "cav0"    — return the row from tesla (V2V4Real default cav-0). Falls
                  back to cluster[0] if no tesla row in the cluster.
      "average" — return a synthetic row with position/yaw/dimensions averaged
                  across all cluster members. Reduces per-vehicle calibration
                  noise; aligns with the "true" object center.
    """
    if not cluster:
        raise ValueError("empty cluster")

    if strategy == "sort":
        return dict(cluster[0])

    if strategy == "cav0":
        # V2V4Real default: tesla is cav-0 (without --swap_ego)
        for r in cluster:
            if r["vehicle_id"] == "tesla":
                return dict(r)
        return dict(cluster[0])

    if strategy == "average":
        n = len(cluster)
        out = dict(cluster[0])  # copy ass_id, label, vehicle_id, obj_id from anchor
        out["x"]      = sum(r["x"]      for r in cluster) / n
        out["y"]      = sum(r["y"]      for r in cluster) / n
        out["z"]      = sum(r["z"]      for r in cluster) / n
        out["yaw"]    = _circular_mean([r["yaw"] for r in cluster])
        out["width"]  = sum(r["width"]  for r in cluster) / n
        out["length"] = sum(r["length"] for r in cluster) / n
        out["height"] = sum(r["height"] for r in cluster) / n
        return out

    raise ValueError(f"unknown merge strategy: {strategy}")


def _row_to_gt_object(r: Dict, track_id: int) -> cmr_ground_truth.GroundTruthObject:
    """Convert a (possibly-merged) GT row dict to a GroundTruthObject."""
    cx, cy, ang = r["x"], r["y"], r["yaw"]
    hl = r["length"] / 2.0
    hw = r["width"] / 2.0
    c, s = math.cos(ang), math.sin(ang)
    corners = [
        (cx + (-hl) * c - (-hw) * s, cy + (-hl) * s + (-hw) * c),
        (cx + ( hl) * c - (-hw) * s, cy + ( hl) * s + (-hw) * c),
        (cx + ( hl) * c - ( hw) * s, cy + ( hl) * s + ( hw) * c),
        (cx + (-hl) * c - ( hw) * s, cy + (-hl) * s + ( hw) * c),
    ]
    return cmr_ground_truth.GroundTruthObject(
        vehicle_id=track_id,
        vehicle_type=r["label"],
        position=[cx, cy],
        velocity_vector=[0.0, 0.0],
        bounding_box=corners,
        angle_rad=ang,
        width=r["width"],
        length=r["length"],
    )


def _dedup_gt_for_frame(
    gt_rows: List[Dict],
    ego_poses: Optional[List[Tuple[float, float]]] = None,
    spatial_gate_m: float = 2.0,
    ego_exclusion_gate_m: float = 0.0,
    merge_strategy: str = "sort",
) -> List[cmr_ground_truth.GroundTruthObject]:
    """
    Build a spatially-clustered + merged GT list for one frame.

    V2V4Real annotation conventions (discovered while debugging):
      1. ~84% of GT rows in a scenario have ``ass_id=-1`` ("not associated
         across vehicles"). Earlier dedup-by-ass_id collapsed all of these to
         a single entry, dropping the majority of legitimate GT. Fix: spatial
         clustering by ``spatial_gate_m`` (2 m) for *every* row, with
         positive ass_ids prioritised when multiple rows fall in one cluster.
      2. Each ego is annotated as a ``car`` GT in the *other* ego's frame.
         V2V4Real treats these as legitimate tracking targets, so we do NOT
         exclude them by default. Pass ``ego_exclusion_gate_m`` > 0 to drop
         them (for "self-only" eval variants).

    merge_strategy controls how each cluster is collapsed to one GT entry:
      - "sort"    (default for backwards-compat): pick the anchor row
                  (alphabetical cav, ass_id-positive preferred). Effectively
                  picks astuff in the typical case.
      - "cav0"    pick tesla's row (V2V4Real default cav-0). This matches
                  V2V4Real's official KITTI single-instance GT convention.
      - "average" average position/yaw/dimensions across cluster members.
                  Most rigorous: reduces per-vehicle calibration noise.
    """
    clusters = _cluster_gt_rows(gt_rows, ego_poses, spatial_gate_m, ego_exclusion_gate_m)
    out: List[cmr_ground_truth.GroundTruthObject] = []
    for cluster in clusters:
        rep = _pick_cluster_representative(cluster, merge_strategy)
        # Stable track_id: ass_id if positive, else hash of (vehicle_id, obj_id)
        # of the anchor (cluster[0]) so HOTA association is consistent across
        # strategies sharing the same clustering.
        anchor = cluster[0]
        if anchor["ass_id"] >= 0:
            track_id = int(anchor["ass_id"])
        else:
            track_id = hash((anchor["vehicle_id"], int(anchor["obj_id"]))) & 0x7FFFFFFF
        out.append(_row_to_gt_object(rep, track_id))
    return out


def _per_ego_gt_for_frame(gt_rows: List[Dict]) -> Dict[str, List[Tuple]]:
    """
    Build V2V4Real's per-ego GT view for one frame: no spatial dedup, no ego
    exclusion. Returns dict[ego_id, [(gt_id, x, y), ...]].

    Per-ego GT IDs are made unique by namespacing on ego_id so cross-ego IDS
    counting doesn't bleed (an IDS for "the other ego" tracked by tesla is a
    separate event from the same physical event seen by astuff).
    """
    per_ego: Dict[str, List[Tuple]] = {}
    for r in gt_rows:
        ego = r["vehicle_id"]
        # Namespace gt_id by ego so id-switches stay per-ego.
        gid = (ego, int(r["ass_id"]) if r["ass_id"] >= 0 else -int(r["obj_id"]))
        # Hash to a stable int (compute_ab3dmot_metrics expects hashable id).
        gid_int = hash(gid) & 0x7FFFFFFF
        per_ego.setdefault(ego, []).append((gid_int, r["x"], r["y"]))
    return per_ego


# --------------------------------------------------------------------------
# Main replay
# --------------------------------------------------------------------------

def _build_fusion_streams(static_match_cov: Optional[np.ndarray],
                          p_tp_birth_gate: Optional[float] = None,
                          p_tp_birth_gate_by_source: Optional[Dict[int, float]] = None,
                          lifecycle_mode: str = "legacy",
                          per_cov_lifecycle: Optional[Dict[str, str]] = None,
                          ab3dmot_min_hits: int = 3,
                          ab3dmot_max_age: int = 2,
                          enable_inactive_preservation: bool = False,
                          inactive_grace_s: float = 1.0,
                          confirm_log_odds: Optional[float] = None,
                          confirm_log_odds_by_source: Optional[Dict[int, float]] = None,
                          kill_log_odds: Optional[float] = None,
                          log_lr_miss: Optional[float] = None,
                          log_lr_miss_by_source: Optional[Dict[int, float]] = None,
                          mahal_weight: float = 0.6,
                          iou_weight: float = 0.4,
                          mahal_gate: float = 13.82,
                          iou_gate: float = 0.0,
                          tape_score_miss_decay: Optional[float] = None,
                          streams_keep: Optional[List[str]] = None,
                          ) -> Dict[str, sensor_fusion.Fusion]:
    """Create the 30 fusion streams keyed by STREAM_KEYS.

    p_tp_birth_gate (Phase A): if set, every fusion stream applies the same
    P(TP)-based birth gate. None = no gate.
    lifecycle_mode (Phase B): "legacy" (count-based) or "log_odds"
    (IPDA-style S_t-based confirm/kill, with miss-frame decay).

    per_cov_lifecycle (optional): per-cov-mode lifecycle override mapping
        cov_mode → lifecycle_mode. e.g. {"baseline": "ab3dmot"} runs the
        baseline cov stream with count-based AB3DMOT lifecycle (no p_tp
        dependency) while other cov modes use the suite-default. Useful
        because flat-cov baseline can't exploit per-detection p_tp signal,
        so its native lifecycle is count-based."""
    streams: Dict[str, sensor_fusion.Fusion] = {}
    # Pull defaults from sensor_fusion module if caller passed None.
    if confirm_log_odds is None:
        confirm_log_odds = sensor_fusion.DEFAULT_CONFIRM_LOG_ODDS
    if kill_log_odds is None:
        kill_log_odds = sensor_fusion.DEFAULT_KILL_LOG_ODDS
    if log_lr_miss is None:
        log_lr_miss = sensor_fusion.DEFAULT_LOG_LR_MISS
    overrides = per_cov_lifecycle or {}
    keep_set = set(streams_keep) if streams_keep else None
    for filter_name, filter_cls in FILTER_CLASSES:
        for cov in COV_MODES:
            key = _stream_key(filter_name, cov)
            if keep_set is not None and key not in keep_set:
                continue
            stream_lifecycle = overrides.get(cov, lifecycle_mode)
            # Per-cov-mode birth gate: with ab3dmot lifecycle the birth gate
            # is irrelevant (count-based confirmation), so disable it for
            # those streams to keep the comparison clean.
            stream_birth_gate = (None if stream_lifecycle == "ab3dmot"
                                 else p_tp_birth_gate)
            # Per-source birth-gate map is irrelevant under ab3dmot lifecycle
            # (count-based confirmation), so disable it for those streams too —
            # mirrors the scalar stream_birth_gate handling above.
            stream_birth_gate_by_source = (None if stream_lifecycle == "ab3dmot"
                                           else p_tp_birth_gate_by_source)
            streams[key] = sensor_fusion.Fusion(
                sensor_fusion.max_id,
                use_trust_scoring=False,
                passthrough_covariance=False,
                static_matching_cov=static_match_cov,
                filter_class=filter_cls,
                p_tp_birth_gate=stream_birth_gate,
                p_tp_birth_gate_by_source=stream_birth_gate_by_source,
                lifecycle_mode=stream_lifecycle,
                ab3dmot_min_hits=ab3dmot_min_hits,
                ab3dmot_max_age=ab3dmot_max_age,
                enable_inactive_preservation=enable_inactive_preservation,
                inactive_grace_s=inactive_grace_s,
                confirm_log_odds=confirm_log_odds,
                confirm_log_odds_by_source=confirm_log_odds_by_source,
                kill_log_odds=kill_log_odds,
                log_lr_miss=log_lr_miss,
                log_lr_miss_by_source=log_lr_miss_by_source,
                mahal_weight=mahal_weight,
                iou_weight=iou_weight,
                mahal_gate=mahal_gate,
                iou_gate=iou_gate,
                tape_score_miss_decay=tape_score_miss_decay,
            )
    return streams


def run_scenario(
    scenario_dir,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Replay one V2V4Real scenario through 30 fusion streams.

    Args:
        scenario_dir: Path to a scenario directory containing ego_state.csv,
                      detections.csv, gt_objects.csv, meta.yaml.
        config: Optional dict with the same keys traci_interface.run_simulation
                accepts. Recognised keys (with defaults):
                  detector_name (str)        = "centerpoint_100m_on_v2v4real"
                  localizer_name (str)       = "rtk_v2v4real"
                  detector_max_range (float) = 70.0
                  variance_floor (float|None)= None
                  use_static_matching (bool) = True
                  cleanup_time (float)       = 0.5
                  score_threshold (float)    = 0.3   # CenterPoint score floor;
                                                     # below this is mostly noise
                                                     # and pollutes the matcher.
                  vehicle_classes (list|None) = ['car','truck','bus',
                                                 'construction_vehicle']
                                                # nuScenes-trained CenterPoint
                                                # emits 10 classes; V2V4Real GT
                                                # only annotates vehicles. Drop
                                                # barriers/cones/peds — they're
                                                # always FPs.
                  class_thresholds (dict|None) = None
                                                # Optional per-class score
                                                # threshold; overrides the
                                                # global score_threshold for
                                                # listed classes.

    Returns:
        Metrics dict in the same shape as traci_interface.run_simulation when
        gpem_triple_fusion=True (see traci_interface.py:1370-1490).
    """
    cfg = config or {}
    # When set, the run includes a `match_tape_per_stream` field in the
    # returned dict containing the per-frame [(track_id, x, y, score), …]
    # records collected during the loop — useful for debug visualisation
    # (overlaying our fused tracks onto a per-frame video) or for offline
    # re-evaluation. Pass a list of STREAM_KEYS to include.
    record_tape_for: List[str] = cfg.get("record_tape_for", []) or []
    # When True, the returned dict also carries the full per-ego match tape for
    # EVERY stream (`match_tape_per_ego`), the exact input to
    # `compute_ab3dmot_metrics_iou`. The suite persists it so AMOTP can be
    # re-scored offline (e.g. after a metric-convention change) without
    # re-running the tracker. See scripts/rescore_v2v4real_amotp_from_tapes.py.
    dump_match_tapes: bool = bool(cfg.get("dump_match_tapes", False))
    # streams_keep: optional list of STREAM_KEYS to build/evaluate. None = all 30.
    # Used for fast-iteration dev where you only need one cov/filter combo.
    # Validated against STREAM_KEYS — unknown keys raise loudly so typos don't
    # silently degrade to an empty-stream run.
    streams_keep_cfg = cfg.get("streams_keep", None)
    if streams_keep_cfg is not None:
        streams_keep_cfg = list(streams_keep_cfg)
        unknown = [k for k in streams_keep_cfg if k not in STREAM_KEYS]
        if unknown:
            raise ValueError(
                f"streams_keep contains unknown keys: {unknown}. "
                f"Valid STREAM_KEYS: {STREAM_KEYS}"
            )
        if not streams_keep_cfg:
            raise ValueError("streams_keep is empty — at least one stream must be kept")
    active_keys: List[str] = streams_keep_cfg if streams_keep_cfg else list(STREAM_KEYS)
    active_key_set = set(active_keys)
    # Cov modes actually consumed by an active stream. Building the rest is
    # wasted work, and — crucially — forces loading cov data a detector's CSV
    # may not carry (e.g. an older GPEM CSV with no polar columns) even when
    # that mode isn't in the kept streams. Restrict to what's needed.
    active_cov_modes = _active_cov_modes(active_key_set)
    # V2X self-report: each ego broadcasts its own RTK pose to the fusion
    # as a high-confidence "detection" of itself. Convention in real V2X
    # systems — every CAV emits CAM/BSM messages with their own position,
    # so cooperators don't need to *detect* each other to track them. This
    # plugs the asymmetric ego-visibility hole we observed with
    # PointPillar (tesla detects astuff in only 4% of frames despite
    # astuff being a large car in clear view).
    self_report_egos = cfg.get("self_report_egos", True)
    # detector_name supports a "{vehicle}" placeholder so a single config can
    # bind per-ego detector models. e.g. "pointpillar_v2v4real_{vehicle}" maps
    # tesla detections to "pointpillar_v2v4real_tesla" and astuff to
    # "pointpillar_v2v4real_astuff" — matches the V2V4Real fine-tune output.
    detector_name = cfg.get("detector_name", "centerpoint_100m_on_v2v4real")
    localizer_name = cfg.get("localizer_name", "rtk_v2v4real")
    # V2V4Real-exact range gating (rectangular, in ego frame). These are the
    # ranges they hard-code in OpenCOOD's config + datasets/__init__.py:
    #   detector lidar_range:  [-70.4, -40, -5, 70.4, 40, 3] (training)
    #   eval GT_RANGE:         [-100, -40, -5, 100, 40, 3]
    # Both rectangles, in (x_min, y_min, z_min, x_max, y_max, z_max) form,
    # all relative to each ego's heading-aligned frame. Anything outside is
    # dropped — including z-out-of-range (we don't *use* z but we drop on it
    # to match their evaluation set semantics).
    lidar_range = cfg.get("lidar_range", [-70.4, -40, -5, 70.4, 40, 3])
    gt_range    = cfg.get("gt_range",    [-100, -40, -5, 100, 40, 3])
    # Legacy circular fallback (used only by callers that didn't pass new ranges).
    max_range = cfg.get("detector_max_range", 70.0)
    variance_floor = cfg.get("variance_floor", None)
    use_static_matching = cfg.get("use_static_matching", True)
    cleanup_time = cfg.get("cleanup_time", DEFAULT_CLEANUP_TIME)
    score_threshold = cfg.get("score_threshold", 0.3)
    # Per-vehicle score floor (mixed-detector configs): each CAV's detections
    # are filtered at ITS OWN detector's threshold (e.g. astuff=PP@0.2,
    # tesla=CPft@0.3), keyed by vehicle_id string. Falls back to the scalar
    # score_threshold for vehicles not listed / single-detector runs. The
    # thresholds never cross vehicles — each CAV uses only its own.
    score_threshold_by_vehicle = cfg.get("score_threshold_by_vehicle") or {}
    # Source vehicles in this scenario, read straight from meta.yaml. The
    # per-source data-driven lifecycle/gate blocks below substitute each of
    # these into a "{vehicle}" detector_name and key their maps by the
    # vehicle's participant id, so the de-hardcoded path generalises across
    # datasets (V2V4Real [astuff, tesla], DAIR-V2X-Seq [veh, inf], …). This
    # mirrors the GPEM error-model path (`for vid in scenario.meta["vehicles"]`
    # ~line 870). The full scenario is loaded later (heavier); meta.yaml alone
    # is cheap. Fall back to the legacy V2V4Real pair if meta is unreadable so
    # nothing regresses on a malformed export.
    try:
        with open(Path(scenario_dir) / "meta.yaml") as _mf:
            _meta_vehicles = yaml.safe_load(_mf).get("vehicles") or ["astuff", "tesla"]
    except (OSError, AttributeError, yaml.YAMLError):
        _meta_vehicles = ["astuff", "tesla"]
    source_vehicles: tuple = tuple(_meta_vehicles)
    # Data-driven per-range birth gate (GPEM-style curve fit on TP/FP score
    # distributions). When `data_driven_gate_alpha` is set (1.0 or 2.0), an
    # additional per-range score floor is applied on top of `score_threshold`:
    #     gate(d) = a·d² + b·d + c   (quadratic fit per detector)
    # Curves loaded from results/birth_gate_calibration/birth_gate_curves.csv
    # (produced by scripts/derive_optimal_birth_gate.py). Default None = off.
    #
    # PER-VEHICLE LOOKUP for mix configs: detector_name may contain `{vehicle}`,
    # in which case each vehicle (astuff, tesla) substitutes its own name and
    # uses its own calibration curve. This keeps heterogeneous fleets (e.g.
    # mix_pp_cpft) using the right per-detector gate per vehicle.
    data_driven_gate_alpha = cfg.get("data_driven_gate_alpha", None)
    # Sigmoid steepness around the gate. p_tp = sigmoid(k · (score − gate(d))):
    # k=10 default → score=gate ± 0.1 maps to p_tp ≈ 0.27 / 0.73; k=20 sharper
    # transition; k=5 gentler. Score = gate exactly → p_tp = 0.5 (neutral).
    data_driven_gate_k = float(cfg.get("data_driven_gate_k", 10.0))
    # Fit form for the curve: "static" / "linear" / "quadratic" / "polar".
    # Static/linear/quad use the fitted coefficients (lightweight). Polar uses
    # per-(range, angle) bin lookup at runtime — most-faithful, no smoothing,
    # mirrors GPEM cov polar mode.
    data_driven_gate_fit = cfg.get("data_driven_gate_fit", "quadratic")
    data_driven_gate_curves: Dict[str, "BirthGateCurve"] = {}
    if data_driven_gate_alpha is not None:
        from birth_gate_curves import load_birth_gate_curve
        det_name = cfg.get("detector_name", "")
        # Optional cfg-level override for which curves CSV to load from.
        # Lets us A/B alternate aggregations (e.g. P25 vs P50) without
        # touching the production CSV path. Same Path passes through to
        # load_birth_gate_curve(..., curve_csv=...).
        _curve_csv = cfg.get("data_driven_gate_csv", None)
        # Optional override for the polar per-(range,angle) bin table directory.
        # Defaults to None → load_birth_gate_curve uses its default
        # results/birth_gate_calibration/amota_bayes location. Instant-tune
        # configs point this at results/instant_tune/<det>/<proto>/ so the
        # per-source <detector>_polar.csv files emitted alongside the JSON are
        # consumed directly.
        _polar_dir = cfg.get("data_driven_gate_polar_dir", None)
        if "{vehicle}" in det_name:
            for vname in source_vehicles:
                lookup = det_name.replace("{vehicle}", vname)
                c = load_birth_gate_curve(lookup, float(data_driven_gate_alpha), curve_csv=_curve_csv, polar_dir=_polar_dir)
                if c is not None:
                    data_driven_gate_curves[vname] = c
                    print(f"  [data-driven-gate] {vname}: {lookup} α={data_driven_gate_alpha} "
                          f"(quad {c.a_quad:.5f}·d² + {c.b_quad:.5f}·d + {c.c_quad:.5f})")
                else:
                    print(f"  [data-driven-gate] {vname}: no curve for {lookup} α={data_driven_gate_alpha} — gate inactive for this vehicle")
        else:
            c = load_birth_gate_curve(det_name, float(data_driven_gate_alpha), curve_csv=_curve_csv, polar_dir=_polar_dir)
            if c is not None:
                # Apply same curve to all vehicles (single-detector fleet).
                for vname in source_vehicles:
                    data_driven_gate_curves[vname] = c
                print(f"  [data-driven-gate] {det_name} α={data_driven_gate_alpha} "
                      f"(quad {c.a_quad:.5f}·d² + {c.b_quad:.5f}·d + {c.c_quad:.5f})")
            else:
                print(f"  [data-driven-gate] no curve for {det_name} α={data_driven_gate_alpha} — disabled")
    # Tier 1 data-driven lifecycle. When `data_driven_lifecycle=True`, look up
    # per-detector log_odds lifecycle params from recommended_lifecycle.csv
    # (produced by scripts/solve_amota_bayes_gate.py). Overrides any explicit
    # p_tp_birth_gate / kill_log_odds / log_lr_miss in the cfg.
    #
    # Per-detector params are derived from characterization data:
    #   |log_lr_miss| = (logit(p_birth) - kill_log_odds) / (L_TP · p_miss_TP)
    # so each detector's lifecycle matches its natural track lifetime and
    # miss rate. Long-tail detectors (cobevt) get gentle decay; short-tail
    # detectors (centerpoint nuScenes) get aggressive decay.
    data_driven_lifecycle_cfg = cfg.get("data_driven_lifecycle", False)
    data_driven_lifecycle_params: Optional[object] = None
    # Per-source LIFECYCLE BLEND: instead of first-match-wins (one global
    # LifecycleParams), load params for BOTH astuff and tesla substitutions
    # and build per-source dicts keyed by participant id (1=astuff, 2=tesla,
    # see VEHICLE_PARTICIPANT_IDS). Track-level lifecycle then routes per the
    # detection's source vehicle, exactly as the per-DETECTION economics
    # (GPEM R, birth evidence) already do. For a bare (single-detector) name
    # both source keys get the same params, so the blend degenerates to the
    # legacy single-scalar path (byte-for-byte-equivalent decisions).
    # kill_log_odds stays a single global floor (not detector-specific).
    p_tp_birth_gate_by_source: Optional[Dict[int, float]] = None
    confirm_log_odds_by_source: Optional[Dict[int, float]] = None
    log_lr_miss_by_source: Optional[Dict[int, float]] = None
    if data_driven_lifecycle_cfg:
        from lifecycle_params import load_lifecycle_params
        # Optional cfg-level override for which lifecycle CSV to load from.
        # Same plumbing as data_driven_gate_csv above.
        _lifecycle_csv = cfg.get("data_driven_lifecycle_csv", None)
        det_name = cfg.get("detector_name", "")
        # Resolve the lookup name per source vehicle. For a "{vehicle}" name
        # each source substitutes its own vehicle; for a bare name every
        # source uses the same lookup (degenerate / single-detector).
        if "{vehicle}" in det_name:
            source_lookup = {vname: det_name.replace("{vehicle}", vname)
                             for vname in source_vehicles}
        else:
            source_lookup = {vname: det_name for vname in source_vehicles}

        _gate_map: Dict[int, float] = {}
        _confirm_map: Dict[int, float] = {}
        _miss_map: Dict[int, float] = {}
        for vname, cand in source_lookup.items():
            lp = load_lifecycle_params(cand, lifecycle_csv=_lifecycle_csv)
            if lp is None:
                print(f"  [data-driven-lifecycle] {vname}: no lifecycle row "
                      f"for {cand} — source falls back to global scalars")
                continue
            pid = VEHICLE_PARTICIPANT_IDS.get(vname)
            if pid is not None:
                _gate_map[pid] = lp.p_birth
                _confirm_map[pid] = lp.confirm_log_odds
                _miss_map[pid] = lp.log_lr_miss
            # First successful load also seeds the global scalar fallbacks
            # (preserves today's behaviour: kill/p_birth/confirm/miss scalars
            # come from the first-matched detector's params).
            if data_driven_lifecycle_params is None:
                data_driven_lifecycle_params = lp
            print(f"  [data-driven-lifecycle] {vname} ({cand}): "
                  f"p_birth={lp.p_birth:.3f}, log_lr_miss={lp.log_lr_miss:.3f}, "
                  f"confirm={lp.confirm_log_odds:.3f}, kill={lp.kill_log_odds:.1f}, "
                  f"(p_miss_obs={lp.p_miss_tp_observed:.2f}, "
                  f"L_TP_obs={lp.l_tp_observed:.1f}, n={lp.n_tracks_observed})")
        # Only attach the per-source maps when they actually differentiate
        # sources (>=2 distinct keys with differing values). A single-detector
        # run leaves them None so Fusion uses the scalar path exactly.
        if len(_gate_map) >= 2 and len(set(_gate_map.values())) > 1:
            p_tp_birth_gate_by_source = _gate_map
        if len(_confirm_map) >= 2 and len(set(_confirm_map.values())) > 1:
            confirm_log_odds_by_source = _confirm_map
        if len(_miss_map) >= 2 and len(set(_miss_map.values())) > 1:
            log_lr_miss_by_source = _miss_map
        if data_driven_lifecycle_params is None:
            print(f"  [data-driven-lifecycle] no lifecycle row for {det_name} — using manual defaults")
        elif (p_tp_birth_gate_by_source or confirm_log_odds_by_source
              or log_lr_miss_by_source):
            print(f"  [data-driven-lifecycle] per-source BLEND active: "
                  f"gate_by_source={p_tp_birth_gate_by_source}, "
                  f"confirm_by_source={confirm_log_odds_by_source}, "
                  f"log_lr_miss_by_source={log_lr_miss_by_source}")

    # Per-CAV NMS IoU threshold. Default 1.0 (disabled): empirically hurts
    # cross-vehicle fusion streams (CI/BICI/SABRE) by ~3 AMOTA — they use
    # within-CAV redundancy as multiple measurements per car. Lower values
    # (e.g. 0.15) match V2V4Real's reference but only help the simple
    # baseline streams. Kept as an opt-in ablation knob.
    per_ego_nms_iou = cfg.get("per_ego_nms_iou", 1.0)
    # Combined-set NMS (cross-CAV) — applied AFTER per-CAV filters but BEFORE
    # Hungarian matching. Each surviving detection is routed back to its
    # original vehicle_id for normal per-stream submission. Mirrors V2V4Real's
    # concat-then-NMS recipe but keeps our per-stream Hungarian + Kalman path.
    # 1.0 = disabled.
    combined_nms_iou = cfg.get("combined_nms_iou", 1.0)
    # nuScenes' 10-class detector emits barrier/cone/pedestrian/etc. that
    # V2V4Real never annotates as GT — keeping them turns CenterPoint's road
    # furniture into ~50% of "tracked" objects, which dominate the FP count.
    # Default to the four vehicle classes V2V4Real actually evaluates against.
    vehicle_classes = cfg.get(
        "vehicle_classes",
        ["car", "truck", "bus", "construction_vehicle"],
    )
    if vehicle_classes is not None:
        vehicle_classes = set(vehicle_classes)
    class_thresholds = cfg.get("class_thresholds", None) or {}
    t0_wall = _time.time()
    scenario = load_scenario(scenario_dir)

    # --- Detector error models in 4 GPEM modes (cached) ---
    # Per-vehicle support: if detector_name contains "{vehicle}", load one set
    # of models per vehicle ID. Otherwise load a single shared set. The
    # tesla/astuff PointPillar fine-tune ships per-vehicle CSVs, so we let
    # the user select them with detector_name="pointpillar_v2v4real_{vehicle}".
    if "{vehicle}" in detector_name:
        per_vehicle_em: Dict[str, Dict[str, object]] = {}
        per_vehicle_calib: Dict[str, object] = {}
        for vid in scenario.meta.get("vehicles", []):
            resolved = detector_name.format(vehicle=vid)
            em_s, em_l, em_q, em_p = _get_detector_error_models(resolved, max_range)
            per_vehicle_em[vid] = {
                "static":         em_s,
                "gpem_linear":    em_l,
                "gpem_quadratic": em_q,
                "gpem_polar":     em_p,
            }
            # Calibration table for this detector model (None if not characterized).
            try:
                from sensor_model_loader import load_sensor_model as _load_sm
                per_vehicle_calib[vid] = _load_sm(resolved).calibration
            except Exception:
                per_vehicle_calib[vid] = None
        cov_to_em = None  # Picked up by vehicle below.
        shared_calib = None
    else:
        em_static, em_linear, em_quadratic, em_polar = _get_detector_error_models(
            detector_name, max_range,
        )
        cov_to_em = {
            "static":         em_static,
            "gpem_linear":    em_linear,
            "gpem_quadratic": em_quadratic,
            "gpem_polar":     em_polar,
        }
        per_vehicle_em = None
        per_vehicle_calib = None
        try:
            from sensor_model_loader import load_sensor_model as _load_sm
            shared_calib = _load_sm(detector_name).calibration
        except Exception:
            shared_calib = None

    # --- Localizer(s): the localization covariance (loc_cov) added to each
    # detection's perception covariance (Eq. 18). It depends only on velocity +
    # yaw, not the filter, so one instance per source is reused across all modes.
    # Per-source support mirrors the per-vehicle detector path: a "{vehicle}"
    # placeholder in localizer_name binds one localizer model per source
    # (DAIR-V2X-Seq's "dair_loc_{vehicle}" -> dair_loc_veh / dair_loc_inf), so a
    # poorly-localized roadside infrastructure source carries a larger loc_cov
    # than the RTK-localized vehicle and the filter down-weights it in fusion. No
    # placeholder -> one shared localizer for every source (V2V4Real: rtk_v2v4real),
    # byte-identical to the pre-per-source behaviour. ---
    dummy_err = ErrorPackage(ErrorType.LOCALIZATION, 0)
    if "{vehicle}" in localizer_name:
        per_vehicle_loc: Optional[Dict[str, object]] = {
            vid: cmr_localizer.get_localizer(
                localizer_name.format(vehicle=vid), dummy_err,
                use_gpem_model=False, use_quadratic=False)
            for vid in scenario.meta.get("vehicles", [])
        }
        # Shared fallback (self-report path / any source absent from meta).
        loc = next(iter(per_vehicle_loc.values()), None)
    else:
        per_vehicle_loc = None
        loc = cmr_localizer.get_localizer(localizer_name, dummy_err, use_gpem_model=False, use_quadratic=False)

    # --- Static matching covariance (mirrors traci_interface.py:419-425) ---
    # When using per-vehicle detectors, take the max std across the vehicle
    # set to keep matching gates conservative.
    static_match_cov = None
    if use_static_matching:
        if per_vehicle_em is not None:
            ds, ps = 0.0, 0.0
            for vid, ems in per_vehicle_em.items():
                ds = max(ds, ems["static"].get_distal_std(30.0))
                ps = max(ps, ems["static"].get_perpendicular_std(30.0))
            d_std, p_std = ds, ps
        else:
            d_std = em_static.get_distal_std(30.0)
            p_std = em_static.get_perpendicular_std(30.0)
        static_match_cov = np.diag([max(d_std, p_std) ** 2, max(d_std, p_std) ** 2])

    # --- Baseline (flat) covariance (mirrors traci_interface.py:472) ---
    baseline_cov = _compute_baseline_covariance()

    # --- Fusion streams + per-stream accumulators ---
    # Phase A: optional P(TP) birth gate (None = legacy / off).
    p_tp_birth_gate = cfg.get("p_tp_birth_gate", None)
    # Phase B: track lifecycle mode (legacy / log_odds / ab3dmot).
    lifecycle_mode = cfg.get("lifecycle_mode", "legacy")
    # Phase B.6: per-cov-mode lifecycle override. flat-cov "baseline" mode
    # has no per-detection p_tp signal to feed log-odds, so its native
    # lifecycle is count-based. Default suite-wide; override per cov mode
    # via cfg["per_cov_lifecycle"] = {"baseline": "ab3dmot", ...}.
    per_cov_lifecycle = cfg.get("per_cov_lifecycle", None)
    ab3dmot_min_hits = int(cfg.get("ab3dmot_min_hits", 3))
    ab3dmot_max_age  = int(cfg.get("ab3dmot_max_age",  2))
    # Phase B.5: inactive-track preservation. Default OFF for clean A/B.
    enable_inactive_preservation = bool(cfg.get("enable_inactive_preservation", False))
    inactive_grace_s = float(cfg.get("inactive_grace_s", 1.0))
    confirm_log_odds_cfg = cfg.get("confirm_log_odds", None)
    kill_log_odds_cfg    = cfg.get("kill_log_odds", None)
    log_lr_miss_cfg      = cfg.get("log_lr_miss", None)
    tape_score_miss_decay_cfg = cfg.get("tape_score_miss_decay", None)
    # Data-driven lifecycle overrides the manual lifecycle knobs if loaded.
    # Note: p_tp_birth_gate is also overridden — for the dd-lifecycle path
    # we want the gate to come from the same calibration that determined the
    # log_lr_miss budget (consistency between birth state and decay).
    if data_driven_lifecycle_params is not None:
        kill_log_odds_cfg = data_driven_lifecycle_params.kill_log_odds
        log_lr_miss_cfg   = data_driven_lifecycle_params.log_lr_miss
        p_tp_birth_gate   = data_driven_lifecycle_params.p_birth
        # Tier 1.5: confirm threshold + coast-track decay (only override if
        # the loaded LifecycleParams carries non-default values — older CSVs
        # without these columns load with canonical defaults and shouldn't
        # silently overwrite cfg's manual values).
        confirm_log_odds_cfg = data_driven_lifecycle_params.confirm_log_odds
        tape_score_miss_decay_cfg = data_driven_lifecycle_params.tape_score_miss_decay
    # Pluggable Hungarian cost: default = our hybrid (Mahalanobis + IoU).
    # Pure-IoU (AB3DMOT-style) → mahal_weight=0, iou_weight=1, large gate.
    mahal_weight_cfg = float(cfg.get("mahal_weight", 0.6))
    iou_weight_cfg   = float(cfg.get("iou_weight",   0.4))
    mahal_gate_cfg   = float(cfg.get("mahal_gate",   13.82))
    iou_gate_cfg     = float(cfg.get("iou_gate",     0.0))
    streams = _build_fusion_streams(static_match_cov,
                                    p_tp_birth_gate=p_tp_birth_gate,
                                    p_tp_birth_gate_by_source=p_tp_birth_gate_by_source,
                                    lifecycle_mode=lifecycle_mode,
                                    per_cov_lifecycle=per_cov_lifecycle,
                                    ab3dmot_min_hits=ab3dmot_min_hits,
                                    ab3dmot_max_age=ab3dmot_max_age,
                                    enable_inactive_preservation=enable_inactive_preservation,
                                    inactive_grace_s=inactive_grace_s,
                                    confirm_log_odds=confirm_log_odds_cfg,
                                    confirm_log_odds_by_source=confirm_log_odds_by_source,
                                    kill_log_odds=kill_log_odds_cfg,
                                    log_lr_miss=log_lr_miss_cfg,
                                    log_lr_miss_by_source=log_lr_miss_by_source,
                                    mahal_weight=mahal_weight_cfg,
                                    iou_weight=iou_weight_cfg,
                                    mahal_gate=mahal_gate_cfg,
                                    iou_gate=iou_gate_cfg,
                                    tape_score_miss_decay=tape_score_miss_decay_cfg,
                                    streams_keep=streams_keep_cfg)
    # Master metric grid: range_mode × merge_strategy.
    #   range_mode: "per_ego" (V2V4Real-exact, restrictive)
    #               "world"   (cooperative gold standard, KITTI ±100×40m union)
    #   merge_strategy: "sort" / "cav0" / "average" — see _pick_cluster_representative()
    # Total: 2 × 3 = 6 (range, strategy) combinations. Each gets its own HOTA
    # accumulator + paper match tape + per-frame AMOTA/AMOTP totals so the
    # user can compare every protocol variant side-by-side without re-running.
    RANGE_MODES = ("per_ego", "world")
    MERGE_STRATEGIES = ("sort", "cav0", "average")
    PRIMARY_RANGE = "per_ego"
    PRIMARY_STRATEGY = "sort"

    hota_accs_by_combo: Dict[Tuple[str, str], Dict[str, HotaAccumulator]] = {
        (rm, s): {k: HotaAccumulator() for k in STREAM_KEYS}
        for rm in RANGE_MODES for s in MERGE_STRATEGIES
    }
    metric_totals_by_combo: Dict[Tuple[str, str], Dict[str, Dict]] = {
        (rm, s): {k: _create_metrics_dict() for k in STREAM_KEYS}
        for rm in RANGE_MODES for s in MERGE_STRATEGIES
    }
    amota_frames_by_combo: Dict[Tuple[str, str], Dict[str, int]] = {
        (rm, s): {k: 0 for k in STREAM_KEYS}
        for rm in RANGE_MODES for s in MERGE_STRATEGIES
    }
    amotp_frames_by_combo: Dict[Tuple[str, str], Dict[str, int]] = {
        (rm, s): {k: 0 for k in STREAM_KEYS}
        for rm in RANGE_MODES for s in MERGE_STRATEGIES
    }
    paper_match_tape_merged_by_combo: Dict[Tuple[str, str], Dict[str, List[Dict]]] = {
        (rm, s): {k: [] for k in STREAM_KEYS}
        for rm in RANGE_MODES for s in MERGE_STRATEGIES
    }
    # Backward-compat aliases — point at the primary (per_ego, sort) so the
    # existing un-suffixed metrics in the output dict still reflect the
    # V2V4Real-exact view that downstream code consumed before this refactor.
    PRIMARY_COMBO = (PRIMARY_RANGE, PRIMARY_STRATEGY)
    hota_accs              = hota_accs_by_combo[PRIMARY_COMBO]
    metric_totals          = metric_totals_by_combo[PRIMARY_COMBO]
    amota_frames           = amota_frames_by_combo[PRIMARY_COMBO]
    amotp_frames           = amotp_frames_by_combo[PRIMARY_COMBO]
    paper_match_tape_merged = paper_match_tape_merged_by_combo[PRIMARY_COMBO]
    # Per-strategy aliases (for the existing per-strategy fields on the
    # primary range_mode = per_ego). These will be deprecated once the
    # master-table generator catches up.
    hota_accs_by_strategy = {s: hota_accs_by_combo[(PRIMARY_RANGE, s)] for s in MERGE_STRATEGIES}
    metric_totals_by_strategy = {s: metric_totals_by_combo[(PRIMARY_RANGE, s)] for s in MERGE_STRATEGIES}
    amota_frames_by_strategy = {s: amota_frames_by_combo[(PRIMARY_RANGE, s)] for s in MERGE_STRATEGIES}
    amotp_frames_by_strategy = {s: amotp_frames_by_combo[(PRIMARY_RANGE, s)] for s in MERGE_STRATEGIES}
    paper_match_tape_merged_by_strategy = {
        s: paper_match_tape_merged_by_combo[(PRIMARY_RANGE, s)] for s in MERGE_STRATEGIES
    }
    # Per-ego match tape (V2V4Real per-ego GT, no merging — single set):
    paper_match_tape_per_ego: Dict[str, List[Dict]] = {k: [] for k in STREAM_KEYS}

    # --- Frame loop ---
    iterations = 0
    for fid in scenario.frame_ids:
        frame = scenario.frames[fid]
        timestamp = fid * 0.1  # V2V4Real is 10 Hz

        # ---- Pre-pass: per-vehicle filter (+ optional per-CAV NMS), then
        #      optional combined-set NMS across all CAVs. Survivors are
        #      grouped back by their original vehicle_id for normal
        #      per-stream submit below. This is the V2V4Real-style
        #      concat-then-NMS recipe but without giving up our per-stream
        #      Hungarian + Kalman fusion path.
        filtered_per_vehicle: Dict[str, List[Dict]] = {}
        for vehicle_id, ego in frame["vehicles"].items():
            ego_pose = frame["vehicles"][vehicle_id]
            def _keep(d, _ego=ego_pose, _vid=vehicle_id):
                if vehicle_classes is not None and d["label"] not in vehicle_classes:
                    return False
                veh_thr = score_threshold_by_vehicle.get(_vid, score_threshold)
                thr = class_thresholds.get(d["label"], veh_thr)
                if d["score"] < thr:
                    return False
                if lidar_range is not None and not _in_ego_range(
                    d["x"], d["y"], d.get("z", 0.0),
                    _ego["x"], _ego["y"], _ego["yaw"],
                    lidar_range,
                ):
                    return False
                return True
            det_rows = [d for d in frame["detections"].get(vehicle_id, []) if _keep(d)]
            if per_ego_nms_iou < 1.0:
                det_rows = _per_ego_nms(det_rows, per_ego_nms_iou)
            filtered_per_vehicle[vehicle_id] = det_rows

        # Combined cross-CAV NMS (Option A). Concat all surviving detections,
        # tag with origin vehicle_id, NMS the union, then split survivors back
        # by tag. Cross-CAV duplicates of the same physical car get reduced
        # to a single highest-score detection — closer to V2V4Real's recipe
        # but without losing our Hungarian/Kalman fusion path.
        if combined_nms_iou < 1.0:
            tagged: List[Dict] = []
            for vid, dets in filtered_per_vehicle.items():
                for d in dets:
                    d_t = dict(d)
                    d_t["_origin_vehicle_id"] = vid
                    tagged.append(d_t)
            kept = _per_ego_nms(tagged, combined_nms_iou)
            filtered_per_vehicle = {vid: [] for vid in filtered_per_vehicle}
            for d in kept:
                filtered_per_vehicle[d["_origin_vehicle_id"]].append(d)

        # Submit detections from each vehicle to all 30 streams.
        for vehicle_id, ego in frame["vehicles"].items():
            participant_id = VEHICLE_PARTICIPANT_IDS.get(vehicle_id, hash(vehicle_id) & 0xFF)
            ego_x, ego_y, ego_yaw = ego["x"], ego["y"], ego["yaw"]
            velocity = math.hypot(ego["vx"], ego["vy"])
            det_rows = filtered_per_vehicle[vehicle_id]

            # Build a list of DetectedObjects per cov-mode. Each cov-mode list
            # gets fed to all 6 filter classes that share that cov-mode (so the
            # same input goes into Kalman + CI + AKF + PF + BICI + SABRE).
            per_cov_observations: Dict[str, List[cmr_sensor.DetectedObject]] = {
                cov: [] for cov in COV_MODES
            }
            # Pick per-vehicle detector models if configured, else shared set.
            ems_for_vehicle = (
                per_vehicle_em[vehicle_id] if per_vehicle_em is not None else cov_to_em
            )
            # Pick per-source localizer (loc_cov) the same way: a poorly-localized
            # source (DAIR inf) carries a larger covariance than an RTK source
            # (DAIR veh). Single shared localizer for non-templated configs.
            loc_for_vehicle = (
                per_vehicle_loc[vehicle_id] if per_vehicle_loc is not None else loc
            )

            # V2X self-report: this ego broadcasts its own RTK pose as a
            # tight-covariance "detection" of itself, prepended to the
            # detector-output list. Both egos see this in their fused track
            # set, so the asymmetric "tesla can't detect astuff" problem is
            # bypassed by astuff *telling* tesla where it is.
            if self_report_egos:
                self_loc_cov = loc_for_vehicle.get_localization_covariance(velocity, ego_yaw)
                self_det = cmr_sensor.DetectedObject(
                    vehicle_id=None,                     # set by Fusion later
                    vehicle_type="car",
                    detected_bbox=None,
                    centroid=[ego_x, ego_y],
                    width=ego["width"],
                    length=ego["length"],
                    angle=ego_yaw,
                    expected_error_gaussian=None,
                    velocity_vector=[ego["vx"], ego["vy"]],
                    error_covariance=self_loc_cov,
                    width_std=0.05,
                    length_std=0.05,
                    yaw_variance=1e-4,
                )
                # Self-report is V2X RTK pose, not a detector observation.
                # Always pass the gate.
                self_det.p_tp = 1.0
                for cov_mode in active_cov_modes:
                    per_cov_observations[cov_mode].append(self_det)

            for det_row in det_rows:
                raw_det = _make_raw_detected_object(det_row)

                dx = det_row["x"] - ego_x
                dy = det_row["y"] - ego_y
                distance   = math.hypot(dx, dy)
                angle_deg  = math.degrees(math.atan2(dy, dx) - ego_yaw)

                # Per-cov-mode P(TP) hierarchy:
                #   baseline → raw detector score (no GPEM; fair baseline
                #              since no characterisation is applied).
                #   static / linear / quadratic / polar → each mode uses
                #              its own error model's detection_probability
                #              (GPEM regression over range & angle), so the
                #              gate matches the covariance tier exactly.
                # baseline: flat cov, no spatial gate (score threshold already
                # applied upstream; baseline claims no GPEM characterisation).
                if "baseline" in active_cov_modes:
                    base_obj = _detection_with_flat_covariance(raw_det, baseline_cov)
                    base_obj.p_tp = 1.0   # baseline = no GPEM characterisation
                    base_obj.det_score = float(det_row["score"])
                    per_cov_observations["baseline"].append(base_obj)
                # static / linear / quadratic / polar: per-detection cov from GPEM
                _gate_curve_here = (data_driven_gate_curves.get(vehicle_id)
                                    if data_driven_gate_curves else None)
                _det_score = float(det_row["score"])
                for cov in ("static", "gpem_linear", "gpem_quadratic", "gpem_polar"):
                    if cov not in active_cov_modes:
                        continue
                    obj = _detection_with_covariance_from_model(
                        raw_det, ego_x, ego_y, ego_yaw,
                        ems_for_vehicle[cov], loc_for_vehicle, velocity,
                        variance_floor=variance_floor,
                        use_mae_covariance=False,
                    )
                    if _gate_curve_here is not None:
                        # GPEM-derived calibrated confidence via sigmoid centered
                        # at the gate:  p_tp = σ(k · (score − gate(d, [angle]))).
                        # score=gate → p_tp=0.5 (neutral). Default k=10 gives
                        # a ±0.1-score-window transition. fit form picks how
                        # gate(d) is computed: static/linear/quad use the
                        # fitted coefficients; polar does per-(range, angle)
                        # bin lookup using `angle_rad`.
                        gate_at_d = _gate_curve_here.evaluate(
                            distance,
                            angle_rad=math.radians(angle_deg),
                            fit=data_driven_gate_fit,
                        )
                        z = data_driven_gate_k * (_det_score - gate_at_d)
                        # Numerically stable sigmoid: clamp z to avoid overflow
                        # in the exp() → equivalent to clipping p_tp at extremes.
                        if z > 50:
                            obj.p_tp = 1.0 - 1e-12
                        elif z < -50:
                            obj.p_tp = 1e-12
                        else:
                            obj.p_tp = 1.0 / (1.0 + math.exp(-z))
                    else:
                        # Legacy path: recall-based (P(detection | object)).
                        # NOT the same as P(TP | score) — kept for back-compat
                        # but the data-driven gate above is what we want.
                        obj.p_tp = max(1e-6, ems_for_vehicle[cov].detection_probability(
                            distance, angle_deg,
                        ))
                    obj.det_score = _det_score
                    per_cov_observations[cov].append(obj)

            for filter_name, _ in FILTER_CLASSES:
                for cov in COV_MODES:
                    key = _stream_key(filter_name, cov)
                    if key not in active_key_set:
                        continue
                    streams[key].processDetectionFrame(
                        timestamp,
                        per_cov_observations[cov],
                        cleanup_time,
                        source_participant_id=participant_id,
                    )

        # Both vehicles' detections submitted: fuse all 30 streams.
        per_stream_dets: Dict[str, List[cmr_sensor.DetectedObject]] = {}
        per_stream_track_records: Dict[str, List[Tuple]] = {}
        for key, fusion in streams.items():
            _result, fused_dets, _, _ = fusion.fuseDetectionFrame(timestamp, monitor=False)
            per_stream_dets[key] = fused_dets
            # V2V4Real-metric track tape: emit-eligible tracks scored by the
            # most recent matched detection's confidence, multiplicatively
            # decayed each miss frame (TAPE_SCORE_MISS_DECAY). This gives
            # AMOTA's recall sweep a rankable score that reflects detector
            # quality + freshness, so stale phantom tracks fall out naturally.
            # Lifecycle (emit/kill) is independent — it's still S_t-based in
            # log_odds mode and fusion_steps-based in legacy.
            if fusion.lifecycle_mode == "log_odds":
                emit_predicate = lambda t: t.S_t > fusion.confirm_log_odds
            elif fusion.lifecycle_mode == "ab3dmot":
                emit_predicate = lambda t: t.hits >= fusion.ab3dmot_min_hits
            else:
                emit_predicate = lambda t: t.fusion_steps > fusion.trackShowThreshold
            # Tape entry: (id, x, y, w, l, yaw, score, height, z) — the 3D layout
            # match_frame_3d_iou expects (h at index 7, z at index 8). The
            # center-distance and BEV-IoU evaluators read only indices 0–6, so the
            # trailing vertical fields are inert for them. height/z are EMA estimates
            # from the matched detections (None until a track sees a 3D detection).
            track_records = [
                (track.id, track.x, track.y,
                 track.width, track.length, track.angle,
                 float(track.tape_score),
                 track.height, track.z)
                for track in fusion.tracked_list
                if emit_predicate(track)
            ]
            per_stream_track_records[key] = track_records

        # Build per_ego_gt_dict (V2V4Real-exact view: each ego's own GT
        # filtered to that ego's GT_RANGE box) — used by paper_per_ego_*
        # and v2v4real_* metrics. Unchanged from before.
        per_ego_gt_filtered: List[Dict] = []
        per_ego_gt_dict: Dict[str, List[Tuple]] = {}
        for ego_id, ego in frame["vehicles"].items():
            for g in frame["gt"]:
                if g["vehicle_id"] != ego_id:
                    continue
                if gt_range is not None and not _in_ego_range(
                    g["x"], g["y"], g.get("z", 0.0),
                    ego["x"], ego["y"], ego["yaw"],
                    gt_range,
                ):
                    continue
                per_ego_gt_filtered.append(g)
            per_ego_gt_dict[ego_id] = []
        for g in per_ego_gt_filtered:
            ego = g["vehicle_id"]
            gid = (ego, int(g["ass_id"]) if g["ass_id"] >= 0 else -int(g["obj_id"]))
            gid_int = hash(gid) & 0x7FFFFFFF
            per_ego_gt_dict.setdefault(ego, []).append(
                (gid_int, g["x"], g["y"], g["width"], g["length"], g["yaw"])
            )

        # Build per-range_mode GT pools for the merged-GT metrics.
        #
        #   "per_ego"  : V2V4Real-exact — row kept only if within ITS own ego's
        #                cfg-provided `gt_range` box. Most restrictive; matches
        #                V2V4Real's published eval where their cav_lidar_range
        #                limits both detections and GT to cav-0's view.
        #
        #   "world"    : Gold-standard cooperative eval — row kept if within
        #                EITHER ego's KITTI ±100×40m annotation box. Captures
        #                the FULL union of both vehicles' annotation extent so
        #                far-away objects only one ego could see still count.
        #                Uses fixed ±100×40m (V2V4Real's annotation range)
        #                regardless of cfg's gt_range, so this view is stable
        #                across configs that use smaller eval ranges.
        WORLD_RANGE = [-100.0, -40.0, -5.0, 100.0, 40.0, 5.0]
        ego_pose_list = [(ego["x"], ego["y"], ego["yaw"])
                         for ego in frame["vehicles"].values()]
        gt_pool_by_range: Dict[str, List[Dict]] = {
            "per_ego": per_ego_gt_filtered,
            "world": [
                g for g in frame["gt"]
                if any(_in_ego_range(g["x"], g["y"], g.get("z", 0.0),
                                     ex, ey, eyaw, WORLD_RANGE)
                       for ex, ey, eyaw in ego_pose_list)
            ],
        }

        # Build merged GT for the 3 range_modes × 3 merge_strategies grid.
        # gt_objs_by_combo[(range_mode, strategy)] -> List[GroundTruthObject]
        gt_objs_by_combo: Dict[Tuple[str, str], List] = {}
        gt_records_by_combo: Dict[Tuple[str, str], List] = {}
        for rm in RANGE_MODES:
            pool = gt_pool_by_range[rm]
            for s in MERGE_STRATEGIES:
                objs = _dedup_gt_for_frame(pool, merge_strategy=s)
                gt_objs_by_combo[(rm, s)] = objs
                gt_records_by_combo[(rm, s)] = [
                    (g.vehicle_id, g.centroid[0], g.centroid[1],
                     g.dimensions[0], g.dimensions[1], g.angle)
                    for g in objs
                ]
        # Backward-compat aliases — primary view = (per_ego, sort)
        gt_objs_by_strategy = {s: gt_objs_by_combo[("per_ego", s)] for s in MERGE_STRATEGIES}
        gt_records_by_strategy = {s: gt_records_by_combo[("per_ego", s)] for s in MERGE_STRATEGIES}

        # Pre-compute the per-frame ego-poses list once (used for track filter)
        ego_poses_for_frame = [
            (e["x"], e["y"], e["yaw"])
            for e in frame["vehicles"].values()
        ]

        # Update HOTA + per-frame AMOTA/AMOTP for each stream, across all
        # (range_mode, merge_strategy) combos.
        # Pre-compute track-filter sets per range_mode (used as tape "tracks"):
        #   - per_ego: tracks within union of egos' cfg gt_range box
        #   - world  : tracks within union of egos' KITTI ±100×40m world box
        WORLD_RANGE_FOR_TRACKS = [-100.0, -40.0, -5.0, 100.0, 40.0, 5.0]
        for key in active_keys:
            fused_dets = per_stream_dets[key]
            tracks_for_key = per_stream_track_records[key]

            tracks_filtered_by_range = {
                "per_ego": [
                    rec for rec in tracks_for_key
                    if gt_range is None or any(
                        _in_ego_range(rec[1], rec[2], 0.0, ex, ey, eyaw, gt_range)
                        for ex, ey, eyaw in ego_poses_for_frame)
                ],
                "world": [
                    rec for rec in tracks_for_key
                    if any(_in_ego_range(rec[1], rec[2], 0.0, ex, ey, eyaw, WORLD_RANGE_FOR_TRACKS)
                           for ex, ey, eyaw in ego_poses_for_frame)
                ],
            }

            for rm in RANGE_MODES:
                tracks_for_rm = tracks_filtered_by_range[rm]
                for s in MERGE_STRATEGIES:
                    gt_objs = gt_objs_by_combo[(rm, s)]
                    gt_records = gt_records_by_combo[(rm, s)]

                    # CMR HOTA + per-frame AMOTA/AMOTP — per (range, strategy).
                    hota_accs_by_combo[(rm, s)][key].update(fused_dets, gt_objs)

                    mota = amota.calculate_amota(fused_dets, gt_objs)
                    if mota is not None:
                        metric_totals_by_combo[(rm, s)][key]["amota"] += mota
                        amota_frames_by_combo[(rm, s)][key] += 1

                    motp = amota.calculate_amotp(fused_dets, gt_objs)
                    if fused_dets and gt_objs and motp > 0.0:
                        metric_totals_by_combo[(rm, s)][key]["amotp"] += motp
                        amotp_frames_by_combo[(rm, s)][key] += 1

                    # AB3DMOT match tape — per (range, strategy).
                    paper_match_tape_merged_by_combo[(rm, s)][key].append({
                        "frame_idx": fid,
                        "tracks": tracks_for_rm,
                        "gts": gt_records,
                    })
            # AB3DMOT match tape — per-ego view. Each (ego, frame) becomes its
            # own entry; the fused tracks are filtered to those within
            # `max_range` of *this* ego (mimics V2V4Real's per-ego protocol
            # where each ego's tracker only emits tracks in its sensor field
            # of view). Without this filter, a fused track at a real car
            # position seen only by ego A counts as TP in A's eval AND FP in
            # B's eval — inflating the FP count ~2.5×.
            for ego_id, gts_for_ego in per_ego_gt_dict.items():
                ego = frame["vehicles"].get(ego_id)
                if ego is None:
                    continue
                # V2V4Real cooperative-method convention: same fused track
                # output is delivered to both egos, each ego's eval matches
                # against its own GT. Filter tracks to that ego's GT_RANGE
                # (±100/±40m) — tracks beyond can never be matched to GT
                # (which is also GT_RANGE-filtered), so counting them as
                # FPs would unfairly punish the cross-fused output.
                # The structural FP doubling within GT_RANGE (a track at
                # "real car X seen only by ego A" → TP for A, FP for B if
                # B's GT didn't annotate it) is intrinsic to per-ego eval
                # of cross-fused trackers — V2V4Real's leaderboard methods
                # all carry this same penalty.
                tracks_for_ego = [
                    rec
                    for rec in tracks_for_key
                    if _in_ego_range(rec[1], rec[2], 0.0,
                                     ego["x"], ego["y"], ego["yaw"],
                                     gt_range)
                ]
                paper_match_tape_per_ego[key].append({
                    "frame_idx": fid * 1000 + (1 if ego_id == "tesla" else 2),
                    "tracks": tracks_for_ego,
                    "gts": gts_for_ego,
                })

        iterations += 1

    # --- Build metrics dict in run_simulation()-compatible shape ---
    # Primary (un-suffixed) metrics use the PRIMARY_STRATEGY ("sort") for
    # backward-compat with downstream consumers. Per-strategy variants are
    # emitted alongside as suffixed dicts (see triple_*_by_strategy below).
    triple_global_metrics = {k: metric_totals[k] for k in STREAM_KEYS}
    triple_amota_counts = {k: amota_frames[k] for k in STREAM_KEYS}
    triple_amotp_counts = {k: amotp_frames[k] for k in STREAM_KEYS}
    triple_hota = {k: hota_accs[k].compute() for k in STREAM_KEYS}

    # Per-(range, strategy) versions (full 2x3 = 6 combos). Keyed by string
    # `f"{range_mode}_{merge_strategy}"` for JSON serializability.
    def _combo_key(rm: str, s: str) -> str:
        return f"{rm}_{s}"
    triple_global_metrics_by_combo = {
        _combo_key(rm, s): {k: metric_totals_by_combo[(rm, s)][k] for k in STREAM_KEYS}
        for rm in RANGE_MODES for s in MERGE_STRATEGIES
    }
    triple_amota_counts_by_combo = {
        _combo_key(rm, s): {k: amota_frames_by_combo[(rm, s)][k] for k in STREAM_KEYS}
        for rm in RANGE_MODES for s in MERGE_STRATEGIES
    }
    triple_amotp_counts_by_combo = {
        _combo_key(rm, s): {k: amotp_frames_by_combo[(rm, s)][k] for k in STREAM_KEYS}
        for rm in RANGE_MODES for s in MERGE_STRATEGIES
    }
    triple_hota_by_combo = {
        _combo_key(rm, s): {k: hota_accs_by_combo[(rm, s)][k].compute() for k in STREAM_KEYS}
        for rm in RANGE_MODES for s in MERGE_STRATEGIES
    }
    # Backward-compat: per-strategy variants on PRIMARY_RANGE only
    triple_global_metrics_by_strategy = {
        s: triple_global_metrics_by_combo[_combo_key(PRIMARY_RANGE, s)] for s in MERGE_STRATEGIES
    }
    triple_amota_counts_by_strategy = {
        s: triple_amota_counts_by_combo[_combo_key(PRIMARY_RANGE, s)] for s in MERGE_STRATEGIES
    }
    triple_amotp_counts_by_strategy = {
        s: triple_amotp_counts_by_combo[_combo_key(PRIMARY_RANGE, s)] for s in MERGE_STRATEGIES
    }
    triple_hota_by_strategy = {
        s: triple_hota_by_combo[_combo_key(PRIMARY_RANGE, s)] for s in MERGE_STRATEGIES
    }

    # --- V2V4Real-paper metrics (AB3DMOT protocol) ---
    # paper_*: merged GT (FP-counted, our paper protocol).
    # paper_per_ego_*: per-ego GT (V2V4Real exact, no cross-merge).
    # v2v4real_*: per-ego + FP-ignore (V2V4Real's published protocol).
    # v2v4real_merged_*: V2V FP-ignore but on merged GT (apples-to-apples
    #   with V2V4Real Table 4 since their official eval uses single-instance GT).
    triple_paper_metrics_per_ego: Dict[str, Dict] = {}
    triple_v2v4real_protocol: Dict[str, Dict] = {}

    # Per-(range, strategy) versions of the merged-GT metrics (full 2x3 grid).
    triple_paper_metrics_by_combo: Dict[str, Dict[str, Dict]] = {
        _combo_key(rm, s): {} for rm in RANGE_MODES for s in MERGE_STRATEGIES
    }
    triple_v2v4real_merged_protocol_by_combo: Dict[str, Dict[str, Dict]] = {
        _combo_key(rm, s): {} for rm in RANGE_MODES for s in MERGE_STRATEGIES
    }
    for key in STREAM_KEYS:
        triple_paper_metrics_per_ego[key] = compute_ab3dmot_metrics(paper_match_tape_per_ego[key])
        triple_v2v4real_protocol[key]     = compute_ab3dmot_metrics_iou(
            paper_match_tape_per_ego[key],
            iou_threshold=0.25,
            ignore_unmatched_fps=True,
        )
        for rm in RANGE_MODES:
            for s in MERGE_STRATEGIES:
                tape = paper_match_tape_merged_by_combo[(rm, s)][key]
                ck = _combo_key(rm, s)
                triple_paper_metrics_by_combo[ck][key] = compute_ab3dmot_metrics(tape)
                triple_v2v4real_merged_protocol_by_combo[ck][key] = compute_ab3dmot_metrics_iou(
                    tape, iou_threshold=0.25, ignore_unmatched_fps=True,
                )

    # Backward-compat aliases:
    #   *_by_strategy: per_ego range only, indexed by strategy string
    #   *_metrics:     primary combo (per_ego, sort)
    triple_paper_metrics_by_strategy = {
        s: triple_paper_metrics_by_combo[_combo_key(PRIMARY_RANGE, s)] for s in MERGE_STRATEGIES
    }
    triple_v2v4real_merged_protocol_by_strategy = {
        s: triple_v2v4real_merged_protocol_by_combo[_combo_key(PRIMARY_RANGE, s)] for s in MERGE_STRATEGIES
    }
    triple_paper_metrics = triple_paper_metrics_by_strategy[PRIMARY_STRATEGY]
    triple_v2v4real_merged_protocol = triple_v2v4real_merged_protocol_by_strategy[PRIMARY_STRATEGY]

    out = {
        "iterations": iterations,
        "recorded_steps": iterations,
        "total_active_cavs": len(scenario.meta.get("vehicles", [])),
        # CAV/CIS metrics aren't meaningful in V2V4Real (everyone's a CAV);
        # provide empty dicts so RunResult.load_from_csv doesn't choke.
        "total_cav_metrics": _create_metrics_dict(),
        "total_cis_metrics": _create_metrics_dict(),
        "total_global_metrics": _create_metrics_dict(),
        "cav_amota_frames_count": 0,
        "cis_amota_frames_count": 0,
        "global_amota_frames_count": iterations,
        "cav_amotp_frames_count": 0,
        "cis_amotp_frames_count": 0,
        "global_amotp_frames_count": iterations,
        "avg_step_time": (_time.time() - t0_wall) / max(iterations, 1),
        "avg_cis_detection_time": 0.0,
        "avg_cav_detection_time": 0.0,
        "avg_global_fusion_time": 0.0,
        "perception_scores": {},
        "perception_anomalies": {},
        "perception_summary": {},
        "global_hota": {"hota": 0.0, "deta": 0.0, "assa": 0.0},

        # Triple-fusion outputs (the meat).
        "triple_global_metrics": triple_global_metrics,
        "triple_global_amota_frames_count": triple_amota_counts,
        "triple_global_amotp_frames_count": triple_amotp_counts,
        "triple_global_hota": triple_hota,
        # Per-frame match tapes for selected streams (debug overlay use only).
        # Each entry: list of {frame_idx, tracks: [(id,x,y,score), ...], gts: [...]}.
        "match_tape_per_stream": {
            k: paper_match_tape_merged[k] for k in record_tape_for
            if k in paper_match_tape_merged
        },
        # Full per-ego match tape for every stream (the exact input to
        # compute_ab3dmot_metrics_iou) — only when dump_match_tapes is set, so
        # normal runs don't pay the serialization cost. Enables cheap offline
        # AMOTP re-scoring (see scripts/rescore_v2v4real_amotp_from_tapes.py).
        **({"match_tape_per_ego": paper_match_tape_per_ego} if dump_match_tapes else {}),
        # V2V4Real-paper metrics (AB3DMOT protocol). One entry per stream
        # key, each a dict from compute_ab3dmot_metrics. Two views:
        #   triple_paper_metrics         — merged GT (cross-fusion natural).
        #   triple_paper_metrics_per_ego — per-ego GT (V2V4Real exact).
        "triple_paper_metrics": triple_paper_metrics,
        "triple_paper_metrics_per_ego": triple_paper_metrics_per_ego,
        # V2V4Real-protocol metrics (DMSTrack KITTI-evaluator behavior:
        # IoU @ 0.25 + ignore_unmatched_fps=True). Apples-to-apples with
        # V2V4Real Table 4 numbers.
        "triple_v2v4real_protocol": triple_v2v4real_protocol,
        # Same protocol but on merged GT (single-instance via spatial dedup).
        # Closer to AB3DMOT's official eval which uses single-instance KITTI GT.
        "triple_v2v4real_merged_protocol": triple_v2v4real_merged_protocol,

        # Per-merge-strategy versions of the merged-GT metrics.
        # Strategies: "sort" (alphabetical/anchor pick — primary, default),
        #             "cav0"  (tesla-only — V2V4Real official convention),
        #             "average" (averaged across cluster — most rigorous).
        # Use these to compare protocols side-by-side without re-running.
        "triple_paper_metrics_by_strategy":           triple_paper_metrics_by_strategy,
        "triple_v2v4real_merged_protocol_by_strategy": triple_v2v4real_merged_protocol_by_strategy,
        "triple_global_hota_by_strategy":             triple_hota_by_strategy,
        "triple_global_metrics_by_strategy":          triple_global_metrics_by_strategy,
        "triple_global_amota_frames_count_by_strategy": triple_amota_counts_by_strategy,
        "triple_global_amotp_frames_count_by_strategy": triple_amotp_counts_by_strategy,
        # Per-(range_mode, merge_strategy) full 2x3 = 6-combo grid.
        # Keys: "per_ego_sort", "per_ego_cav0", "per_ego_average",
        #       "world_sort",   "world_cav0",   "world_average".
        # Use these for the master-table dump comparing all eval protocols.
        "triple_paper_metrics_by_combo":             triple_paper_metrics_by_combo,
        "triple_v2v4real_merged_protocol_by_combo":  triple_v2v4real_merged_protocol_by_combo,
        "triple_global_hota_by_combo":               triple_hota_by_combo,
        "triple_global_metrics_by_combo":            triple_global_metrics_by_combo,
        "triple_global_amota_frames_count_by_combo": triple_amota_counts_by_combo,
        "triple_global_amotp_frames_count_by_combo": triple_amotp_counts_by_combo,

        # Scenario provenance for downstream aggregation.
        "scenario_id": scenario.meta.get("scenario_id"),
        "scenario_split": scenario.meta.get("split"),
        "scenario_dir": str(scenario.scenario_dir),
        "wall_time_seconds": _time.time() - t0_wall,
    }
    return out
