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

# Same canonical 30 keys (in stable order) as traci_interface.run_simulation.
STREAM_KEYS: List[str] = [
    _stream_key(fname, cov) for fname, _ in FILTER_CLASSES for cov in COV_MODES
]

# Cleanup time (matches config_templates.py:70). Stale tracks > this many
# seconds without an update get dropped.
DEFAULT_CLEANUP_TIME = 0.5

# Stable per-vehicle int IDs (used for fusion's source_participant_id encoding
# and for tracking which CAV produced each measurement; trust scoring is off).
VEHICLE_PARTICIPANT_IDS: Dict[str, int] = {
    "astuff": 1,
    "tesla":  2,
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


def _dedup_gt_for_frame(
    gt_rows: List[Dict],
    ego_poses: Optional[List[Tuple[float, float]]] = None,
    spatial_gate_m: float = 2.0,
    ego_exclusion_gate_m: float = 0.0,
) -> List[cmr_ground_truth.GroundTruthObject]:
    """
    Build a spatially-deduplicated GT list for one frame ("merged" view).

    V2V4Real annotation conventions (discovered while debugging):
      1. ~84% of GT rows in a scenario have ``ass_id=-1`` ("not associated
         across vehicles"). Earlier dedup-by-ass_id collapsed all of these to
         a single entry, dropping the majority of legitimate GT. Fix: spatial
         clustering by ``spatial_gate_m`` (2 m) for *every* row, with
         positive ass_ids prioritised when multiple rows fall in one cluster.
      2. Each ego is annotated as a ``car`` GT in the *other* ego's frame.
         V2V4Real treats these as legitimate tracking targets (the other
         vehicle is a real car visible in the scene), so we do NOT exclude
         them by default. Pass ``ego_exclusion_gate_m`` > 0 to drop them
         (for "self-only" eval variants).
    """
    if not gt_rows:
        return []

    rows = list(gt_rows)

    # 1. Drop rows that are within `ego_exclusion_gate_m` of any ego pose
    #    (the other ego is annotated in our GT but isn't a tracking target).
    if ego_poses:
        kept = []
        for r in rows:
            is_ego = False
            for ex, ey in ego_poses:
                if (r["x"] - ex) ** 2 + (r["y"] - ey) ** 2 < ego_exclusion_gate_m ** 2:
                    is_ego = True
                    break
            if not is_ego:
                kept.append(r)
        rows = kept

    # 2. Spatial dedup. Greedy: prefer rows with positive ass_id, then break
    #    ties deterministically by (vehicle_id, obj_id) so the same ego's row
    #    always wins the same cluster across frames — required for stable GT IDs.
    rows_sorted = sorted(rows, key=lambda r: (r["ass_id"] < 0, r["ass_id"],
                                              r["vehicle_id"], int(r["obj_id"])))
    seen: Dict[Tuple[float, float], Dict] = {}
    kept_rows: List[Dict] = []
    for r in rows_sorted:
        merged = False
        for k in seen.keys():
            if (r["x"] - k[0]) ** 2 + (r["y"] - k[1]) ** 2 < spatial_gate_m ** 2:
                merged = True
                break
        if not merged:
            seen[(r["x"], r["y"])] = r
            kept_rows.append(r)

    # 3. Stable, unique vehicle_ids for HOTA's id-based association.
    #    Use ass_id when positive (cross-frame stable), otherwise use
    #    (vehicle_id, obj_id) — obj_id is stable within one ego across all
    #    frames, and the deterministic sort above ensures the same ego always
    #    wins each cluster, so the GT ID is stable even for fast-moving vehicles.
    #    (The old 1 m-grid position quantisation aliased to a new ID every frame
    #    at typical driving speeds, producing hundreds of spurious IDS.)
    seen = {}
    for r in kept_rows:
        if r["ass_id"] >= 0:
            r["_track_id"] = int(r["ass_id"])
        else:
            r["_track_id"] = hash((r["vehicle_id"], int(r["obj_id"]))) & 0x7FFFFFFF
    out: List[cmr_ground_truth.GroundTruthObject] = []
    for r in kept_rows:
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
        out.append(cmr_ground_truth.GroundTruthObject(
            vehicle_id=r["_track_id"],
            vehicle_type=r["label"],
            position=[cx, cy],
            velocity_vector=[0.0, 0.0],
            bounding_box=corners,
            angle_rad=ang,
            width=r["width"],
            length=r["length"],
        ))
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
                          lifecycle_mode: str = "legacy",
                          ab3dmot_min_hits: int = 3,
                          ab3dmot_max_age: int = 2,
                          enable_inactive_preservation: bool = False,
                          inactive_grace_s: float = 1.0,
                          confirm_log_odds: Optional[float] = None,
                          kill_log_odds: Optional[float] = None,
                          log_lr_miss: Optional[float] = None,
                          mahal_weight: float = 0.6,
                          iou_weight: float = 0.4,
                          mahal_gate: float = 13.82,
                          iou_gate: float = 0.0,
                          tape_score_miss_decay: Optional[float] = None,
                          ) -> Dict[str, sensor_fusion.Fusion]:
    """Create the 30 fusion streams keyed by STREAM_KEYS.

    p_tp_birth_gate (Phase A): if set, every fusion stream applies the same
    P(TP)-based birth gate. None = no gate.
    lifecycle_mode (Phase B): "legacy" (count-based) or "log_odds"
    (IPDA-style S_t-based confirm/kill, with miss-frame decay)."""
    streams: Dict[str, sensor_fusion.Fusion] = {}
    # Pull defaults from sensor_fusion module if caller passed None.
    if confirm_log_odds is None:
        confirm_log_odds = sensor_fusion.DEFAULT_CONFIRM_LOG_ODDS
    if kill_log_odds is None:
        kill_log_odds = sensor_fusion.DEFAULT_KILL_LOG_ODDS
    if log_lr_miss is None:
        log_lr_miss = sensor_fusion.DEFAULT_LOG_LR_MISS
    for filter_name, filter_cls in FILTER_CLASSES:
        for cov in COV_MODES:
            key = _stream_key(filter_name, cov)
            streams[key] = sensor_fusion.Fusion(
                sensor_fusion.max_id,
                use_trust_scoring=False,
                passthrough_covariance=False,
                static_matching_cov=static_match_cov,
                filter_class=filter_cls,
                p_tp_birth_gate=p_tp_birth_gate,
                lifecycle_mode=lifecycle_mode,
                ab3dmot_min_hits=ab3dmot_min_hits,
                ab3dmot_max_age=ab3dmot_max_age,
                enable_inactive_preservation=enable_inactive_preservation,
                inactive_grace_s=inactive_grace_s,
                confirm_log_odds=confirm_log_odds,
                kill_log_odds=kill_log_odds,
                log_lr_miss=log_lr_miss,
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

    # --- Localizer (one instance reused across modes; covariance only depends
    # on velocity + yaw, not on filter type) ---
    dummy_err = ErrorPackage(ErrorType.LOCALIZATION, 0)
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
    ab3dmot_min_hits = int(cfg.get("ab3dmot_min_hits", 3))
    ab3dmot_max_age  = int(cfg.get("ab3dmot_max_age",  2))
    # Phase B.5: inactive-track preservation. Default OFF for clean A/B.
    enable_inactive_preservation = bool(cfg.get("enable_inactive_preservation", False))
    inactive_grace_s = float(cfg.get("inactive_grace_s", 1.0))
    confirm_log_odds_cfg = cfg.get("confirm_log_odds", None)
    kill_log_odds_cfg    = cfg.get("kill_log_odds", None)
    log_lr_miss_cfg      = cfg.get("log_lr_miss", None)
    # Pluggable Hungarian cost: default = our hybrid (Mahalanobis + IoU).
    # Pure-IoU (AB3DMOT-style) → mahal_weight=0, iou_weight=1, large gate.
    mahal_weight_cfg = float(cfg.get("mahal_weight", 0.6))
    iou_weight_cfg   = float(cfg.get("iou_weight",   0.4))
    mahal_gate_cfg   = float(cfg.get("mahal_gate",   13.82))
    iou_gate_cfg     = float(cfg.get("iou_gate",     0.0))
    tape_score_miss_decay_cfg = cfg.get("tape_score_miss_decay", None)
    streams = _build_fusion_streams(static_match_cov,
                                    p_tp_birth_gate=p_tp_birth_gate,
                                    lifecycle_mode=lifecycle_mode,
                                    ab3dmot_min_hits=ab3dmot_min_hits,
                                    ab3dmot_max_age=ab3dmot_max_age,
                                    enable_inactive_preservation=enable_inactive_preservation,
                                    inactive_grace_s=inactive_grace_s,
                                    confirm_log_odds=confirm_log_odds_cfg,
                                    kill_log_odds=kill_log_odds_cfg,
                                    log_lr_miss=log_lr_miss_cfg,
                                    mahal_weight=mahal_weight_cfg,
                                    iou_weight=iou_weight_cfg,
                                    mahal_gate=mahal_gate_cfg,
                                    iou_gate=iou_gate_cfg,
                                    tape_score_miss_decay=tape_score_miss_decay_cfg)
    hota_accs: Dict[str, HotaAccumulator] = {k: HotaAccumulator() for k in STREAM_KEYS}
    metric_totals: Dict[str, Dict] = {k: _create_metrics_dict() for k in STREAM_KEYS}
    amota_frames: Dict[str, int] = {k: 0 for k in STREAM_KEYS}
    amotp_frames: Dict[str, int] = {k: 0 for k in STREAM_KEYS}
    # V2V4Real-paper-metric tape: per-stream list of {frame_idx, tracks, gts}.
    # tracks = [(track_id, x, y, score), ...]; gts = [(gt_id, x, y), ...].
    # Two views — "merged" (spatial dedup, our cross-fusion natural eval) and
    # "per_ego" (V2V4Real's exact convention: each ego's own GT, no cross-merge).
    paper_match_tape_merged:  Dict[str, List[Dict]] = {k: [] for k in STREAM_KEYS}
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
            def _keep(d, _ego=ego_pose):
                if vehicle_classes is not None and d["label"] not in vehicle_classes:
                    return False
                thr = class_thresholds.get(d["label"], score_threshold)
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

            # V2X self-report: this ego broadcasts its own RTK pose as a
            # tight-covariance "detection" of itself, prepended to the
            # detector-output list. Both egos see this in their fused track
            # set, so the asymmetric "tesla can't detect astuff" problem is
            # bypassed by astuff *telling* tesla where it is.
            if self_report_egos:
                self_loc_cov = loc.get_localization_covariance(velocity, ego_yaw)
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
                for cov_mode in COV_MODES:
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
                base_obj = _detection_with_flat_covariance(raw_det, baseline_cov)
                base_obj.p_tp = 1.0
                base_obj.det_score = float(det_row["score"])
                per_cov_observations["baseline"].append(base_obj)
                # static / linear / quadratic / polar: per-detection cov from GPEM
                for cov in ("static", "gpem_linear", "gpem_quadratic", "gpem_polar"):
                    obj = _detection_with_covariance_from_model(
                        raw_det, ego_x, ego_y, ego_yaw,
                        ems_for_vehicle[cov], loc, velocity,
                        variance_floor=variance_floor,
                        use_mae_covariance=False,
                    )
                    obj.p_tp = max(1e-6, ems_for_vehicle[cov].detection_probability(
                        distance, angle_deg,
                    ))
                    obj.det_score = float(det_row["score"])
                    per_cov_observations[cov].append(obj)

            for filter_name, _ in FILTER_CLASSES:
                for cov in COV_MODES:
                    key = _stream_key(filter_name, cov)
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
            # Tape entry: (id, x, y, w, l, yaw, score). Extra bbox fields are
            # ignored by the center-distance evaluator but read by the
            # V2V4Real-protocol IoU evaluator.
            track_records = [
                (track.id, track.x, track.y,
                 track.width, track.length, track.angle,
                 float(track.tape_score))
                for track in fusion.tracked_list
                if emit_predicate(track)
            ]
            per_stream_track_records[key] = track_records

        # Build BOTH GT views for this frame, with V2V4Real's GT_RANGE
        # rectangular ego-frame filter applied:
        #
        #   per_ego_gt_dict: each ego's GT, filtered to that ego's own GT_RANGE.
        #     Matches V2V4Real's exact eval semantics — base_postprocessor uses
        #     `cav_lidar_range if train else GT_RANGE` and filters per-ego.
        #   merged_gt_objs:  spatial dedup across egos AFTER per-ego range
        #     filtering — so a GT in scope for either ego is kept once.
        #
        # Egos themselves are legitimate tracking targets per V2V4Real
        # annotation convention, so no ego self-exclusion.
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
        # Build the per-ego GT records (V2V4Real-exact view) from the
        # filtered set, namespacing IDs by ego.
        for g in per_ego_gt_filtered:
            ego = g["vehicle_id"]
            gid = (ego, int(g["ass_id"]) if g["ass_id"] >= 0 else -int(g["obj_id"]))
            gid_int = hash(gid) & 0x7FFFFFFF
            # Per-ego tape entry: (id, x, y, w, l, yaw). Bbox fields used by
            # IoU evaluator; ignored by center-distance.
            per_ego_gt_dict.setdefault(ego, []).append(
                (gid_int, g["x"], g["y"], g["width"], g["length"], g["yaw"])
            )
        # Merged view: spatial dedup over the already-range-filtered set.
        gt_objs_merged = _dedup_gt_for_frame(per_ego_gt_filtered)
        gt_records_merged = [
            (g.vehicle_id, g.centroid[0], g.centroid[1],
             g.dimensions[0], g.dimensions[1], g.angle)
            for g in gt_objs_merged
        ]

        # Update HOTA + per-frame AMOTA/AMOTP for each stream + paper tapes.
        for key in STREAM_KEYS:
            fused_dets = per_stream_dets[key]
            tracks_for_key = per_stream_track_records[key]

            # CMR HOTA + per-frame AMOTA/AMOTP — uses merged GT.
            hota_accs[key].update(fused_dets, gt_objs_merged)

            mota = amota.calculate_amota(fused_dets, gt_objs_merged)
            if mota is not None:
                metric_totals[key]["amota"] += mota
                amota_frames[key] += 1

            motp = amota.calculate_amotp(fused_dets, gt_objs_merged)
            if fused_dets and gt_objs_merged and motp > 0.0:
                metric_totals[key]["amotp"] += motp
                amotp_frames[key] += 1

            # AB3DMOT match tape — merged view.
            # Filter tracks to the union of all egos' GT_RANGE so that dying
            # ghost tracks outside any ego's sensing zone don't count as FPs.
            ego_poses_for_frame = [
                (e["x"], e["y"], e["yaw"])
                for e in frame["vehicles"].values()
            ]
            tracks_merged = [
                rec for rec in tracks_for_key
                if any(_in_ego_range(rec[1], rec[2], 0.0, ex, ey, eyaw, gt_range)
                       for ex, ey, eyaw in ego_poses_for_frame)
            ]
            paper_match_tape_merged[key].append({
                "frame_idx": fid,
                "tracks": tracks_merged,
                "gts": gt_records_merged,
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
    triple_global_metrics = {k: metric_totals[k] for k in STREAM_KEYS}
    triple_amota_counts = {k: amota_frames[k] for k in STREAM_KEYS}
    triple_amotp_counts = {k: amotp_frames[k] for k in STREAM_KEYS}
    triple_hota = {k: hota_accs[k].compute() for k in STREAM_KEYS}

    # --- V2V4Real-paper metrics (AB3DMOT protocol) ---
    # Computed per-stream against BOTH GT views:
    #   triple_paper_metrics          — merged GT (spatial dedup, our default).
    #   triple_paper_metrics_per_ego  — V2V4Real exact (per-ego GT, no merge).
    # AB3DMOT spec: 40 recall thresholds, 2 m center-distance gate, AMOTP↑.
    triple_paper_metrics: Dict[str, Dict] = {}
    triple_paper_metrics_per_ego: Dict[str, Dict] = {}
    # V2V4Real-protocol metrics (DMSTrack-style): IoU @ 0.25 + FP-ignore via
    # min_height filter. See docs/V2V4REAL_REPRODUCIBILITY_NOTES.md "smoking
    # gun" section. This is what the published Table 4 numbers actually
    # measure — apples-to-apples comparison with their leaderboard.
    triple_v2v4real_protocol: Dict[str, Dict] = {}
    triple_v2v4real_merged_protocol: Dict[str, Dict] = {}
    for key in STREAM_KEYS:
        # paper_*: merged GT (spatial dedup, stable IDs via vehicle_id/obj_id).
        # paper_per_ego_*: per-ego GT, for reference.
        # v2v4real_*: per-ego + FP-ignore — matches DMSTrack's published protocol.
        # v2v4real_merged_*: V2V protocol (FP-ignore) but on merged GT —
        #   apples-to-apples with V2V4Real Table 4 when single-ego export is used,
        #   since V2V4Real's official KITTI eval uses single-instance GT (≈ merged).
        triple_paper_metrics[key]         = compute_ab3dmot_metrics(paper_match_tape_merged[key])
        triple_paper_metrics_per_ego[key] = compute_ab3dmot_metrics(paper_match_tape_per_ego[key])
        triple_v2v4real_protocol[key]     = compute_ab3dmot_metrics_iou(
            paper_match_tape_per_ego[key],
            iou_threshold=0.25,
            ignore_unmatched_fps=True,
        )
        triple_v2v4real_merged_protocol[key] = compute_ab3dmot_metrics_iou(
            paper_match_tape_merged[key],
            iou_threshold=0.25,
            ignore_unmatched_fps=True,
        )

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

        # Scenario provenance for downstream aggregation.
        "scenario_id": scenario.meta.get("scenario_id"),
        "scenario_split": scenario.meta.get("split"),
        "scenario_dir": str(scenario.scenario_dir),
        "wall_time_seconds": _time.time() - t0_wall,
    }
    return out
