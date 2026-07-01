import math
import random
from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from scipy.optimize import linear_sum_assignment
import bisect
from typing import Dict, Tuple, List, Optional

import utils
from sensor import DetectedObject
from filters.kalman_ctrv import ResizableKalman


max_id = 10000


# IPDA log-odds defaults (Phase B). All in natural log units.
#   confirm: P(real) > 20/21 ≈ 0.952
#   kill:    P(real) <  1/21 ≈ 0.048
DEFAULT_CONFIRM_LOG_ODDS = math.log(20.0)        #  +2.996
DEFAULT_KILL_LOG_ODDS    = math.log(1.0 / 20.0)  #  −2.996
# log(P_miss / (1 − P_miss)) — per-frame miss-evidence decay applied to any
# matched-but-unmatched-this-frame track. P_miss = 0.30 → −0.847; mild.
DEFAULT_LOG_LR_MISS      = math.log(0.30 / 0.70)
# Clamp on individual log-likelihood-ratio updates to keep one strong
# detection from dominating the entire history.
LOG_LR_CLAMP = 6.0
# Numerical safety on Bayesian log-odds (P_TP exactly 0 or 1 → ±inf).
_P_EPS = 1e-6
# Per-frame multiplicative decay on `tape_score` when a track is unmatched.
# 0.5 = rapid decay (drops to ~0.06 in 4 frames; stale tracks coast out
#       of the AMOTA recall sweep). EMPIRICALLY THE BEST DEFAULT for our
#       pipeline — confirmed in single-scenario A/B (decay=0.5 beats
#       0.7, 0.9, 1.0 across every fusion stream).
# 1.0 = no decay (matches AB3DMOT's behaviour where coasting tracks keep
#       their latest matched det.score — but our pipeline emits noisier
#       coasting tracks that benefit from being demoted in the recall
#       sort). Available as opt-in for ablation.
TAPE_SCORE_MISS_DECAY = 0.5
# Phase B.5 — inactive-track preservation grace period (seconds). When a
# track's S_t falls below kill_log_odds, instead of deleting we mark it
# inactive and keep it in tracked_list for this many seconds. During that
# window the Hungarian matcher still considers it as a candidate; a fresh
# detection that falls within its Mahalanobis gate revives the same ID
# (S_t bumped above confirm threshold). Beyond the window the track is
# actually deleted. 1.0 s ≈ 10 frames at V2V4Real's 10Hz — long enough to
# bridge typical occlusions, short enough that stale tracks don't pollute
# the matching gate forever.
DEFAULT_INACTIVE_GRACE_S = 1.0

# EMA gain for the per-track vertical (z, height) smoother. CMR fuses in 2D, so
# z/height ride alongside the planar state as a fixed-gain 1D smoother (decoupled
# from the 2D filter → BEV fusion/metrics unchanged): new = old + gain·(meas − old).
# A fixed gain (not GPEM inverse-variance weighting) is used DELIBERATELY for this
# first pass. The GPEM DETECTOR vertical std is fit in each sensor's own frame and
# is correct there (veh 0.085–0.14 ≈ measured). The EXTRA z spread the roadside-inf
# source shows in the WORLD frame (measured 0.189 vs 0.085 sensor-fit) is
# LOCALIZATION/transform error — the inf pose + system_error_offset residual
# projected through the tilted mount (horizontal error rotates into z). That belongs
# in the LOCALIZER covariance (loc_cov, added to perc_cov in the cov-helper), which
# is intentionally ~0 here (localizer_name="rtk_v2v4real" stand-in; DAIR localization
# is S2/S3 scope). So inverse-variance weighting on the detector-only std over-trusts
# inf and underperforms; the fixed-gain smoother is robust to the missing loc term.
# Once DAIR localization is modeled, GPEM-R vertical fusion falls out for free.
# 0.5 balances denoising vs tracking gentle z drift. w/l ARE GPEM-variance-fused
# (their stds are frame-stable) in ResizableKalman/CI — see kalman_ctrv.py:612.
VERTICAL_EMA_GAIN = 0.5


def _as_float_or_none(v):
    """float(v), or None if v is None/NaN/non-numeric (detector didn't report it)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _blend_vertical(current, measurement):
    """EMA-blend a vertical scalar (z or height). First valid obs initialises;
    a missing measurement leaves the estimate unchanged."""
    m = _as_float_or_none(measurement)
    if m is None:
        return current
    if current is None:
        return m
    return current + VERTICAL_EMA_GAIN * (m - current)


def _logit(p: float) -> float:
    """log(p / (1 − p)) clamped against the [eps, 1 − eps] range."""
    p = max(_P_EPS, min(1.0 - _P_EPS, p))
    return math.log(p / (1.0 - p))


def _source_id_of(detected_object) -> int:
    """Recover the source participant id of a detection.

    Fusion.processDetectionFrame encodes the source as
    `det.vehicle_id = source_participant_id * max_id + idx`, so the
    participant id (1=astuff, 2=tesla in the V2V4Real replay; 0 for
    single-detector / sim runs) is `vehicle_id // max_id`. Detections
    whose vehicle_id is unset/None resolve to source 0 — the same key the
    single-detector path uses, so the per-source maps degenerate cleanly.
    """
    vid = getattr(detected_object, "vehicle_id", None)
    if vid is None:
        return 0
    try:
        return int(vid) // max_id
    except (TypeError, ValueError):
        return 0


class MatchClass:
    def __init__(self, id, x, y, covariance, dx, dy, d_confidence, confidence, trust_score, object_type, time, width, length, angle, width_std=0.5, length_std=0.5, yaw_variance=None):
        self.x = x
        self.y = y
        self.covariance = covariance
        self.dx = dx
        self.dy = dy
        self.velocity_confidence = d_confidence
        self.type = object_type
        self.last_measurement = time
        self.id = id
        self.confidence = confidence
        self.trust_score = trust_score
        self.width = width
        self.length = length
        self.angle = angle
        self.width_std = width_std
        self.length_std = length_std
        self.yaw_variance = yaw_variance  # MSE for heading from GPEM




class GlobalTracked:
    def __init__(self, detected_object, time, track_id, fusion_mode, use_trust_scoring=True, passthrough_covariance=False, filter_class=None,
                 confirm_log_odds=DEFAULT_CONFIRM_LOG_ODDS,
                 confirm_log_odds_by_source=None,
                 log_lr_miss_by_source=None):
        self.x = detected_object.centroid[0]
        self.y = detected_object.centroid[1]
        self.passthrough_covariance = passthrough_covariance
        self._filter_class = filter_class if filter_class is not None else ResizableKalman
        self.dx = 0
        self.dy = 0
        self.error_covariance = np.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype='float')
        self.last_measurement = time
        self.last_update = time
        self.id = track_id
        self.idx = 0
        self.min_size = 0.5
        self.track_count = 0
        self.d_covariance = np.array([[2.0, 0.0], [0.0, 2.0]], dtype='float')
        self.match_list = []
        self.fusion_steps = 0
        self.error_monitor = []
        self.num_trackers = 0

        # IPDA log-odds existence score (Phase B). Initialized from the
        # spawning detection's calibrated P(TP); for detections without
        # calibration (p_tp absent), defaults to logit(0.5) = 0 — the
        # legacy "neutral start, gain confidence by re-detection" behavior.
        self.S_t = _logit(getattr(detected_object, "p_tp", 0.5))

        # --- Per-source LIFECYCLE BLEND state -------------------------------
        # Mixed-detector configs (astuff=PP, tesla=CPft) route per-detection
        # economics per source already; the blend extends that to track-level
        # lifecycle. Each track records:
        #   * birth_source_id  — participant id of the detection that spawned
        #                        it (vehicle_id // max_id). Its confirm
        #                        threshold is inherited from this source.
        #   * source_hits      — Counter of source ids over every matched
        #                        update (spawn counts as the first hit). Used
        #                        to blend log_lr_miss by the source mix.
        # The per-source maps are dicts {source_id -> float}; a None map or a
        # missing key falls back to the global scalar, so single-detector
        # runs degenerate to the legacy scalar path exactly.
        self._confirm_log_odds_by_source = confirm_log_odds_by_source
        self._log_lr_miss_by_source = log_lr_miss_by_source
        self.birth_source_id = _source_id_of(detected_object)
        self.source_hits = Counter()
        self.source_hits[self.birth_source_id] += 1
        # Track-own confirm threshold, inherited from the birthing source.
        if confirm_log_odds_by_source:
            self.confirm_log_odds = confirm_log_odds_by_source.get(
                self.birth_source_id, confirm_log_odds)
        else:
            self.confirm_log_odds = confirm_log_odds

        # AMOTA-tape score: latest matched detection's confidence. Decayed
        # multiplicatively on miss-frames (TAPE_SCORE_MISS_DECAY) so a track
        # that goes stale ranks low in V2V4Real's recall sweep — analogous
        # to how Kalman covariance inflates during prediction.
        self.tape_score = float(getattr(detected_object, "det_score", 1.0))

        # AB3DMOT-style counters (lifecycle_mode="ab3dmot"). Track confirms
        # after `min_hits` matched frames, dies after `max_age` consecutive
        # missed frames. Spawn counts as the first hit.
        self.hits = 1
        self.misses_since_match = 0

        # Phase B.5 — inactive-track preservation. is_inactive=True means
        # this track has had S_t collapse but Fusion is keeping it around
        # for `inactive_grace_s` so that a fresh detection landing in its
        # Mahalanobis gate can revive the same ID (no ID switch on brief
        # occlusions). Tracks emit only when is_inactive is False.
        self.is_inactive = False
        self.inactive_since = None  # set when a track becomes inactive
        self.revive_count = 0       # stats: # of times this track was revived
        self.width = detected_object.dimensions[0]
        self.length = detected_object.dimensions[1]
        self.angle = detected_object.angle

        # Vertical state (z gravity-center, box height). CMR fuses in 2D BEV, so
        # z/height are not in any filter's state vector; instead each track keeps
        # a lightweight EMA of the matched detections' z/height (a steady-state 1D
        # smoother, decoupled from the 2D filter — BEV fusion/metrics are unchanged).
        # This denoises the per-frame detector z/height that a 2D tracker would
        # otherwise have to take at face value, giving genuine 3D track boxes.
        self.z = _as_float_or_none(getattr(detected_object, "z", None))
        self.height = _as_float_or_none(getattr(detected_object, "height", None))

        # Type voting: track counts of each type seen
        # Uses a sliding window approach with exponential decay
        self.type_votes = {}  # {type_value: vote_count}
        self.type = detected_object.type  # Current best estimate
        self._update_type_vote(detected_object.type)

        # Trupercept stuff
        self.trupercept_list = []

        # Add this first match
        new_match = MatchClass(detected_object.vehicle_id, detected_object.centroid[0], detected_object.centroid[1], detected_object.error_covariance,
                               detected_object.velocity_vector[0], detected_object.velocity_vector[1], detected_object.error_covariance, 1.0, 1.0,
                               detected_object.type, time,
                               detected_object.dimensions[0], detected_object.dimensions[1], detected_object.angle,
                               detected_object.width_std, detected_object.length_std,
                               getattr(detected_object, 'yaw_variance', None))
        self.match_list.append(new_match)

        # Kalman stuff
        self.fusion_mode = fusion_mode
        self.kalman = self._filter_class(time, self.x, self.y, fusion_mode,
                                         initial_width=self.width, initial_length=self.length,
                                         use_trust_scoring=use_trust_scoring,
                                         passthrough_covariance=passthrough_covariance)
    
    def _update_type_vote(self, detected_type):
        """Update type voting with a new detection's type classification."""
        if detected_type is None:
            return
        
        # Decay existing votes (exponential decay for recency weighting)
        decay_factor = 0.95
        for t in self.type_votes:
            self.type_votes[t] *= decay_factor
        
        # Add vote for this detection's type
        if detected_type in self.type_votes:
            self.type_votes[detected_type] += 1.0
        else:
            self.type_votes[detected_type] = 1.0
        
        # Determine winning type (majority vote with recency weighting)
        if self.type_votes:
            self.type = max(self.type_votes, key=self.type_votes.get)

    def update(self, detected_object, time):
        new_match = MatchClass(detected_object.vehicle_id, detected_object.centroid[0], detected_object.centroid[1], detected_object.error_covariance,
                               detected_object.velocity_vector[0], detected_object.velocity_vector[1], detected_object.error_covariance, 1.0, 1.0,
                               detected_object.type, time,
                               detected_object.dimensions[0], detected_object.dimensions[1], detected_object.angle,
                               detected_object.width_std, detected_object.length_std,
                               getattr(detected_object, 'yaw_variance', None))
        self.match_list.append(new_match)

        # Update type voting
        self._update_type_vote(detected_object.type)

        self.last_measurement = time

        # AB3DMOT counters
        self.hits += 1
        self.misses_since_match = 0

        # Per-source lifecycle blend: record which source fed this update so
        # the miss-decay can be weighted by the track's source mix.
        self.source_hits[_source_id_of(detected_object)] += 1

        # Refresh tape_score with the latest matched detection's confidence.
        # Falls back to the prior tape_score if det_score not set (keeps
        # legacy/sim pipeline backward-compatible).
        self.tape_score = float(getattr(detected_object, "det_score", self.tape_score))

        # Blend the matched detection's z/height into the track's vertical EMA
        # (denoise; first observation initialises). Decoupled from the 2D filter.
        self.z = _blend_vertical(self.z, getattr(detected_object, "z", None))
        self.height = _blend_vertical(self.height, getattr(detected_object, "height", None))

    # Gets our position in an array form so we can use it in the BallTree
    def getPosition(self):
        # ORDER: [x, y, width, length, angle]
        # REVERTED: Matching logic expects width first
        return [
            [self.x, self.y, self.width, self.length, self.angle]
        ]

    def getPositionPredicted(self, timestamp):
        # Call the Kalman filter's prediction method
        predicted_x, predicted_y, a, b, phi = self.kalman.getKalmanPred(timestamp)

        # Dynamically adjust the bounding box size based on Kalman filter's uncertainty
        # Use 2 standard deviations for the matching gate to help catch detections 
        # that might be slightly offset from prediction
        adjusted_width = self.width + (a * 2.0)
        adjusted_length = self.length + (b * 2.0)

        # Return the predicted position with the dynamically adjusted width and length
        # ORDER: [x, y, width, length, angle]
        # REVERTED: Matching logic expects width first
        return [
            [predicted_x, predicted_y, adjusted_width, adjusted_length, self.angle]
        ]
    
    def getPositionPredictedWithCovariance(self, timestamp):
        """
        Get predicted position with full covariance matrix for Hungarian matching.
        Returns position, bounding box, and prediction covariance.
        """
        predicted_x, predicted_y, P_pred = self.kalman.getKalmanPredWithCovariance(timestamp)
        
        # Ellipse parameters for bbox adjustment
        a, b, phi = utils.ellipsify(P_pred[0:2, 0:2], 2.0)
        adjusted_width = self.width + (a * 2.0)
        adjusted_length = self.length + (b * 2.0)
        
        return {
            'x': predicted_x,
            'y': predicted_y,
            'P_pred': P_pred,
            'bbox': [predicted_x, predicted_y, adjusted_width, adjusted_length, self.angle],
            'base_bbox': [predicted_x, predicted_y, self.width, self.length, self.angle]
        }

    def fusion(self, time, monitor, participant_trust_scores: Dict[int, float] = None):
        """
        Fuse all matches for this track using Kalman filter.
        
        Args:
            time: Current timestamp
            monitor: Whether to collect cooperative monitoring data
            participant_trust_scores: Dict mapping participant_id to trust score (for global fusion)
        """
        # Measurement ordering strategy depends on covariance knowledge:
        # - Baseline (flat R): randomize — no quality info, prevents arrival-order bias
        # - GPEM (per-measurement R): sort by trace(R) ascending — process the
        #   tightest measurement first for largest Kalman gain, reflecting the
        #   practical advantage of knowing per-measurement quality
        if self.passthrough_covariance:
            ordered = sorted(self.match_list,
                             key=lambda m: np.trace(m.covariance) if m.covariance is not None else float('inf'),
                             reverse=True)
        else:
            ordered = list(self.match_list)
            random.shuffle(ordered)
        self.kalman.fusion(ordered, time, monitor, participant_trust_scores)
        self.x = self.kalman.x
        self.y = self.kalman.y
        self.error_covariance = self.kalman.error_covariance
        self.dx = self.kalman.dx
        self.dy = self.kalman.dy
        self.d_covariance = self.kalman.d_covariance
        self.error_monitor = self.kalman.error_tracker_temp
        self.num_trackers = len(self.kalman.localTrackersIDList)
        self.trupercept_list = self.kalman.trupercept_list
        self.fusion_steps += 1
        
        # Get Kalman-tracked width and length
        self.width = self.kalman.width
        self.length = self.kalman.length
        
        # Use measured angle from detections (weighted circular mean)
        # Velocity direction != vehicle heading (especially during turns)
        if len(self.match_list) > 0:
            if len(self.match_list) == 1:
                self.angle = self.match_list[0].angle
            else:
                # Weighted circular mean for angle (handles wraparound)
                sin_sum = 0.0
                cos_sum = 0.0
                total_weight = 0.0
                for i, match in enumerate(self.match_list):
                    weight = 2.0 ** i  # More recent = higher weight
                    sin_sum += weight * math.sin(match.angle)
                    cos_sum += weight * math.cos(match.angle)
                    total_weight += weight
                if total_weight > 0:
                    self.angle = math.atan2(sin_sum / total_weight, cos_sum / total_weight)

    def update_log_odds(self, log_lr: float) -> None:
        """Add a clamped log-likelihood-ratio update to the existence score."""
        self.S_t += max(-LOG_LR_CLAMP, min(LOG_LR_CLAMP, log_lr))

    def blended_log_lr_miss(self, default_log_lr_miss: float) -> float:
        """Source-mix-weighted per-miss log-odds decay for this track.

        A track fed by multiple sources decays on a miss by the
        source-count-weighted average of each source's per-source
        log_lr_miss. With no per-source map (single-detector run) or a track
        whose sources are all absent from the map, this returns the global
        scalar unchanged — exact legacy behaviour.
        """
        by_source = self._log_lr_miss_by_source
        if not by_source or not self.source_hits:
            return default_log_lr_miss
        total = 0.0
        weight = 0.0
        for src, n in self.source_hits.items():
            total += n * by_source.get(src, default_log_lr_miss)
            weight += n
        if weight <= 0:
            return default_log_lr_miss
        return total / weight

    @property
    def tracking_score(self) -> float:
        """sigmoid(S_t) ∈ [0, 1] — calibrated existence probability for
        AB3DMOT-style recall sweeps."""
        # Stable sigmoid.
        if self.S_t >= 0:
            ez = math.exp(-self.S_t)
            return 1.0 / (1.0 + ez)
        ez = math.exp(self.S_t)
        return ez / (1.0 + ez)

    def clearLastFrame(self):
        self.match_list = []


class Fusion:
    # Fusion is a special class for matching and fusing detections for a variety of sources.
    # The input is scalable and therefore must be generated before being fed into this class.
    # A unique list of detections is required from each individual sensor or pre-fused device
    # output or it will not be matched. Detections too close to each other may be combined.
    # This is a modified version of the frame-by-frame tracker seen in:
    # https://github.com/eandert/Jetson_Nano_Camera_Vehicle_Tracker
    def __init__(self, id, use_trust_scoring=True, passthrough_covariance=False, filter_class=None,
                 static_matching_cov=None, p_tp_birth_gate=None,
                 p_tp_birth_gate_by_source=None,
                 lifecycle_mode="legacy",
                 confirm_log_odds=DEFAULT_CONFIRM_LOG_ODDS,
                 confirm_log_odds_by_source=None,
                 kill_log_odds=DEFAULT_KILL_LOG_ODDS,
                 log_lr_miss=DEFAULT_LOG_LR_MISS,
                 log_lr_miss_by_source=None,
                 ab3dmot_min_hits=3,
                 ab3dmot_max_age=2,
                 enable_inactive_preservation=False,
                 inactive_grace_s=DEFAULT_INACTIVE_GRACE_S,
                 mahal_weight=0.6, iou_weight=0.4, mahal_gate=13.82,
                 iou_gate=0.0,
                 tape_score_miss_decay=None):
        # Set other parameters for the class
        self.tracked_list = []
        self.id = id
        self.current_track_id = 0
        self.min_size = 0.5
        self.trackShowThreshold = 4
        self.fusion_mode = 2  # CTRV model: uses heading measurement for better turn tracking
        self.use_trust_scoring = use_trust_scoring  # Flag to enable/disable trust-based filtering
        self.passthrough_covariance = passthrough_covariance  # Pass measurement covariance directly
        self.filter_class = filter_class  # Pluggable filter; None → ResizableKalman (default)
        self.static_matching_cov = static_matching_cov  # If set, use this fixed cov for matching only
        # Phase A: minimum P(TP) for a detection to spawn a new tentative track.
        # None = disabled (legacy behaviour: every unmatched detection spawns).
        # Detections without a `.p_tp` attribute are treated as P_TP=1.0
        # (calibration unavailable → no gating, identical to current pipeline).
        self.p_tp_birth_gate = p_tp_birth_gate

        # --- Per-source LIFECYCLE BLEND maps --------------------------------
        # For mixed-detector configs (astuff=PP, tesla=CPft) the track-level
        # lifecycle params are routed per the DETECTION'S source vehicle
        # rather than the old first-match-wins astuff-global scalar.
        #   p_tp_birth_gate_by_source  {source_id -> gate}   birth gate (#1)
        #   confirm_log_odds_by_source {source_id -> thresh} confirm (#2)
        #   log_lr_miss_by_source      {source_id -> decay}  miss decay (#3)
        # Each is keyed by participant id (vehicle_id // max_id; 1=astuff,
        # 2=tesla, 0=single-detector/sim). None = use the scalar fallback,
        # which reproduces the legacy single-scalar path exactly.
        # kill_log_odds stays a single global floor (not detector-specific).
        self.p_tp_birth_gate_by_source = p_tp_birth_gate_by_source
        self.confirm_log_odds_by_source = confirm_log_odds_by_source
        self.log_lr_miss_by_source = log_lr_miss_by_source

        # Phase B: track-lifecycle mode.
        #   "legacy"   — fusion_steps > trackShowThreshold (count-based, current).
        #   "log_odds" — IPDA-style: confirm at S_t > confirm_log_odds; kill
        #                at S_t < kill_log_odds; miss-frame decay each step.
        #   "ab3dmot"  — V2V4Real/AB3DMOT-style: confirm after min_hits matched
        #                frames, kill after max_age consecutive missed frames.
        #                Hard count rules; combines well with our P(TP) birth
        #                gate (Phase A) and tape_score-with-decay (AMOTA).
        self.lifecycle_mode = lifecycle_mode
        self.confirm_log_odds = confirm_log_odds
        self.kill_log_odds    = kill_log_odds
        self.log_lr_miss      = log_lr_miss
        self.ab3dmot_min_hits = ab3dmot_min_hits
        self.ab3dmot_max_age  = ab3dmot_max_age

        # Phase B.5: when a track's S_t collapses below kill_log_odds, do
        # we delete it (False, classic) or move it to an inactive state
        # for `inactive_grace_s` seconds (True) so a Mahalanobis re-match
        # can revive the same track ID? Inactive tracks don't emit but
        # still appear in the matching cost matrix. Only takes effect in
        # log_odds lifecycle mode.
        self.enable_inactive_preservation = enable_inactive_preservation
        self.inactive_grace_s = inactive_grace_s

        # Pluggable Hungarian cost. Default = our hybrid (Mahalanobis + IoU).
        # Setting mahal_weight=0, iou_weight=1 reproduces AB3DMOT's pure-IoU
        # matching. Setting mahal_weight=1, iou_weight=0 gives pure
        # Mahalanobis. mahal_gate is the chi² gate on Mahalanobis distance
        # (13.82 = 99.9% / 2 DOF); set to a large value to effectively
        # disable the Mahalanobis gate (useful when iou_weight=1).
        self.mahal_weight = float(mahal_weight)
        self.iou_weight   = float(iou_weight)
        self.mahal_gate   = float(mahal_gate)
        self.iou_gate     = float(iou_gate)

        # Per-frame multiplicative tape_score decay applied to coasting tracks.
        # 1.0 = no decay (matches AB3DMOT's "score persists across coast"
        # behaviour). Lower = faster decay. None falls back to module default.
        self.tape_score_miss_decay = (
            float(tape_score_miss_decay) if tape_score_miss_decay is not None
            else TAPE_SCORE_MISS_DECAY
        )
        
        # Trust scoring: per-participant trust scores (default 1.0 = fully trusted)
        # Key: participant_id (extracted from vehicle_id // max_id)
        # Value: trust score (1.0 = normal, >1.2 = anomalous/untrusted)
        self.participant_trust_scores: Dict[int, float] = {}
    
    def get_participant_trust_score(self, vehicle_id: int) -> float:
        """
        Get the trust score for a participant based on their vehicle_id.
        
        Args:
            vehicle_id: The universal ID (participant_id * max_id + track_id)
            
        Returns:
            Trust score (1.0 = normal, >1.0 = less trusted)
        """
        participant_id = vehicle_id // max_id
        return self.participant_trust_scores.get(participant_id, 1.0)
    
    def update_trust_scores(self, scores: Dict[int, float]) -> None:
        """
        Update trust scores from PerceptionScorer output.

        Args:
            scores: Dict mapping participant_id to their normalized trust score
        """
        self.participant_trust_scores.update(scores)

    def _birth_gate_for(self, det) -> Optional[float]:
        """Resolve the P(TP) birth gate for a detection's SOURCE (blend #1).

        Mixed-detector configs gate each detection against the gate of its
        own source vehicle (vehicle_id // max_id). With no per-source map the
        single scalar self.p_tp_birth_gate applies to every source — exact
        legacy behaviour. A per-source map that lacks a source's key falls
        back to the scalar too.
        """
        if self.p_tp_birth_gate_by_source:
            return self.p_tp_birth_gate_by_source.get(
                _source_id_of(det), self.p_tp_birth_gate)
        return self.p_tp_birth_gate

    def _spawn_track(self, det, timestamp, track_id):
        """Construct a GlobalTracked, threading the per-source lifecycle maps
        so the new track records its birth source + resolves its own confirm
        threshold (blend #2) and can blend its miss-decay (blend #3)."""
        return GlobalTracked(
            det,
            timestamp,
            track_id,
            self.fusion_mode,
            use_trust_scoring=self.use_trust_scoring,
            passthrough_covariance=self.passthrough_covariance,
            filter_class=self.filter_class,
            confirm_log_odds=self.confirm_log_odds,
            confirm_log_odds_by_source=self.confirm_log_odds_by_source,
            log_lr_miss_by_source=self.log_lr_miss_by_source,
        )

    def fuseDetectionFrame(self, time, monitor=False):
        result = []
        cooperative_monitoring = []
        trupercept_monitoring = []
        detected_objects = []  # New list to store DetectedObject instances

        # Per-frame miss handling, applied once per fused frame BEFORE the
        # kill check / output. A track is "missed in this frame" iff its
        # last_measurement is strictly before the current frame's timestamp.
        for track in self.tracked_list:
            if track.last_measurement < time:
                # Tape-score decay (per-Fusion-instance knob; default 1.0 =
                # no decay, matches AB3DMOT). Lower values demote stale
                # tracks down the AMOTA recall sort.
                track.tape_score *= self.tape_score_miss_decay
                # AB3DMOT-style miss counter
                track.misses_since_match += 1
                # Phase B: log-odds miss-evidence accumulation. The decay is
                # the track's source-mix-weighted log_lr_miss (blend #3); with
                # no per-source map it equals the global scalar self.log_lr_miss.
                if self.lifecycle_mode == "log_odds":
                    track.update_log_odds(track.blended_log_lr_miss(self.log_lr_miss))
        if self.lifecycle_mode == "log_odds":
            survivors = []
            for t in self.tracked_list:
                if t.S_t >= self.kill_log_odds:
                    survivors.append(t)
                else:
                    # S_t collapsed. Either delete (preservation off) or
                    # park in inactive state (preservation on, log_odds only).
                    if self.enable_inactive_preservation:
                        if not t.is_inactive:
                            t.is_inactive = True
                            t.inactive_since = time
                        # Drop only if grace has expired
                        if (time - (t.inactive_since or time)) <= self.inactive_grace_s:
                            survivors.append(t)
            self.tracked_list = survivors
        elif self.lifecycle_mode == "ab3dmot":
            self.tracked_list = [
                t for t in self.tracked_list
                if t.misses_since_match < self.ab3dmot_max_age
            ]

        # First pass: run Kalman fusion on all tracks
        for track in self.tracked_list:
            trust_scores = self.participant_trust_scores if self.use_trust_scoring else None
            track.fusion(time, monitor, trust_scores)

        # Clean duplicate tracks BEFORE building output (so duplicates are excluded)
        if len(self.tracked_list) >= 1:
            self.cleanDetections()

        # Second pass: build output from surviving tracks.
        # Emit gate depends on lifecycle_mode:
        #   "legacy"   — fusion_steps > trackShowThreshold (count-based)
        #   "log_odds" — S_t > confirm_log_odds (Bayesian existence threshold)
        #   "ab3dmot"  — hits >= ab3dmot_min_hits  (AB3DMOT count rule)
        # Inactive tracks (Phase B.5) never emit — they wait for revival.
        for track in self.tracked_list:
            if track.is_inactive:
                continue
            if self.lifecycle_mode == "log_odds":
                # Confirm against the track's OWN threshold, inherited from
                # its birthing source (blend #2). Falls back to the global
                # self.confirm_log_odds for single-detector runs.
                emit = track.S_t > track.confirm_log_odds
            elif self.lifecycle_mode == "ab3dmot":
                emit = track.hits >= self.ab3dmot_min_hits
            else:
                emit = track.fusion_steps > self.trackShowThreshold
            if emit:
                result.append([track.id, track.x, track.y, track.error_covariance.tolist(
                ), track.dx, track.dy, track.d_covariance.tolist(), track.num_trackers])
                if track.error_monitor:
                    cooperative_monitoring.append(track.error_monitor)
                    trupercept_monitoring.append(track.trupercept_list)

                detected_object = DetectedObject(
                    vehicle_id=track.id,
                    vehicle_type=track.type,
                    detected_bbox=None,
                    centroid=[track.x, track.y],
                    width=track.width,
                    length=track.length,
                    angle=track.angle,
                    expected_error_gaussian=None,
                    error_covariance=track.error_covariance,
                    velocity_vector=[track.dx, track.dy]
                )
                detected_object.last_measurement = track.last_measurement
                # Carry existence confidence so the next-level tracker (global fusion)
                # can accumulate log-odds evidence correctly. Confirmed local tracks
                # are reliable, so p_tp is the track's sigmoid(S_t) in log_odds mode
                # (high for well-confirmed tracks) or 0.9 for count-based modes.
                if self.lifecycle_mode == "log_odds":
                    detected_object.p_tp = track.tracking_score
                else:
                    detected_object.p_tp = 0.9
                detected_object.det_score = track.tape_score
                detected_objects.append(detected_object)

                bbox = detected_object.detected_bbox_corners
                detected_object.detected_bbox = bbox

            track.clearLastFrame()

        return result, detected_objects, cooperative_monitoring, trupercept_monitoring

    def processDetectionFrame(self, timestamp, observations, cleanupTime, source_participant_id: int = 0):
        """
        Process a frame of detections from a single participant (CAV/CIS).
        
        Args:
            timestamp: Current simulation time
            observations: List of DetectedObject instances
            cleanupTime: Time threshold for removing stale tracks
            source_participant_id: ID of the CAV/CIS that produced these detections
                                   Used for trust scoring (participant_id * max_id + track_id)
        """
        detections_list_positions = []
        detections_list = []
        for idx, det in enumerate(observations):
            # ORDER: [x, y, width, length, angle]
            # REVERTED: Matching logic expects width first
            detections_list_positions.append(
                [det.centroid[0], det.centroid[1], det.dimensions[0], det.dimensions[1], det.angle])
            
            # Encode source participant ID into the detection for trust scoring
            # This allows us to track which participant contributed each detection
            # Format: encoded_id = source_participant_id * max_id + detection_index
            # Note: vehicle_id may be a string (SUMO ID), so we use detection index instead
            det.vehicle_id = source_participant_id * max_id + idx
            
            detections_list.append(det)

        # Call the matching function to modify our detections in tracked_list
        self.matchDetections(detections_list_positions, detections_list,
                             timestamp, cleanupTime)

    def matchDetections(self, detections_list_positions, detection_list, timestamp, cleanupTime):
        """
        Match detections to existing tracks using the Hungarian algorithm with
        a hybrid cost function combining Mahalanobis distance and IOU.
        
        This replaces the greedy BallTree approach with globally optimal assignment.
        """
        # Check if there are any detections
        if len(detections_list_positions) > 0:
            
            # Check if there are any existing tracks
            if len(self.tracked_list) > 0:
                num_tracks = len(self.tracked_list)
                num_detections = len(detections_list_positions)

                IMPOSSIBLE_COST = 1e9
                cost_matrix = np.full((num_tracks, num_detections), IMPOSSIBLE_COST)

                # Pre-compute track predictions (one Kalman prediction per track)
                track_predictions = [
                    track.getPositionPredictedWithCovariance(timestamp)
                    for track in self.tracked_list
                ]

                # --- VECTORIZED COST MATRIX ---
                # Extract track positions and prediction covariances as arrays
                pred_xy = np.empty((num_tracks, 2), dtype=np.float64)
                pred_Ps = np.empty((num_tracks, 2, 2), dtype=np.float64)
                for i, p in enumerate(track_predictions):
                    pred_xy[i, 0] = p['x']
                    pred_xy[i, 1] = p['y']
                    pred_Ps[i] = p['P_pred'][0:2, 0:2]

                # Pre-extract detection positions and covariances as arrays
                det_xy = np.empty((num_detections, 2), dtype=np.float64)
                det_covs = np.empty((num_detections, 2, 2), dtype=np.float64)
                _eye2 = np.eye(2, dtype=np.float64)
                for j, det in enumerate(detection_list):
                    det_xy[j, 0] = det.centroid[0]
                    det_xy[j, 1] = det.centroid[1]
                    if self.static_matching_cov is not None:
                        dc = np.asarray(self.static_matching_cov, dtype=np.float64).reshape(2, 2)
                    elif det.error_covariance is None or np.asarray(det.error_covariance).size < 4:
                        dc = _eye2
                    else:
                        dc = np.asarray(det.error_covariance, dtype=np.float64).reshape(2, 2)
                    det_covs[j] = dc

                # Batch Euclidean pre-filter: (T, D) squared-distance matrix
                diff = pred_xy[:, np.newaxis, :] - det_xy[np.newaxis, :, :]  # (T, D, 2)
                dist2 = (diff * diff).sum(axis=2)                              # (T, D)
                t_near, d_near = np.where(dist2 <= 900.0)

                if len(t_near) > 0:
                    # Batch innovation covariance S = P_pred[t] + R[d]
                    S_batch = pred_Ps[t_near] + det_covs[d_near]  # (N, 2, 2)
                    S_batch[:, 0, 0] += 1e-8
                    S_batch[:, 1, 1] += 1e-8
                    # Analytic batch 2×2 inverse (avoids LAPACK overhead)
                    S_inv_batch = utils.inv2x2(S_batch)           # (N, 2, 2)

                    # Batch Mahalanobis: d²[n] = diffs[n] @ S_inv[n] @ diffs[n]
                    diffs = diff[t_near, d_near]                   # (N, 2)
                    tmp = np.einsum('nij,nj->ni', S_inv_batch, diffs)
                    mahal = np.sqrt(np.maximum(0.0, (diffs * tmp).sum(axis=1)))  # (N,)

                    # Fill cost for pairs passing the Mahalanobis gate
                    # IoU (expensive Sutherland-Hodgman) only runs on this small subset
                    gate_k = np.where(mahal <= self.mahal_gate)[0]
                    for k in gate_k:
                        t_idx = int(t_near[k])
                        d_idx = int(d_near[k])
                        m = float(mahal[k])

                        if self.iou_weight > 0:
                            iou = utils.rotated_box_iou(
                                detections_list_positions[d_idx],
                                track_predictions[t_idx]['bbox'])
                            if self.iou_gate > 0.0 and iou < self.iou_gate:
                                continue
                        else:
                            iou = 0.0

                        cost_matrix[t_idx, d_idx] = (
                            self.mahal_weight * (m / self.mahal_gate) +
                            self.iou_weight * (1.0 - iou)
                        )

                # Run Hungarian algorithm for optimal assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Filter out assignments that exceed cost threshold
                matched_track_indices = set()
                matched_det_indices = set()
                
                for t_idx, d_idx in zip(row_ind, col_ind):
                    if cost_matrix[t_idx, d_idx] < IMPOSSIBLE_COST:
                        # Valid match
                        det = detection_list[d_idx]
                        track = self.tracked_list[t_idx]
                        track.update(det, timestamp)
                        # Phase B: log-odds match-evidence update.
                        if self.lifecycle_mode == "log_odds":
                            track.update_log_odds(_logit(getattr(det, "p_tp", 0.5)))
                            # Phase B.5: revive an inactive track that just
                            # got matched. Same ID restored; bump S_t back
                            # above confirm threshold so the track re-emits
                            # immediately rather than re-accumulating from 0.
                            if track.is_inactive:
                                track.is_inactive = False
                                track.inactive_since = None
                                track.S_t = max(track.S_t, track.confirm_log_odds + 0.5)
                                track.revive_count += 1
                        matched_track_indices.add(t_idx)
                        matched_det_indices.add(d_idx)
                # Note: per-frame miss decay is applied in fuseDetectionFrame,
                # not here — matchDetections runs per-ego, but a track is
                # "missed in this frame" only if no ego observed it.
                
                # Create new tracks for unmatched detections (Phase A gate).
                for d_idx in range(num_detections):
                    if d_idx not in matched_det_indices:
                        det = detection_list[d_idx]
                        _gate = self._birth_gate_for(det)
                        if _gate is not None and \
                           getattr(det, "p_tp", 1.0) < _gate:
                            continue
                        new = self._spawn_track(
                            det,
                            timestamp,
                            (max_id * self.id) + self.current_track_id,
                        )
                        if self.current_track_id < max_id:
                            self.current_track_id += 1
                        else:
                            self.current_track_id = 0
                        self.tracked_list.append(new)

            else:
                # No existing tracks: spawn for each detection (Phase A gate).
                for dl in detection_list:
                    _gate = self._birth_gate_for(dl)
                    if _gate is not None and \
                       getattr(dl, "p_tp", 1.0) < _gate:
                        continue
                    new = self._spawn_track(
                        dl, timestamp, (max_id * self.id) + self.current_track_id)
                    if self.current_track_id < max_id:
                        self.current_track_id += 1
                    else:
                        self.current_track_id = 0
                    self.tracked_list.append(new)

        # Clean up old tracks. Stale-time deletion always applies UNLESS the
        # track is currently inactive (Phase B.5 owns that lifetime — its
        # grace period takes precedence over `cleanupTime` once a track
        # has been parked). Log-odds kill is gated on B.5 being OFF; when
        # B.5 is on, the kill→inactive transition is handled exclusively
        # in fuseDetectionFrame.
        remove = []
        for idx, track in enumerate(self.tracked_list):
            track.relations = []
            if track.is_inactive:
                continue
            if track.last_measurement <= (timestamp - cleanupTime):
                remove.append(idx)
            elif self.lifecycle_mode == "log_odds" and \
                 not self.enable_inactive_preservation and \
                 track.S_t < self.kill_log_odds:
                remove.append(idx)

        for delete in reversed(remove):
            self.tracked_list.pop(delete)

    def cleanDetections(self):
        """
        Remove duplicate tracks that represent the same physical object.
        
        Uses Mahalanobis distance combined with IOU to identify overlapping tracks,
        then keeps the track with more fusion history (more stable).
        """
        if len(self.tracked_list) < 2:
            return
        
        num_tracks = len(self.tracked_list)
        
        # Build pairwise similarity matrix
        # Only consider upper triangle since similarity is symmetric
        merge_pairs = []  # List of (i, j, cost) where i < j
        
        for i in range(num_tracks):
            track_i = self.tracked_list[i]
            # Inactive tracks (Phase B.5) don't participate in the merge pool —
            # their position is stale and they're already not emitting.
            if track_i.is_inactive:
                continue
            xi, yi = track_i.x, track_i.y
            # Lazy-compute bbox_i: deferred until a nearby candidate is found
            bbox_i = None

            for j in range(i + 1, num_tracks):
                track_j = self.tracked_list[j]
                if track_j.is_inactive:
                    continue
                # Fast Euclidean pre-filter: skip expensive polygon/Mahalanobis ops for
                # clearly distant pairs. 15m covers max Mahalanobis<4.0 with σ≤2.5m tracks.
                dx = track_j.x - xi
                dy = track_j.y - yi
                if dx * dx + dy * dy > 225.0:
                    continue

                # Lazy-init bbox_i on first nearby candidate
                if bbox_i is None:
                    a_i, b_i, _ = utils.ellipsify(track_i.error_covariance, 2.0)
                    bbox_i = [xi, yi, track_i.width + a_i, track_i.length + b_i, track_i.angle]

                a_j, b_j, _ = utils.ellipsify(track_j.error_covariance, 2.0)
                bbox_j = [track_j.x, track_j.y, track_j.width + a_j, track_j.length + b_j, track_j.angle]

                # Compute IOU
                iou = utils.rotated_box_iou(bbox_i, bbox_j)
                
                # Compute Mahalanobis distance using combined covariance
                combined_cov = track_i.error_covariance + track_j.error_covariance
                try:
                    S_inv = np.linalg.inv(combined_cov)
                except np.linalg.LinAlgError:
                    S_inv = np.linalg.inv(combined_cov + 1e-6 * np.eye(2))
                
                pos_i = [track_i.x, track_i.y]
                pos_j = [track_j.x, track_j.y]
                mahal_dist = utils.mahalanobis_distance(pos_i, pos_j, S_inv)
                
                # Consider merging if:
                # 1. IOU > 0.3 (decent overlap), OR
                # 2. Mahalanobis distance < 4.0 (statistically close given uncertainties)
                if iou > 0.3 or mahal_dist < 4.0:
                    # Combined score (lower = more similar)
                    cost = (1.0 - iou) * 0.5 + (mahal_dist / 10.0) * 0.5
                    merge_pairs.append((i, j, cost))
        
        # Sort by cost (most similar first)
        merge_pairs.sort(key=lambda x: x[2])
        
        # Greedily merge: keep track with more fusion history
        remove_set = set()
        for i, j, cost in merge_pairs:
            if i in remove_set or j in remove_set:
                continue
            
            track_i = self.tracked_list[i]
            track_j = self.tracked_list[j]
            
            # Only merge if at least one track is established. Establishment
            # rule depends on lifecycle_mode.
            if self.lifecycle_mode == "log_odds":
                est_i = track_i.S_t > self.confirm_log_odds
                est_j = track_j.S_t > self.confirm_log_odds
            elif self.lifecycle_mode == "ab3dmot":
                est_i = track_i.hits >= self.ab3dmot_min_hits
                est_j = track_j.hits >= self.ab3dmot_min_hits
            else:
                est_i = track_i.fusion_steps >= self.trackShowThreshold
                est_j = track_j.fusion_steps >= self.trackShowThreshold
            if not est_i and not est_j:
                continue

            # Keep the more confident track. Confidence = S_t in log_odds, hits
            # in ab3dmot, fusion_steps in legacy. Tiebreaker: most recent.
            if self.lifecycle_mode == "log_odds":
                key_i, key_j = track_i.S_t, track_j.S_t
            elif self.lifecycle_mode == "ab3dmot":
                key_i, key_j = track_i.hits, track_j.hits
            else:
                key_i, key_j = track_i.fusion_steps, track_j.fusion_steps
            if key_i > key_j:
                remove_set.add(j)
            elif key_j > key_i:
                remove_set.add(i)
            elif track_i.last_measurement >= track_j.last_measurement:
                remove_set.add(j)
            else:
                remove_set.add(i)
        
        # Remove merged tracks (in reverse order to maintain indices)
        for idx in sorted(remove_set, reverse=True):
            self.tracked_list.pop(idx)
