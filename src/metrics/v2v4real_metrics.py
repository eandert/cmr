"""
V2V4Real-paper-equivalent tracking metrics (AB3DMOT protocol).

V2V4Real (CVPR 2023) reports its tracking benchmark using AB3DMOT's evaluation
suite (Weng et al. IROS 2020 / arXiv 2008.08063), not nuScenes Tracking. The
two protocols differ in subtle but important ways:

  - AB3DMOT uses **40 recall thresholds** (0.025, 0.050, ..., 1.000); nuScenes
    uses 10.
  - AB3DMOT defines **AMOTP↑** as `mean(1 - dist/threshold)` over recalls
    (higher is better — i.e. closer matches score higher), while nuScenes
    AMOTP is mean translation error (lower is better).
  - AB3DMOT adds **sAMOTA**: average of sMOTA(r)=max(0,(TP-FP-IDS)/(r*GT)) over recall thresholds.
    This corrects for the fact that the recall sweep can't always reach all
    targets — without scaling, methods with a lower ceiling on recall get
    artificially deflated AMOTA scores.
  - AB3DMOT reports **MOTA at full data** (no recall sweep), plus **MT**
    (Mostly Tracked GTs, > 80 %) and **ML** (Mostly Lost GTs, < 20 %).

V2V4Real's leaderboard reports all six numbers in **percent**.

Inputs (per-frame match tape) collected by `v2v4real_replay`:

    PerFrameRecord = {
        "frame_idx": int,
        "tracks":    [(track_id, x, y, tracking_score), ...],
        "gts":       [(gt_id, x, y), ...],
    }

Output (returned dict):
    amota_pct, amotp_pct, samota_pct, mota_pct, mt_pct, ml_pct  -- the headline
                                       leaderboard numbers (in percent).
    amota, amotp, samota, mota, mt, ml                          -- same in [0,1]
                                       for downstream code that prefers
                                       fractions.
    mota_per_recall, motp_per_recall   -- arrays, len(recall_thresholds).
    tp_total, fp_total, fn_total, ids_total, gt_total            -- counts at full
                                       recall (no thresholding).
    num_frames, recall_thresholds, match_threshold_m             -- protocol info.

References:
- AB3DMOT paper: https://arxiv.org/abs/2008.08063
- AB3DMOT code:  https://github.com/xinshuoweng/AB3DMOT
- V2V4Real paper, Section 4 (cites AB3DMOT for tracking eval).
"""
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


CENTER_DIST_THRESHOLD_M = 2.0
# AB3DMOT default: 40 recall thresholds in 0.025 steps.
RECALL_THRESHOLDS = np.arange(0.025, 1.0001, 0.025)
# Mostly-Tracked / Mostly-Lost thresholds (Bernardin & Stiefelhagen 2008,
# adopted by MOT16 and AB3DMOT).
MT_THRESHOLD = 0.8
ML_THRESHOLD = 0.2


# --------------------------------------------------------------------------
# Per-frame matching
# --------------------------------------------------------------------------

def match_frame(
    tracks: List[Tuple],
    gts: List[Tuple],
    threshold: float = CENTER_DIST_THRESHOLD_M,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Hungarian assignment by center distance, gated at `threshold` metres."""
    if not tracks or not gts:
        return [], list(range(len(tracks))), list(range(len(gts)))

    track_xy = np.array([(t[1], t[2]) for t in tracks])
    gt_xy = np.array([(g[1], g[2]) for g in gts])
    dist = np.linalg.norm(track_xy[:, None, :] - gt_xy[None, :, :], axis=-1)

    BIG = 1e6
    cost = np.where(dist <= threshold, dist, BIG)
    rows, cols = linear_sum_assignment(cost)

    matches: List[Tuple[int, int, float]] = []
    matched_t, matched_g = set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] < BIG:
            matches.append((int(r), int(c), float(cost[r, c])))
            matched_t.add(int(r))
            matched_g.add(int(c))

    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_t]
    unmatched_gts = [i for i in range(len(gts)) if i not in matched_g]
    return matches, unmatched_tracks, unmatched_gts


def match_frame_3d_iou(
    tracks: List[Tuple],     # (id, x, y, w, l, yaw, score, h, z_center)
    gts: List[Tuple],        # (id, x, y, w, l, yaw, h, z_center)
    iou_threshold: float = 0.25,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Like match_frame_iou but uses full 3D IoU (BEV footprint × height interval).

    Track tuples must have h at index 7 and z_center at index 8.
    GT tuples must have h at index 6 and z_center at index 7.
    """
    from geometry import iou_3d_box

    if not tracks or not gts:
        return [], list(range(len(tracks))), list(range(len(gts)))

    BIG = 1e6
    n_t, n_g = len(tracks), len(gts)
    cost = np.full((n_t, n_g), BIG)
    for i, t in enumerate(tracks):
        t_box = (t[1], t[2], t[3], t[4], t[5], t[8], t[7])  # cx,cy,w,l,yaw,z,h
        for j, g in enumerate(gts):
            g_box = (g[1], g[2], g[3], g[4], g[5], g[7], g[6])  # cx,cy,w,l,yaw,z,h
            iou = iou_3d_box(t_box, g_box)
            if iou >= iou_threshold:
                cost[i, j] = 1.0 - iou
    rows, cols = linear_sum_assignment(cost)

    matches: List[Tuple[int, int, float]] = []
    matched_t, matched_g = set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] < BIG:
            matches.append((int(r), int(c), float(cost[r, c])))
            matched_t.add(int(r))
            matched_g.add(int(c))

    unmatched_tracks = [i for i in range(n_t) if i not in matched_t]
    unmatched_gts    = [i for i in range(n_g) if i not in matched_g]
    return matches, unmatched_tracks, unmatched_gts


def match_frame_iou(
    tracks: List[Tuple],     # (id, x, y, w, l, yaw, score)
    gts: List[Tuple],        # (id, x, y, w, l, yaw)
    iou_threshold: float = 0.25,
    convex_hull: bool = False,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """KITTI-style BEV-IoU Hungarian assignment, gated at `iou_threshold`.

    `tracks` and `gts` carry full bbox info: (id, x, y, width, length,
    yaw[, score]). Returns matches as (track_idx, gt_idx, 1-IoU).

    convex_hull: use scipy ConvexHull for intersection area (matches DMSTrack).
    """
    from utils import rotated_box_iou

    if not tracks or not gts:
        return [], list(range(len(tracks))), list(range(len(gts)))

    BIG = 1e6
    n_t, n_g = len(tracks), len(gts)
    cost = np.full((n_t, n_g), BIG)
    iou_mat = np.zeros((n_t, n_g))
    for i, t in enumerate(tracks):
        t_box = [t[1], t[2], t[3], t[4], t[5]]
        for j, g in enumerate(gts):
            g_box = [g[1], g[2], g[3], g[4], g[5]]
            iou = rotated_box_iou(t_box, g_box, convex_hull=convex_hull)
            iou_mat[i, j] = iou
            if iou >= iou_threshold:
                cost[i, j] = 1.0 - iou
    rows, cols = linear_sum_assignment(cost)

    matches: List[Tuple[int, int, float]] = []
    matched_t, matched_g = set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] < BIG:
            matches.append((int(r), int(c), float(cost[r, c])))
            matched_t.add(int(r))
            matched_g.add(int(c))

    unmatched_tracks = [i for i in range(n_t) if i not in matched_t]
    unmatched_gts    = [i for i in range(n_g) if i not in matched_g]
    return matches, unmatched_tracks, unmatched_gts


def _get_thresholds(
    tp_scores: List[float],
    num_gt: int,
    num_sample_pts: int = 41,
) -> Tuple[List[float], List[float]]:
    """Port of AB3DMOT's getThresholds.

    Given the scores of ALL true-positive detections (from the unrestricted full
    evaluation) and the total GT count, returns (confidence_thresholds,
    recall_levels) — a list of at most num_sample_pts-1 pairs.  The i-th pair is
    the confidence threshold T_i at which recall first crosses level r_i =
    i/(num_sample_pts-1).  The result is used to re-run the full evaluation at
    each T_i and then average the MOTA values (AMOTA).
    """
    scores = np.array(tp_scores, dtype=np.float64)
    scores[::-1].sort()  # descending in-place  (sort then flip)
    scores = scores[::-1]  # make a descending view (already done above but be explicit)
    scores = np.sort(tp_scores)[::-1]  # descending

    current_recall = 0.0
    thresholds: List[float] = []
    recalls: List[float] = []
    n = len(scores)
    for i, score in enumerate(scores):
        l_recall = (i + 1) / float(num_gt)
        r_recall = ((i + 2) / float(num_gt)) if i < n - 1 else l_recall
        # Skip if the next TP is strictly closer to current_recall target
        if (r_recall - current_recall) < (current_recall - l_recall) and i < n - 1:
            continue
        thresholds.append(float(score))
        recalls.append(current_recall)
        current_recall += 1.0 / (num_sample_pts - 1.0)

    # Throw the first entry (recall=0) per the original implementation
    return thresholds[1:], recalls[1:]


def compute_ab3dmot_metrics_iou(
    per_frame: List[Dict],
    iou_threshold: float = 0.25,
    recall_thresholds: np.ndarray = RECALL_THRESHOLDS,
    min_gt_lifetime_frames: int = 1,
    max_eval_range_m: Optional[float] = None,
    ego_xy_per_frame: Optional[Dict] = None,
    ignore_unmatched_fps: bool = False,
    match_3d: bool = False,
    convex_hull: bool = False,
    use_track_avg_score: bool = False,
    ab3dmot_rematching: bool = False,
) -> Dict:
    """KITTI-V2V4Real-style AMOTA evaluator: BEV-IoU @ iou_threshold gate.

    Drop-in replacement for compute_ab3dmot_metrics() but with full-bbox
    tracks & gts and IoU-based match gate. Mirrors DMSTrack's
    `evaluate_v2v4real` evaluator behaviour as closely as the BEV
    approximation allows.

    `per_frame` schema:
        {"frame_idx": int,
         "tracks": [(track_id, x, y, width, length, yaw, score), ...],
         "gts":    [(gt_id, x, y, width, length, yaw), ...]}

    `min_gt_lifetime_frames` is the KITTI-style proxy: drop GT IDs that
    appear in fewer than this many frames (rough stand-in for KITTI's
    `min_height=25` filter, which removes GTs too small to track
    reliably). Setting >1 mimics the V2V4Real GT pruning loosely.
    """
    # Optional GT lifetime filter (KITTI min_height proxy v1).
    if min_gt_lifetime_frames > 1:
        gt_seen = {}
        for f in per_frame:
            for g in f["gts"]:
                gt_seen[g[0]] = gt_seen.get(g[0], 0) + 1
        keep_gt = {gid for gid, n in gt_seen.items() if n >= min_gt_lifetime_frames}
        filtered: List[Dict] = []
        for f in per_frame:
            filtered.append({
                "frame_idx": f["frame_idx"],
                "tracks": f["tracks"],
                "gts": [g for g in f["gts"] if g[0] in keep_gt],
            })
        per_frame = filtered

    # KITTI min_height proxy v2: drop tracks AND GT beyond max_eval_range_m
    # (relative to nearest ego in that frame). At KITTI's typical 720px focal
    # length, min_height=25 ≈ 1.5*720/25 = 43m for a 1.5m car. Setting
    # max_eval_range_m=43.0 approximates DMSTrack's evaluate_v2v4real range
    # cut without per-box truncation/occlusion annotations.
    if max_eval_range_m is not None and ego_xy_per_frame is not None:
        import math as _math
        filtered2: List[Dict] = []
        for f in per_frame:
            egos = ego_xy_per_frame.get(f["frame_idx"], [])
            def _in(x, y, _egos=egos, _r=max_eval_range_m):
                if not _egos:
                    return True
                return any(_math.hypot(x - ex, y - ey) <= _r for ex, ey in _egos)
            filtered2.append({
                "frame_idx": f["frame_idx"],
                "tracks": [t for t in f["tracks"] if _in(t[1], t[2])],
                "gts":    [g for g in f["gts"]    if _in(g[1], g[2])],
            })
        per_frame = filtered2

    n_frames = len(per_frame)
    gt_total = sum(len(f["gts"]) for f in per_frame)

    all_track_entries: List[Tuple[float, int, int, Optional[int], Optional[float]]] = []
    per_gt_lifeline: Dict[int, Dict[int, int]] = {}
    per_gt_total_frames: Dict[int, int] = {}
    all_gt_ids: set = set()

    for f in per_frame:
        fidx = f["frame_idx"]
        tracks = f["tracks"]
        gts = f["gts"]
        for g in gts:
            gid = g[0]
            per_gt_total_frames[gid] = per_gt_total_frames.get(gid, 0) + 1
            all_gt_ids.add(gid)

        if match_3d:
            matches, _ut, _ug = match_frame_3d_iou(tracks, gts, iou_threshold)
        else:
            matches, _ut, _ug = match_frame_iou(tracks, gts, iou_threshold, convex_hull=convex_hull)

        track_match: Dict[int, Tuple[Optional[int], Optional[float]]] = {}
        for t_idx, g_idx, d in matches:
            track_match[t_idx] = (gts[g_idx][0], d)
            gt_id = gts[g_idx][0]
            track_id = tracks[t_idx][0]
            per_gt_lifeline.setdefault(gt_id, {})[fidx] = track_id

        for t_idx, t in enumerate(tracks):
            track_id = t[0]
            score = t[6] if len(t) > 6 else 1.0
            gt_id, dist = track_match.get(t_idx, (None, None))
            all_track_entries.append((score, fidx, track_id, gt_id, dist))

    # AB3DMOT-style track-averaged scoring: replace per-detection score with the
    # mean score of the track across all its frames before sorting.  This matches
    # AB3DMOT's evaluate.py which replaces trk.score with id_average_score[id].
    if use_track_avg_score:
        _track_score_sum: Dict[int, float] = {}
        _track_score_cnt: Dict[int, int] = {}
        for score, fidx, track_id, gt_id, dist in all_track_entries:
            _track_score_sum[track_id] = _track_score_sum.get(track_id, 0.0) + score
            _track_score_cnt[track_id] = _track_score_cnt.get(track_id, 0) + 1
        _track_avg: Dict[int, float] = {
            tid: _track_score_sum[tid] / _track_score_cnt[tid]
            for tid in _track_score_sum
        }
        all_track_entries = [
            (_track_avg[track_id], fidx, track_id, gt_id, dist)
            for _, fidx, track_id, gt_id, dist in all_track_entries
        ]

    if ab3dmot_rematching:
        # AB3DMOT exact mode: port of evaluate.py's getThresholds + per-threshold
        # compute3rdPartyMetrics.  At each of 40 confidence thresholds the full
        # Hungarian matching is re-run with only the tracks whose average score
        # meets the threshold.  This matches the official paper numbers exactly.

        # Ensure we have track avg scores for filtering (may have been computed above)
        if not use_track_avg_score:
            _ts: Dict[int, float] = {}; _tc: Dict[int, int] = {}
            for f in per_frame:
                for t in f["tracks"]:
                    _ts[t[0]] = _ts.get(t[0], 0.0) + (t[6] if len(t) > 6 else 1.0)
                    _tc[t[0]] = _tc.get(t[0], 0) + 1
            _track_avg = {tid: _ts[tid] / _tc[tid] for tid in _ts}
        # else: _track_avg was set by the use_track_avg_score block above

        # TP scores for getThresholds come from the full-data matching stored in
        # all_track_entries (after potential avg replacement).
        _tp_scores = [e[0] for e in all_track_entries if e[3] is not None]

        _threshold_list, _recall_list = _get_thresholds(_tp_scores, gt_total)

        _iou_dist_max = 1.0 - iou_threshold
        _mota_vals: List[float] = []
        _smota_vals: List[float] = []
        _motp_vals: List[float] = []

        for _conf_thresh, _recall_target in zip(_threshold_list, _recall_list):
            _tp = _fp = _fn = 0
            _tp_dist = 0.0
            _matched: List[Tuple] = []

            for f in per_frame:
                fidx = f["frame_idx"]
                filt_t = [t for t in f["tracks"] if _track_avg.get(t[0], 0.0) >= _conf_thresh]
                gts_f = f["gts"]

                if match_3d:
                    _m, _ut, _ug = match_frame_3d_iou(filt_t, gts_f, iou_threshold)
                else:
                    _m, _ut, _ug = match_frame_iou(filt_t, gts_f, iou_threshold, convex_hull=convex_hull)

                for ti, gi, cost in _m:
                    _tp += 1
                    _tp_dist += cost
                    _matched.append((fidx, filt_t[ti][0], gts_f[gi][0]))

                if not ignore_unmatched_fps:
                    _fp += len(_ut)
                _fn += len(_ug)

            _ids = _count_ids(_matched)
            _mota = max(0.0, 1.0 - (_fn + _fp + _ids) / max(gt_total, 1))
            _smota = min(1.0, max(0.0, (_tp - _fp - _ids) / max(_recall_target * gt_total, 1e-9)))
            _avg_d = (_tp_dist / _tp) if _tp > 0 else _iou_dist_max
            _motp = max(0.0, 1.0 - _avg_d / max(_iou_dist_max, 1e-6))
            _mota_vals.append(_mota)
            _smota_vals.append(_smota)
            _motp_vals.append(_motp)

        n_recalls = len(recall_thresholds)
        while len(_mota_vals) < n_recalls:
            _mota_vals.append(0.0); _smota_vals.append(0.0); _motp_vals.append(0.0)

        _amota  = float(np.sum(_mota_vals[:n_recalls]))  / max(n_recalls, 1)
        _amotp  = float(np.sum(_motp_vals[:n_recalls]))  / max(n_recalls, 1)
        _samota = float(np.sum(_smota_vals[:n_recalls])) / max(n_recalls, 1)

        # Headline MOTA from full-data matching (no threshold)
        _tp_full = [e for e in all_track_entries if e[3] is not None]
        _fp_full = 0 if ignore_unmatched_fps else len([e for e in all_track_entries if e[3] is None])
        _fn_full = gt_total - len(_tp_full)
        _ids_full = _count_ids([(e[1], e[2], e[3]) for e in _tp_full])
        _mota_full = max(0.0, 1.0 - (_ids_full + _fp_full + _fn_full) / max(gt_total, 1))

        mt = ml = 0
        for gt_id in all_gt_ids:
            total = per_gt_total_frames.get(gt_id, 1)
            matched = len(per_gt_lifeline.get(gt_id, {}))
            ratio = matched / total if total else 0
            if ratio >= 0.8:   mt += 1
            elif ratio < 0.2:  ml += 1

        return {
            "amota": _amota, "amotp": _amotp, "samota": _samota, "mota": _mota_full,
            "mt": mt / max(len(all_gt_ids), 1), "ml": ml / max(len(all_gt_ids), 1),
            "max_recall_reached": len(_tp_full) / max(gt_total, 1),
            "iou_threshold": iou_threshold, "min_gt_lifetime_frames": min_gt_lifetime_frames,
            "max_eval_range_m": max_eval_range_m, "n_frames": n_frames, "gt_total": gt_total,
            "tp_total": len(_tp_full), "fp_total": _fp_full,
            "ids_total": _ids_full,
        }

    # Recall sweep (same as compute_ab3dmot_metrics but operating on
    # 1-IoU "distances" gated at 1-iou_threshold).
    all_track_entries.sort(key=lambda e: -e[0])

    cum_tp = 0; cum_fp = 0; cum_tp_dist = 0.0
    matched_so_far: List[Tuple[int, int, int]] = []
    mota_per_recall: List[float] = []
    smota_per_recall: List[float] = []
    motp_per_recall: List[float] = []
    target_recalls = list(recall_thresholds)
    target_idx = 0

    iou_dist_max = 1.0 - iou_threshold  # max valid "distance" under this gate
    for entry in all_track_entries:
        score, fidx, track_id, gt_id, dist = entry
        if gt_id is not None:
            cum_tp += 1
            cum_tp_dist += dist if dist is not None else 0.0
            matched_so_far.append((fidx, track_id, gt_id))
        else:
            # DMSTrack/V2V4Real evaluator quirk: unmatched tracker entries
            # whose 2D bbox height < min_height get IGNORED (not counted as
            # FPs). For V2V4Real where 2D bbox is always 0,0,0,0, this means
            # ALL unmatched tracks are ignored — FPs effectively don't count.
            # Setting ignore_unmatched_fps=True reproduces this hack.
            if ignore_unmatched_fps:
                continue
            cum_fp += 1
        recall_now = (cum_tp / gt_total) if gt_total else 0.0
        while target_idx < len(target_recalls) and recall_now >= target_recalls[target_idx]:
            tp, fp = cum_tp, cum_fp
            fn = gt_total - tp
            ids = _count_ids(matched_so_far)
            r = target_recalls[target_idx]
            mota_r = max(0.0, 1.0 - (ids + fp + fn) / max(gt_total, 1))
            smota_r = min(1.0, max(0.0, (tp - fp - ids) / max(r * gt_total, 1e-9)))
            mota_per_recall.append(mota_r)
            smota_per_recall.append(smota_r)
            avg_dist = (cum_tp_dist / tp) if tp > 0 else iou_dist_max
            motp_r = max(0.0, 1.0 - avg_dist / max(iou_dist_max, 1e-6))
            motp_per_recall.append(motp_r)
            target_idx += 1

    while target_idx < len(target_recalls):
        mota_per_recall.append(0.0)
        smota_per_recall.append(0.0)
        motp_per_recall.append(0.0)
        target_idx += 1

    n_recalls = len(recall_thresholds)
    amota = float(np.sum(mota_per_recall)) / max(n_recalls, 1)
    amotp = float(np.sum(motp_per_recall)) / max(n_recalls, 1)
    max_recall_reached = (cum_tp / gt_total) if gt_total else 0.0
    samota = float(np.sum(smota_per_recall)) / max(n_recalls, 1)
    # Headline MOTA: at full recall, no recall sweep.
    fn_full = gt_total - cum_tp
    ids_full = _count_ids(matched_so_far)
    mota_full = max(0.0, 1.0 - (ids_full + cum_fp + fn_full) / max(gt_total, 1))
    # MT/ML — match the standard compute_ab3dmot_metrics convention.
    mt = ml = 0
    for gt_id in all_gt_ids:
        total = per_gt_total_frames.get(gt_id, 1)
        matched = len(per_gt_lifeline.get(gt_id, {}))
        ratio = matched / total if total else 0
        if ratio >= 0.8:
            mt += 1
        elif ratio < 0.2:
            ml += 1

    return {
        "amota": amota,
        "amotp": amotp,
        "samota": samota,
        "mota": mota_full,
        "mt": mt / max(len(all_gt_ids), 1),
        "ml": ml / max(len(all_gt_ids), 1),
        "max_recall_reached": max_recall_reached,
        "iou_threshold": iou_threshold,
        "min_gt_lifetime_frames": min_gt_lifetime_frames,
        "max_eval_range_m": max_eval_range_m,
        "n_frames": n_frames,
        "gt_total": gt_total,
        "tp_total": cum_tp,
        "fp_total": cum_fp,
        "ids_total": ids_full,
    }


# --------------------------------------------------------------------------
# AB3DMOT-style metric computation
# --------------------------------------------------------------------------

def compute_ab3dmot_metrics(
    per_frame: List[Dict],
    threshold: float = CENTER_DIST_THRESHOLD_M,
    recall_thresholds: np.ndarray = RECALL_THRESHOLDS,
) -> Dict:
    """
    Compute V2V4Real-paper / AB3DMOT tracking metrics.

    Args:
        per_frame: list of {"frame_idx": int,
                            "tracks": [(track_id, x, y, score), ...],
                            "gts":    [(gt_id, x, y), ...]}
        threshold: center-distance gate (m, default 2.0).
        recall_thresholds: 1D array of recall targets, default 40 in 0.025 steps.

    Returns:
        Dict (see module docstring).
    """
    n_frames = len(per_frame)
    gt_total = sum(len(f["gts"]) for f in per_frame)

    # ---- Per-frame Hungarian matching ----
    # For each frame, store the matched (track_idx, gt_idx, dist) and the
    # unmatched-track / unmatched-gt index lists. We also build:
    #   - all_track_entries: flat list across all frames for the recall sweep.
    #   - per_gt_lifeline: dict gt_id -> {frame_idx -> matched_track_id}
    all_track_entries: List[Tuple[float, int, int, Optional[int], Optional[float]]] = []
    per_gt_lifeline: Dict[int, Dict[int, int]] = {}
    # Per-GT presence count (how many frames a GT appears, total).
    per_gt_total_frames: Dict[int, int] = {}
    # Track count of unique GTs.
    all_gt_ids: set = set()

    for f in per_frame:
        fidx = f["frame_idx"]
        tracks = f["tracks"]
        gts = f["gts"]
        for g in gts:
            gid = g[0]
            per_gt_total_frames[gid] = per_gt_total_frames.get(gid, 0) + 1
            all_gt_ids.add(gid)

        matches, _ut, _ug = match_frame(tracks, gts, threshold)

        track_match: Dict[int, Tuple[Optional[int], Optional[float]]] = {}
        for t_idx, g_idx, d in matches:
            track_match[t_idx] = (gts[g_idx][0], d)
            gt_id = gts[g_idx][0]
            track_id = tracks[t_idx][0]
            per_gt_lifeline.setdefault(gt_id, {})[fidx] = track_id

        # Tape entries may be either (id, x, y, score) [4-tuple] or
        # (id, x, y, w, l, yaw, score) [7-tuple]. Score is always last.
        for t_idx, t in enumerate(tracks):
            track_id = t[0]
            score = t[-1] if len(t) >= 4 else 1.0
            gt_id, dist = track_match.get(t_idx, (None, None))
            all_track_entries.append((score, fidx, track_id, gt_id, dist))

    # ---- AB3DMOT recall sweep ----
    all_track_entries.sort(key=lambda e: -e[0])  # by score, descending

    cum_tp = 0
    cum_fp = 0
    cum_tp_dist = 0.0
    matched_so_far: List[Tuple[int, int, int]] = []

    mota_per_recall: List[float] = []
    smota_per_recall: List[float] = []
    motp_per_recall: List[float] = []
    target_idx = 0

    target_recalls = list(recall_thresholds)
    for entry in all_track_entries:
        score, fidx, track_id, gt_id, dist = entry
        if gt_id is not None:
            cum_tp += 1
            cum_tp_dist += dist
            matched_so_far.append((fidx, track_id, gt_id))
        else:
            cum_fp += 1

        if gt_total == 0:
            recall_now = 0.0
        else:
            recall_now = cum_tp / gt_total

        while target_idx < len(target_recalls) and recall_now >= target_recalls[target_idx]:
            tp, fp = cum_tp, cum_fp
            fn = gt_total - tp
            ids = _count_ids(matched_so_far)
            r = target_recalls[target_idx]
            # AB3DMOT MOTA (per recall): clamped to >= 0.
            mota_r = max(0.0, 1.0 - (ids + fp + fn) / max(gt_total, 1))
            smota_r = min(1.0, max(0.0, (tp - fp - ids) / max(r * gt_total, 1e-9)))
            mota_per_recall.append(mota_r)
            smota_per_recall.append(smota_r)
            # AB3DMOT MOTP (per recall): 1 - avg(dist)/threshold (higher = closer).
            avg_dist = (cum_tp_dist / tp) if tp > 0 else threshold
            motp_r = max(0.0, 1.0 - avg_dist / threshold)
            motp_per_recall.append(motp_r)
            target_idx += 1

    # Pad unreached recalls with 0 (AB3DMOT convention — unreached recall
    # contributes 0 to the average, not NaN).
    while target_idx < len(target_recalls):
        mota_per_recall.append(0.0)
        smota_per_recall.append(0.0)
        motp_per_recall.append(0.0)
        target_idx += 1

    # Headline AMOTA / AMOTP / sAMOTA (in [0, 1]).
    n_recalls = len(recall_thresholds)
    amota = float(np.sum(mota_per_recall)) / max(n_recalls, 1)
    amotp = float(np.sum(motp_per_recall)) / max(n_recalls, 1)
    # sAMOTA: reference AB3DMOT formula — sMOTA(r) = max(0, (TP-FP-IDS)/(r*GT))
    # averaged over all recall thresholds (same denominator as AMOTA).
    max_recall = (cum_tp / gt_total) if gt_total > 0 else 0.0
    samota = float(np.sum(smota_per_recall)) / max(n_recalls, 1)

    # ---- Single-point MOTA (no recall sweep) ----
    final_tp = cum_tp
    final_fp = cum_fp
    final_fn = gt_total - cum_tp
    final_ids = _count_ids(matched_so_far)
    mota = max(0.0, 1.0 - (final_ids + final_fp + final_fn) / max(gt_total, 1))

    # ---- MT / ML ----
    n_gts = len(all_gt_ids)
    mt_count = 0
    ml_count = 0
    for gt_id in all_gt_ids:
        total_frames = per_gt_total_frames.get(gt_id, 0)
        matched_frames = len(per_gt_lifeline.get(gt_id, {}))
        if total_frames == 0:
            continue
        frac = matched_frames / total_frames
        if frac >= MT_THRESHOLD:
            mt_count += 1
        elif frac <= ML_THRESHOLD:
            ml_count += 1
    mt = mt_count / max(n_gts, 1)
    ml = ml_count / max(n_gts, 1)

    return {
        # Percent versions (V2V4Real leaderboard format).
        "amota_pct":   100.0 * amota,
        "amotp_pct":   100.0 * amotp,
        "samota_pct":  100.0 * samota,
        "mota_pct":    100.0 * mota,
        "mt_pct":      100.0 * mt,
        "ml_pct":      100.0 * ml,
        # Same values in [0, 1].
        "amota":  amota,
        "amotp":  amotp,
        "samota": samota,
        "mota":   mota,
        "mt":     mt,
        "ml":     ml,
        # Diagnostics.
        "mota_per_recall": mota_per_recall,
        "motp_per_recall": motp_per_recall,
        "tp_total":  final_tp,
        "fp_total":  final_fp,
        "fn_total":  final_fn,
        "ids_total": final_ids,
        "gt_total":  gt_total,
        "n_unique_gts": n_gts,
        "max_recall": max_recall,
        "num_frames": n_frames,
        "recall_thresholds": list(recall_thresholds),
        "match_threshold_m": threshold,
    }


def compute_hota_iou(
    per_frame: List[Dict],
    iou_threshold: float = 0.25,
    ignore_unmatched_fps: bool = False,
    match_3d: bool = False,
    convex_hull: bool = False,
) -> Dict:
    """Compute HOTA and its components (DetA, AssA, LocA) using BEV-IoU matching.

    Follows Luiten et al. IJCV 2021 at a single IoU threshold (not the
    multi-alpha average used in pedestrian MOT benchmarks).

    Args:
        per_frame: same tape format as compute_ab3dmot_metrics_iou —
            [{"frame_idx": int, "tracks": [(id,x,y,w,l,yaw,score), ...],
              "gts": [(id,x,y,w,l,yaw), ...]}]
        iou_threshold: BEV-IoU gate for TP matching.
        ignore_unmatched_fps: if True, unmatched tracks don't count as FP
            (V2V4Real protocol).

    Returns:
        dict with keys: hota, deta, assa, loca  (all in [0, 1])
    """
    frame_matches: List[List[Tuple]] = []   # (track_id, gt_id, iou) per frame
    frame_fp_counts: List[int] = []
    frame_fn_counts: List[int] = []
    frame_track_id_sets: List[set] = []
    frame_gt_id_sets: List[set] = []

    for f in per_frame:
        tracks = f["tracks"]
        gts = f["gts"]
        if match_3d:
            matches, unmatched_t, unmatched_g = match_frame_3d_iou(tracks, gts, iou_threshold)
        else:
            matches, unmatched_t, unmatched_g = match_frame_iou(tracks, gts, iou_threshold, convex_hull=convex_hull)

        id_matches = []
        for t_idx, g_idx, cost in matches:
            iou = 1.0 - cost
            id_matches.append((tracks[t_idx][0], gts[g_idx][0], iou))

        fp = 0 if ignore_unmatched_fps else len(unmatched_t)
        fn = len(unmatched_g)

        frame_matches.append(id_matches)
        frame_fp_counts.append(fp)
        frame_fn_counts.append(fn)
        frame_track_id_sets.append({t[0] for t in tracks})
        frame_gt_id_sets.append({g[0] for g in gts})

    total_tp = sum(len(m) for m in frame_matches)
    total_fp = sum(frame_fp_counts)
    total_fn = sum(frame_fn_counts)
    deta = total_tp / max(1, total_tp + total_fp + total_fn)

    if total_tp == 0:
        return {"hota": 0.0, "deta": deta, "assa": 0.0, "loca": 0.0}

    all_ious = [iou for fm in frame_matches for _, _, iou in fm]
    loca = float(np.mean(all_ious)) if all_ious else 0.0

    # Build frame-presence index per track/gt id
    track_frame_idx: Dict = defaultdict(list)
    gt_frame_idx: Dict = defaultdict(list)
    for t, tids in enumerate(frame_track_id_sets):
        for tid in tids:
            track_frame_idx[tid].append(t)
    for t, gids in enumerate(frame_gt_id_sets):
        for gid in gids:
            gt_frame_idx[gid].append(t)

    frame_match_pair_sets = [
        {(tid, gid) for tid, gid, _ in fm} for fm in frame_matches
    ]

    # A(c) per unique (track_id, gt_id) pair — cache to avoid recomputing
    pair_assa: Dict[Tuple, float] = {}
    pair_tp_count: Dict[Tuple, int] = {}

    for fm in frame_matches:
        for track_id, gt_id, _ in fm:
            pair = (track_id, gt_id)
            pair_tp_count[pair] = pair_tp_count.get(pair, 0) + 1
            if pair in pair_assa:
                continue
            p_frames = set(track_frame_idx[track_id])
            g_frames = set(gt_frame_idx[gt_id])
            tpa = fpa = fna = 0
            for f in p_frames | g_frames:
                p_here = track_id in frame_track_id_sets[f]
                g_here = gt_id in frame_gt_id_sets[f]
                if p_here and g_here and pair in frame_match_pair_sets[f]:
                    tpa += 1
                else:
                    if p_here:
                        fpa += 1
                    if g_here:
                        fna += 1
            pair_assa[pair] = tpa / max(1, tpa + fpa + fna)

    total_assa = sum(pair_assa[p] * cnt for p, cnt in pair_tp_count.items())
    assa = total_assa / max(1, total_tp)
    hota = math.sqrt(deta * assa)

    return {"hota": hota, "deta": deta, "assa": assa, "loca": loca}


# Backwards-compatible alias for any existing callers.
def compute_amota_amotp(per_frame, threshold=CENTER_DIST_THRESHOLD_M,
                       recall_thresholds=RECALL_THRESHOLDS):
    """Compatibility shim — calls compute_ab3dmot_metrics()."""
    return compute_ab3dmot_metrics(per_frame, threshold, recall_thresholds)


def _count_ids(matched: List[Tuple[int, int, int]]) -> int:
    """Count GT identity switches across consecutive frames."""
    if not matched:
        return 0
    by_gt: Dict[int, List[Tuple[int, int]]] = {}
    for fidx, tid, gid in matched:
        by_gt.setdefault(gid, []).append((fidx, tid))
    ids = 0
    for gid, entries in by_gt.items():
        entries.sort(key=lambda x: x[0])
        prev_tid = None
        for fidx, tid in entries:
            if prev_tid is not None and tid != prev_tid:
                ids += 1
            prev_tid = tid
    return ids
