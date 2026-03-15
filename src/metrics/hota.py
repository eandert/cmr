"""
HOTA (Higher Order Tracking Accuracy) metric implementation.

HOTA decomposes tracking performance into:
  - DetA (Detection Accuracy): How well detections match ground truth per-frame
  - AssA (Association Accuracy): How consistently tracks maintain correct identity over time
  - HOTA = sqrt(DetA * AssA)

Reference: Luiten et al., "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking",
           International Journal of Computer Vision, 2021.

This is a standalone implementation that reuses the IoU matching from amota.py.
"""

import math
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from metrics.amota import _match_detections_to_gt


class HotaAccumulator:
    """
    Accumulates per-frame matching data and computes HOTA at the end.

    Usage:
        acc = HotaAccumulator()
        for each frame:
            acc.update(detected_objects, ground_truth_objects)
        results = acc.compute()  # {'hota': ..., 'deta': ..., 'assa': ...}
    """

    def __init__(self, iou_threshold: float = 0.1, bbox_tolerance: float = 0.5):
        self.iou_threshold = iou_threshold
        self.bbox_tolerance = bbox_tolerance

        # Per-frame storage: list of (matches, fp_count, fn_count)
        # matches = list of (pred_id, gt_id) using actual object IDs
        self._frame_matches: List[List[Tuple]] = []
        self._frame_fp_counts: List[int] = []
        self._frame_fn_counts: List[int] = []

        # Track presence per frame: which pred_ids and gt_ids exist
        self._frame_pred_ids: List[set] = []
        self._frame_gt_ids: List[set] = []

    def update(self, detected_objects, ground_truth_objects):
        """
        Process one frame of detections vs ground truth.

        Args:
            detected_objects: list of DetectedObject (from fusion output)
            ground_truth_objects: list of GroundTruthObject
        """
        if len(ground_truth_objects) == 0 and len(detected_objects) == 0:
            return

        matches_idx, unmatched_dets, unmatched_gts = _match_detections_to_gt(
            detected_objects, ground_truth_objects,
            self.iou_threshold, self.bbox_tolerance
        )

        # Convert index-based matches to ID-based matches
        id_matches = []
        for d_idx, g_idx in matches_idx:
            pred_id = detected_objects[d_idx].vehicle_id
            gt_id = ground_truth_objects[g_idx].vehicle_id
            id_matches.append((pred_id, gt_id))

        # Track all pred/gt IDs present this frame
        pred_ids = {d.vehicle_id for d in detected_objects}
        gt_ids = {g.vehicle_id for g in ground_truth_objects}

        self._frame_matches.append(id_matches)
        self._frame_fp_counts.append(len(unmatched_dets))
        self._frame_fn_counts.append(len(unmatched_gts))
        self._frame_pred_ids.append(pred_ids)
        self._frame_gt_ids.append(gt_ids)

    def compute(self) -> Dict[str, float]:
        """
        Compute HOTA, DetA, and AssA from accumulated frame data.

        Returns:
            dict with keys: 'hota', 'deta', 'assa'
            All values in [0, 1]. Returns zeros if no data.
        """
        n_frames = len(self._frame_matches)
        if n_frames == 0:
            return {'hota': 0.0, 'deta': 0.0, 'assa': 0.0}

        # --- DetA: Detection Accuracy ---
        # DetA = total_TP / (total_TP + total_FP + total_FN)
        total_tp = sum(len(m) for m in self._frame_matches)
        total_fp = sum(self._frame_fp_counts)
        total_fn = sum(self._frame_fn_counts)

        deta = total_tp / max(1, total_tp + total_fp + total_fn)

        if total_tp == 0:
            return {'hota': 0.0, 'deta': deta, 'assa': 0.0}

        # --- AssA: Association Accuracy ---
        # For each TP match (pred_id, gt_id) at frame t, compute:
        #   TPA(c) = frames where pred_id and gt_id are matched to each other
        #   FPA(c) = frames where pred_id exists but is NOT matched to gt_id
        #   FNA(c) = frames where gt_id exists but is NOT matched to pred_id
        #   A(c) = TPA / (TPA + FPA + FNA)
        # AssA = mean of A(c) over all TP matches across all frames

        # Build lookup: for each frame, the set of (pred_id, gt_id) matched pairs
        frame_match_sets = [set(m) for m in self._frame_matches]

        # Build per-pred and per-gt match lookups for efficiency
        # pred_to_gt[frame_idx] = {pred_id: gt_id} for matched preds
        # gt_to_pred[frame_idx] = {gt_id: pred_id} for matched gts
        pred_to_gt: List[Dict] = []
        gt_to_pred: List[Dict] = []
        for matches in self._frame_matches:
            p2g = {}
            g2p = {}
            for pid, gid in matches:
                p2g[pid] = gid
                g2p[gid] = pid
            pred_to_gt.append(p2g)
            gt_to_pred.append(g2p)

        # For efficiency, build frame indices where each pred_id / gt_id appears
        pred_frames: Dict[object, List[int]] = defaultdict(list)
        gt_frames: Dict[object, List[int]] = defaultdict(list)
        for t in range(n_frames):
            for pid in self._frame_pred_ids[t]:
                pred_frames[pid].append(t)
            for gid in self._frame_gt_ids[t]:
                gt_frames[gid].append(t)

        # Compute A(c) for each TP match and accumulate
        total_association_score = 0.0
        total_tp_matches = 0

        for t, matches in enumerate(self._frame_matches):
            for pred_id, gt_id in matches:
                # Frames where pred_id exists
                p_frames = set(pred_frames[pred_id])
                # Frames where gt_id exists
                g_frames = set(gt_frames[gt_id])
                # Union of frames where either exists
                all_relevant_frames = p_frames | g_frames

                tpa = 0
                fpa = 0
                fna = 0
                for f in all_relevant_frames:
                    pred_present = f < len(self._frame_pred_ids) and pred_id in self._frame_pred_ids[f]
                    gt_present = f < len(self._frame_gt_ids) and gt_id in self._frame_gt_ids[f]

                    if pred_present and gt_present and (pred_id, gt_id) in frame_match_sets[f]:
                        tpa += 1
                    else:
                        if pred_present:
                            fpa += 1
                        if gt_present:
                            fna += 1

                a_c = tpa / max(1, tpa + fpa + fna)
                total_association_score += a_c
                total_tp_matches += 1

        assa = total_association_score / max(1, total_tp_matches)

        hota = math.sqrt(deta * assa)

        return {'hota': hota, 'deta': deta, 'assa': assa}

    def reset(self):
        """Clear all accumulated data."""
        self._frame_matches.clear()
        self._frame_fp_counts.clear()
        self._frame_fn_counts.clear()
        self._frame_pred_ids.clear()
        self._frame_gt_ids.clear()
