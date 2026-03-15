import math
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


class MatchClass:
    def __init__(self, id, x, y, covariance, dx, dy, d_confidence, confidence, trust_score, object_type, time, width, length, angle, width_std=0.5, length_std=0.5):
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




class GlobalTracked:
    def __init__(self, detected_object, time, track_id, fusion_mode, use_trust_scoring=True, passthrough_covariance=False, filter_class=None):
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
        self.width = detected_object.dimensions[0]
        self.length = detected_object.dimensions[1]
        self.angle = detected_object.angle

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
                               detected_object.width_std, detected_object.length_std)
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
                               detected_object.width_std, detected_object.length_std)
        self.match_list.append(new_match)
        
        # Update type voting
        self._update_type_vote(detected_object.type)

        self.last_measurement = time

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
        self.kalman.fusion(self.match_list, time, monitor, participant_trust_scores)
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
                 static_matching_cov=None):
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

    def fuseDetectionFrame(self, time, monitor=False):
        result = []
        cooperative_monitoring = []
        trupercept_monitoring = []
        detected_objects = []  # New list to store DetectedObject instances

        # First pass: run Kalman fusion on all tracks
        for track in self.tracked_list:
            trust_scores = self.participant_trust_scores if self.use_trust_scoring else None
            track.fusion(time, monitor, trust_scores)

        # Clean duplicate tracks BEFORE building output (so duplicates are excluded)
        if len(self.tracked_list) >= 1:
            self.cleanDetections()

        # Second pass: build output from surviving tracks
        for track in self.tracked_list:
            if track.fusion_steps > self.trackShowThreshold:
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
                
                # Build cost matrix: rows = tracks, cols = detections
                # Using a large finite value for "impossible" assignments (gated out)
                # Note: scipy's linear_sum_assignment can't handle inf values
                IMPOSSIBLE_COST = 1e9
                cost_matrix = np.full((num_tracks, num_detections), IMPOSSIBLE_COST)
                
                # Pre-compute track predictions with covariance
                track_predictions = []
                for track in self.tracked_list:
                    pred = track.getPositionPredictedWithCovariance(timestamp)
                    track_predictions.append(pred)
                
                # Build cost matrix with hybrid Mahalanobis + IOU metric
                for t_idx, (track, pred) in enumerate(zip(self.tracked_list, track_predictions)):
                    for d_idx, det in enumerate(detection_list):
                        det_pos = [det.centroid[0], det.centroid[1]]
                        det_bbox = detections_list_positions[d_idx]
                        # Use static matching cov if set, otherwise detection's own cov
                        if self.static_matching_cov is not None:
                            det_cov = self.static_matching_cov
                        else:
                            det_cov = det.error_covariance

                        # Ensure det_cov is 2x2
                        if det_cov is None or np.array(det_cov).size < 4:
                            det_cov = np.eye(2)
                        else:
                            det_cov = np.array(det_cov).reshape(2, 2)

                        # Compute hybrid cost
                        cost, mahal_dist, iou = utils.compute_hybrid_cost(
                            detection_pos=det_pos,
                            detection_cov=det_cov,
                            predicted_pos=[pred['x'], pred['y']],
                            P_pred=pred['P_pred'],
                            det_bbox=det_bbox,
                            pred_bbox=pred['bbox'],
                            mahal_weight=0.6,  # Weight Mahalanobis more for far detections
                            iou_weight=0.4,    # IOU helps when boxes overlap
                            mahal_gate=13.82   # 99.9% chi-squared with 2 DOF
                        )
                        
                        cost_matrix[t_idx, d_idx] = cost
                
                # Run Hungarian algorithm for optimal assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Filter out assignments that exceed cost threshold
                matched_track_indices = set()
                matched_det_indices = set()
                
                for t_idx, d_idx in zip(row_ind, col_ind):
                    if cost_matrix[t_idx, d_idx] < IMPOSSIBLE_COST:
                        # Valid match
                        self.tracked_list[t_idx].update(detection_list[d_idx], timestamp)
                        matched_track_indices.add(t_idx)
                        matched_det_indices.add(d_idx)
                
                # Create new tracks for unmatched detections
                for d_idx in range(num_detections):
                    if d_idx not in matched_det_indices:
                        new = GlobalTracked(
                            detection_list[d_idx],
                            timestamp,
                            (max_id * self.id) + self.current_track_id,
                            self.fusion_mode,
                            use_trust_scoring=self.use_trust_scoring,
                            passthrough_covariance=self.passthrough_covariance,
                            filter_class=self.filter_class,
                        )
                        if self.current_track_id < max_id:
                            self.current_track_id += 1
                        else:
                            self.current_track_id = 0
                        self.tracked_list.append(new)

            else:
                # If there are no existing tracks, create new tracks for all detections
                for dl in detection_list:
                    new = GlobalTracked(dl, timestamp, (max_id * self.id) + self.current_track_id, self.fusion_mode,
                                        use_trust_scoring=self.use_trust_scoring,
                                        passthrough_covariance=self.passthrough_covariance,
                                        filter_class=self.filter_class)
                    if self.current_track_id < max_id:
                        self.current_track_id += 1
                    else:
                        self.current_track_id = 0
                    self.tracked_list.append(new)

        # Clean up old tracks
        remove = []
        for idx, track in enumerate(self.tracked_list):
            track.relations = []
            if track.last_measurement <= (timestamp - cleanupTime):
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
            # Get position with covariance inflation
            a_i, b_i, _ = utils.ellipsify(track_i.error_covariance, 2.0)
            bbox_i = [track_i.x, track_i.y, track_i.width + a_i, track_i.length + b_i, track_i.angle]
            
            for j in range(i + 1, num_tracks):
                track_j = self.tracked_list[j]
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
            
            # Only merge if at least one track is established
            if track_i.fusion_steps < self.trackShowThreshold and track_j.fusion_steps < self.trackShowThreshold:
                continue
            
            # Keep the track with more history (more fusion steps)
            # Tiebreaker: more recent last_measurement
            if track_i.fusion_steps > track_j.fusion_steps:
                remove_set.add(j)
            elif track_j.fusion_steps > track_i.fusion_steps:
                remove_set.add(i)
            elif track_i.last_measurement >= track_j.last_measurement:
                remove_set.add(j)
            else:
                remove_set.add(i)
        
        # Remove merged tracks (in reverse order to maintain indices)
        for idx in sorted(remove_set, reverse=True):
            self.tracked_list.pop(idx)
