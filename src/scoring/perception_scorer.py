"""
Perception Scorer - Implements Conclave scoring algorithm.

This module aggregates per-frame error measurements into perception scores
for each participant (CAV/CIS) using a revolving buffer approach.

Based on the original implementation:
https://github.com/eandert/One_Tenth_Scale_Autonomous_Vehicle/blob/devel/road_side_unit/src/rsu.py

The scoring algorithm:
1. Each participant's error_std (actual_error / expected_error) is collected per frame
2. Errors are aggregated in a revolving buffer (default 200 frames)
3. Cross-participant normalization identifies relative deviations
4. Comparison to baseline (captured after warmup) detects anomalies
5. 20% deviation from baseline triggers anomaly flag
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class ParticipantBuffer:
    """Revolving buffer for a single participant's error measurements."""
    count: int = 0  # Number of measurements added
    buffer_idx: int = 0  # Current index in revolving buffer
    error_stds: List[float] = field(default_factory=list)  # Error std values
    num_errors: List[int] = field(default_factory=list)  # Detection counts per frame
    
    def is_ready(self, min_count: int) -> bool:
        """Check if buffer has enough data for scoring."""
        return self.count >= min_count
    
    def get_average(self) -> float:
        """Get average error_std from buffer."""
        if self.count == 0:
            return 0.0
        return sum(self.error_stds) / self.count


@dataclass
class TruPerceptBuffer:
    """Revolving buffer for TruPercept confidence values."""
    count: int = 0
    buffer_idx: int = 0
    confidences: List[float] = field(default_factory=list)
    
    def is_ready(self, min_count: int) -> bool:
        return self.count >= min_count
    
    def get_average(self) -> float:
        if self.count == 0:
            return 0.0
        return sum(self.confidences) / self.count


class PerceptionScorer:
    """
    Aggregates per-frame errors into perception scores using Conclave method.
    
    The scorer maintains revolving buffers for each participant and calculates
    normalized scores that can detect anomalous perception behavior.
    
    Attributes:
        buffer_size: Size of the revolving buffer (default 200 frames)
        min_detections: Minimum detections per frame to count (default 3)
        missed_detection_penalty: Penalty for sparse detections (default 3.0)
        anomaly_threshold: Score ratio that triggers anomaly (default 1.2 = 20% worse)
    """
    
    def __init__(
        self,
        buffer_size: int = 200,
        min_detections: int = 3,
        missed_detection_penalty: float = 3.0,
        anomaly_threshold: float = 1.2,
        warmup_frames: int = 25  # Frames to skip before collecting data
    ):
        self.buffer_size = buffer_size
        self.min_detections = min_detections
        self.missed_detection_penalty = missed_detection_penalty
        self.anomaly_threshold = anomaly_threshold
        self.warmup_frames = warmup_frames
        
        # Frame counter
        self.frame_count = 0
        
        # Conclave buffers: {participant_id: ParticipantBuffer}
        self.conclave_buffers: Dict[str, ParticipantBuffer] = {}
        
        # TruPercept buffers: {participant_id: TruPerceptBuffer}
        self.trupercept_buffers: Dict[str, TruPerceptBuffer] = {}
        
        # Baseline scores (captured after warmup period ends)
        self.baseline_scores: Dict[str, float] = {}
        self.baseline_captured: bool = False
        
        # Current scores and anomaly flags
        self.current_scores: Dict[str, float] = {}
        self.current_normalized_scores: Dict[str, float] = {}
        self.anomalies: Dict[str, bool] = {}
        
        # Track when anomalies were first detected
        self.anomaly_detection_times: Dict[str, int] = {}
        
        # Statistics
        self.total_frames_processed = 0
    
    def add_frame(self, participant_id: str, error_std: float, num_detections: int) -> None:
        """
        Add error measurement to revolving buffer (Conclave algorithm).
        
        Args:
            participant_id: Unique ID of the sensing participant (CAV/CIS)
            error_std: Standardized error (actual_error / expected_error)
            num_detections: Number of detections used to compute this error
        """
        # Skip warmup period
        if self.frame_count < self.warmup_frames:
            return
        
        # Require minimum detections for quality
        if num_detections < self.min_detections:
            if num_detections == 1:
                # Penalize very sparse detections (likely missed objects)
                self._add_to_buffer(participant_id, self.missed_detection_penalty, 10)
            return
        
        # Clamp error_std to reasonable range to prevent extreme outliers
        # from completely dominating the buffer average
        clamped_error_std = max(0.1, min(5.0, error_std))
        
        self._add_to_buffer(participant_id, clamped_error_std, num_detections)
    
    def _add_to_buffer(self, participant_id: str, error_std: float, num_detections: int) -> None:
        """Internal method to add data to the revolving buffer."""
        if participant_id not in self.conclave_buffers:
            self.conclave_buffers[participant_id] = ParticipantBuffer()
        
        buf = self.conclave_buffers[participant_id]
        
        if buf.count < self.buffer_size:
            # Still filling the buffer
            buf.count += 1
            buf.error_stds.append(error_std)
            buf.num_errors.append(num_detections)
            buf.buffer_idx += 1
        else:
            # Revolving buffer - replace oldest entry
            idx = buf.buffer_idx % self.buffer_size
            buf.error_stds[idx] = error_std
            buf.num_errors[idx] = num_detections
            buf.buffer_idx += 1
    
    def add_trupercept_frame(self, participant_id: str, confidence: float) -> None:
        """
        Add TruPercept confidence to revolving buffer.
        
        Args:
            participant_id: Unique ID of the sensing participant
            confidence: Detection confidence value
        """
        if self.frame_count < self.warmup_frames:
            return
        
        if participant_id not in self.trupercept_buffers:
            self.trupercept_buffers[participant_id] = TruPerceptBuffer()
        
        buf = self.trupercept_buffers[participant_id]
        
        if buf.count < self.buffer_size:
            buf.count += 1
            buf.confidences.append(confidence)
            buf.buffer_idx += 1
        else:
            idx = buf.buffer_idx % self.buffer_size
            buf.confidences[idx] = confidence
            buf.buffer_idx += 1
    
    def step(self) -> None:
        """Increment frame counter. Call this once per simulation step."""
        self.frame_count += 1
        self.total_frames_processed += 1
    
    def calculate_scores(self) -> Dict[str, float]:
        """
        Calculate normalized perception scores for all participants.
        
        Returns:
            Dictionary mapping participant_id to their current score.
            Score of 1.0 = matches baseline
            Score > 1.0 = BETTER than baseline (more trusted)
            Score < 1.0 = WORSE than baseline (less trusted)
        """
        min_buffer_fill = self.buffer_size // 2  # Require at least half-full buffer
        
        # Step 1: Calculate normalization denominator (global mean of all error_stds)
        total_sum = 0.0
        total_count = 0
        
        for pid, buf in self.conclave_buffers.items():
            if buf.is_ready(min_buffer_fill):
                total_sum += sum(buf.error_stds)
                total_count += buf.count
        
        normalizer = total_sum / total_count if total_count > 0 else 1.0
        
        # Prevent division by zero
        if normalizer == 0.0:
            normalizer = 1.0
        
        # Step 2: Calculate each participant's normalized score
        for pid, buf in self.conclave_buffers.items():
            if buf.is_ready(min_buffer_fill):
                avg_error_std = buf.get_average()
                normalized = avg_error_std / normalizer
                
                self.current_normalized_scores[pid] = normalized
                
                # Step 3: Compare to baseline (if captured)
                # INVERTED: higher score = better (more trusted)
                # final_score = baseline / normalized
                # - If normalized < baseline (performing better): score > 1.0
                # - If normalized = baseline: score = 1.0
                # - If normalized > baseline (performing worse): score < 1.0
                if self.baseline_captured and pid in self.baseline_scores:
                    baseline = self.baseline_scores[pid]
                    if normalized > 0:
                        final_score = baseline / normalized
                    else:
                        final_score = 1.0  # Avoid division by zero
                else:
                    # No baseline yet - use inverse of normalized (lower error = higher score)
                    final_score = 1.0 / normalized if normalized > 0 else 1.0
                
                self.current_scores[pid] = final_score
                
                # Step 4: Check for anomaly (score drops below threshold = worse than baseline)
                # anomaly_threshold of 0.8 means 20% worse than baseline triggers anomaly
                is_anomaly = final_score <= (1.0 / self.anomaly_threshold)
                
                # Track first anomaly detection time
                if is_anomaly and pid not in self.anomaly_detection_times:
                    self.anomaly_detection_times[pid] = self.frame_count
                
                self.anomalies[pid] = is_anomaly
        
        return self.current_scores
    
    def calculate_trupercept_scores(self) -> Dict[str, float]:
        """Calculate TruPercept scores for all participants."""
        min_buffer_fill = self.buffer_size // 2
        scores = {}
        
        for pid, buf in self.trupercept_buffers.items():
            if buf.is_ready(min_buffer_fill):
                scores[pid] = buf.get_average()
        
        return scores
    
    def capture_baseline(self) -> None:
        """
        Capture current scores as baseline.
        
        Call this after the warmup period ends to establish "healthy" scores
        that future measurements will be compared against.
        """
        # Calculate current scores
        self.calculate_scores()
        
        # Store as baseline
        self.baseline_scores = self.current_normalized_scores.copy()
        self.baseline_captured = True
        
        # Reset anomaly tracking
        self.anomalies = {}
        self.anomaly_detection_times = {}
    
    def get_anomalies(self) -> Dict[str, bool]:
        """
        Return dict of participants with anomaly flags.
        
        Returns:
            Dictionary mapping participant_id to anomaly flag (True = anomalous)
        """
        return self.anomalies.copy()
    
    def get_anomaly_participants(self) -> List[str]:
        """Return list of participant IDs that are currently flagged as anomalous."""
        return [pid for pid, is_anomaly in self.anomalies.items() if is_anomaly]
    
    def get_time_to_detection(self, participant_id: str) -> Optional[int]:
        """
        Get the number of frames from baseline capture to anomaly detection.
        
        Args:
            participant_id: The participant to check
            
        Returns:
            Number of frames, or None if not yet detected as anomalous
        """
        if participant_id in self.anomaly_detection_times:
            # Time from when baseline was captured
            return self.anomaly_detection_times[participant_id] - self.warmup_frames
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current scoring state.
        
        Returns:
            Dictionary with scoring summary information
        """
        return {
            "total_frames": self.total_frames_processed,
            "baseline_captured": self.baseline_captured,
            "num_participants": len(self.conclave_buffers),
            "num_anomalies": sum(1 for a in self.anomalies.values() if a),
            "scores": self.current_scores.copy(),
            "anomalies": self.anomalies.copy(),
            "detection_times": self.anomaly_detection_times.copy(),
        }
    
    def process_cooperative_monitoring(self, cooperative_monitoring: List[List]) -> None:
        """
        Process cooperative monitoring data from sensor fusion.
        
        This is a convenience method that takes the cooperative_monitoring
        output from Fusion.fuseDetectionFrame() and adds all entries to the scorer.
        
        Args:
            cooperative_monitoring: List of track error lists, where each entry is
                                    [participant_id, error_std, num_detections]
        """
        for track_errors in cooperative_monitoring:
            for entry in track_errors:
                if len(entry) >= 3:
                    participant_id, error_std, num_detections = entry[0], entry[1], entry[2]
                    self.add_frame(str(participant_id), error_std, num_detections)
    
    def process_trupercept_monitoring(self, trupercept_monitoring: List[List]) -> None:
        """
        Process TruPercept monitoring data from sensor fusion.
        
        Args:
            trupercept_monitoring: List of track confidence lists, where each entry is
                                   [participant_id, confidence]
        """
        for track_confidences in trupercept_monitoring:
            for entry in track_confidences:
                if len(entry) >= 2:
                    participant_id, confidence = entry[0], entry[1]
                    self.add_trupercept_frame(str(participant_id), confidence)
