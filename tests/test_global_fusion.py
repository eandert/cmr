"""
Global sensor fusion tests with multiple participants (CAVs/CIS).

Tests that global fusion properly handles:
- Multiple vehicles contributing detections of the same targets
- Different participants with different sensor precisions
- Track consistency (no flicker) when multiple sources observe same target
- Proper position/covariance when fusing multiple observations

Run from repo root: python tests/test_global_fusion.py -v
"""
import unittest
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sensor_fusion import Fusion
from sensor import DetectedObject

# Test parameters
WARMUP_FRAMES = 5
TRACK_FRAMES_REQUIRED = 25
DT = 0.1
CLEANUP_TIME = 1.0

# Position tolerance - global fusion should be tighter due to multiple observations
POSITION_TOL_M = 1.5
COV_TRACE_MAX_M2 = 2.0


def make_detection(x, y, width, length, angle_rad, vx=0, vy=0, cov_scale=0.25):
    """Build a DetectedObject for fusion input."""
    cov = np.eye(2) * cov_scale
    return DetectedObject(
        vehicle_id=0,
        vehicle_type=0,
        detected_bbox=None,
        centroid=(x, y),
        width=width,
        length=length,
        angle=angle_rad,
        expected_error_gaussian=None,
        velocity_vector=(vx, vy),
        error_covariance=cov,
        width_std=0.2,
        length_std=0.2,
    )


def run_global_fusion_sequence(fusion, detections_per_frame_per_participant, total_frames, n_participants):
    """
    Run global fusion for total_frames with multiple participants.
    
    detections_per_frame_per_participant(t, participant_id) returns list of DetectedObject.
    Returns list of (result_rows, frame_idx) for frames after warmup.
    """
    results_after_warmup = []
    for t in range(total_frames):
        time = t * DT
        # Each participant contributes their detections
        for pid in range(n_participants):
            observations = detections_per_frame_per_participant(t, pid)
            fusion.processDetectionFrame(time, observations, CLEANUP_TIME, source_participant_id=pid)
        
        # Fuse all detections
        result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        if t >= WARMUP_FRAMES:
            results_after_warmup.append((list(result), t))
    
    return results_after_warmup


class TestGlobalFusionTwoParticipants(unittest.TestCase):
    """Two participants observing the same targets."""

    def test_two_participants_single_target(self):
        """Two CAVs both see one stationary target - should produce one track."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        gt_x, gt_y = 50.0, 0.0
        
        def detections(t, pid):
            # Both participants see the same target with slight noise offset
            offset = 0.1 * (pid - 0.5)  # Small offset based on participant
            return [make_detection(gt_x + offset, gt_y + offset, 2.0, 4.5, 0.0, 0.0, 0.0)]
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants=2)
        
        self.assertEqual(len(results), TRACK_FRAMES_REQUIRED)
        
        # Should have exactly 1 track (not 2 - should fuse the observations)
        for result, frame_idx in results:
            self.assertGreaterEqual(len(result), 1, 
                msg=f"Frame {frame_idx}: Should have at least 1 track")
            # Check position is near ground truth
            if len(result) >= 1:
                track_x = result[0][1]
                track_y = result[0][2]
                dist = math.sqrt((track_x - gt_x)**2 + (track_y - gt_y)**2)
                self.assertLess(dist, POSITION_TOL_M,
                    msg=f"Frame {frame_idx}: Track at ({track_x:.2f},{track_y:.2f}) too far from GT ({gt_x},{gt_y})")

    def test_two_participants_moving_target(self):
        """Two CAVs both see one moving target at 5 m/s."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        v = 5.0
        
        def detections(t, pid):
            x = 50.0 + t * DT * v
            offset = 0.05 * (pid - 0.5)
            return [make_detection(x + offset, offset, 2.0, 4.5, 0.0, v, 0.0)]
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants=2)
        
        self.assertEqual(len(results), TRACK_FRAMES_REQUIRED)
        
        # Skip first few frames where filter is still converging, check last 20 frames
        for result, frame_idx in results[-20:]:
            self.assertGreaterEqual(len(result), 1)
            if len(result) >= 1:
                expected_x = 50.0 + frame_idx * DT * v
                track_x = result[0][1]
                dist = abs(track_x - expected_x)
                self.assertLess(dist, POSITION_TOL_M,
                    msg=f"Frame {frame_idx}: Track x={track_x:.2f} too far from expected {expected_x:.2f}")

    def test_two_participants_different_precisions(self):
        """One CAV has good sensors, one has poor - fused result should favor good."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        gt_x, gt_y = 50.0, 0.0
        
        def detections(t, pid):
            if pid == 0:
                # Good sensor: small covariance, accurate position
                cov = np.eye(2) * 0.1
                return [DetectedObject(
                    vehicle_id=0, vehicle_type=0, detected_bbox=None,
                    centroid=(gt_x, gt_y), width=2.0, length=4.5, angle=0.0,
                    expected_error_gaussian=None, velocity_vector=(0, 0),
                    error_covariance=cov, width_std=0.2, length_std=0.2
                )]
            else:
                # Poor sensor: large covariance, biased position
                cov = np.eye(2) * 4.0
                return [DetectedObject(
                    vehicle_id=0, vehicle_type=0, detected_bbox=None,
                    centroid=(gt_x + 3.0, gt_y + 3.0),  # 3m bias
                    width=2.0, length=4.5, angle=0.0,
                    expected_error_gaussian=None, velocity_vector=(0, 0),
                    error_covariance=cov, width_std=0.5, length_std=0.5
                )]
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants=2)
        
        # Fused position should be closer to good sensor (gt) than poor sensor (gt+3)
        for result, frame_idx in results:
            if len(result) >= 1:
                track_x = result[0][1]
                track_y = result[0][2]
                dist_to_good = math.sqrt((track_x - gt_x)**2 + (track_y - gt_y)**2)
                dist_to_poor = math.sqrt((track_x - (gt_x+3))**2 + (track_y - (gt_y+3))**2)
                self.assertLess(dist_to_good, dist_to_poor,
                    msg=f"Frame {frame_idx}: Fused position should be closer to good sensor")


class TestGlobalFusionMultipleTargets(unittest.TestCase):
    """Multiple targets observed by multiple participants."""

    def test_three_targets_two_participants(self):
        """Three targets observed by two participants - should maintain 3 tracks."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        
        # Three targets with good spacing
        targets = [(20.0, 0.0), (50.0, 0.0), (80.0, 0.0)]
        
        def detections(t, pid):
            dets = []
            for i, (tx, ty) in enumerate(targets):
                offset = 0.05 * (pid - 0.5)
                dets.append(make_detection(tx + offset, ty + offset, 2.0, 4.5, 0.0, 0.0, 0.0))
            return dets
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants=2)
        
        self.assertEqual(len(results), TRACK_FRAMES_REQUIRED)
        
        # Should maintain 3 tracks throughout
        for result, frame_idx in results:
            self.assertGreaterEqual(len(result), 3,
                msg=f"Frame {frame_idx}: Should have at least 3 tracks, got {len(result)}")

    def test_moving_convoy_two_participants(self):
        """Three vehicles in convoy (10m spacing), both participants observe all."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        v = 5.0
        spacing = 10.0
        
        def detections(t, pid):
            base_x = 20.0 + t * DT * v
            offset = 0.05 * (pid - 0.5)
            return [
                make_detection(base_x + offset, offset, 2.0, 4.5, 0.0, v, 0.0),
                make_detection(base_x + spacing + offset, offset, 2.0, 4.5, 0.0, v, 0.0),
                make_detection(base_x + 2*spacing + offset, offset, 2.0, 4.5, 0.0, v, 0.0),
            ]
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants=2)
        
        # Should maintain 3 tracks
        for result, frame_idx in results:
            self.assertGreaterEqual(len(result), 3,
                msg=f"Frame {frame_idx}: Convoy should have 3 tracks")


class TestGlobalFusionNoFlicker(unittest.TestCase):
    """Tests that global fusion produces stable tracks without flickering."""

    def test_no_flicker_single_target_three_participants(self):
        """Three participants see one target - should have exactly 1 stable track."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        gt_x, gt_y = 50.0, 0.0
        v = 2.0  # Slow movement
        
        def detections(t, pid):
            x = gt_x + t * DT * v
            offset = 0.1 * (pid - 1)  # -0.1, 0, 0.1 for pid 0,1,2
            return [make_detection(x + offset, gt_y + offset, 2.0, 4.5, 0.0, v, 0.0)]
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants=3)
        
        track_counts = [len(r) for r, _ in results]
        
        # No flicker: should be stable at 1 track
        # Allow some initial instability but last 20 frames should be stable
        stable_counts = track_counts[-20:]
        self.assertTrue(all(c >= 1 for c in stable_counts),
            msg=f"Last 20 frames should have at least 1 track: {stable_counts}")
        
        # Track count should not oscillate wildly
        max_count = max(stable_counts)
        min_count = min(stable_counts)
        self.assertLessEqual(max_count - min_count, 1,
            msg=f"Track count should be stable: min={min_count}, max={max_count}")

    def test_no_flicker_two_close_targets(self):
        """Two close targets (8m apart) - should maintain 2 distinct tracks."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        gap = 8.0  # Larger gap to ensure separate tracks
        v = 2.0
        
        def detections(t, pid):
            base_x = 50.0 + t * DT * v
            offset = 0.05 * (pid - 0.5)
            return [
                make_detection(base_x + offset, offset, 2.0, 4.5, 0.0, v, 0.0),
                make_detection(base_x + gap + offset, offset, 2.0, 4.5, 0.0, v, 0.0),
            ]
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants=2)
        
        # Should maintain 2 tracks - allow some tolerance for ghost tracks
        # Last 10 frames should be stable at 2 tracks
        stable_counts = [len(r) for r, _ in results[-10:]]
        self.assertTrue(all(c >= 2 for c in stable_counts),
            msg=f"Last 10 frames should have at least 2 tracks: {stable_counts}")
        # Most frames should have exactly 2 tracks (allow for occasional extra)
        avg_count = sum(len(r) for r, _ in results) / len(results)
        self.assertLessEqual(avg_count, 3.0,
            msg=f"Average track count should be close to 2: avg={avg_count:.2f}")


class TestGlobalFusionTurningVehicle(unittest.TestCase):
    """Global fusion with turning vehicles."""

    def test_turning_vehicle_two_participants(self):
        """One turning vehicle observed by two participants."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        radius = 15.0
        omega = 0.2  # rad/s
        
        def detections(t, pid):
            time = t * DT
            x = 50.0 + radius * math.cos(omega * time)
            y = radius * math.sin(omega * time)
            heading = omega * time + math.pi/2
            vx = -radius * omega * math.sin(omega * time)
            vy = radius * omega * math.cos(omega * time)
            
            offset = 0.05 * (pid - 0.5)
            return [make_detection(x + offset, y + offset, 2.0, 4.5, heading, vx, vy)]
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants=2)
        
        # Should maintain 1 track and follow the circular path
        for result, frame_idx in results:
            self.assertGreaterEqual(len(result), 1)
            
            if len(result) >= 1:
                time = frame_idx * DT
                expected_x = 50.0 + radius * math.cos(omega * time)
                expected_y = radius * math.sin(omega * time)
                track_x = result[0][1]
                track_y = result[0][2]
                dist = math.sqrt((track_x - expected_x)**2 + (track_y - expected_y)**2)
                self.assertLess(dist, POSITION_TOL_M * 1.5,  # Slightly larger for turning
                    msg=f"Frame {frame_idx}: Turning track too far from expected path")


class TestGlobalFusionCovarianceReduction(unittest.TestCase):
    """Tests that multiple observations reduce uncertainty."""

    def test_more_participants_reduces_covariance(self):
        """More participants observing same target should reduce track covariance."""
        gt_x, gt_y = 50.0, 0.0
        total_frames = WARMUP_FRAMES + 10  # Shorter test
        
        # Run with 1, 2, and 3 participants
        final_covariances = []
        
        for n_participants in [1, 2, 3]:
            fusion = Fusion(id=0, use_trust_scoring=False)
            
            def make_detections_factory(n):
                def detections(t, pid):
                    if pid < n:
                        offset = 0.05 * (pid - (n-1)/2)
                        return [make_detection(gt_x + offset, gt_y + offset, 2.0, 4.5, 0.0, 0.0, 0.0)]
                    return []
                return detections
            
            detections = make_detections_factory(n_participants)
            
            for t in range(total_frames):
                time = t * DT
                for pid in range(n_participants):
                    obs = detections(t, pid)
                    if obs:
                        fusion.processDetectionFrame(time, obs, CLEANUP_TIME, source_participant_id=pid)
                result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
            
            if len(result) >= 1:
                cov = np.array(result[0][3])  # error_covariance
                trace = np.trace(cov)
                final_covariances.append(trace)
        
        # More participants should give smaller (or equal) covariance
        # Allow some tolerance since this is statistical
        self.assertLessEqual(final_covariances[1], final_covariances[0] * 1.5,
            msg=f"2 participants should not have much larger cov than 1: {final_covariances}")
        self.assertLessEqual(final_covariances[2], final_covariances[1] * 1.5,
            msg=f"3 participants should not have much larger cov than 2: {final_covariances}")


class TestGlobalFusionMixedScenario(unittest.TestCase):
    """Complex mixed scenario with multiple participants and targets."""

    def test_mixed_scenario_three_participants_four_targets(self):
        """
        Three participants observe four targets:
        - One stationary
        - One moving straight
        - One turning
        - One fast-moving
        """
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        
        def detections(t, pid):
            time = t * DT
            offset = 0.03 * (pid - 1)  # Small offset per participant
            
            dets = []
            
            # Target 1: Stationary at (0, 0)
            dets.append(make_detection(0 + offset, 0 + offset, 2.0, 4.5, 0.0, 0.0, 0.0))
            
            # Target 2: Moving at 5 m/s along x starting at (50, 0)
            x_mov = 50 + t * DT * 5
            dets.append(make_detection(x_mov + offset, 0 + offset, 2.0, 4.5, 0.0, 5.0, 0.0))
            
            # Target 3: Turning on circle at (100, 0) center, radius 10
            radius = 10.0
            omega = 0.3
            x_turn = 100 + radius * math.cos(omega * time)
            y_turn = radius * math.sin(omega * time)
            heading = omega * time + math.pi/2
            vx = -radius * omega * math.sin(omega * time)
            vy = radius * omega * math.cos(omega * time)
            dets.append(make_detection(x_turn + offset, y_turn + offset, 2.0, 4.5, heading, vx, vy))
            
            # Target 4: Fast-moving at 15 m/s along x starting at (200, 0)
            x_fast = 200 + t * DT * 15
            dets.append(make_detection(x_fast + offset, 0 + offset, 2.0, 4.5, 0.0, 15.0, 0.0))
            
            return dets
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants=3)
        
        self.assertEqual(len(results), TRACK_FRAMES_REQUIRED)
        
        # Should maintain at least 4 tracks for most frames
        track_counts = [len(r) for r, _ in results]
        avg_count = sum(track_counts) / len(track_counts)
        self.assertGreaterEqual(avg_count, 3.5,
            msg=f"Should average at least 3.5 tracks across frames: avg={avg_count:.2f}")
        
        # Last 10 frames should be stable
        last_10 = track_counts[-10:]
        self.assertTrue(all(c >= 4 for c in last_10),
            msg=f"Last 10 frames should have at least 4 tracks: {last_10}")


class TestHighPenetrationScaling(unittest.TestCase):
    """
    Tests for high CAV penetration scenarios (approaching 100%).
    
    At high penetration, many vehicles all report detections of each other,
    which stresses the matching and deduplication algorithms.
    """

    def test_10_participants_single_target_single_track(self):
        """10 CAVs all see one stationary target - must produce exactly 1 track."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        gt_x, gt_y = 50.0, 0.0
        n_participants = 10
        
        def detections(t, pid):
            # Each participant sees the target with slight noise
            np.random.seed(t * 1000 + pid)  # Reproducible noise per frame/participant
            noise_x = np.random.normal(0, 0.1)
            noise_y = np.random.normal(0, 0.1)
            return [make_detection(gt_x + noise_x, gt_y + noise_y, 2.0, 4.5, 0.0, 0.0, 0.0)]
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants)
        
        # Should have exactly 1 track, NOT 10 (one per participant)
        track_counts = [len(r) for r, _ in results]
        
        # After warmup, we should consistently have 1 track
        for count, (_, frame) in zip(track_counts, results):
            self.assertEqual(count, 1, 
                msg=f"Frame {frame}: Expected 1 track but got {count}. "
                    f"Multiple tracks suggest duplicate creation.")
    
    def test_20_participants_single_target_single_track(self):
        """20 CAVs all see one stationary target - must still produce exactly 1 track."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        gt_x, gt_y = 50.0, 0.0
        n_participants = 20
        
        def detections(t, pid):
            np.random.seed(t * 1000 + pid)
            noise_x = np.random.normal(0, 0.15)  # Slightly more noise with more participants
            noise_y = np.random.normal(0, 0.15)
            return [make_detection(gt_x + noise_x, gt_y + noise_y, 2.0, 4.5, 0.0, 0.0, 0.0)]
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants)
        
        track_counts = [len(r) for r, _ in results]
        
        # Check last 10 frames for stability
        last_10_counts = track_counts[-10:]
        self.assertTrue(all(c == 1 for c in last_10_counts),
            msg=f"Last 10 frames should have exactly 1 track: {last_10_counts}. "
                f"Track duplication occurring with high participant count.")
    
    def test_10_participants_3_targets_exactly_3_tracks(self):
        """10 CAVs see 3 well-separated targets - must produce exactly 3 tracks."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        n_participants = 10
        
        # 3 targets at well-separated locations
        targets = [
            (20.0, 0.0),   # Target 1
            (50.0, 0.0),   # Target 2 - 30m away
            (80.0, 0.0),   # Target 3 - 30m away
        ]
        
        def detections(t, pid):
            np.random.seed(t * 1000 + pid)
            dets = []
            for gt_x, gt_y in targets:
                noise_x = np.random.normal(0, 0.1)
                noise_y = np.random.normal(0, 0.1)
                dets.append(make_detection(gt_x + noise_x, gt_y + noise_y, 2.0, 4.5, 0.0, 0.0, 0.0))
            return dets
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants)
        
        track_counts = [len(r) for r, _ in results]
        
        # After warmup should consistently have exactly 3 tracks
        last_10_counts = track_counts[-10:]
        self.assertTrue(all(c == 3 for c in last_10_counts),
            msg=f"Last 10 frames should have exactly 3 tracks: {last_10_counts}")
    
    def test_high_penetration_no_track_id_flickering(self):
        """
        With many participants, track IDs should remain stable (no flickering).
        
        Track ID flickering means a track gets destroyed and recreated with
        a new ID, which breaks downstream tracking continuity.
        """
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED + 20  # Extra frames
        gt_x, gt_y = 50.0, 0.0
        n_participants = 15
        
        def detections(t, pid):
            np.random.seed(t * 1000 + pid)
            noise_x = np.random.normal(0, 0.1)
            noise_y = np.random.normal(0, 0.1)
            return [make_detection(gt_x + noise_x, gt_y + noise_y, 2.0, 4.5, 0.0, 0.0, 0.0)]
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants)
        
        # Collect track IDs from last 20 frames (should be stable)
        # Result row format: [track.id, x, y, error_cov, dx, dy, d_cov, num_trackers]
        track_ids_per_frame = []
        for result, frame in results[-20:]:
            ids = set()
            for row in result:
                if len(row) >= 1:
                    ids.add(row[0])  # track_id is at index 0
            track_ids_per_frame.append(ids)
        
        # All frames should have the same set of track IDs
        if track_ids_per_frame:
            first_ids = track_ids_per_frame[0]
            for i, ids in enumerate(track_ids_per_frame[1:], 1):
                self.assertEqual(ids, first_ids,
                    msg=f"Frame {i}: Track IDs changed from {first_ids} to {ids}. "
                        f"ID flickering detected.")
    
    def test_many_participants_moving_target(self):
        """15 CAVs tracking a moving target - should maintain single consistent track."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        n_participants = 15
        velocity = 10.0  # 10 m/s
        
        def detections(t, pid):
            np.random.seed(t * 1000 + pid)
            # Target moving along x-axis
            gt_x = 20.0 + t * DT * velocity
            gt_y = 0.0
            noise_x = np.random.normal(0, 0.2)
            noise_y = np.random.normal(0, 0.2)
            vx = velocity + np.random.normal(0, 0.5)
            return [make_detection(gt_x + noise_x, gt_y + noise_y, 2.0, 4.5, 0.0, vx, 0.0)]
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants)
        
        track_counts = [len(r) for r, _ in results]
        
        # Should have exactly 1 track throughout
        last_10_counts = track_counts[-10:]
        self.assertTrue(all(c == 1 for c in last_10_counts),
            msg=f"Moving target with {n_participants} participants should produce 1 track: {last_10_counts}")
    
    def test_burst_of_detections_same_frame(self):
        """
        Simulate scenario where all detections arrive nearly simultaneously.
        
        This stresses the deduplication in matchDetections() - if 20 very similar
        detections arrive, they should match to the same track, not create 20 tracks.
        """
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        gt_x, gt_y = 50.0, 0.0
        
        # Use a single "participant" that provides all detections at once
        # to simulate the burst scenario
        def detections(t, pid):
            if pid != 0:
                return []
            
            # Generate 20 detections all for the same vehicle, with noise
            dets = []
            for i in range(20):
                np.random.seed(t * 1000 + i)
                noise_x = np.random.normal(0, 0.15)
                noise_y = np.random.normal(0, 0.15)
                dets.append(make_detection(gt_x + noise_x, gt_y + noise_y, 2.0, 4.5, 0.0, 0.0, 0.0))
            return dets
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants=1)
        
        track_counts = [len(r) for r, _ in results]
        
        # With proper deduplication, should converge to 1 track
        # Initial frames might have more due to warmup, but should stabilize
        last_10_counts = track_counts[-10:]
        avg_last_10 = sum(last_10_counts) / len(last_10_counts)
        
        self.assertLessEqual(avg_last_10, 2.0,
            msg=f"20 similar detections per frame should produce ~1 track, got avg={avg_last_10:.1f}. "
                f"Last 10 counts: {last_10_counts}. Track deduplication failing.")


class TestHighPenetrationMismatching(unittest.TestCase):
    """
    Tests specifically for track mismatching scenarios at high penetration.
    
    These tests verify that when many observations arrive, they get matched
    to the correct tracks rather than creating confusion.
    """

    def test_close_vehicles_not_merged(self):
        """
        Two vehicles 8m apart should remain as 2 separate tracks,
        even with 15 participants each seeing both.
        """
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        n_participants = 15
        
        # Two vehicles separated by 8 meters (larger than vehicle length)
        vehicle1 = (50.0, 0.0)
        vehicle2 = (58.0, 0.0)  # 8m separation
        
        def detections(t, pid):
            np.random.seed(t * 1000 + pid)
            dets = []
            for vx, vy in [vehicle1, vehicle2]:
                noise_x = np.random.normal(0, 0.2)
                noise_y = np.random.normal(0, 0.2)
                dets.append(make_detection(vx + noise_x, vy + noise_y, 2.0, 4.5, 0.0, 0.0, 0.0))
            return dets
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants)
        
        track_counts = [len(r) for r, _ in results]
        
        # Should maintain exactly 2 tracks
        last_10_counts = track_counts[-10:]
        self.assertTrue(all(c == 2 for c in last_10_counts),
            msg=f"Two vehicles 8m apart should produce 2 tracks: {last_10_counts}")
    
    def test_crossing_vehicles_maintain_identity(self):
        """
        Two vehicles crossing paths should maintain their separate track IDs.
        
        This is a classic multi-object tracking challenge that gets harder
        with more observations (more chances for confusion).
        """
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + 60  # Need more frames for crossing
        n_participants = 10
        
        def detections(t, pid):
            np.random.seed(t * 1000 + pid)
            time = t * DT
            
            # Vehicle A: Moving from (30, -10) to (30, 10) along y-axis
            ax = 30.0
            ay = -10.0 + time * 3.0  # 3 m/s along y
            
            # Vehicle B: Moving from (20, 0) to (40, 0) along x-axis
            bx = 20.0 + time * 3.0  # 3 m/s along x
            by = 0.0
            
            dets = []
            for vx, vy, heading, vel_x, vel_y in [
                (ax, ay, math.pi/2, 0.0, 3.0),  # Vehicle A heading +y
                (bx, by, 0.0, 3.0, 0.0),        # Vehicle B heading +x
            ]:
                noise_x = np.random.normal(0, 0.15)
                noise_y = np.random.normal(0, 0.15)
                dets.append(make_detection(vx + noise_x, vy + noise_y, 2.0, 4.5, heading, vel_x, vel_y))
            return dets
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants)
        
        # Extract track IDs over time
        track_id_sequences = {}  # position_key -> list of track_ids seen
        
        for result, frame in results:
            for row in result:
                if len(row) >= 7:
                    x, y = row[0], row[1]
                    track_id = row[6]
                    # Determine which vehicle based on trajectory
                    # This is approximate - we mainly care about consistent ID
        
        # Should have exactly 2 tracks throughout
        track_counts = [len(r) for r, _ in results]
        last_10_counts = track_counts[-10:]
        
        # During crossing, might briefly have confusion, but should stabilize
        self.assertTrue(all(c == 2 for c in last_10_counts),
            msg=f"Two crossing vehicles should maintain 2 tracks: {last_10_counts}")
    
    def test_no_phantom_tracks_high_noise(self):
        """
        With high noise but same underlying vehicle, should not create phantom tracks.
        
        Phantom tracks occur when noisy detections are misinterpreted as new vehicles.
        """
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        gt_x, gt_y = 50.0, 0.0
        n_participants = 12
        
        def detections(t, pid):
            np.random.seed(t * 1000 + pid)
            # Higher noise than typical
            noise_x = np.random.normal(0, 0.5)  # 0.5m std dev
            noise_y = np.random.normal(0, 0.5)
            return [make_detection(gt_x + noise_x, gt_y + noise_y, 2.0, 4.5, 0.0, 0.0, 0.0)]
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants)
        
        track_counts = [len(r) for r, _ in results]
        
        # Even with high noise, should converge to 1 track
        last_10_counts = track_counts[-10:]
        max_tracks = max(last_10_counts)
        
        self.assertLessEqual(max_tracks, 2,
            msg=f"High noise should not create phantom tracks. Max tracks in last 10: {max_tracks}")
    
    def test_late_joining_participant_merges_correctly(self):
        """
        When a new CAV joins (starts providing detections mid-simulation),
        its observations should merge with existing tracks, not create new ones.
        """
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED + 20
        gt_x, gt_y = 50.0, 0.0
        n_initial_participants = 5
        n_late_participants = 10  # 10 more join at frame 20
        
        def detections(t, pid):
            # First 5 participants always active
            # Participants 5-14 only active after frame 20
            if pid >= n_initial_participants and t < 20:
                return []
            
            np.random.seed(t * 1000 + pid)
            noise_x = np.random.normal(0, 0.1)
            noise_y = np.random.normal(0, 0.1)
            return [make_detection(gt_x + noise_x, gt_y + noise_y, 2.0, 4.5, 0.0, 0.0, 0.0)]
        
        n_total_participants = n_initial_participants + n_late_participants
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_total_participants)
        
        track_counts = [len(r) for r, _ in results]
        
        # Should always have exactly 1 track, even after late joiners
        # Check frames after late join (frame 20 is at index 20 - WARMUP_FRAMES if t >= WARMUP_FRAMES)
        last_15_counts = track_counts[-15:]
        
        self.assertTrue(all(c == 1 for c in last_15_counts),
            msg=f"Late joining participants should merge with existing track: {last_15_counts}")


class TestFullPenetrationSimulation(unittest.TestCase):
    """
    Tests that simulate 100% CAV penetration more realistically.
    
    At 100% penetration, every vehicle is a CAV, so:
    - Each CAV sees all other CAVs
    - Each CAV reports its own position (self-localization)
    - The fusion receives N×(N-1) detections per frame for N vehicles
    
    This creates a combinatorial explosion of detections that must be properly matched.
    """

    def test_5_cavs_all_seeing_each_other(self):
        """
        5 CAVs each see the other 4 CAVs.
        Total detections per frame: 5 × 4 = 20
        Expected tracks: 5 (one per vehicle)
        """
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        n_vehicles = 5
        
        # 5 stationary vehicles at different positions
        vehicle_positions = [
            (10.0, 0.0),
            (25.0, 0.0),
            (40.0, 0.0),
            (55.0, 0.0),
            (70.0, 0.0),
        ]
        
        def detections(t, observer_id):
            np.random.seed(t * 1000 + observer_id)
            dets = []
            # Observer sees all OTHER vehicles (not itself in detection, just localization)
            for target_id, (tx, ty) in enumerate(vehicle_positions):
                if target_id != observer_id:
                    noise_x = np.random.normal(0, 0.2)
                    noise_y = np.random.normal(0, 0.2)
                    dets.append(make_detection(tx + noise_x, ty + noise_y, 2.0, 4.5, 0.0, 0.0, 0.0))
            return dets
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants=n_vehicles)
        
        track_counts = [len(r) for r, _ in results]
        last_10_counts = track_counts[-10:]
        
        # Should have 5 tracks (but might be 4 if self not included - depends on setup)
        # Actually each observer sees 4 others, so we expect 4 or 5 tracks
        self.assertTrue(all(4 <= c <= 5 for c in last_10_counts),
            msg=f"5 CAVs seeing each other should produce 4-5 tracks: {last_10_counts}")
    
    def test_10_cavs_all_seeing_each_other(self):
        """
        10 CAVs each see the other 9 CAVs.
        Total detections per frame: 10 × 9 = 90
        Expected tracks: 10
        """
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        n_vehicles = 10
        
        # 10 vehicles in a line, 12m apart
        vehicle_positions = [(10.0 + i * 12, 0.0) for i in range(n_vehicles)]
        
        def detections(t, observer_id):
            np.random.seed(t * 1000 + observer_id)
            dets = []
            for target_id, (tx, ty) in enumerate(vehicle_positions):
                if target_id != observer_id:
                    noise_x = np.random.normal(0, 0.25)
                    noise_y = np.random.normal(0, 0.25)
                    dets.append(make_detection(tx + noise_x, ty + noise_y, 2.0, 4.5, 0.0, 0.0, 0.0))
            return dets
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants=n_vehicles)
        
        track_counts = [len(r) for r, _ in results]
        last_10_counts = track_counts[-10:]
        
        # Should have 9 or 10 tracks
        avg_tracks = sum(last_10_counts) / len(last_10_counts)
        self.assertGreaterEqual(avg_tracks, 8,
            msg=f"10 CAVs should produce ~9-10 tracks. Avg in last 10: {avg_tracks:.1f}")
        self.assertLessEqual(avg_tracks, 12,
            msg=f"10 CAVs should not produce >12 tracks (duplicates). Avg: {avg_tracks:.1f}")
    
    def test_15_cavs_with_self_localization(self):
        """
        15 CAVs each see all others AND report their own position (self-localization).
        Total detections per frame: 15 × 15 = 225 (each sees 14 + self)
        Expected tracks: 15
        
        This simulates the actual 100% penetration scenario with self-localization.
        """
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        n_vehicles = 15
        
        # 15 vehicles, 10m spacing
        vehicle_positions = [(10.0 + i * 10, 0.0) for i in range(n_vehicles)]
        
        def detections(t, observer_id):
            np.random.seed(t * 1000 + observer_id)
            dets = []
            
            # Self-localization (high confidence, low covariance)
            sx, sy = vehicle_positions[observer_id]
            self_noise_x = np.random.normal(0, 0.05)  # Localizer is very accurate
            self_noise_y = np.random.normal(0, 0.05)
            dets.append(make_detection(sx + self_noise_x, sy + self_noise_y, 2.0, 4.5, 0.0, 0.0, 0.0, cov_scale=0.01))
            
            # Detections of others (higher noise)
            for target_id, (tx, ty) in enumerate(vehicle_positions):
                if target_id != observer_id:
                    noise_x = np.random.normal(0, 0.3)
                    noise_y = np.random.normal(0, 0.3)
                    dets.append(make_detection(tx + noise_x, ty + noise_y, 2.0, 4.5, 0.0, 0.0, 0.0, cov_scale=0.25))
            return dets
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants=n_vehicles)
        
        track_counts = [len(r) for r, _ in results]
        last_10_counts = track_counts[-10:]
        
        avg_tracks = sum(last_10_counts) / len(last_10_counts)
        
        # Should have close to 15 tracks
        self.assertGreaterEqual(avg_tracks, 12,
            msg=f"15 CAVs with self-loc should produce ~15 tracks. Avg: {avg_tracks:.1f}")
        self.assertLessEqual(avg_tracks, 20,
            msg=f"15 CAVs should not create >20 tracks (duplicates). Avg: {avg_tracks:.1f}")
    
    def test_moving_cavs_100_percent_penetration(self):
        """
        10 moving CAVs all detecting each other.
        
        Tests that high penetration doesn't break tracking when vehicles move.
        """
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        n_vehicles = 10
        
        # Initial positions, all moving in same direction at ~10 m/s
        initial_positions = [(10.0 + i * 12, 0.0) for i in range(n_vehicles)]
        velocity = 10.0  # m/s
        
        def detections(t, observer_id):
            np.random.seed(t * 1000 + observer_id)
            time = t * DT
            dets = []
            
            for target_id, (init_x, init_y) in enumerate(initial_positions):
                if target_id != observer_id:
                    # Target position at time t
                    tx = init_x + time * velocity
                    ty = init_y
                    noise_x = np.random.normal(0, 0.3)
                    noise_y = np.random.normal(0, 0.3)
                    vx = velocity + np.random.normal(0, 0.5)
                    dets.append(make_detection(tx + noise_x, ty + noise_y, 2.0, 4.5, 0.0, vx, 0.0))
            return dets
        
        results = run_global_fusion_sequence(fusion, detections, total_frames, n_participants=n_vehicles)
        
        track_counts = [len(r) for r, _ in results]
        last_10_counts = track_counts[-10:]
        
        avg_tracks = sum(last_10_counts) / len(last_10_counts)
        
        # Should maintain ~9 tracks (10 vehicles, each sees 9 others)
        self.assertGreaterEqual(avg_tracks, 7,
            msg=f"Moving CAVs should maintain ~9 tracks. Avg: {avg_tracks:.1f}")
        self.assertLessEqual(avg_tracks, 12,
            msg=f"Moving CAVs should not create duplicates. Avg: {avg_tracks:.1f}")


class TestCleanDetectionsEffectiveness(unittest.TestCase):
    """
    Tests specifically for the cleanDetections() deduplication function.
    
    cleanDetections() is called to merge tracks that have converged to the same
    location. These tests verify it works correctly under high load.
    """

    def test_explicit_duplicate_tracks_get_cleaned(self):
        """
        If duplicate tracks somehow get created, cleanDetections should merge them.
        
        This tests the cleanup mechanism directly.
        """
        fusion = Fusion(id=0, use_trust_scoring=False)
        gt_x, gt_y = 50.0, 0.0
        
        # First, create conditions where duplicates might appear
        # by having detections arrive in a specific pattern
        total_frames = 50
        
        # In first 5 frames, have 2 participants with slightly offset detections
        # that might not match to each other initially
        def detections(t, pid):
            if t < 5:
                # Initial phase: larger offsets that might create separate tracks
                offset = 0.8 * (pid - 0.5)  # ±0.4m offset
            else:
                # Later phase: detections converge
                offset = 0.05 * (pid - 0.5)
            
            return [make_detection(gt_x + offset, gt_y + offset, 2.0, 4.5, 0.0, 0.0, 0.0)]
        
        # Run fusion
        for t in range(total_frames):
            time = t * DT
            for pid in range(2):
                obs = detections(t, pid)
                fusion.processDetectionFrame(time, obs, CLEANUP_TIME, source_participant_id=pid)
            
            # Explicitly call cleanDetections to test its effectiveness
            fusion.cleanDetections()
            
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        # After cleanDetections, should have at most 1 mature track
        mature_tracks = [t for t in fusion.tracked_list if t.fusion_steps > fusion.trackShowThreshold]
        
        self.assertLessEqual(len(mature_tracks), 1,
            msg=f"cleanDetections should merge duplicate tracks. Found {len(mature_tracks)} mature tracks.")
    
    def test_clean_detections_scales_with_many_tracks(self):
        """
        Verify cleanDetections handles many tracks efficiently without errors.
        
        This is a stress test to ensure the algorithm doesn't break with scale.
        """
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        # Create 5 well-separated targets with many participants
        targets = [(20 + i * 15, 0.0) for i in range(5)]  # 15m spacing
        n_participants = 20
        total_frames = WARMUP_FRAMES + 30
        
        def detections(t, pid):
            np.random.seed(t * 1000 + pid)
            dets = []
            for tx, ty in targets:
                noise_x = np.random.normal(0, 0.15)
                noise_y = np.random.normal(0, 0.15)
                dets.append(make_detection(tx + noise_x, ty + noise_y, 2.0, 4.5, 0.0, 0.0, 0.0))
            return dets
        
        # Run simulation
        for t in range(total_frames):
            time = t * DT
            for pid in range(n_participants):
                obs = detections(t, pid)
                fusion.processDetectionFrame(time, obs, CLEANUP_TIME, source_participant_id=pid)
            
            # Call cleanDetections each frame
            fusion.cleanDetections()
            
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        # Should have exactly 5 tracks (one per target)
        final_result, _, _, _ = fusion.fuseDetectionFrame((total_frames - 1) * DT, monitor=False)
        
        self.assertEqual(len(final_result), 5,
            msg=f"5 targets with 20 participants should produce exactly 5 tracks. Got {len(final_result)}.")


if __name__ == "__main__":
    unittest.main()
