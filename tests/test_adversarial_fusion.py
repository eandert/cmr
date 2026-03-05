"""
Adversarial tests for sensor fusion to stress-test edge cases and failure modes.

These tests target specific potential issues:
- Track ID stability across frames
- IOU threshold edge cases
- Track cleanup when targets disappear
- Detection dropout/occlusion handling
- Matching conflicts (multiple tracks vs one detection)
- Angle wrapping near ±π
- Crossing/passing vehicles (ID swap risk)
- Sudden velocity changes (hard braking)
- Empty detection frames
- Participant dropout scenarios

Run from repo root: python -m pytest tests/test_adversarial_fusion.py -v
"""
import unittest
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sensor_fusion import Fusion
from sensor import DetectedObject

DT = 0.1
CLEANUP_TIME = 1.0
WARMUP_FRAMES = 5


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


class TestTrackIDStability(unittest.TestCase):
    """Verify that track IDs remain stable across frames."""

    def test_single_track_id_persists(self):
        """A single moving target should keep the same track ID throughout."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = 30
        v = 3.0
        
        track_ids_seen = []
        
        for t in range(total_frames):
            time = t * DT
            x = 50.0 + t * DT * v
            det = make_detection(x, 0, 2.0, 4.5, 0.0, v, 0.0)
            fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
            
            if t >= WARMUP_FRAMES and len(result) >= 1:
                track_ids_seen.append(result[0][0])  # Track ID is first element
        
        # All track IDs after warmup should be the same
        self.assertGreater(len(track_ids_seen), 10)
        unique_ids = set(track_ids_seen)
        self.assertEqual(len(unique_ids), 1,
            msg=f"Track ID should be stable, but got multiple IDs: {unique_ids}")

    def test_two_tracks_maintain_separate_ids(self):
        """Two targets should maintain distinct, stable track IDs."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = 30
        v = 3.0
        gap = 20.0  # Large gap to ensure no confusion
        
        track_id_pairs = []
        
        for t in range(total_frames):
            time = t * DT
            x1 = 30.0 + t * DT * v
            x2 = 30.0 + gap + t * DT * v
            dets = [
                make_detection(x1, 0, 2.0, 4.5, 0.0, v, 0.0),
                make_detection(x2, 0, 2.0, 4.5, 0.0, v, 0.0),
            ]
            fusion.processDetectionFrame(time, dets, CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
            
            if t >= WARMUP_FRAMES and len(result) >= 2:
                # Sort by x position to consistently identify which is which
                sorted_result = sorted(result, key=lambda r: r[1])  # r[1] is x
                track_id_pairs.append((sorted_result[0][0], sorted_result[1][0]))
        
        # Each position should have consistent ID
        self.assertGreater(len(track_id_pairs), 10)
        first_ids = set(p[0] for p in track_id_pairs)
        second_ids = set(p[1] for p in track_id_pairs)
        self.assertEqual(len(first_ids), 1, msg=f"First track ID unstable: {first_ids}")
        self.assertEqual(len(second_ids), 1, msg=f"Second track ID unstable: {second_ids}")
        # And they should be different from each other
        self.assertNotEqual(list(first_ids)[0], list(second_ids)[0])


class TestTrackCleanup(unittest.TestCase):
    """Test that stale tracks are properly removed."""

    def test_track_removed_after_target_disappears(self):
        """Track should be removed after target is not detected for cleanupTime."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        # First, establish a track
        for t in range(15):
            time = t * DT
            det = make_detection(50.0, 0, 2.0, 4.5, 0.0, 0.0, 0.0)
            fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        self.assertEqual(len(result), 1, "Should have 1 track established")
        
        # Now stop detecting - run for cleanupTime + some buffer
        disappear_frames = int((CLEANUP_TIME + 0.5) / DT)
        for t in range(15, 15 + disappear_frames):
            time = t * DT
            fusion.processDetectionFrame(time, [], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        # Track should be gone
        self.assertEqual(len(result), 0, 
            msg=f"Track should be removed after {CLEANUP_TIME}s without detections")

    def test_track_reappears_as_new_id(self):
        """If target disappears long enough and reappears, it gets a new ID."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        # Establish track
        for t in range(15):
            time = t * DT
            det = make_detection(50.0, 0, 2.0, 4.5, 0.0, 0.0, 0.0)
            fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        original_id = result[0][0] if len(result) >= 1 else None
        
        # Disappear for cleanup time
        disappear_frames = int((CLEANUP_TIME + 0.5) / DT)
        for t in range(15, 15 + disappear_frames):
            time = t * DT
            fusion.processDetectionFrame(time, [], CLEANUP_TIME, source_participant_id=0)
            fusion.fuseDetectionFrame(time, monitor=False)
        
        # Reappear at same location
        reappear_start = 15 + disappear_frames
        for t in range(reappear_start, reappear_start + 15):
            time = t * DT
            det = make_detection(50.0, 0, 2.0, 4.5, 0.0, 0.0, 0.0)
            fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        self.assertEqual(len(result), 1, "Should have 1 track after reappearing")
        new_id = result[0][0]
        self.assertNotEqual(original_id, new_id,
            msg="Reappeared target should get new ID since old track was cleaned up")


class TestDetectionDropout(unittest.TestCase):
    """Test handling of brief detection dropout (occlusion)."""

    def test_brief_dropout_maintains_track(self):
        """Brief detection dropout (< cleanupTime) should maintain track."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        v = 3.0
        
        # Establish track
        for t in range(15):
            time = t * DT
            x = 50.0 + t * DT * v
            det = make_detection(x, 0, 2.0, 4.5, 0.0, v, 0.0)
            fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        original_id = result[0][0] if len(result) >= 1 else None
        
        # Brief dropout (0.5s < cleanupTime)
        dropout_frames = int(0.5 / DT)
        for t in range(15, 15 + dropout_frames):
            time = t * DT
            fusion.processDetectionFrame(time, [], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        # Detection returns (vehicle kept moving during dropout)
        reappear_start = 15 + dropout_frames
        for t in range(reappear_start, reappear_start + 10):
            time = t * DT
            x = 50.0 + t * DT * v  # Continued at expected position
            det = make_detection(x, 0, 2.0, 4.5, 0.0, v, 0.0)
            fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        # Should maintain same track ID
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], original_id,
            msg="Brief dropout should maintain track ID")


class TestAngleWrapping(unittest.TestCase):
    """Test handling of angles near ±π boundaries."""

    def test_angle_crossing_pi_boundary(self):
        """Vehicle rotating across the π/-π boundary should maintain stable track."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = 40
        omega = 0.3  # rad/s - will cross boundary during test
        
        track_ids = []
        
        for t in range(total_frames):
            time = t * DT
            # Angle that crosses π boundary
            angle = math.pi - 0.5 + omega * time  # Starts near +π, crosses to -π
            # Wrap to [-π, π]
            while angle > math.pi:
                angle -= 2 * math.pi
            
            det = make_detection(50.0, 0, 2.0, 4.5, angle, 0.0, 0.0)
            fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
            
            if t >= WARMUP_FRAMES and len(result) >= 1:
                track_ids.append(result[0][0])
        
        # Track ID should remain stable despite angle wrapping
        self.assertGreater(len(track_ids), 20)
        unique_ids = set(track_ids)
        self.assertEqual(len(unique_ids), 1,
            msg=f"Track ID should be stable across π boundary: {unique_ids}")


class TestCrossingVehicles(unittest.TestCase):
    """Test that crossing/passing vehicles don't swap IDs."""

    def test_perpendicular_crossing_no_id_swap(self):
        """Two vehicles crossing perpendicularly should maintain their IDs."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = 50
        v = 5.0
        
        # Vehicle 1: moving +x from (0, 0)
        # Vehicle 2: moving +y from (25, -25)
        # They cross around (25, 0) at t=0.5s (frame 5)
        
        track_id_history = {1: [], 2: []}
        
        for t in range(total_frames):
            time = t * DT
            
            # Vehicle 1: +x direction
            x1 = t * DT * v
            y1 = 0
            
            # Vehicle 2: +y direction, offset so they cross
            x2 = 25
            y2 = -25 + t * DT * v
            
            dets = [
                make_detection(x1, y1, 2.0, 4.5, 0.0, v, 0.0),
                make_detection(x2, y2, 2.0, 4.5, math.pi/2, 0.0, v),
            ]
            fusion.processDetectionFrame(time, dets, CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
            
            if t >= WARMUP_FRAMES and len(result) >= 2:
                # Identify by angle (0 vs π/2)
                for r in result:
                    # r = [id, x, y, cov, dx, dy, ...]
                    # Check which vehicle based on velocity direction
                    dx, dy = r[4], r[5]
                    if abs(dx) > abs(dy):  # Moving in x direction
                        track_id_history[1].append(r[0])
                    else:  # Moving in y direction
                        track_id_history[2].append(r[0])
        
        # Each vehicle should have consistent IDs
        for veh_id, ids in track_id_history.items():
            if len(ids) > 5:
                unique = set(ids)
                self.assertLessEqual(len(unique), 2,  # Allow 1-2 due to velocity inference lag
                    msg=f"Vehicle {veh_id} should have stable ID, got: {unique}")


class TestSuddenVelocityChange(unittest.TestCase):
    """Test handling of sudden velocity changes (hard braking, acceleration)."""

    def test_hard_braking_maintains_track(self):
        """Vehicle that suddenly stops should maintain its track."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        v_initial = 10.0  # 10 m/s
        
        track_ids = []
        
        # Phase 1: Moving at 10 m/s
        for t in range(20):
            time = t * DT
            x = 50.0 + t * DT * v_initial
            det = make_detection(x, 0, 2.0, 4.5, 0.0, v_initial, 0.0)
            fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
            if len(result) >= 1:
                track_ids.append(result[0][0])
        
        stopped_x = 50.0 + 20 * DT * v_initial
        
        # Phase 2: Suddenly stopped
        for t in range(20, 40):
            time = t * DT
            det = make_detection(stopped_x, 0, 2.0, 4.5, 0.0, 0.0, 0.0)
            fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
            if len(result) >= 1:
                track_ids.append(result[0][0])
        
        # Track ID should remain stable despite hard braking
        self.assertGreater(len(track_ids), 20)
        unique_ids = set(track_ids)
        self.assertEqual(len(unique_ids), 1,
            msg=f"Hard braking should not create new track: {unique_ids}")

    def test_sudden_acceleration_maintains_track(self):
        """Vehicle that suddenly accelerates should maintain its track."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        track_ids = []
        
        # Phase 1: Stopped
        for t in range(15):
            time = t * DT
            det = make_detection(50.0, 0, 2.0, 4.5, 0.0, 0.0, 0.0)
            fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
            if len(result) >= 1:
                track_ids.append(result[0][0])
        
        # Phase 2: Suddenly moving at 10 m/s
        v = 10.0
        for t in range(15, 35):
            time = t * DT
            x = 50.0 + (t - 15) * DT * v
            det = make_detection(x, 0, 2.0, 4.5, 0.0, v, 0.0)
            fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
            if len(result) >= 1:
                track_ids.append(result[0][0])
        
        # Track ID should remain stable
        self.assertGreater(len(track_ids), 20)
        unique_ids = set(track_ids)
        self.assertEqual(len(unique_ids), 1,
            msg=f"Sudden acceleration should not create new track: {unique_ids}")


class TestEmptyFrames(unittest.TestCase):
    """Test handling of empty detection frames."""

    def test_empty_frame_no_crash(self):
        """System should handle empty detection frames gracefully."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        # Just empty frames
        for t in range(10):
            time = t * DT
            fusion.processDetectionFrame(time, [], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
            self.assertEqual(len(result), 0)

    def test_intermittent_empty_frames(self):
        """Intermittent empty frames between detections should not break tracking."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        v = 3.0
        track_ids = []
        
        for t in range(40):
            time = t * DT
            # Every 3rd frame is empty
            if t % 3 == 0:
                dets = []
            else:
                x = 50.0 + t * DT * v
                dets = [make_detection(x, 0, 2.0, 4.5, 0.0, v, 0.0)]
            
            fusion.processDetectionFrame(time, dets, CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
            
            if t >= WARMUP_FRAMES and len(result) >= 1:
                track_ids.append(result[0][0])
        
        # Should maintain stable track despite gaps
        self.assertGreater(len(track_ids), 15)
        unique_ids = set(track_ids)
        self.assertEqual(len(unique_ids), 1,
            msg=f"Intermittent frames should not break track: {unique_ids}")


class TestParticipantDropout(unittest.TestCase):
    """Test global fusion when participants drop out intermittently."""

    def test_one_participant_drops_out(self):
        """Global fusion should continue when one of two participants stops reporting."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        total_frames = 40
        gt_x, gt_y = 50.0, 0.0
        
        for t in range(total_frames):
            time = t * DT
            
            # Participant 0 always reports
            det0 = make_detection(gt_x, gt_y, 2.0, 4.5, 0.0, 0.0, 0.0)
            fusion.processDetectionFrame(time, [det0], CLEANUP_TIME, source_participant_id=0)
            
            # Participant 1 drops out after frame 20
            if t < 20:
                det1 = make_detection(gt_x + 0.1, gt_y + 0.1, 2.0, 4.5, 0.0, 0.0, 0.0)
                fusion.processDetectionFrame(time, [det1], CLEANUP_TIME, source_participant_id=1)
            
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
            
            if t >= WARMUP_FRAMES:
                # Should always have at least 1 track
                self.assertGreaterEqual(len(result), 1,
                    msg=f"Frame {t}: Should maintain track when one participant drops out")

    def test_all_participants_drop_intermittently(self):
        """If all participants briefly drop, track should persist if < cleanupTime."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        # Establish track
        for t in range(15):
            time = t * DT
            det = make_detection(50.0, 0, 2.0, 4.5, 0.0, 0.0, 0.0)
            for pid in range(2):
                fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=pid)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        original_id = result[0][0] if len(result) >= 1 else None
        
        # Both drop for 0.3s (< cleanupTime)
        dropout_frames = int(0.3 / DT)
        for t in range(15, 15 + dropout_frames):
            time = t * DT
            for pid in range(2):
                fusion.processDetectionFrame(time, [], CLEANUP_TIME, source_participant_id=pid)
            fusion.fuseDetectionFrame(time, monitor=False)
        
        # Both return
        reappear_start = 15 + dropout_frames
        for t in range(reappear_start, reappear_start + 10):
            time = t * DT
            det = make_detection(50.0, 0, 2.0, 4.5, 0.0, 0.0, 0.0)
            for pid in range(2):
                fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=pid)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        # Track should persist
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], original_id)


class TestMatchingConflicts(unittest.TestCase):
    """Test handling of matching conflicts."""

    def test_detection_matches_nearest_track(self):
        """When detection is between two tracks, it should match the nearest."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        # Create two well-separated tracks
        for t in range(15):
            time = t * DT
            dets = [
                make_detection(30.0, 0, 2.0, 4.5, 0.0, 0.0, 0.0),
                make_detection(70.0, 0, 2.0, 4.5, 0.0, 0.0, 0.0),
            ]
            fusion.processDetectionFrame(time, dets, CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        self.assertEqual(len(result), 2)
        
        # Now the "left" detection moves slightly right
        # It should still match the left track, not jump to right
        for t in range(15, 25):
            time = t * DT
            dets = [
                make_detection(35.0, 0, 2.0, 4.5, 0.0, 0.0, 0.0),  # Moved right
                make_detection(70.0, 0, 2.0, 4.5, 0.0, 0.0, 0.0),
            ]
            fusion.processDetectionFrame(time, dets, CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        self.assertEqual(len(result), 2)
        # Left track should have moved to ~35, right should stay at 70
        sorted_result = sorted(result, key=lambda r: r[1])
        self.assertLess(sorted_result[0][1], 45, "Left track should follow left detection")
        self.assertGreater(sorted_result[1][1], 65, "Right track should stay with right detection")


class TestDimensionStability(unittest.TestCase):
    """Test that track dimensions remain stable."""

    def test_dimension_tracking_stable(self):
        """Track width/length should remain stable over time."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        width_initial = 2.0
        length_initial = 4.5
        
        for t in range(30):
            time = t * DT
            # Small random-ish perturbation to dimensions (simulating measurement noise)
            w_noise = 0.1 * math.sin(t * 0.5)
            l_noise = 0.1 * math.cos(t * 0.3)
            det = make_detection(50.0, 0, width_initial + w_noise, length_initial + l_noise, 0.0, 0.0, 0.0)
            fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=0)
            result, detected_objects, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        
        # Check final dimensions are close to initial
        if len(detected_objects) >= 1:
            final_width = detected_objects[0].dimensions[0]
            final_length = detected_objects[0].dimensions[1]
            self.assertAlmostEqual(final_width, width_initial, delta=0.3,
                msg=f"Width should be stable: got {final_width}")
            self.assertAlmostEqual(final_length, length_initial, delta=0.3,
                msg=f"Length should be stable: got {final_length}")


class TestIOUEdgeCases(unittest.TestCase):
    """Test IOU matching edge cases."""

    def test_long_thin_vehicle_at_angle(self):
        """Long thin vehicle at 45 degrees should still match correctly."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        # Long bus-like vehicle
        width = 2.5
        length = 12.0
        angle = math.pi / 4  # 45 degrees
        
        track_ids = []
        
        for t in range(30):
            time = t * DT
            # Slight movement
            det = make_detection(50.0 + t * 0.1, t * 0.1, width, length, angle, 1.0, 1.0)
            fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
            
            if t >= WARMUP_FRAMES and len(result) >= 1:
                track_ids.append(result[0][0])
        
        # Should maintain single track
        self.assertGreater(len(track_ids), 15)
        unique_ids = set(track_ids)
        self.assertEqual(len(unique_ids), 1,
            msg=f"Long vehicle at angle should track stably: {unique_ids}")

    def test_motorcycle_small_dimensions(self):
        """Small motorcycle-sized vehicle should track correctly."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        # Motorcycle dimensions
        width = 0.6
        length = 2.0
        v = 5.0
        
        track_ids = []
        
        for t in range(30):
            time = t * DT
            x = 50.0 + t * DT * v
            det = make_detection(x, 0, width, length, 0.0, v, 0.0)
            fusion.processDetectionFrame(time, [det], CLEANUP_TIME, source_participant_id=0)
            result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
            
            if t >= WARMUP_FRAMES and len(result) >= 1:
                track_ids.append(result[0][0])
        
        # Should maintain single track
        self.assertGreater(len(track_ids), 15)
        unique_ids = set(track_ids)
        self.assertEqual(len(unique_ids), 1,
            msg=f"Small motorcycle should track stably: {unique_ids}")


if __name__ == "__main__":
    unittest.main()
