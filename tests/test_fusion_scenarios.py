"""
Multi-step local sensor fusion tests with synthetic scenarios.

Runs Fusion for multiple steps with created detection sequences:
- Non-moving vehicle
- Moving vehicle
- Fast moving vehicle
- Turning vehicle(s)
- Vehicles with reasonable spacing
- Close-in vehicles

Asserts that after a warmup period we maintain tracking for 25 frames per scenario.
Run from repo root: python tests/test_fusion_scenarios.py -v
"""
import unittest
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sensor_fusion import Fusion
from sensor import DetectedObject

# Fusion shows a track in output when fusion_steps > trackShowThreshold (4)
TRACK_SHOW_THRESHOLD = 4
WARMUP_FRAMES = 5
TRACK_FRAMES_REQUIRED = 25
DT = 0.1
CLEANUP_TIME = 1.0  # seconds; keep tracks alive if we miss a frame
SOURCE_PARTICIPANT_ID = 0

# Tightness: fused position must be within this (m) of expected.
# If tests fail here, tune the Kalman filter rather than loosening.
POSITION_TOL_M = 1.0
# Fast-moving: slightly more tolerant due to prediction lag at high speeds
POSITION_TOL_FAST_MOVING_M = 2.0
# Covariance: position error covariance trace (var_x + var_y) must be below this (m^2)
COV_TRACE_MAX_M2 = 2.0

# Close multi-vehicle separation test: strict so Kalman tuning is validated (don't loosen)
CLOSE_VEHICLES_POSITION_TOL_M = 2.0   # fused position within 2 m of GT
CLOSE_VEHICLES_COV_TRACE_MAX_M2 = 2.0  # same as global
CLOSE_VEHICLES_GAP_M = 4.0             # lateral or longitudinal gap between vehicles
CLOSE_VEHICLES_SPEED_MS = 2.0          # slow so filter lag is small


def make_detection(x, y, width, length, angle_rad, vx=0, vy=0, cov_scale=0.1):
    """Build a DetectedObject for fusion input (vehicle_id set by Fusion from index)."""
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


def run_fusion_sequence(fusion, detections_per_frame, total_frames):
    """
    Run fusion for total_frames. detections_per_frame(t) returns list of DetectedObject.
    Returns list of track counts per frame (only for t >= WARMUP_FRAMES).
    """
    track_counts_after_warmup = []
    for t in range(total_frames):
        time = t * DT
        observations = detections_per_frame(t)
        fusion.processDetectionFrame(time, observations, CLEANUP_TIME, SOURCE_PARTICIPANT_ID)
        result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        if t >= WARMUP_FRAMES:
            track_counts_after_warmup.append(len(result))
    return track_counts_after_warmup


def run_fusion_sequence_with_ground_truth(fusion, detections_per_frame, expected_positions_fn, total_frames):
    """
    Run fusion and return full result rows and expected positions for t >= WARMUP_FRAMES.
    - detections_per_frame(t): list of DetectedObject.
    - expected_positions_fn(t): list of (x, y) in stable order (e.g. by x).
    Returns: list of (result_rows, expected_xy_list) for each frame after warmup.
    result_rows: each row is [id, x, y, error_covariance, dx, dy, d_covariance, num_trackers].
    """
    out = []
    for t in range(total_frames):
        time = t * DT
        observations = detections_per_frame(t)
        fusion.processDetectionFrame(time, observations, CLEANUP_TIME, SOURCE_PARTICIPANT_ID)
        result, _, _, _ = fusion.fuseDetectionFrame(time, monitor=False)
        if t >= WARMUP_FRAMES:
            expected = expected_positions_fn(t)
            out.append((list(result), list(expected)))
    return out


def _pair_tracks_to_expected(result_rows, expected_xy_list):
    """
    Pair result rows to expected (x,y) by sorting both by x then y so order is stable.
    result_rows: list of [id, x, y, cov, ...]; cov is 2x2 list.
    Returns: list of ((tx, ty), cov, (ex, ey)) for each pair. Requires len(result_rows) == len(expected_xy_list).
    """
    if len(result_rows) != len(expected_xy_list):
        return None
    # Sort by x then y
    sorted_result = sorted(result_rows, key=lambda r: (r[1], r[2]))
    sorted_expected = sorted(expected_xy_list, key=lambda p: (p[0], p[1]))
    return [
        ((r[1], r[2]), np.array(r[3]), (e[0], e[1]))
        for r, e in zip(sorted_result, sorted_expected)
    ]


def _pair_tracks_to_expected_allow_extra(result_rows, expected_xy_list):
    """
    When len(result_rows) >= len(expected_xy_list), pair each expected position to its nearest
    track (each track used at most once). Ensures we have enough tracks and that each GT has a
    nearby track. Returns list of ((tx, ty), cov, (ex, ey)) of length len(expected_xy_list), or None
    if fewer tracks than expected.
    """
    n_exp = len(expected_xy_list)
    if len(result_rows) < n_exp:
        return None
    # Greedy: for each expected (in fixed order), pick the unused track nearest to it
    expected_sorted = sorted(expected_xy_list, key=lambda p: (p[0], p[1]))
    result = list(result_rows)
    pairs = []
    for (ex, ey) in expected_sorted:
        best_idx = None
        best_dist = float("inf")
        for idx, row in enumerate(result):
            tx, ty = row[1], row[2]
            d = math.sqrt((tx - ex) ** 2 + (ty - ey) ** 2)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_idx is None:
            return None
        row = result.pop(best_idx)
        pairs.append(((row[1], row[2]), np.array(row[3]), (ex, ey)))
    return pairs


def assert_positions_and_covariance(frame_pairs, position_tol_m=POSITION_TOL_M, cov_trace_max_m2=COV_TRACE_MAX_M2):
    """Assert every paired track is within position_tol of expected and has covariance trace <= cov_trace_max."""
    for frame_idx, (result_rows, expected_list) in enumerate(frame_pairs):
        pairs = _pair_tracks_to_expected(result_rows, expected_list)
        assert pairs is not None, (
            f"Frame {frame_idx}: got {len(result_rows)} tracks but {len(expected_list)} expected positions"
        )
        for (tx, ty), cov, (ex, ey) in pairs:
            dist = math.sqrt((tx - ex) ** 2 + (ty - ey) ** 2)
            assert dist <= position_tol_m, (
                f"Frame {frame_idx}: track at ({tx:.3f},{ty:.3f}) expected ({ex:.3f},{ey:.3f}), dist={dist:.3f} > {position_tol_m}"
            )
            trace = float(np.trace(cov))
            assert trace <= cov_trace_max_m2, (
                f"Frame {frame_idx}: position covariance trace {trace:.3f} > {cov_trace_max_m2}"
            )


def assert_positions_and_covariance_allow_extra(
    frame_pairs, position_tol_m=POSITION_TOL_M, cov_trace_max_m2=COV_TRACE_MAX_M2
):
    """Like assert_positions_and_covariance but allows result to have more tracks than expected (pairs by nearest)."""
    for frame_idx, (result_rows, expected_list) in enumerate(frame_pairs):
        pairs = _pair_tracks_to_expected_allow_extra(result_rows, expected_list)
        assert pairs is not None, (
            f"Frame {frame_idx}: got {len(result_rows)} tracks but need at least {len(expected_list)}"
        )
        for (tx, ty), cov, (ex, ey) in pairs:
            dist = math.sqrt((tx - ex) ** 2 + (ty - ey) ** 2)
            assert dist <= position_tol_m, (
                f"Frame {frame_idx}: track at ({tx:.3f},{ty:.3f}) expected ({ex:.3f},{ey:.3f}), dist={dist:.3f} > {position_tol_m}"
            )
            trace = float(np.trace(cov))
            assert trace <= cov_trace_max_m2, (
                f"Frame {frame_idx}: position covariance trace {trace:.3f} > {cov_trace_max_m2}"
            )


class TestFusionScenarios(unittest.TestCase):
    def test_non_moving_vehicle(self):
        """One vehicle at fixed position for warmup + 25 frames."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        n_expected = 1
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        gt_x, gt_y = 10.0, 0.0

        def detections(t):
            return [make_detection(gt_x, gt_y, 2.0, 4.0, 0.0, 0.0, 0.0)]

        def expected_positions(t):
            return [(gt_x, gt_y)]

        frame_pairs = run_fusion_sequence_with_ground_truth(
            fusion, detections, expected_positions, total_frames
        )
        self.assertEqual(len(frame_pairs), TRACK_FRAMES_REQUIRED)
        for _, (rows, exp) in enumerate(frame_pairs):
            self.assertGreaterEqual(len(rows), n_expected)
        assert_positions_and_covariance(frame_pairs)

    def test_moving_vehicle(self):
        """One vehicle moving at 5 m/s along x."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        v = 5.0
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED

        def detections(t):
            x = 0 + t * DT * v
            return [make_detection(x, 0.0, 2.0, 4.0, 0.0, v, 0.0)]

        def expected_positions(t):
            x = 0 + t * DT * v
            return [(x, 0.0)]

        frame_pairs = run_fusion_sequence_with_ground_truth(
            fusion, detections, expected_positions, total_frames
        )
        self.assertEqual(len(frame_pairs), TRACK_FRAMES_REQUIRED)
        for _, (rows, exp) in enumerate(frame_pairs):
            self.assertGreaterEqual(len(rows), 1)
        assert_positions_and_covariance_allow_extra(frame_pairs)

    def test_fast_moving_vehicle(self):
        """One vehicle at 15 m/s along x."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        v = 15.0
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED

        def detections(t):
            x = 0 + t * DT * v
            return [make_detection(x, 0.0, 2.0, 4.0, 0.0, v, 0.0)]

        def expected_positions(t):
            x = 0 + t * DT * v
            return [(x, 0.0)]

        frame_pairs = run_fusion_sequence_with_ground_truth(
            fusion, detections, expected_positions, total_frames
        )
        self.assertEqual(len(frame_pairs), TRACK_FRAMES_REQUIRED)
        for _, (rows, exp) in enumerate(frame_pairs):
            self.assertGreaterEqual(len(rows), 1)
        assert_positions_and_covariance_allow_extra(
            frame_pairs, position_tol_m=POSITION_TOL_FAST_MOVING_M
        )

    def test_close_vehicles_separation_no_flicker(self):
        """
        Two close vehicles moving slowly: must maintain exactly 2 tracks every frame (no flicker),
        positions within CLOSE_VEHICLES_POSITION_TOL_M of GT, covariance bounded.
        If this fails, tune Kalman/matching rather than loosening tolerances.
        """
        fusion = Fusion(id=0, use_trust_scoring=False)
        gap = CLOSE_VEHICLES_GAP_M
        v = CLOSE_VEHICLES_SPEED_MS
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        n_expected = 2

        def detections(t):
            base_x = t * DT * v
            return [
                make_detection(base_x, 0.0, 2.0, 4.0, 0.0, v, 0.0),
                make_detection(base_x + gap, 0.0, 2.0, 4.0, 0.0, v, 0.0),
            ]

        def expected_positions(t):
            base_x = t * DT * v
            return [(base_x, 0.0), (base_x + gap, 0.0)]

        frame_pairs = run_fusion_sequence_with_ground_truth(
            fusion, detections, expected_positions, total_frames
        )
        self.assertEqual(len(frame_pairs), TRACK_FRAMES_REQUIRED)

        # No flicker: exactly 2 tracks every frame
        for frame_idx, (result_rows, expected_list) in enumerate(frame_pairs):
            self.assertEqual(
                len(result_rows),
                n_expected,
                msg=(
                    f"Frame {frame_idx}: expected exactly {n_expected} tracks (no flicker), got {len(result_rows)}. "
                    "Tune Kalman/matching if tracks merge or duplicate."
                ),
            )

        # Strict position and covariance (reasonable tuning targets)
        assert_positions_and_covariance(
            frame_pairs,
            position_tol_m=CLOSE_VEHICLES_POSITION_TOL_M,
            cov_trace_max_m2=CLOSE_VEHICLES_COV_TRACE_MAX_M2,
        )

    def test_turning_vehicle(self):
        """One vehicle on circular path (constant speed, changing heading)."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        radius = 20.0
        omega = 0.3  # rad/s
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED

        def detections(t):
            time = t * DT
            x = radius * math.cos(omega * time)
            y = radius * math.sin(omega * time)
            angle = omega * time + math.pi / 2
            speed = radius * omega
            vx = -speed * math.sin(omega * time)
            vy = speed * math.cos(omega * time)
            return [make_detection(x, y, 2.0, 4.0, angle, vx, vy)]

        def expected_positions(t):
            time = t * DT
            x = radius * math.cos(omega * time)
            y = radius * math.sin(omega * time)
            return [(x, y)]

        frame_pairs = run_fusion_sequence_with_ground_truth(
            fusion, detections, expected_positions, total_frames
        )
        self.assertEqual(len(frame_pairs), TRACK_FRAMES_REQUIRED)
        for _, (rows, exp) in enumerate(frame_pairs):
            self.assertGreaterEqual(len(rows), 1)
        assert_positions_and_covariance_allow_extra(frame_pairs)

    def test_vehicles_reasonable_spacing(self):
        """Three vehicles with 15 m spacing, moving at 5 m/s."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        v = 5.0
        spacing = 15.0
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        n_expected = 3

        def detections(t):
            base_x = t * DT * v
            return [
                make_detection(base_x + 0, 0.0, 2.0, 4.0, 0.0, v, 0.0),
                make_detection(base_x + spacing, 0.0, 2.0, 4.0, 0.0, v, 0.0),
                make_detection(base_x + 2 * spacing, 0.0, 2.0, 4.0, 0.0, v, 0.0),
            ]

        def expected_positions(t):
            base_x = t * DT * v
            return [
                (base_x + 0, 0.0),
                (base_x + spacing, 0.0),
                (base_x + 2 * spacing, 0.0),
            ]

        frame_pairs = run_fusion_sequence_with_ground_truth(
            fusion, detections, expected_positions, total_frames
        )
        self.assertEqual(len(frame_pairs), TRACK_FRAMES_REQUIRED)
        for _, (rows, exp) in enumerate(frame_pairs):
            self.assertGreaterEqual(len(rows), n_expected)
        assert_positions_and_covariance_allow_extra(frame_pairs)

    def test_close_in_vehicles(self):
        """Two vehicles 3 m apart (close), same heading, moving."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        v = 5.0
        gap = 3.0
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        n_expected = 2

        def detections(t):
            base_x = t * DT * v
            return [
                make_detection(base_x, 0.0, 2.0, 4.0, 0.0, v, 0.0),
                make_detection(base_x + gap, 0.0, 2.0, 4.0, 0.0, v, 0.0),
            ]

        def expected_positions(t):
            base_x = t * DT * v
            return [(base_x, 0.0), (base_x + gap, 0.0)]

        frame_pairs = run_fusion_sequence_with_ground_truth(
            fusion, detections, expected_positions, total_frames
        )
        self.assertEqual(len(frame_pairs), TRACK_FRAMES_REQUIRED)
        for _, (rows, exp) in enumerate(frame_pairs):
            self.assertGreaterEqual(len(rows), n_expected)
        assert_positions_and_covariance_allow_extra(frame_pairs)

    def test_mixed_scenario(self):
        """One stationary, one moving, one turning — all tracked for 25 frames after warmup."""
        fusion = Fusion(id=0, use_trust_scoring=False)
        radius = 15.0
        omega = 0.2
        v_move = 4.0
        total_frames = WARMUP_FRAMES + TRACK_FRAMES_REQUIRED
        n_expected = 3

        def detections(t):
            time = t * DT
            stationary = make_detection(0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 0.0)
            moving = make_detection(50.0 + t * DT * v_move, 0.0, 2.0, 4.0, 0.0, v_move, 0.0)
            x_turn = 100 + radius * math.cos(omega * time)
            y_turn = 0 + radius * math.sin(omega * time)
            angle_turn = omega * time + math.pi / 2
            vx_turn = -radius * omega * math.sin(omega * time)
            vy_turn = radius * omega * math.cos(omega * time)
            turning = make_detection(x_turn, y_turn, 2.0, 4.0, angle_turn, vx_turn, vy_turn)
            return [stationary, moving, turning]

        def expected_positions(t):
            time = t * DT
            x_move = 50.0 + t * DT * v_move
            x_turn = 100 + radius * math.cos(omega * time)
            y_turn = 0 + radius * math.sin(omega * time)
            return [(0.0, 0.0), (x_move, 0.0), (x_turn, y_turn)]

        frame_pairs = run_fusion_sequence_with_ground_truth(
            fusion, detections, expected_positions, total_frames
        )
        self.assertEqual(len(frame_pairs), TRACK_FRAMES_REQUIRED)
        for _, (rows, exp) in enumerate(frame_pairs):
            self.assertGreaterEqual(len(rows), n_expected)
        # Allow extra tracks (ghosts) so we only check that each GT has a nearby track
        assert_positions_and_covariance_allow_extra(frame_pairs)


if __name__ == "__main__":
    unittest.main()
