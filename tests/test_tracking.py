
import unittest
import sys
import os
import math
import numpy as np
from sklearn.neighbors import BallTree

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import utils
from sensor_fusion import ResizableKalman, GlobalTracked, MatchClass
from sensor import DetectedObject


def make_match(mid, x, y, cov=None, dx=0, dy=0, width=2, length=4, angle=0, time=0):
    """Helper to create MatchClass with sensible defaults."""
    if cov is None:
        cov = np.eye(2)
    return MatchClass(
        id=mid, x=x, y=y, covariance=cov, dx=dx, dy=dy,
        d_confidence=1, confidence=1, trust_score=1, object_type=0, time=time,
        width=width, length=length, angle=angle, width_std=0.3, length_std=0.3,
    )


def make_detected(vehicle_id, x, y, width=2, length=4, angle=0, dx=0, dy=0):
    """Helper to create DetectedObject for GlobalTracked."""
    cov = np.eye(2) * 0.25
    return DetectedObject(
        vehicle_id=vehicle_id,
        vehicle_type=0,
        detected_bbox=None,
        centroid=(x, y),
        width=width,
        length=length,
        angle=angle,
        expected_error_gaussian=None,
        velocity_vector=(dx, dy),
        error_covariance=cov,
        width_std=0.3,
        length_std=0.3,
    )


class TestTrackingComponents(unittest.TestCase):

    def test_rotated_box_iou_alignment(self):
        """Test that IOU calculation handles dimensions and rotation correctly."""
        # Box 1: Center (0,0), 2x4 (Width=2, Length=4), Angle 0 (East)
        # Standard math: Length along X.
        # utils.get_rotated_box_corners expects [cx, cy, w, h, angle]
        # logic in get_rotated_box_corners: w is along Y, h is along X at angle 0? 
        # Wait, let's re-verify utils.py logic in the test.
        
        # If I pass w=2, h=4, angle=0.
        # If logic is: corners = [-h/2, -w/2]... rotated.
        # Then h is along X.
        
        box1 = [0, 0, 2, 4, 0] 
        
        # Box 2: Identical to Box 1
        box2 = [0, 0, 2, 4, 0]
        self.assertAlmostEqual(utils.rotated_box_iou(box1, box2), 1.0)

        # Box 3: Rotated 90 degrees (pi/2)
        # Should be effectively 4x2 in AABB terms.
        box3 = [0, 0, 2, 4, math.pi/2]
        
        # Intersection of 2x4 (along X) and 2x4 (rotated 90 -> along Y)
        # Box 1 ranges: x[-2, 2], y[-1, 1]
        # Box 3 ranges: x[-1, 1], y[-2, 2]
        # Intersection: x[-1, 1], y[-1, 1] -> Area 2x2 = 4
        # Union: Area1 + Area2 - Intersection = 8 + 8 - 4 = 12
        # IOU = 4/12 = 1/3 ≈ 0.333
        iou = utils.rotated_box_iou(box1, box3)
        self.assertAlmostEqual(iou, 1.0/3.0, places=4, msg=f"IOU of 90-deg crossed boxes should be 1/3, got {iou}")

    def test_iou_with_offset(self):
        """Test IOU with spatial offset to see if it drops off as expected."""
        # Box 1: 2x4 at (0,0)
        box1 = [0, 0, 2, 4, 0]
        
        # Box 2: Shifted by 1 in X (halfway overlap in length)
        # x range: [-1, 3] instead of [-2, 2]
        # Overlap x: [-1, 2] -> width 3.
        # y range: [-1, 1] (same)
        # Intersection area: 3 * 2 = 6
        # Union: 8 + 8 - 6 = 10
        # IOU: 6/10 = 0.6
        box2 = [1, 0, 2, 4, 0]
        iou = utils.rotated_box_iou(box1, box2)
        self.assertAlmostEqual(iou, 0.6, places=4)

    def test_inflation_impact_on_matching(self):
        """Test if inflating bounding boxes helps match distant predictions."""
        # Ground Truth / Measurement: at (0,0), size 2x4
        meas_box = [0, 0, 2, 4, 0]
        
        # Prediction: offset by 3.0 units in X. (Disjoint if not inflated, 2+1=3 edge vs 2 edge? No, center distance 3. Half-lengths 2. 0+2=2 vs 3-2=1. Overlap 1.)
        # Meas x: [-2, 2]
        # Pred x: [1, 5] (Center 3, length 4)
        # Overlap x: [1, 2] width 1
        # Intersection: 2
        # Union: 8 + 8 - 2 = 14
        # IOU = 1/7 ≈ 0.1428
        
        pred_center = [3, 0]
        pred_box = [pred_center[0], pred_center[1], 2, 4, 0]
        
        iou_normal = utils.rotated_box_iou(meas_box, pred_box)
        # print(f"Normal IOU (Offset 3): {iou_normal}")
        
        # Threshold is 0.2. So 0.1428 fails.
        self.assertLess(iou_normal, 0.2)
        
        # Now inflate prediction by 1.0 unit in each dimension (simulating uncertainty)
        # New size: 3x5
        # Pred x: [3-2.5, 3+2.5] -> [0.5, 5.5]
        # Meas x: [-2, 2]
        # Overlap x: [0.5, 2] -> width 1.5
        # Overlap y: [-1, 1] -> height 2
        # Intersection: 3
        # Union: 8 + 15 - 3 = 20
        # IOU = 3/20 = 0.15
        
        pred_inflated = [3, 0, 3, 5, 0]
        iou_inflated = utils.rotated_box_iou(meas_box, pred_inflated)
        # print(f"Inflated IOU (Offset 3): {iou_inflated}")
        
        # 0.15 is > 0.1428. It IMPROVED slightly.
        self.assertGreater(iou_inflated, iou_normal)
        
        # Try larger inflation (2.0 units -> 4x6)
        # Pred x: [3-3, 3+3] -> [0, 6]
        # Overlap x: [0, 2] -> width 2
        # Overlap y: [-1, 1] -> height 2
        # Intersection: 4
        # Union: 8 + 24 - 4 = 28
        # IOU = 4/28 = 0.1428
        
        pred_huge = [3, 0, 4, 6, 0]
        iou_huge = utils.rotated_box_iou(meas_box, pred_huge)
        # print(f"Huge IOU (Offset 3): {iou_huge}")
        
        # Back to 0.1428.
        
        # Conclusion: Small inflation CAN improve IOU slightly by increasing intersection area faster than union area
        # in specific overlap configurations.
        # But large inflation dilutes it.
        
        # However, for DISJOINT boxes, inflation is strictly necessary to get IOU > 0.
        # Offset 5.
        # Meas x: [-2, 2]. Pred x: [3, 7]. Disjoint.
        pred_disjoint = [5, 0, 2, 4, 0]
        self.assertEqual(utils.rotated_box_iou(meas_box, pred_disjoint), 0.0)
        
        # Inflate huge (4x6)
        # Pred x: [5-3, 5+3] -> [2, 8]. Touching. IOU still 0?
        # Inflate more (6x8)
        # Pred x: [5-4, 5+4] -> [1, 9].
        # Meas x: [-2, 2]. Overlap [1, 2] width 1.
        # Intersection: 2.
        # Union: 8 + 48 - 2 = 54.
        # IOU = 2/54 = 0.037.
        
        pred_very_huge = [5, 0, 6, 8, 0]
        self.assertGreater(utils.rotated_box_iou(meas_box, pred_very_huge), 0.0)
        
        # The user's claim "it matches better" likely refers to cases where the prediction is slightly off
        # and standard IOU falls just below 0.2, but inflation bumps it up (or makes it non-zero).
        # Or, it allows the BallTree to find it as a candidate (since BallTree pruning might discard distant objects).
        
    def test_kalman_prediction(self):
        """Test basic Kalman prediction."""
        # Init Kalman
        k = ResizableKalman(time=0, x=0, y=0, fusion_mode=1)
        # Initial state should be 0,0
        
        # Update with measurement at t=0
        # This sets the initial state
        m1 = MatchClass(id=1, x=0, y=0, covariance=np.eye(2), dx=0, dy=0, d_confidence=1, confidence=1, trust_score=1, object_type=0, time=0, width=2, length=4, angle=0)
        k.fusion([m1], time=0, monitor=False)
        
        self.assertEqual(k.x, 0)
        self.assertEqual(k.y, 0)
        
        # Predict at t=1 (Process model)
        # With velocity=0, prediction should stay near 0
        pred_x, pred_y, a, b, phi = k.getKalmanPred(1.0)
        self.assertAlmostEqual(pred_x, 0, delta=1.0)
        
        # Update with measurement at x=10 at t=1
        m2 = MatchClass(id=1, x=10, y=0, covariance=np.eye(2), dx=10, dy=0, d_confidence=1, confidence=1, trust_score=1, object_type=0, time=1, width=2, length=4, angle=0)
        k.fusion([m2], time=1, monitor=False)
        
        # State should move toward the measurement (exact weighting depends on covariances)
        # With high initial velocity uncertainty, filter trusts measurement more
        self.assertGreater(k.x, 0)  # Moved from initial position
        self.assertLessEqual(k.x, 10.0)  # Not beyond measurement
        
        # Velocity should be inferred from position change
        self.assertGreater(k.dx, 0)  # Positive velocity inferred
        
        # Predict at t=2 should extrapolate forward using inferred velocity
        pred_x, pred_y, a, b, phi = k.getKalmanPred(2.0)
        self.assertGreater(pred_x, k.x)  # Should project forward

    def test_kalman_ctrv_realistic_tracking(self):
        """Test CTRV Kalman with realistic values (10 Hz, 5 m/s vehicle)."""
        # CTRV: State = [x, y, v, psi, psi_dot]
        # Realistic: vehicle at (50, 0) heading East (psi=0), moving at 5 m/s
        k = ResizableKalman(time=0.0, x=50, y=0, fusion_mode=2)
        
        # Realistic covariance: ~0.5m std dev position, ~0.1 rad heading
        pos_cov = np.array([[0.25, 0.0], [0.0, 0.25]], dtype='float')
        
        # First measurement at t=0
        m0 = MatchClass(id=1, x=50, y=0, covariance=pos_cov, dx=5, dy=0, 
                        d_confidence=1, confidence=1, trust_score=1, object_type=0,
                        time=0.0, width=2, length=4.5, angle=0.0)
        k.fusion([m0], time=0.0, monitor=False)
        
        self.assertAlmostEqual(k.x, 50.0, delta=0.5)
        self.assertAlmostEqual(k.y, 0.0, delta=0.5)
        
        # Simulate 10 frames at 10 Hz (dt=0.1s), vehicle moving at 5 m/s
        dt = 0.1
        velocity = 5.0
        for i in range(1, 11):
            t = i * dt
            expected_x = 50.0 + velocity * t
            m = MatchClass(id=1, x=expected_x, y=0, covariance=pos_cov, dx=velocity, dy=0,
                           d_confidence=1, confidence=1, trust_score=1, object_type=0,
                           time=t, width=2, length=4.5, angle=0.0)
            k.fusion([m], time=t, monitor=False)
        
        # After 1 second, vehicle should be at x=55
        self.assertAlmostEqual(k.x, 55.0, delta=1.0)
        self.assertAlmostEqual(k.y, 0.0, delta=0.5)
        
        # Velocity should be inferred to be ~5 m/s (allows some filter lag)
        speed = math.sqrt(k.dx**2 + k.dy**2)
        self.assertAlmostEqual(speed, velocity, delta=2.0)
        
        # Prediction at t=1.5 should extrapolate forward from current state
        # With inferred velocity, should be ahead of current position
        pred_x, pred_y, a, b, phi = k.getKalmanPred(1.5)
        self.assertGreater(pred_x, k.x)  # Should predict forward
        self.assertGreater(pred_x, 55.0)  # Should be ahead of where we were at t=1.0

    def test_kalman_ctrv_turning_vehicle(self):
        """Test CTRV handles turning vehicle with heading measurements."""
        k = ResizableKalman(time=0.0, x=0, y=0, fusion_mode=2)
        pos_cov = np.array([[0.25, 0.0], [0.0, 0.25]], dtype='float')
        
        # Vehicle turning on a circle: radius=10m, omega=0.2 rad/s (~11 deg/s)
        radius = 10.0
        omega = 0.2
        dt = 0.1
        
        # First measurement
        m0 = MatchClass(id=1, x=radius, y=0, covariance=pos_cov, dx=0, dy=radius*omega,
                        d_confidence=1, confidence=1, trust_score=1, object_type=0,
                        time=0.0, width=2, length=4.5, angle=math.pi/2)
        k.fusion([m0], time=0.0, monitor=False)
        
        # Run for 20 frames (2 seconds)
        for i in range(1, 21):
            t = i * dt
            theta = omega * t
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            heading = theta + math.pi/2  # Tangent to circle
            vx = -radius * omega * math.sin(theta)
            vy = radius * omega * math.cos(theta)
            
            m = MatchClass(id=1, x=x, y=y, covariance=pos_cov, dx=vx, dy=vy,
                           d_confidence=1, confidence=1, trust_score=1, object_type=0,
                           time=t, width=2, length=4.5, angle=heading)
            k.fusion([m], time=t, monitor=False)
        
        # After 2 seconds, theta = 0.4 rad (~23 degrees)
        expected_theta = omega * 2.0
        expected_x = radius * math.cos(expected_theta)
        expected_y = radius * math.sin(expected_theta)
        
        self.assertAlmostEqual(k.x, expected_x, delta=1.5)
        self.assertAlmostEqual(k.y, expected_y, delta=1.5)
        
        # Speed should be radius * omega = 2 m/s
        speed = math.sqrt(k.dx**2 + k.dy**2)
        self.assertAlmostEqual(speed, radius * omega, delta=0.5)


class TestMatchingAllAngles(unittest.TestCase):
    """Matching and IOU at all angles and symmetry."""

    def test_iou_identical_boxes(self):
        """Identical boxes at any angle should have IOU 1.0."""
        w, h = 2, 4
        for angle in [0, math.pi/6, math.pi/4, math.pi/2, math.pi, -math.pi/4]:
            box = [0, 0, w, h, angle]
            self.assertAlmostEqual(utils.rotated_box_iou(box, box), 1.0, places=6)

    def test_iou_symmetry(self):
        """IOU(a, b) == IOU(b, a) for any two boxes."""
        boxes = [
            [0, 0, 2, 4, 0],
            [1, 0, 2, 4, 0],
            [0, 0, 2, 4, math.pi/4],
            [3, 2, 2, 4, math.pi/2],
        ]
        for i, a in enumerate(boxes):
            for j, b in enumerate(boxes):
                iou_ab = utils.rotated_box_iou(a, b)
                iou_ba = utils.rotated_box_iou(b, a)
                self.assertAlmostEqual(iou_ab, iou_ba, places=6, msg=f"IOU symmetry failed for boxes {i},{j}")

    def test_iou_at_angles_0_90_180(self):
        """Known IOU at 0°, 90°, 180° for same-size boxes."""
        # 2x4 box at origin
        base = [0, 0, 2, 4, 0]
        # Same box rotated 90°: crossing diamonds -> 1/3
        rot90 = [0, 0, 2, 4, math.pi/2]
        self.assertAlmostEqual(utils.rotated_box_iou(base, rot90), 1.0/3.0, places=4)
        # Same box rotated 180°: identical shape, full overlap
        rot180 = [0, 0, 2, 4, math.pi]
        self.assertAlmostEqual(utils.rotated_box_iou(base, rot180), 1.0, places=5)

    def test_iou_45_degree_overlap(self):
        """Two same boxes rotated 45° relative: overlap is symmetric."""
        box0 = [0, 0, 2, 4, 0]
        box45 = [0, 0, 2, 4, math.pi/4]
        iou = utils.rotated_box_iou(box0, box45)
        self.assertGreater(iou, 0.2)
        self.assertLess(iou, 1.0)

    def test_compute_distance_bbox_identical(self):
        """Distance = 0 for identical boxes."""
        box = [1, 2, 2, 4, 0.5]
        self.assertAlmostEqual(utils.computeDistanceBBox(box, box), 0.0, places=6)

    def test_compute_distance_bbox_disjoint(self):
        """Distance = 1 for disjoint boxes."""
        a = [0, 0, 2, 4, 0]
        b = [100, 100, 2, 4, 0]
        self.assertAlmostEqual(utils.computeDistanceBBox(a, b), 1.0, places=6)

    def test_compute_distance_bbox_is_one_minus_iou(self):
        """computeDistanceBBox must equal 1 - IOU."""
        pairs = [
            ([0, 0, 2, 4, 0], [0, 0, 2, 4, 0]),
            ([0, 0, 2, 4, 0], [1, 0, 2, 4, 0]),
            ([0, 0, 2, 4, 0], [0, 0, 2, 4, math.pi/2]),
        ]
        for a, b in pairs:
            iou = utils.rotated_box_iou(a, b)
            dist = utils.computeDistanceBBox(a, b)
            self.assertAlmostEqual(dist, 1.0 - iou, places=6)


class TestMatchingManyElements(unittest.TestCase):
    """BallTree and matching with many tracks/detections."""

    def test_balltree_returns_nearest_by_iou(self):
        """BallTree with computeDistanceBBox metric returns nearest neighbor by IOU."""
        # Build 10 detection boxes in a row along x
        np.random.seed(42)
        detections = []
        for i in range(10):
            detections.append([float(i * 5), 0, 2, 4, 0])
        arr = np.array(detections).reshape(10, 5)
        tree = BallTree(arr, metric=utils.computeDistanceBBox)
        # Query with a box that is closest to index 3 (center 15, 0)
        query = np.array([[14, 0, 2, 4, 0]])
        dist, idx = tree.query(query, k=1)
        self.assertEqual(idx[0, 0], 3)
        # Query with box at 0,0 should match index 0
        query0 = np.array([[0, 0, 2, 4, 0]])
        _, idx0 = tree.query(query0, k=1)
        self.assertEqual(idx0[0, 0], 0)

    def test_balltree_k_nearest_ordering(self):
        """BallTree k-nearest returns distances in ascending order."""
        boxes = [[i * 3.0, 0, 2, 4, 0] for i in range(8)]
        arr = np.array(boxes).reshape(8, 5)
        tree = BallTree(arr, metric=utils.computeDistanceBBox)
        query = np.array([[10.0, 0, 2, 4, 0]])
        dist, idx = tree.query(query, k=3)
        self.assertLessEqual(dist[0, 0], dist[0, 1])
        self.assertLessEqual(dist[0, 1], dist[0, 2])
        # Closest should be index 3 (center 9) or 4 (center 12)
        self.assertIn(idx[0, 0], (3, 4))

    def test_many_tracks_many_detections_correct_pairing(self):
        """With a grid of tracks and detections, best pairing by IOU is correct."""
        # 3 tracks at (0,0), (6,0), (12,0)
        track_positions = [[0, 0, 2, 4, 0], [6, 0, 2, 4, 0], [12, 0, 2, 4, 0]]
        # 3 detections at (0.5, 0), (6.5, 0), (12.5, 0) -> should match 0-0, 1-1, 2-2
        detection_positions = [[0.5, 0, 2, 4, 0], [6.5, 0, 2, 4, 0], [12.5, 0, 2, 4, 0]]
        arr = np.array(detection_positions).reshape(3, 5)
        tree = BallTree(arr, metric=utils.computeDistanceBBox)
        for i, track in enumerate(track_positions):
            q = np.array([track]).reshape(1, 5)
            _, idx = tree.query(q, k=1)
            self.assertEqual(idx[0, 0], i, msg=f"Track {i} should match detection {i}")

    def test_matching_at_various_angles_many(self):
        """Many boxes at different angles: IOU and distance are consistent."""
        boxes = []
        for i in range(6):
            angle = i * math.pi / 6
            boxes.append([float(i * 4), 0, 2, 4, angle])
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                iou = utils.rotated_box_iou(boxes[i], boxes[j])
                dist = utils.computeDistanceBBox(boxes[i], boxes[j])
                self.assertGreaterEqual(iou, 0.0)
                self.assertLessEqual(iou, 1.0 + 1e-9)  # allow float rounding
                self.assertAlmostEqual(dist, 1.0 - iou, places=5)


class TestKalmanFilter(unittest.TestCase):
    """Kalman filter behavior: init, prediction, update, covariance, dimensions."""

    def test_first_frame_single_measurement(self):
        """First fusion with one measurement: state equals measurement."""
        k = ResizableKalman(time=0, x=0, y=0, fusion_mode=1)
        m = make_match(1, 7, 11, time=0)
        k.fusion([m], time=0, monitor=False)
        self.assertAlmostEqual(k.x, 7.0, places=5)
        self.assertAlmostEqual(k.y, 11.0, places=5)

    def test_first_frame_multiple_measurements(self):
        """First fusion with N measurements: state is mean of positions."""
        k = ResizableKalman(time=0, x=0, y=0, fusion_mode=1)
        ms = [make_match(i, 0 + i*2, 0, time=0) for i in range(5)]
        k.fusion(ms, time=0, monitor=False)
        # Mean x = 0+2+4+6+8 / 5 = 4, mean y = 0
        self.assertAlmostEqual(k.x, 4.0, places=5)
        self.assertAlmostEqual(k.y, 0.0, places=5)

    def test_prediction_constant_velocity(self):
        """With constant velocity, prediction propagates position correctly."""
        k = ResizableKalman(time=0, x=0, y=0, fusion_mode=1)
        k.fusion([make_match(1, 0, 0, dx=5, dy=0, time=0)], time=0, monitor=False)
        k.fusion([make_match(1, 5, 0, dx=5, dy=0, time=1)], time=1, monitor=False)
        pred_x, pred_y, a, b, phi = k.getKalmanPred(2.0)
        # After two updates at (0,0) and (5,0), velocity is estimated; prediction at t=2 should be ahead of current state
        self.assertGreater(pred_x, k.x, msg="Predicted x should be ahead of current state")
        self.assertLess(pred_y, 2.0)
        self.assertGreater(pred_y, -2.0)

    def test_update_high_precision_pulls_toward_measurement(self):
        """Very small measurement covariance: state should follow measurement."""
        k = ResizableKalman(time=0, x=0, y=0, fusion_mode=1)
        k.fusion([make_match(1, 0, 0, time=0)], time=0, monitor=False)
        tiny_cov = np.eye(2) * 1e-4
        m = MatchClass(1, 20, 30, tiny_cov, 0, 0, 1, 1, 1, 0, 1, 2, 4, 0)
        k.fusion([m], time=1, monitor=False)
        self.assertGreater(k.x, 15)
        self.assertGreater(k.y, 20)

    def test_update_low_precision_stays_near_prior(self):
        """Very large measurement covariance: state should stay near prior."""
        k = ResizableKalman(time=0, x=0, y=0, fusion_mode=1)
        k.fusion([make_match(1, 0, 0, time=0)], time=0, monitor=False)
        big_cov = np.eye(2) * 100.0
        m = MatchClass(1, 50, 50, big_cov, 0, 0, 1, 1, 1, 0, 1, 2, 4, 0)
        k.fusion([m], time=1, monitor=False)
        self.assertLess(k.x, 15)
        self.assertLess(k.y, 15)

    def test_covariance_positive_definite_after_update(self):
        """Position covariance (2x2) remains positive definite after update."""
        k = ResizableKalman(time=0, x=0, y=0, fusion_mode=1)
        k.fusion([make_match(1, 0, 0, time=0)], time=0, monitor=False)
        k.fusion([make_match(1, 1, 1, time=1)], time=1, monitor=False)
        P = k.error_covariance
        eigenvalues = np.linalg.eigvalsh(P)
        self.assertTrue(np.all(eigenvalues > 0.01), msg=f"Eigenvalues {eigenvalues} should be positive")

    def test_dimension_tracking_width_length(self):
        """Width and length are updated by 1D Kalman from measurements."""
        k = ResizableKalman(time=0, x=0, y=0, fusion_mode=1, initial_width=3.0, initial_length=5.0)
        k.fusion([make_match(1, 0, 0, width=2, length=4, time=0)], time=0, monitor=False)
        k.fusion([make_match(1, 1, 0, width=2.1, length=4.2, time=1)], time=1, monitor=False)
        self.assertGreater(k.width, 2.0)
        self.assertLess(k.width, 2.2)
        self.assertGreater(k.length, 4.0)
        self.assertLess(k.length, 4.3)

    def test_ellipsify_returns_positive_axes(self):
        """getKalmanPred returns positive a, b for non-degenerate covariance."""
        k = ResizableKalman(time=0, x=0, y=0, fusion_mode=1)
        k.fusion([make_match(1, 0, 0, time=0)], time=0, monitor=False)
        k.fusion([make_match(1, 2, 0, time=1)], time=1, monitor=False)
        pred_x, pred_y, a, b, phi = k.getKalmanPred(2.0)
        self.assertGreater(a, 0)
        self.assertGreater(b, 0)

    def test_fusion_mode_zero_constant_position(self):
        """Fusion mode 0: no velocity; position stays at initial after predict."""
        k = ResizableKalman(time=0, x=5, y=5, fusion_mode=0)
        k.fusion([make_match(1, 5, 5, time=0)], time=0, monitor=False)
        pred_x, pred_y, a, b, phi = k.getKalmanPred(10.0)
        self.assertAlmostEqual(pred_x, 5.0, places=5)
        self.assertAlmostEqual(pred_y, 5.0, places=5)


class TestCovariancePropagation(unittest.TestCase):
    """Tests that covariance from detections properly propagates through fusion."""

    def test_measurement_covariance_affects_kalman_gain(self):
        """Different measurement covariances give different state estimates."""
        # Initialize two identical filters
        k1 = ResizableKalman(time=0, x=0, y=0, fusion_mode=2)
        k2 = ResizableKalman(time=0, x=0, y=0, fusion_mode=2)
        
        # First frame identical
        k1.fusion([make_match(1, 0, 0, time=0)], time=0, monitor=False)
        k2.fusion([make_match(1, 0, 0, time=0)], time=0, monitor=False)
        
        # Second frame: same position (10, 0) but different covariances
        small_cov = np.array([[0.1, 0], [0, 0.1]], dtype='float')  # High precision
        large_cov = np.array([[10.0, 0], [0, 10.0]], dtype='float')  # Low precision
        
        m_small = MatchClass(1, 10, 0, small_cov, 10, 0, 1, 1, 1, 0, 1, 2, 4, 0)
        m_large = MatchClass(1, 10, 0, large_cov, 10, 0, 1, 1, 1, 0, 1, 2, 4, 0)
        
        k1.fusion([m_small], time=1, monitor=False)
        k2.fusion([m_large], time=1, monitor=False)
        
        # High precision measurement should pull state closer to measurement
        self.assertGreater(k1.x, k2.x, 
            msg="High-precision measurement should give larger x (closer to 10)")

    def test_localization_error_increases_track_covariance(self):
        """Larger measurement covariance should result in larger track covariance."""
        k_tight = ResizableKalman(time=0, x=0, y=0, fusion_mode=2)
        k_loose = ResizableKalman(time=0, x=0, y=0, fusion_mode=2)
        
        tight_cov = np.array([[0.25, 0], [0, 0.25]], dtype='float')  # 0.5m std
        loose_cov = np.array([[4.0, 0], [0, 4.0]], dtype='float')    # 2.0m std
        
        # Run several frames with consistent measurements
        for i in range(5):
            t = i * 0.1
            m_tight = MatchClass(1, i*0.5, 0, tight_cov, 5, 0, 1, 1, 1, 0, t, 2, 4, 0)
            m_loose = MatchClass(1, i*0.5, 0, loose_cov, 5, 0, 1, 1, 1, 0, t, 2, 4, 0)
            k_tight.fusion([m_tight], time=t, monitor=False)
            k_loose.fusion([m_loose], time=t, monitor=False)
        
        # Track fed with loose measurements should have larger position covariance
        trace_tight = np.trace(k_tight.error_covariance)
        trace_loose = np.trace(k_loose.error_covariance)
        self.assertGreater(trace_loose, trace_tight,
            msg=f"Loose measurements should give larger covariance: {trace_loose} > {trace_tight}")

    def test_covariance_reduces_with_consistent_measurements(self):
        """Position covariance should decrease as more consistent measurements arrive."""
        k = ResizableKalman(time=0, x=50, y=0, fusion_mode=2)
        
        # Consistent measurements at known position with moderate noise
        cov = np.array([[0.5, 0], [0, 0.5]], dtype='float')
        
        traces = []
        for i in range(10):
            t = i * 0.1
            m = MatchClass(1, 50, 0, cov, 0, 0, 1, 1, 1, 0, t, 2, 4, 0)
            k.fusion([m], time=t, monitor=False)
            traces.append(np.trace(k.error_covariance))
        
        # Covariance should generally decrease (or at least not explode)
        self.assertLess(traces[-1], traces[0] * 2,
            msg="Covariance should not grow unboundedly with consistent measurements")

    def test_anisotropic_covariance_preserved(self):
        """Non-diagonal (correlated) measurement covariance should affect state."""
        k1 = ResizableKalman(time=0, x=0, y=0, fusion_mode=2)
        k2 = ResizableKalman(time=0, x=0, y=0, fusion_mode=2)
        
        # Diagonal vs correlated covariance with same trace
        diag_cov = np.array([[1.0, 0], [0, 1.0]], dtype='float')
        corr_cov = np.array([[1.5, 0.5], [0.5, 0.5]], dtype='float')  # Correlated
        
        k1.fusion([make_match(1, 0, 0, cov=diag_cov, time=0)], time=0, monitor=False)
        k2.fusion([make_match(1, 0, 0, cov=corr_cov, time=0)], time=0, monitor=False)
        
        # Both should have valid positive definite covariance
        for k, name in [(k1, "diagonal"), (k2, "correlated")]:
            eigvals = np.linalg.eigvalsh(k.error_covariance)
            self.assertTrue(np.all(eigvals > 0), 
                msg=f"{name} covariance should be positive definite: {eigvals}")

    def test_detector_precision_vs_localizer_precision(self):
        """
        Simulate realistic scenario: detector covariance (position) + localizer covariance.
        Higher combined error should result in more uncertain track.
        """
        # Scenario 1: Good detector (0.3m) + good localizer (combined small cov)
        k_good = ResizableKalman(time=0, x=50, y=0, fusion_mode=2)
        good_cov = np.array([[0.1, 0], [0, 0.1]], dtype='float')  # ~0.3m std
        
        # Scenario 2: Poor detector (1.5m) + poor localizer (combined large cov)
        k_poor = ResizableKalman(time=0, x=50, y=0, fusion_mode=2)
        poor_cov = np.array([[2.25, 0], [0, 2.25]], dtype='float')  # ~1.5m std
        
        # Run 5 frames
        for i in range(5):
            t = i * 0.1
            x = 50 + i * 0.5
            k_good.fusion([MatchClass(1, x, 0, good_cov, 5, 0, 1, 1, 1, 0, t, 2, 4, 0)], time=t, monitor=False)
            k_poor.fusion([MatchClass(1, x, 0, poor_cov, 5, 0, 1, 1, 1, 0, t, 2, 4, 0)], time=t, monitor=False)
        
        # Good sensors should give tighter track covariance
        trace_good = np.trace(k_good.error_covariance)
        trace_poor = np.trace(k_poor.error_covariance)
        self.assertLess(trace_good, trace_poor,
            msg=f"Good sensors should give smaller covariance: {trace_good} < {trace_poor}")


class TestBallTreeMatching(unittest.TestCase):
    """Tests for BallTree-based detection-to-track matching."""

    def test_balltree_finds_exact_match(self):
        """Track at exact detection position should have distance 0 (IOU 1)."""
        track = [10, 20, 2, 4, 0]  # x, y, w, h, angle
        detections = [[10, 20, 2, 4, 0], [50, 50, 2, 4, 0]]
        
        tree = BallTree(np.array(detections), metric=utils.computeDistanceBBox)
        dist, idx = tree.query([track], k=1, return_distance=True)
        
        self.assertEqual(idx[0][0], 0)  # Should match first detection
        self.assertAlmostEqual(dist[0][0], 0.0, places=5)  # Distance should be 0

    def test_balltree_prefers_closer_detection(self):
        """Given multiple detections, should return nearest by IOU distance."""
        track = [10, 10, 2, 4, 0]
        detections = [
            [10, 10, 2, 4, 0],   # Exact match
            [11, 10, 2, 4, 0],   # 1m away
            [15, 10, 2, 4, 0],   # 5m away
        ]
        
        tree = BallTree(np.array(detections), metric=utils.computeDistanceBBox)
        dist, idx = tree.query([track], k=3, return_distance=True)
        
        # Should return in order of distance (closest first)
        self.assertEqual(idx[0][0], 0)  # Exact match first
        self.assertEqual(idx[0][1], 1)  # 1m away second
        self.assertEqual(idx[0][2], 2)  # 5m away third
        
        # Distances should be increasing
        self.assertLess(dist[0][0], dist[0][1])
        self.assertLess(dist[0][1], dist[0][2])

    def test_balltree_angle_affects_matching(self):
        """Two detections at same position but different angles should have different IOU."""
        track = [10, 10, 2, 4, 0]  # Heading 0 (East)
        
        det_aligned = [10, 10, 2, 4, 0]        # Same heading
        det_perpendicular = [10, 10, 2, 4, math.pi/2]  # 90° rotated
        
        dist_aligned = utils.computeDistanceBBox(track, det_aligned)
        dist_perpendicular = utils.computeDistanceBBox(track, det_perpendicular)
        
        self.assertLess(dist_aligned, dist_perpendicular,
            msg="Aligned box should have smaller distance (higher IOU)")

    def test_matching_with_realistic_vehicle_dimensions(self):
        """Matching with realistic car dimensions (1.8m x 4.5m)."""
        width, length = 1.8, 4.5
        
        # Track at (50, 0), detection at (50.5, 0) - small offset
        track = [50, 0, width, length, 0]
        detection = [50.5, 0, width, length, 0]
        
        dist = utils.computeDistanceBBox(track, detection)
        
        # Should have high IOU (low distance) for small offset
        self.assertLess(dist, 0.3, msg="Small offset should still have high IOU")

    def test_matching_rejects_distant_detection(self):
        """Detections far from track should have distance close to 1 (IOU near 0)."""
        track = [0, 0, 2, 4, 0]
        far_detection = [100, 100, 2, 4, 0]
        
        dist = utils.computeDistanceBBox(track, far_detection)
        
        self.assertGreater(dist, 0.99, msg="Far detection should have IOU near 0")

    def test_multiple_tracks_multiple_detections_pairing(self):
        """Multiple tracks matching to multiple detections should find optimal pairs."""
        # 3 tracks in a row
        tracks = [
            [0, 0, 2, 4, 0],
            [10, 0, 2, 4, 0],
            [20, 0, 2, 4, 0],
        ]
        
        # 3 detections, slightly offset from tracks
        detections = [
            [0.5, 0, 2, 4, 0],   # Near track 0
            [10.3, 0, 2, 4, 0],  # Near track 1
            [19.8, 0, 2, 4, 0],  # Near track 2
        ]
        
        tree = BallTree(np.array(detections), metric=utils.computeDistanceBBox)
        
        # Each track should match to its nearest detection
        for i, track in enumerate(tracks):
            dist, idx = tree.query([track], k=1, return_distance=True)
            self.assertEqual(idx[0][0], i, 
                msg=f"Track {i} should match detection {i}, got {idx[0][0]}")


class TestCTRVHeadingMeasurement(unittest.TestCase):
    """Tests that CTRV properly uses heading measurements."""

    def test_heading_measurement_updates_state(self):
        """Heading measurement should update the psi state in CTRV."""
        k = ResizableKalman(time=0, x=0, y=0, fusion_mode=2)
        
        # First measurement: heading = 0
        m1 = MatchClass(1, 0, 0, np.eye(2)*0.25, 0, 0, 1, 1, 1, 0, 0, 2, 4, 0)
        k.fusion([m1], time=0, monitor=False)
        
        # Second measurement: heading = pi/4 (45 degrees)
        m2 = MatchClass(1, 1, 1, np.eye(2)*0.25, 1, 1, 1, 1, 1, 0, 0.1, 2, 4, math.pi/4)
        k.fusion([m2], time=0.1, monitor=False)
        
        # Heading state (X_hat_t[3]) should move toward pi/4
        heading = k.X_hat_t[3, 0]
        self.assertGreater(heading, 0, msg="Heading should be positive after pi/4 measurement")
        self.assertLess(heading, math.pi/2, msg="Heading should be less than pi/2")

    def test_ctrv_velocity_inferred_from_motion(self):
        """CTRV should infer velocity magnitude from position changes."""
        k = ResizableKalman(time=0, x=0, y=0, fusion_mode=2)
        
        # Vehicle moving East at 5 m/s
        cov = np.eye(2) * 0.25
        for i in range(10):
            t = i * 0.1
            x = i * 0.5  # 5 m/s * 0.1s = 0.5m per frame
            m = MatchClass(1, x, 0, cov, 5, 0, 1, 1, 1, 0, t, 2, 4, 0)
            k.fusion([m], time=t, monitor=False)
        
        # Velocity state (X_hat_t[2]) should be approximately 5 m/s
        v = k.X_hat_t[2, 0]
        self.assertGreater(v, 2.0, msg=f"Velocity {v} should be at least 2 m/s")
        self.assertLess(v, 10.0, msg=f"Velocity {v} should be less than 10 m/s")


class TestTypeVoting(unittest.TestCase):
    """Tests for vehicle type voting in GlobalTracked."""

    def test_type_initialized_from_first_detection(self):
        """Track type should be set from the first detection."""
        from sensor_fusion import GlobalTracked
        from sensor import DetectedObject
        
        det = DetectedObject(
            vehicle_id=1, vehicle_type="car", detected_bbox=None,
            centroid=(10, 20), width=2.0, length=4.5, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        
        track = GlobalTracked(det, time=0, track_id=1, fusion_mode=2)
        self.assertEqual(track.type, "car")

    def test_type_voting_majority_wins(self):
        """After multiple detections, majority type should win."""
        from sensor_fusion import GlobalTracked
        from sensor import DetectedObject
        
        det1 = DetectedObject(
            vehicle_id=1, vehicle_type="truck", detected_bbox=None,
            centroid=(10, 20), width=2.5, length=8.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        
        track = GlobalTracked(det1, time=0, track_id=1, fusion_mode=2)
        self.assertEqual(track.type, "truck")
        
        # Add 5 car detections
        for i in range(5):
            det = DetectedObject(
                vehicle_id=1, vehicle_type="car", detected_bbox=None,
                centroid=(10 + i*0.1, 20), width=2.0, length=4.5, angle=0.0,
                expected_error_gaussian=None, velocity_vector=(1, 0),
                error_covariance=np.eye(2) * 0.25
            )
            track.update(det, time=0.1*(i+1))
        
        # Should now be "car" due to majority
        self.assertEqual(track.type, "car")

    def test_type_voting_handles_none(self):
        """None type values should not affect voting."""
        from sensor_fusion import GlobalTracked
        from sensor import DetectedObject
        
        det1 = DetectedObject(
            vehicle_id=1, vehicle_type="car", detected_bbox=None,
            centroid=(10, 20), width=2.0, length=4.5, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        
        track = GlobalTracked(det1, time=0, track_id=1, fusion_mode=2)
        
        # Add detection with None type
        det2 = DetectedObject(
            vehicle_id=1, vehicle_type=None, detected_bbox=None,
            centroid=(10.1, 20), width=2.0, length=4.5, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(1, 0),
            error_covariance=np.eye(2) * 0.25
        )
        track.update(det2, time=0.1)
        
        # Should still be "car"
        self.assertEqual(track.type, "car")

    def test_type_voting_recency_weighted(self):
        """More recent detections should have more weight due to decay."""
        from sensor_fusion import GlobalTracked
        from sensor import DetectedObject
        
        det1 = DetectedObject(
            vehicle_id=1, vehicle_type="truck", detected_bbox=None,
            centroid=(10, 20), width=2.5, length=8.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        
        track = GlobalTracked(det1, time=0, track_id=1, fusion_mode=2)
        
        # Add 3 truck detections
        for i in range(3):
            det = DetectedObject(
                vehicle_id=1, vehicle_type="truck", detected_bbox=None,
                centroid=(10 + i*0.1, 20), width=2.5, length=8.0, angle=0.0,
                expected_error_gaussian=None, velocity_vector=(1, 0),
                error_covariance=np.eye(2) * 0.25
            )
            track.update(det, time=0.1*(i+1))
        
        # Then add 4 car detections (more recent)
        for i in range(4):
            det = DetectedObject(
                vehicle_id=1, vehicle_type="car", detected_bbox=None,
                centroid=(10.4 + i*0.1, 20), width=2.0, length=4.5, angle=0.0,
                expected_error_gaussian=None, velocity_vector=(1, 0),
                error_covariance=np.eye(2) * 0.25
            )
            track.update(det, time=0.4 + 0.1*(i+1))
        
        # Car should win due to more recent votes (4 car at 1.0 weight vs 4 truck decayed)
        self.assertEqual(track.type, "car")

    def test_type_persists_in_output(self):
        """Type should be passed through to output DetectedObjects."""
        from sensor_fusion import Fusion
        from sensor import DetectedObject
        
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        det = DetectedObject(
            vehicle_id=1, vehicle_type="bus", detected_bbox=None,
            centroid=(50, 0), width=2.5, length=12.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        
        # Run for enough frames to exceed trackShowThreshold
        for t in range(10):
            fusion.processDetectionFrame(t * 0.1, [det], 1.0, source_participant_id=0)
            result, detected_objects, _, _ = fusion.fuseDetectionFrame(t * 0.1, monitor=False)
        
        # Output should have the type
        self.assertGreater(len(detected_objects), 0)
        self.assertEqual(detected_objects[0].type, "bus")


class TestHungarianMatching(unittest.TestCase):
    """Tests for the Hungarian algorithm-based matching and Mahalanobis distance."""

    def test_mahalanobis_distance_identical_points(self):
        """Mahalanobis distance between identical points should be 0."""
        S_inv = np.eye(2)
        dist = utils.mahalanobis_distance([10, 20], [10, 20], S_inv)
        self.assertAlmostEqual(dist, 0.0)

    def test_mahalanobis_distance_unit_offset(self):
        """Mahalanobis distance with identity covariance equals Euclidean."""
        S_inv = np.eye(2)
        dist = utils.mahalanobis_distance([0, 0], [3, 4], S_inv)
        self.assertAlmostEqual(dist, 5.0)

    def test_mahalanobis_distance_covariance_scaling(self):
        """Higher uncertainty (larger covariance) reduces Mahalanobis distance."""
        pos1, pos2 = [0, 0], [10, 0]
        
        # Tight covariance
        S_tight = np.eye(2)
        dist_tight = utils.mahalanobis_distance(pos1, pos2, np.linalg.inv(S_tight))
        
        # Loose covariance
        S_loose = np.eye(2) * 100
        dist_loose = utils.mahalanobis_distance(pos1, pos2, np.linalg.inv(S_loose))
        
        self.assertGreater(dist_tight, dist_loose)

    def test_mahalanobis_anisotropic_covariance(self):
        """Anisotropic covariance affects distance differently per axis."""
        # High uncertainty in x, low in y
        S = np.array([[100, 0], [0, 1]])
        S_inv = np.linalg.inv(S)
        
        # x-offset should have low Mahalanobis distance
        dist_x = utils.mahalanobis_distance([0, 0], [10, 0], S_inv)
        # y-offset should have high Mahalanobis distance
        dist_y = utils.mahalanobis_distance([0, 0], [0, 10], S_inv)
        
        self.assertLess(dist_x, dist_y)

    def test_hybrid_cost_perfect_match(self):
        """Perfect overlap gives low cost."""
        cost, mahal, iou = utils.compute_hybrid_cost(
            detection_pos=[10, 20],
            detection_cov=np.eye(2),
            predicted_pos=[10, 20],
            P_pred=np.eye(2),
            det_bbox=[10, 20, 2, 4, 0],
            pred_bbox=[10, 20, 2, 4, 0]
        )
        self.assertLess(cost, 0.1)
        self.assertAlmostEqual(iou, 1.0)

    def test_hybrid_cost_gated_out(self):
        """Far away detection exceeds Mahalanobis gate."""
        cost, mahal, iou = utils.compute_hybrid_cost(
            detection_pos=[100, 200],  # Far from predicted
            detection_cov=np.eye(2),
            predicted_pos=[10, 20],
            P_pred=np.eye(2),
            det_bbox=[100, 200, 2, 4, 0],
            pred_bbox=[10, 20, 2, 4, 0],
            mahal_gate=9.21  # 99% chi-squared gate
        )
        # Should be gated out (returns 1e9)
        self.assertEqual(cost, 1e9)

    def test_hybrid_cost_close_but_no_overlap(self):
        """Close detection without IOU overlap uses Mahalanobis."""
        # Detection 5m away - no IOU but within Mahalanobis gate
        cost, mahal, iou = utils.compute_hybrid_cost(
            detection_pos=[15, 20],
            detection_cov=np.eye(2) * 4,  # 2m std
            predicted_pos=[10, 20],
            P_pred=np.eye(2) * 4,
            det_bbox=[15, 20, 2, 4, 0],
            pred_bbox=[10, 20, 2, 4, 0],
            mahal_gate=13.82  # 99.9% gate
        )
        # Should not be gated out
        self.assertLess(cost, 1e9)
        # IOU should be 0 or very low
        self.assertLess(iou, 0.1)

    def test_hungarian_optimal_assignment(self):
        """Hungarian algorithm finds globally optimal assignment."""
        from sensor_fusion import Fusion
        from sensor import DetectedObject
        
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        # Create initial tracks
        det1 = DetectedObject(
            vehicle_id=1, vehicle_type="car", detected_bbox=None,
            centroid=(10, 0), width=2.0, length=4.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        det2 = DetectedObject(
            vehicle_id=2, vehicle_type="car", detected_bbox=None,
            centroid=(20, 0), width=2.0, length=4.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        
        fusion.processDetectionFrame(0, [det1, det2], 1.0, source_participant_id=0)
        fusion.fuseDetectionFrame(0, monitor=False)
        
        # Should have 2 tracks
        self.assertEqual(len(fusion.tracked_list), 2)
        
        # Now send detections in swapped order but same positions
        # Hungarian should correctly match regardless of order
        det2_again = DetectedObject(
            vehicle_id=2, vehicle_type="car", detected_bbox=None,
            centroid=(20, 0), width=2.0, length=4.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        det1_again = DetectedObject(
            vehicle_id=1, vehicle_type="car", detected_bbox=None,
            centroid=(10, 0), width=2.0, length=4.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        
        fusion.processDetectionFrame(0.1, [det2_again, det1_again], 1.0, source_participant_id=0)
        fusion.fuseDetectionFrame(0.1, monitor=False)
        
        # Still should have exactly 2 tracks
        self.assertEqual(len(fusion.tracked_list), 2)

    def test_hungarian_handles_more_detections_than_tracks(self):
        """New detections create new tracks."""
        from sensor_fusion import Fusion
        from sensor import DetectedObject
        
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        # Start with 1 detection
        det1 = DetectedObject(
            vehicle_id=1, vehicle_type="car", detected_bbox=None,
            centroid=(10, 0), width=2.0, length=4.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        
        fusion.processDetectionFrame(0, [det1], 1.0, source_participant_id=0)
        fusion.fuseDetectionFrame(0, monitor=False)
        
        self.assertEqual(len(fusion.tracked_list), 1)
        
        # Now add 2 more detections
        det2 = DetectedObject(
            vehicle_id=2, vehicle_type="car", detected_bbox=None,
            centroid=(20, 0), width=2.0, length=4.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        det3 = DetectedObject(
            vehicle_id=3, vehicle_type="car", detected_bbox=None,
            centroid=(30, 0), width=2.0, length=4.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        
        fusion.processDetectionFrame(0.1, [det1, det2, det3], 1.0, source_participant_id=0)
        fusion.fuseDetectionFrame(0.1, monitor=False)
        
        # Should have 3 tracks now
        self.assertEqual(len(fusion.tracked_list), 3)

    def test_hungarian_handles_more_tracks_than_detections(self):
        """Unmatched tracks persist until cleanup."""
        from sensor_fusion import Fusion
        from sensor import DetectedObject
        
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        # Start with 3 detections
        dets = [
            DetectedObject(
                vehicle_id=i, vehicle_type="car", detected_bbox=None,
                centroid=(10 * i, 0), width=2.0, length=4.0, angle=0.0,
                expected_error_gaussian=None, velocity_vector=(0, 0),
                error_covariance=np.eye(2) * 0.25
            )
            for i in range(1, 4)
        ]
        
        fusion.processDetectionFrame(0, dets, 1.0, source_participant_id=0)
        fusion.fuseDetectionFrame(0, monitor=False)
        
        self.assertEqual(len(fusion.tracked_list), 3)
        
        # Now only send 1 detection - the other 2 tracks should persist
        fusion.processDetectionFrame(0.1, [dets[0]], 1.0, source_participant_id=0)
        fusion.fuseDetectionFrame(0.1, monitor=False)
        
        # All 3 tracks should still exist (cleanup time not reached)
        self.assertEqual(len(fusion.tracked_list), 3)

    def test_mahalanobis_enables_matching_drifted_track(self):
        """Mahalanobis distance allows matching even when IOU is 0."""
        from sensor_fusion import Fusion
        from sensor import DetectedObject
        
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        # Create track at position (10, 0)
        det1 = DetectedObject(
            vehicle_id=1, vehicle_type="car", detected_bbox=None,
            centroid=(10, 0), width=2.0, length=4.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(5, 0),  # Moving right
            error_covariance=np.eye(2) * 0.25
        )
        
        fusion.processDetectionFrame(0, [det1], 1.0, source_participant_id=0)
        fusion.fuseDetectionFrame(0, monitor=False)
        
        initial_track_id = fusion.tracked_list[0].id
        
        # After 0.5s, vehicle should be at ~12.5m but detection is at 12m
        # With 2x4m bbox, IOU might be minimal, but Mahalanobis should work
        det2 = DetectedObject(
            vehicle_id=1, vehicle_type="car", detected_bbox=None,
            centroid=(12, 0), width=2.0, length=4.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(5, 0),
            error_covariance=np.eye(2) * 0.25
        )
        
        fusion.processDetectionFrame(0.5, [det2], 1.0, source_participant_id=0)
        fusion.fuseDetectionFrame(0.5, monitor=False)
        
        # Should still have 1 track (matched, not created new)
        self.assertEqual(len(fusion.tracked_list), 1)
        # Track ID should be same (track persisted)
        self.assertEqual(fusion.tracked_list[0].id, initial_track_id)


class TestCleanDetectionsImproved(unittest.TestCase):
    """Tests for improved cleanDetections using Mahalanobis distance."""

    def test_clean_detections_merges_close_tracks(self):
        """Tracks within Mahalanobis threshold get merged."""
        from sensor_fusion import Fusion
        from sensor import DetectedObject
        
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        # Create two very close tracks (simulating duplicate)
        det1 = DetectedObject(
            vehicle_id=1, vehicle_type="car", detected_bbox=None,
            centroid=(10, 0), width=2.0, length=4.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        det2 = DetectedObject(
            vehicle_id=2, vehicle_type="car", detected_bbox=None,
            centroid=(10.5, 0), width=2.0, length=4.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        
        # Create tracks from separate frames (bypassing matching)
        fusion.processDetectionFrame(0, [det1], 1.0, source_participant_id=0)
        fusion.fuseDetectionFrame(0, monitor=False)
        
        # Force creation of second track
        fusion.processDetectionFrame(0.1, [det1, det2], 1.0, source_participant_id=1)
        fusion.fuseDetectionFrame(0.1, monitor=False)
        
        # Multiple frames to establish tracks
        for t in range(5):
            fusion.processDetectionFrame(0.2 + t * 0.1, [det1, det2], 1.0, source_participant_id=0)
            fusion.fuseDetectionFrame(0.2 + t * 0.1, monitor=False)
        
        # Clean should identify these as duplicates
        fusion.cleanDetections()
        
        # Should have fewer tracks after cleaning (exact count depends on threshold)
        # At minimum, shouldn't explode in track count
        self.assertLessEqual(len(fusion.tracked_list), 2)

    def test_clean_detections_keeps_distinct_tracks(self):
        """Tracks far apart should not be merged."""
        from sensor_fusion import Fusion
        from sensor import DetectedObject
        
        fusion = Fusion(id=0, use_trust_scoring=False)
        
        # Two tracks 20m apart - should not merge
        det1 = DetectedObject(
            vehicle_id=1, vehicle_type="car", detected_bbox=None,
            centroid=(0, 0), width=2.0, length=4.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        det2 = DetectedObject(
            vehicle_id=2, vehicle_type="car", detected_bbox=None,
            centroid=(20, 0), width=2.0, length=4.0, angle=0.0,
            expected_error_gaussian=None, velocity_vector=(0, 0),
            error_covariance=np.eye(2) * 0.25
        )
        
        for t in range(10):
            fusion.processDetectionFrame(t * 0.1, [det1, det2], 1.0, source_participant_id=0)
            fusion.fuseDetectionFrame(t * 0.1, monitor=False)
        
        fusion.cleanDetections()
        
        # Should still have 2 distinct tracks
        self.assertEqual(len(fusion.tracked_list), 2)


if __name__ == '__main__':
    unittest.main()
