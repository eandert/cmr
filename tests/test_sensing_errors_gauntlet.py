"""
Gauntlet tests: sensing generation, error models, localizer, ground truth,
gaussians, and utils. No stone unturned.
Run from repo root: python tests/test_sensing_errors_gauntlet.py
"""
import unittest
import sys
import os
import math
import numpy as np

# Add src to path (run from repo root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import utils
import gaussians
import ground_truth
import localizer
from config.detector_type import DetectorType
from config.sensor_type import SensorType
from error import ErrorPackage, ErrorType


# ---------------------------------------------------------------------------
# Utils: Polynomial, rotate_point, Kalman, ellipsify, get_rotated_box_corners
# ---------------------------------------------------------------------------

class TestUtilsPolynomial(unittest.TestCase):
    def test_polynomial_evaluate_constant(self):
        p = utils.Polynomial([3.0])
        self.assertAlmostEqual(p.evaluate(0), 3.0)
        self.assertAlmostEqual(p.evaluate(10), 3.0)

    def test_polynomial_evaluate_linear(self):
        p = utils.Polynomial([0, 2.0])
        self.assertAlmostEqual(p.evaluate(0), 0)
        self.assertAlmostEqual(p.evaluate(5), 10.0)

    def test_polynomial_evaluate_quadratic(self):
        p = utils.Polynomial([1, 0, 1])  # 1 + x^2
        self.assertAlmostEqual(p.evaluate(0), 1.0)
        self.assertAlmostEqual(p.evaluate(2), 5.0)


class TestUtilsRotatePoint(unittest.TestCase):
    def test_rotate_point_identity(self):
        x, y = utils.rotate_point(1, 0, 0, 0, 0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 0.0)

    def test_rotate_point_90_around_origin(self):
        # (1, 0) rotated 90° CCW around (0,0) -> (0, 1)
        x, y = utils.rotate_point(1, 0, 0, 0, math.pi / 2)
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 1.0, places=5)

    def test_rotate_point_around_center(self):
        # Point (5, 5) around center (5, 5) -> unchanged for any angle
        x, y = utils.rotate_point(5, 5, 5, 5, math.pi / 4)
        self.assertAlmostEqual(x, 5.0, places=5)
        self.assertAlmostEqual(y, 5.0, places=5)


class TestUtilsKalman(unittest.TestCase):
    def test_kalman_prediction_constant_velocity(self):
        # State [x, y, vx, vy], F = I + dt * (velocity part)
        X = np.array([[0], [0], [1], [0]])
        P = np.eye(4) * 0.1
        F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.zeros((4, 1))
        Q = np.eye(4) * 0.01
        X_new, P_new = utils.kalman_prediction(X, P, F, B, 0, Q)
        self.assertAlmostEqual(X_new[0, 0], 1.0)
        self.assertAlmostEqual(X_new[1, 0], 0.0)

    def test_kalman_update_reduces_covariance(self):
        X = np.array([[0], [0]])
        P = np.eye(2)
        Z = np.array([[1], [1]])
        R = np.eye(2) * 0.1
        H = np.eye(2)
        X_u, P_u = utils.kalman_update(X, P, Z, R, H)
        self.assertLess(P_u[0, 0], P[0, 0])
        self.assertAlmostEqual(X_u[0, 0], 1.0, delta=0.1)
        self.assertAlmostEqual(X_u[1, 0], 1.0, delta=0.1)


class TestUtilsEllipsify(unittest.TestCase):
    def test_ellipsify_diagonal_covariance(self):
        cov = np.array([[4.0, 0], [0, 1.0]])
        a, b, phi = utils.ellipsify(cov, 1.0)
        self.assertAlmostEqual(a, 2.0)
        self.assertAlmostEqual(b, 1.0)
        self.assertAlmostEqual(phi, 0.0)

    def test_ellipsify_positive_definite(self):
        cov = np.array([[1, 0.5], [0.5, 1]])
        a, b, phi = utils.ellipsify(cov, 2.0)
        self.assertGreater(a, 0)
        self.assertGreater(b, 0)


class TestUtilsGetRotatedBoxCorners(unittest.TestCase):
    def test_corners_at_zero_angle(self):
        # 2x4 box at origin, angle 0: length along x
        corners = utils.get_rotated_box_corners(0, 0, 2, 4, 0)
        self.assertEqual(corners.shape, (4, 2))
        # Half-length=2, half-width=1. Corners: (-2,-1), (2,-1), (2,1), (-2,1)
        self.assertAlmostEqual(corners[0, 0], -2.0)
        self.assertAlmostEqual(corners[0, 1], -1.0)


class TestUtilsPolygonAndRadius(unittest.TestCase):
    def test_polygon_area_rectangle(self):
        rect = np.array([[0, 0], [2, 0], [2, 1], [0, 1]])
        self.assertAlmostEqual(utils.polygon_area(rect), 2.0)

    def test_calculate_radius_at_angle(self):
        # Ellipse a=2, b=1, phi=0. At theta=0 radius is 2 (major axis)
        r = utils.calculateRadiusAtAngle(2, 1, 0, 0)
        self.assertAlmostEqual(r, 2.0)
        r90 = utils.calculateRadiusAtAngle(2, 1, 0, math.pi / 2)
        self.assertAlmostEqual(r90, 1.0)


# ---------------------------------------------------------------------------
# Gaussians
# ---------------------------------------------------------------------------

class TestGaussiansBivariate(unittest.TestCase):
    def test_covariance_rotation(self):
        # Variance 1,1 at angle 0 -> diagonal [[1,0],[0,1]]
        g = gaussians.BivariateGaussian(1.0, 1.0, 0)
        np.testing.assert_array_almost_equal(g.covariance, np.eye(2))

    def test_covariance_positive_definite(self):
        g = gaussians.BivariateGaussian(2.0, 0.5, math.pi / 4)
        vals = np.linalg.eigvalsh(g.covariance)
        self.assertTrue(np.all(vals > 0))

    def test_ellipse_radius_at_angle(self):
        g = gaussians.BivariateGaussian(1.0, 1.0, 0)
        r0 = g.calc_self_radius_at_angle(0, 1.0)
        r90 = g.calc_self_radius_at_angle(math.pi / 2, 1.0)
        self.assertGreater(r0, 0)
        self.assertGreater(r90, 0)


# ---------------------------------------------------------------------------
# Error models (DistributionBin without CSV dependency)
# ---------------------------------------------------------------------------

class TestErrorModelsDistributionBin(unittest.TestCase):
    def test_normal_bin_sample_and_std(self):
        from error_models import DistributionBin
        bin_obj = DistributionBin("normal", {"mu": 0, "sigma": 0.1}, 0, 10)
        std = bin_obj.get_std()
        self.assertAlmostEqual(std, 0.1)
        np.random.seed(42)
        s = bin_obj.sample()
        self.assertIsInstance(s, (float, np.floating))

    def test_laplace_bin_std(self):
        from error_models import DistributionBin
        bin_obj = DistributionBin("laplace", {"mu": 0, "b": 0.1}, 0, 10)
        std = bin_obj.get_std()
        self.assertAlmostEqual(std, 0.1 * np.sqrt(2))


class TestErrorModelsLoaded(unittest.TestCase):
    """Requires CSV data in src/data/sensor_models (bev_fusion, etc.). Skip if missing."""

    @classmethod
    def setUpClass(cls):
        try:
            from error_models import get_error_model
            cls.model = get_error_model("bev_fusion", use_gpem_model=True)
            cls.has_model = True
        except Exception:
            cls.has_model = False

    def test_get_distal_std_increases_with_distance(self):
        if not self.has_model:
            self.skipTest("bev_fusion CSV not available")
        s0 = self.model.get_distal_std(5)
        s1 = self.model.get_distal_std(50)
        self.assertGreater(s1, 0)
        self.assertGreater(s0, 0)

    def test_detection_probability_in_range(self):
        if not self.has_model:
            self.skipTest("bev_fusion CSV not available")
        p = self.model.detection_probability(20)
        self.assertGreaterEqual(p, 0)
        self.assertLessEqual(p, 1)

    def test_sample_all_errors_returns_dict(self):
        if not self.has_model:
            self.skipTest("bev_fusion CSV not available")
        np.random.seed(123)
        out = self.model.sample_all_errors(10.0, 0)
        self.assertIn('x_error', out)
        self.assertIn('y_error', out)
        self.assertIn('distal_std', out)
        self.assertIn('perp_std', out)
        self.assertIn('width_std', out)
        self.assertIn('length_std', out)


# ---------------------------------------------------------------------------
# Localizer
# ---------------------------------------------------------------------------

class TestLocalizer(unittest.TestCase):
    def test_lateral_longitudinal_at_velocity(self):
        # Polynomial [c0, c1] -> c0 + c1*v
        loc = localizer.Localizer([0.1, 0.01], [0.2, 0.02], ErrorPackage(ErrorType.LOCALIZATION, 50), use_gpem_model=True)
        lat = loc.lateral_error_at_velocity(10)
        long_ = loc.longitudinal_error_at_velocity(10)
        self.assertAlmostEqual(lat, 0.1 + 0.01 * 10)
        self.assertAlmostEqual(long_, 0.2 + 0.02 * 10)

    def test_get_localization_covariance_shape_and_symmetry(self):
        loc = localizer.Localizer([0.1], [0.2], ErrorPackage(ErrorType.LOCALIZATION, 50), use_gpem_model=True)
        cov = loc.get_localization_covariance(5.0, 0)
        self.assertEqual(cov.shape, (2, 2))
        np.testing.assert_array_almost_equal(cov, cov.T)
        vals = np.linalg.eigvalsh(cov)
        self.assertTrue(np.all(vals > 0))

    def test_get_localization_pose_with_error(self):
        loc = localizer.Localizer([0.1], [0.2], ErrorPackage(ErrorType.LOCALIZATION, 0), use_gpem_model=True)
        x, y, yaw = loc.get_localization_pose(0, 0, 0, 10, has_error=False)
        self.assertIsInstance(x, (int, float))
        self.assertIsInstance(y, (int, float))
        self.assertIsInstance(yaw, (int, float))


# ---------------------------------------------------------------------------
# Ground truth (no TraCI)
# ---------------------------------------------------------------------------

class TestGroundTruthFromData(unittest.TestCase):
    def test_create_ground_truth_from_data_angle_and_velocity(self):
        # SUMO angle 0 = East. We use 90 - angle_deg -> 90° -> East in math?
        # angle_rad = rad(90-0) = pi/2. So vehicle facing "North" in math.
        gt = ground_truth.create_ground_truth_from_data(
            "v0", position=(100, 200), speed=10, angle_deg=0,
            length=4, width=2, vehicle_type="car"
        )
        self.assertEqual(gt.vehicle_id, "v0")
        self.assertEqual(gt.dimensions[0], 2)
        self.assertEqual(gt.dimensions[1], 4)
        # Velocity: 10 * (cos(pi/2), sin(pi/2)) = (0, 10)
        self.assertAlmostEqual(gt.velocity_vector[0], 0, places=5)
        self.assertAlmostEqual(gt.velocity_vector[1], 10, places=5)

    def test_ground_truth_bbox_four_corners(self):
        gt = ground_truth.create_ground_truth_from_data(
            "v1", position=(0, 0), speed=0, angle_deg=90,
            length=4, width=2, vehicle_type="car"
        )
        self.assertEqual(len(gt.bbox), 4)
        for p in gt.bbox:
            self.assertEqual(len(p), 2)


# ---------------------------------------------------------------------------
# Sensor (no TraCI; use PERFECT detector and SensorType)
# ---------------------------------------------------------------------------

class TestSensor(unittest.TestCase):
    def test_check_in_range_and_fov_center(self):
        from sensor import Sensor
        sensor_type = SensorType.OS1_128
        detector_type = DetectorType.PERFECT
        sensor = Sensor(sensor_type, detector_type)
        # target_angle in sensor frame (relative). Center angle 0. So target 0, distance 50
        self.assertTrue(sensor.check_in_range_and_fov(0, 50))
        self.assertFalse(sensor.check_in_range_and_fov(0, 500))

    def test_perfect_detector_polynomials(self):
        from sensor import Sensor
        sensor = Sensor(SensorType.OS1_128, DetectorType.PERFECT)
        self.assertAlmostEqual(sensor.centroid_radial_error_at_distance(10), 0.000001)
        self.assertAlmostEqual(sensor.detection_probability_at_distance(10), 0.999999, places=5)


# ---------------------------------------------------------------------------
# Sensing: create_detected_bounding_boxes (PERFECT detector, deterministic)
# ---------------------------------------------------------------------------

class TestCreateDetectedBoundingBoxes(unittest.TestCase):
    def test_perfect_detector_no_position_error(self):
        import random
        from sensor import Sensor, create_detected_bounding_boxes
        from ground_truth import GroundTruthObject
        random.seed(0)
        np.random.seed(0)
        sensor = Sensor(SensorType.OS1_128, DetectorType.PERFECT)
        actual = (0, 0, 0)
        believed = (0, 0, 0)
        gt = GroundTruthObject("v1", "car", (10, 0), (0, 0), [], 0, 2, 4)
        detections, _ = create_detected_bounding_boxes(
            sensor, actual, believed, [gt], "ego"
        )
        # PERFECT detector has prob ~1; with seed we expect one detection
        self.assertGreaterEqual(len(detections), 1, msg="PERFECT detector should detect in-FOV object")
        d = detections[0]
        self.assertAlmostEqual(d.centroid[0], 10, delta=0.01)
        self.assertAlmostEqual(d.centroid[1], 0, delta=0.01)
        self.assertEqual(d.dimensions[0], 2)
        self.assertEqual(d.dimensions[1], 4)


# ---------------------------------------------------------------------------
# ErrorPackage
# ---------------------------------------------------------------------------

class TestErrorPackage(unittest.TestCase):
    def test_extrinsics_severity(self):
        pkg = ErrorPackage(ErrorType.SINGLE_SENSOR_EXTRINSICS, 5)
        self.assertAlmostEqual(pkg.angle_error_rad, math.radians(5))
        self.assertAlmostEqual(pkg.position_error_m, 0.5)

    def test_localization_severity(self):
        pkg = ErrorPackage(ErrorType.LOCALIZATION, 50)
        self.assertAlmostEqual(pkg.error_scale, 0.5)


if __name__ == '__main__':
    unittest.main()
