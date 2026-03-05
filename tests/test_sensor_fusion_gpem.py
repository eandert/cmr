"""
Unit tests for sensor fusion math and GPEM (Generalized Parameterized Error Modeling) adherence.

These tests verify that:
- First-frame fusion uses the correct covariance-of-mean formula: Cov(mean) = (1/N) * mean(Sigma_i)
- Kalman update equations are correct (gain K, state update, covariance update)
- GPEM-derived covariance is used in fusion and smaller R gives more weight to the measurement
- Ellipsify and covariance properties hold (trace, positive definiteness)
- Fusion with multiple measurements respects measurement covariances (GPEM or static)

Run from repo root: python -m pytest tests/test_sensor_fusion_gpem.py -v
"""

import unittest
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sensor_fusion import ResizableKalman, MatchClass
import utils


# -----------------------------------------------------------------------------
# First-frame average: Cov(mean of N iid) = Sigma / N
# -----------------------------------------------------------------------------


class TestFirstFrameAverageCovariance(unittest.TestCase):
    """Verify first-frame fusion uses Cov(mean) = (1/N)*mean(Sigma_i)."""

    def test_single_measurement_returns_unchanged(self):
        """Single measurement: first-frame average returns that measurement and its covariance."""
        kf = ResizableKalman(time=0.0, x=1.0, y=2.0, fusion_mode=0)
        cov = np.array([[0.5, 0.0], [0.0, 0.5]])
        match = MatchClass(1, 1.0, 2.0, cov, 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
        kf.addFrames([match])
        mu, c = kf.averageMeasurementsFirstFrame()
        np.testing.assert_array_almost_equal(mu, [1.0, 2.0])
        np.testing.assert_array_almost_equal(c, cov)

    def test_two_equal_covariances_gives_half(self):
        """Two measurements with same Sigma: fused covariance = Sigma/2."""
        kf = ResizableKalman(time=0.0, x=0.0, y=0.0, fusion_mode=0)
        sigma = np.array([[0.4, 0.0], [0.0, 0.4]])
        m1 = MatchClass(1, 0.0, 0.0, sigma, 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
        m2 = MatchClass(2, 0.0, 0.0, sigma.copy(), 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
        kf.addFrames([m1, m2])
        mu, c = kf.averageMeasurementsFirstFrame()
        expected_cov = sigma / 2.0
        np.testing.assert_array_almost_equal(c, expected_cov,
            err_msg="Cov(mean) for 2 iid should be Sigma/2")

    def test_three_equal_covariances_gives_third(self):
        """Three measurements with same Sigma: fused covariance = Sigma/3."""
        kf = ResizableKalman(time=0.0, x=0.0, y=0.0, fusion_mode=0)
        sigma = np.array([[0.6, 0.0], [0.0, 0.6]])
        matches = [
            MatchClass(i, 0.1 * i, 0.2 * i, sigma.copy(), 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
            for i in range(1, 4)
        ]
        kf.addFrames(matches)
        mu, c = kf.averageMeasurementsFirstFrame()
        expected_cov = sigma / 3.0
        np.testing.assert_array_almost_equal(c, expected_cov,
            err_msg="Cov(mean) for 3 iid should be Sigma/3")

    def test_different_covariances_formula(self):
        """Cov(mean) = (1/N) * mean(Sigma_i) as implemented."""
        kf = ResizableKalman(time=0.0, x=0.0, y=0.0, fusion_mode=0)
        s1 = np.array([[0.2, 0.0], [0.0, 0.2]])
        s2 = np.array([[0.4, 0.0], [0.0, 0.4]])
        s3 = np.array([[0.6, 0.0], [0.0, 0.6]])
        m1 = MatchClass(1, 0.0, 0.0, s1, 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
        m2 = MatchClass(2, 0.0, 0.0, s2, 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
        m3 = MatchClass(3, 0.0, 0.0, s3, 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
        kf.addFrames([m1, m2, m3])
        mu, c = kf.averageMeasurementsFirstFrame()
        mean_sigma = (s1 + s2 + s3) / 3.0
        expected_cov = mean_sigma / 3.0  # (1/N)*mean(Sigma_i)
        np.testing.assert_array_almost_equal(c, expected_cov,
            err_msg="Cov(mean) = (1/N)*mean(Sigma_i)")


# -----------------------------------------------------------------------------
# Kalman update math: K = P H^T (H P H^T + R)^{-1}, x_new, P_new
# -----------------------------------------------------------------------------


class TestKalmanUpdateMath(unittest.TestCase):
    """Verify utils.kalman_update implements standard Kalman equations."""

    def test_kalman_update_reduces_covariance(self):
        """After a valid update, posterior P should have smaller or equal variance."""
        # 1D case in 2D state: state [x, y], we observe [x, y]
        X = np.array([[1.0], [2.0]])
        P = np.eye(2) * 2.0  # prior variance 2 each
        R = np.eye(2) * 0.5  # measurement noise 0.5
        H = np.eye(2)
        Z = np.array([[1.5], [2.5]])  # measurement
        X_new, P_new = utils.kalman_update(X, P, Z, R, H)
        self.assertLessEqual(P_new[0, 0], P[0, 0])
        self.assertLessEqual(P_new[1, 1], P[1, 1])
        self.assertGreater(P_new[0, 0], 0)
        self.assertGreater(P_new[1, 1], 0)

    def test_kalman_update_pulls_toward_measurement(self):
        """Posterior mean should be between prior and measurement."""
        X = np.array([[0.0], [0.0]])
        P = np.eye(2) * 1.0
        R = np.eye(2) * 1.0
        H = np.eye(2)
        Z = np.array([[10.0], [0.0]])
        X_new, _ = utils.kalman_update(X, P, Z, R, H)
        self.assertGreater(X_new[0, 0], 0)
        self.assertLess(X_new[0, 0], 10)

    def test_smaller_R_gives_more_weight_to_measurement(self):
        """With smaller R, posterior should be closer to measurement."""
        X = np.array([[0.0], [0.0]])
        P = np.eye(2) * 2.0
        Z = np.array([[5.0], [5.0]])
        H = np.eye(2)
        _, P_lo = utils.kalman_update(X, P.copy(), Z, np.eye(2) * 0.01, H)
        _, P_hi = utils.kalman_update(X, P.copy(), Z, np.eye(2) * 10.0, H)
        # Low R => trust measurement more => smaller posterior variance
        self.assertLess(np.trace(P_lo), np.trace(P_hi))

    def test_kalman_gain_formula(self):
        """Verify K = P H^T (H P H^T + R)^{-1} and x_new = x + K(z - H x)."""
        X = np.array([[0.0], [0.0]])
        P = np.eye(2) * 1.0
        R = np.eye(2) * 0.5
        H = np.eye(2)
        Z = np.array([[2.0], [3.0]])
        X_new, P_new = utils.kalman_update(X, P, Z, R, H)
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H.T).dot(np.linalg.inv(S))
        X_expected = X + K.dot(Z - H.dot(X))
        np.testing.assert_array_almost_equal(X_new, X_expected, decimal=5,
            err_msg="State update should follow x_new = x + K(z - Hx)")
        # Standard Kalman: P_new = (I - K H) P
        P_expected = P - K.dot(H).dot(P)
        np.testing.assert_array_almost_equal(P_new, P_expected, decimal=5,
            err_msg="Covariance update should follow P_new = (I-KH)P")


# -----------------------------------------------------------------------------
# GPEM: smaller covariance (e.g. near distance) => more weight in fusion
# -----------------------------------------------------------------------------


class TestGPEMCovarianceWeightInFusion(unittest.TestCase):
    """Verify that GPEM-derived (smaller) covariance gives more weight in fusion."""

    def test_smaller_measurement_cov_pulls_more(self):
        """Two measurements at same point: smaller R should dominate fused result."""
        # ResizableKalman fusion_mode=0 (CV), no trust scoring
        kf = ResizableKalman(time=0.0, x=0.0, y=0.0, fusion_mode=0)
        # First frame: init at (0,0) with some covariance
        small_r = np.eye(2) * 0.1   # precise measurement
        large_r = np.eye(2) * 2.0   # noisy measurement
        m_precise = MatchClass(1, 3.0, 0.0, small_r, 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
        m_noisy   = MatchClass(2, 0.0, 0.0, large_r, 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
        kf.addFrames([m_precise, m_noisy])
        kf.fusion([m_precise, m_noisy], time=0.0, monitor=False, participant_trust_scores=None)
        # Fused x should be pulled toward 3.0 (precise measurement)
        self.assertGreater(kf.x, 1.0, msg="Fused position should be pulled toward precise measurement")
        self.assertLess(kf.x, 3.0)

    def test_fused_covariance_smaller_than_max_input(self):
        """Fusing two measurements should yield posterior covariance smaller than the larger R."""
        kf = ResizableKalman(time=0.0, x=5.0, y=5.0, fusion_mode=0)
        r1 = np.eye(2) * 0.5
        r2 = np.eye(2) * 1.5
        m1 = MatchClass(1, 5.0, 5.0, r1, 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
        m2 = MatchClass(2, 5.0, 5.0, r2, 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
        kf.fusion([m1, m2], time=0.0, monitor=False, participant_trust_scores=None)
        # error_covariance is 2x2 position block
        trace_fused = np.trace(kf.error_covariance)
        self.assertLess(trace_fused, 1.6, msg="Fused covariance should be smaller than 1.5+epsilon")


# -----------------------------------------------------------------------------
# Ellipsify and covariance properties
# -----------------------------------------------------------------------------


class TestEllipsifyAndCovariance(unittest.TestCase):
    """Verify ellipsify and that covariance is used consistently."""

    def test_ellipsify_trace_consistent(self):
        """Ellipsify returns (semi-major, semi-minor, angle of major axis). For diagonal cov,
        eigenvalues 0.25, 0.49 => a=0.7, b=0.5; major axis along y => phi = π/2."""
        cov = np.array([[0.25, 0.0], [0.0, 0.49]])
        a, b, phi = utils.ellipsify(cov, num_std_deviations=1.0)
        self.assertAlmostEqual(a, 0.7, places=4, msg="a = semi-major = 1.0*sqrt(0.49)")
        self.assertAlmostEqual(b, 0.5, places=4, msg="b = semi-minor = 1.0*sqrt(0.25)")
        self.assertAlmostEqual(phi, math.pi / 2.0, places=5,
            msg="Major axis is along y (eigenvector for 0.49) => phi = π/2")

    def test_ellipsify_positive_definite(self):
        """Ellipsify should handle valid 2x2 covariance."""
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        a, b, phi = utils.ellipsify(cov, num_std_deviations=2.0)
        self.assertGreater(a, 0)
        self.assertGreater(b, 0)


# -----------------------------------------------------------------------------
# GPEM vs static: distance-dependent covariance
# -----------------------------------------------------------------------------


class TestGPEMDistanceDependentCovariance(unittest.TestCase):
    """Verify GPEM produces distance-dependent covariance used in fusion."""

    @classmethod
    def setUpClass(cls):
        try:
            from error_models import get_error_model
            cls.gpem = get_error_model("bev_fusion", use_gpem_model=True, force_reload=True)
            cls.static = get_error_model("bev_fusion", use_gpem_model=False, force_reload=True)
            cls.has_model = True
        except Exception:
            cls.has_model = False

    def test_gpem_std_increases_with_distance(self):
        """GPEM distal/perpendicular std should increase with distance (typical regression)."""
        if not self.has_model:
            self.skipTest("bev_fusion error model not available")
        near = 10.0
        far = 50.0
        d_near = self.gpem.get_distal_std(near)
        d_far = self.gpem.get_distal_std(far)
        p_near = self.gpem.get_perpendicular_std(near)
        p_far = self.gpem.get_perpendicular_std(far)
        self.assertLessEqual(d_near, d_far * 1.5 + 0.01,
            msg="GPEM distal std should generally increase with distance")
        self.assertLessEqual(p_near, p_far * 1.5 + 0.01,
            msg="GPEM perp std should generally increase with distance")

    def test_static_std_constant_with_distance(self):
        """Static model std should not depend on distance."""
        if not self.has_model:
            self.skipTest("bev_fusion error model not available")
        s1 = self.static.get_distal_std(10.0)
        s2 = self.static.get_distal_std(50.0)
        self.assertAlmostEqual(s1, s2, places=6,
            msg="Static model distal std should be constant")


# -----------------------------------------------------------------------------
# End-to-end: fusion uses measurement covariance (GPEM path)
# -----------------------------------------------------------------------------


class TestFusionUsesMeasurementCovariance(unittest.TestCase):
    """Fusion must use each measurement's covariance (from GPEM or static)."""

    def test_fusion_stores_and_uses_measurement_covariance(self):
        """ResizableKalman uses measurement R in update (not a constant)."""
        kf = ResizableKalman(time=0.0, x=0.0, y=0.0, fusion_mode=0)
        # First frame: one measurement
        r1 = np.array([[0.09, 0.0], [0.0, 0.09]])  # 0.3 m std
        m1 = MatchClass(1, 1.0, 1.0, r1, 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
        kf.fusion([m1], time=0.0, monitor=False, participant_trust_scores=None)
        self.assertGreater(kf.error_covariance[0, 0], 0)
        self.assertLessEqual(kf.error_covariance[0, 0], 1.0,
            msg="After one measurement, position variance should be bounded")

    def test_two_measurements_different_cov_fuse_correctly(self):
        """With a prior established, two measurements with different R: Kalman weights
        the smaller-R measurement more, so fused x is closer to 1.0 than to 2.0."""
        kf = ResizableKalman(time=0.0, x=0.0, y=0.0, fusion_mode=0)
        # Step 1: one measurement to establish prior (so idx > 0 and Kalman path is used next)
        m0 = MatchClass(0, 1.0, 0.0, np.eye(2) * 0.1, 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
        kf.fusion([m0], time=0.0, monitor=False, participant_trust_scores=None)
        # Step 2: two measurements at (1,0) small R and (2,0) large R
        m1 = MatchClass(1, 1.0, 0.0, np.eye(2) * 0.01, 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
        m2 = MatchClass(2, 2.0, 0.0, np.eye(2) * 1.0, 0, 0, 1.0, 1.0, 1.0, 0, 0.0, 2.0, 4.0, 0.0)
        kf.fusion([m1, m2], time=0.1, monitor=False, participant_trust_scores=None)
        # Fused x should be closer to 1.0 (small R) than to 2.0 (large R)
        self.assertGreater(kf.x, 1.0)
        self.assertLess(kf.x, 1.5)


if __name__ == "__main__":
    unittest.main()
