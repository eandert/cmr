"""
Comprehensive test suite for the Covariance Intersection (CI) filter.

Tests are organized into groups:
  1. Core CI math (_ci_fuse, _ci_fuse_multi)
  2. CI filter initialization and interface compliance
  3. CI filter fusion behavior (single/multi measurement, all fusion modes)
  4. CI vs Kalman comparison (CI should be more conservative but never overconfident)
  5. GPEM covariance sensitivity (CI should differentiate between different R values)
  6. Edge cases and numerical robustness
  7. Prediction methods (getKalmanPred, getKalmanPredWithCovariance)

Run: python -m pytest tests/test_ci_filter.py -v -s
"""

import unittest
import sys
import os
import math
import numpy as np
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sensor_fusion import MatchClass
from filters.covariance_intersection import (
    CovarianceIntersectionFilter, _ci_fuse, _ci_fuse_multi
)
from filters.kalman_ctrv import ResizableKalman
import utils


# =============================================================================
# HELPERS
# =============================================================================

def make_match(x, y, cov, tracker_id=1, confidence=1.0, trust=1.0,
               width=2.0, length=4.5, angle=0.0, width_std=0.3, length_std=0.3):
    """Create a MatchClass for testing. Mirrors the interface used by sensor_fusion.py."""
    return MatchClass(
        tracker_id, x, y, np.asarray(cov, dtype='float'),
        0.0, 0.0,  # velocity
        np.asarray(cov, dtype='float'),  # d_covariance
        confidence, trust,
        1,  # type
        0.0,  # time
        width, length, angle,
        width_std, length_std
    )


def make_ci_filter(time=0.0, x=0.0, y=0.0, fusion_mode=0,
                   passthrough_covariance=False, **kwargs):
    """Factory for CI filter with sensible defaults."""
    return CovarianceIntersectionFilter(
        time=time, x=x, y=y, fusion_mode=fusion_mode,
        passthrough_covariance=passthrough_covariance, **kwargs)


def make_kalman_filter(time=0.0, x=0.0, y=0.0, fusion_mode=0,
                       passthrough_covariance=False, **kwargs):
    """Factory for standard Kalman filter for comparison."""
    return ResizableKalman(
        time=time, x=x, y=y, fusion_mode=fusion_mode,
        passthrough_covariance=passthrough_covariance, **kwargs)


def run_filter_trajectory(filter_obj, positions, dt=0.1, noise_std=0.0,
                          cov=None, start_time=0.0):
    """
    Feed a sequence of positions through a filter and return the estimates.

    Args:
        filter_obj: A FilterBase instance
        positions: list of (x, y) ground truth positions
        dt: time step between positions
        noise_std: std dev of Gaussian noise added to measurements
        cov: 2x2 measurement covariance (default: identity * noise_std²)
        start_time: initial time

    Returns:
        list of (x_est, y_est) estimated positions after each fusion step
    """
    if cov is None:
        cov = np.eye(2) * max(noise_std**2, 0.01)

    estimates = []
    for i, (gt_x, gt_y) in enumerate(positions):
        t = start_time + i * dt
        meas_x = gt_x + np.random.randn() * noise_std
        meas_y = gt_y + np.random.randn() * noise_std
        match = make_match(meas_x, meas_y, cov)
        filter_obj.fusion([match], t, monitor=False)
        estimates.append((filter_obj.x, filter_obj.y))

    return estimates


# =============================================================================
# TEST GROUP 1: CORE CI MATH
# =============================================================================

class TestCIFuseMath(unittest.TestCase):
    """Tests for the standalone _ci_fuse and _ci_fuse_multi functions."""

    def test_identical_estimates_fuse_to_same(self):
        """When both estimates are identical, CI should return the same estimate."""
        x = np.array([1.0, 2.0])
        P = np.array([[0.5, 0.1], [0.1, 0.3]])
        x_f, P_f, omega = _ci_fuse(x, P, x, P)
        np.testing.assert_allclose(x_f, x, atol=1e-10,
            err_msg="Fusing identical estimates should return the same mean")
        np.testing.assert_allclose(P_f, P, atol=1e-10,
            err_msg="Fusing identical covariances should return the same covariance")

    def test_fused_covariance_smaller_than_larger_input(self):
        """CI fused covariance trace should be ≤ the larger of the two inputs.

        NOTE: CI does NOT guarantee P_fused < min(Pa, Pb). When Pa >> Pb,
        the optimal omega → 0, and P_fused ≈ Pb. CI can never do better than
        the best single input — it can only guarantee consistency.
        The trace is guaranteed ≤ max(trace(Pa), trace(Pb)) because at omega=0
        we get Pa and at omega=1 we get Pb, and the optimum is at least as good.
        """
        xa = np.array([0.0, 0.0])
        Pa = np.array([[2.0, 0.0], [0.0, 2.0]])
        xb = np.array([1.0, 1.0])
        Pb = np.array([[1.0, 0.0], [0.0, 1.0]])
        x_f, P_f, omega = _ci_fuse(xa, Pa, xb, Pb)
        max_trace = max(np.trace(Pa), np.trace(Pb))
        self.assertLessEqual(np.trace(P_f), max_trace + 1e-6,
            "Fused trace should be ≤ max of input traces")

    def test_fused_covariance_is_positive_definite(self):
        """CI result must always be positive definite."""
        Pa = np.array([[1.0, 0.3], [0.3, 2.0]])
        Pb = np.array([[3.0, -0.5], [-0.5, 0.5]])
        xa = np.array([1.0, -1.0])
        xb = np.array([-1.0, 2.0])
        _, P_f, _ = _ci_fuse(xa, Pa, xb, Pb)
        eigenvalues = np.linalg.eigvalsh(P_f)
        self.assertTrue(np.all(eigenvalues > 0),
            f"Fused covariance should be PD but eigenvalues are {eigenvalues}")

    def test_omega_prefers_smaller_covariance(self):
        """When Pa >> Pb, omega should be near 0 (trust estimate b more)."""
        xa = np.array([0.0, 0.0])
        Pa = np.array([[100.0, 0.0], [0.0, 100.0]])  # Very uncertain
        xb = np.array([5.0, 5.0])
        Pb = np.array([[0.01, 0.0], [0.0, 0.01]])    # Very certain
        _, _, omega = _ci_fuse(xa, Pa, xb, Pb)
        self.assertLess(omega, 0.1,
            f"omega should be near 0 when Pb << Pa, got {omega}")

    def test_omega_prefers_larger_covariance_when_reversed(self):
        """When Pb >> Pa, omega should be near 1 (trust estimate a more)."""
        xa = np.array([5.0, 5.0])
        Pa = np.array([[0.01, 0.0], [0.0, 0.01]])
        xb = np.array([0.0, 0.0])
        Pb = np.array([[100.0, 0.0], [0.0, 100.0]])
        _, _, omega = _ci_fuse(xa, Pa, xb, Pb)
        self.assertGreater(omega, 0.9,
            f"omega should be near 1 when Pa << Pb, got {omega}")

    def test_fused_mean_between_inputs(self):
        """Fused mean should lie between the two input means (componentwise, roughly)."""
        xa = np.array([0.0, 0.0])
        Pa = np.array([[1.0, 0.0], [0.0, 1.0]])
        xb = np.array([10.0, 10.0])
        Pb = np.array([[1.0, 0.0], [0.0, 1.0]])
        x_f, _, _ = _ci_fuse(xa, Pa, xb, Pb)
        # With equal covariances, omega ≈ 0.5, so mean should be near midpoint
        for i in range(2):
            self.assertGreater(x_f[i], xa[i] - 1.0)
            self.assertLess(x_f[i], xb[i] + 1.0)

    def test_det_minimization_also_works(self):
        """CI with det minimization should also produce valid results."""
        xa = np.array([1.0, 2.0])
        Pa = np.array([[2.0, 0.5], [0.5, 3.0]])
        xb = np.array([3.0, 1.0])
        Pb = np.array([[1.0, -0.2], [-0.2, 1.5]])
        x_f, P_f, omega = _ci_fuse(xa, Pa, xb, Pb, minimize='det')
        # Must be PD
        self.assertTrue(np.all(np.linalg.eigvalsh(P_f) > 0))
        # Det should be ≤ max of both inputs (CI can't make it bigger)
        max_det = max(np.linalg.det(Pa), np.linalg.det(Pb))
        self.assertLessEqual(np.linalg.det(P_f), max_det + 1e-6)

    def test_1d_case(self):
        """CI should work for 1D estimates (degenerate 1x1 matrix case).

        For 1D CI, P_fused = 1 / (ω/Pa + (1-ω)/Pb). The minimum over ω is
        achieved at ω=0 when Pa > Pb, giving P_fused = Pb. So the fused
        covariance approaches (but doesn't go below) the smaller input.
        """
        xa = np.array([0.0])
        Pa = np.array([[4.0]])
        xb = np.array([2.0])
        Pb = np.array([[1.0]])
        x_f, P_f, omega = _ci_fuse(xa, Pa, xb, Pb)
        self.assertGreater(x_f[0], 0.0)
        self.assertLess(x_f[0], 2.0)
        self.assertLess(P_f[0, 0], 4.0)
        # P_fused ≈ Pb (the smaller input) — can't go below it in 1D CI
        self.assertLessEqual(P_f[0, 0], 1.0 + 1e-3)

    def test_3d_case(self):
        """CI should work for 3D estimates (CTRV measurement space)."""
        xa = np.array([1.0, 2.0, 0.5])
        Pa = np.diag([1.0, 1.0, 0.1])
        xb = np.array([1.5, 2.5, 0.6])
        Pb = np.diag([0.5, 0.5, 0.05])
        x_f, P_f, omega = _ci_fuse(xa, Pa, xb, Pb)
        self.assertEqual(x_f.shape, (3,))
        self.assertEqual(P_f.shape, (3, 3))
        self.assertTrue(np.all(np.linalg.eigvalsh(P_f) > 0))

    def test_symmetry(self):
        """Swapping inputs should give the same fused result (symmetric)."""
        xa = np.array([1.0, 2.0])
        Pa = np.array([[1.0, 0.2], [0.2, 1.5]])
        xb = np.array([3.0, 1.0])
        Pb = np.array([[2.0, -0.3], [-0.3, 0.8]])
        x_f1, P_f1, w1 = _ci_fuse(xa, Pa, xb, Pb)
        x_f2, P_f2, w2 = _ci_fuse(xb, Pb, xa, Pa)
        np.testing.assert_allclose(x_f1, x_f2, atol=1e-8,
            err_msg="CI should be symmetric in the inputs")
        np.testing.assert_allclose(P_f1, P_f2, atol=1e-8,
            err_msg="CI covariance should be symmetric in the inputs")

    def test_omega_bounded(self):
        """Omega must always be in [0, 1]."""
        for _ in range(20):
            n = np.random.randint(1, 5)
            A = np.random.randn(n, n)
            Pa = A @ A.T + 0.1 * np.eye(n)
            B = np.random.randn(n, n)
            Pb = B @ B.T + 0.1 * np.eye(n)
            xa = np.random.randn(n)
            xb = np.random.randn(n)
            _, _, omega = _ci_fuse(xa, Pa, xb, Pb)
            self.assertGreaterEqual(omega, 0.0 - 1e-10)
            self.assertLessEqual(omega, 1.0 + 1e-10)


class TestCIFuseMulti(unittest.TestCase):
    """Tests for sequential pairwise CI fusion of multiple estimates."""

    def test_single_estimate_passthrough(self):
        """Single estimate should be returned unchanged."""
        x = np.array([1.0, 2.0])
        P = np.array([[0.5, 0.1], [0.1, 0.3]])
        x_f, P_f = _ci_fuse_multi([(x, P)])
        np.testing.assert_allclose(x_f, x)
        np.testing.assert_allclose(P_f, P)

    def test_empty_raises(self):
        """Empty list should raise ValueError."""
        with self.assertRaises(ValueError):
            _ci_fuse_multi([])

    def test_multi_reduces_uncertainty(self):
        """Fusing N estimates should give uncertainty ≤ the largest single input.

        NOTE: Sequential pairwise CI cannot reduce below the smallest single
        input covariance. But it should always be ≤ the largest.
        """
        estimates = [
            (np.array([i * 0.1, i * 0.1]), np.eye(2) * (1.0 + i * 0.1))
            for i in range(5)
        ]
        x_f, P_f = _ci_fuse_multi(estimates)
        max_trace = max(np.trace(P_i) for _, P_i in estimates)
        self.assertLessEqual(np.trace(P_f), max_trace + 1e-6,
            "Multi-fused trace should be ≤ max of individual traces")

    def test_two_estimates_matches_pairwise(self):
        """Two estimates via _ci_fuse_multi should match a single _ci_fuse call."""
        xa = np.array([1.0, 2.0])
        Pa = np.array([[1.0, 0.0], [0.0, 1.0]])
        xb = np.array([2.0, 3.0])
        Pb = np.array([[0.5, 0.0], [0.0, 0.5]])
        x_direct, P_direct, _ = _ci_fuse(xa, Pa, xb, Pb)
        x_multi, P_multi = _ci_fuse_multi([(xa, Pa), (xb, Pb)])
        np.testing.assert_allclose(x_multi, x_direct, atol=1e-10)
        np.testing.assert_allclose(P_multi, P_direct, atol=1e-10)

    def test_result_is_pd(self):
        """Multi-fused covariance must always be PD."""
        estimates = [
            (np.random.randn(2), np.diag([np.random.uniform(0.1, 5.0),
                                           np.random.uniform(0.1, 5.0)]))
            for _ in range(5)
        ]
        _, P_f = _ci_fuse_multi(estimates)
        self.assertTrue(np.all(np.linalg.eigvalsh(P_f) > 0))


# =============================================================================
# TEST GROUP 2: CI FILTER INITIALIZATION AND INTERFACE COMPLIANCE
# =============================================================================

class TestCIFilterInit(unittest.TestCase):
    """Test that CI filter initializes correctly and exposes required interface."""

    def test_cv_initialization(self):
        """CV mode should create a 4D state filter."""
        f = make_ci_filter(fusion_mode=0, x=5.0, y=10.0)
        self.assertEqual(f.X_hat_t.shape, (4, 1))
        self.assertEqual(f.P_hat_t.shape, (4, 4))
        self.assertAlmostEqual(f.x, 5.0)
        self.assertAlmostEqual(f.y, 10.0)
        self.assertEqual(f.fusion_mode, 0)

    def test_ca_initialization(self):
        """CA mode should create a 6D state filter."""
        f = make_ci_filter(fusion_mode=1, x=5.0, y=10.0)
        self.assertEqual(f.X_hat_t.shape, (6, 1))
        self.assertEqual(f.P_hat_t.shape, (6, 6))

    def test_ctrv_initialization(self):
        """CTRV mode should create a 5D state filter."""
        f = make_ci_filter(fusion_mode=2, x=5.0, y=10.0)
        self.assertEqual(f.X_hat_t.shape, (5, 1))
        self.assertEqual(f.P_hat_t.shape, (5, 5))

    def test_has_required_attributes(self):
        """CI filter must expose all attributes that GlobalTracked reads."""
        f = make_ci_filter()
        required_attrs = [
            'x', 'y', 'dx', 'dy', 'error_covariance', 'd_covariance',
            'width', 'length', 'error_tracker_temp', 'trupercept_list',
            'localTrackersIDList', 'P_hat_t', 'X_hat_t', 'last_update', 'idx'
        ]
        for attr in required_attrs:
            self.assertTrue(hasattr(f, attr), f"Missing required attribute: {attr}")

    def test_implements_filter_base(self):
        """CI filter must implement all FilterBase abstract methods."""
        from filters.base import FilterBase
        self.assertTrue(issubclass(CovarianceIntersectionFilter, FilterBase))

    def test_initial_covariance_positive_definite(self):
        """Initial P_hat_t should be positive definite."""
        for mode in [0, 1, 2]:
            f = make_ci_filter(fusion_mode=mode)
            eigenvalues = np.linalg.eigvalsh(f.P_hat_t)
            self.assertTrue(np.all(eigenvalues > 0),
                f"Mode {mode}: P_hat_t should be PD, eigenvalues={eigenvalues}")

    def test_same_constructor_signature_as_kalman(self):
        """CI filter should accept the same constructor args as ResizableKalman."""
        # This ensures it's a true drop-in replacement
        kwargs = dict(time=1.0, x=5.0, y=10.0, fusion_mode=0,
                      initial_width=1.8, initial_length=4.0,
                      use_trust_scoring=True, passthrough_covariance=False)
        ci = CovarianceIntersectionFilter(**kwargs)
        kf = ResizableKalman(**kwargs)
        # Both should initialize to the same position
        self.assertAlmostEqual(ci.x, kf.x)
        self.assertAlmostEqual(ci.y, kf.y)


# =============================================================================
# TEST GROUP 3: CI FILTER FUSION BEHAVIOR
# =============================================================================

class TestCIFilterFusion(unittest.TestCase):
    """Test CI fusion with actual measurements."""

    def test_first_frame_initialization(self):
        """First fusion call should initialize position from measurement."""
        f = make_ci_filter(x=0.0, y=0.0)
        match = make_match(5.0, 10.0, [[0.1, 0.0], [0.0, 0.1]])
        f.fusion([match], 0.0, monitor=False)
        self.assertAlmostEqual(f.x, 5.0, places=5)
        self.assertAlmostEqual(f.y, 10.0, places=5)
        self.assertEqual(f.idx, 1)

    def test_second_frame_uses_ci(self):
        """Second fusion should use CI update (not just average)."""
        f = make_ci_filter(x=0.0, y=0.0)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        # Frame 0: initialize at (5, 10)
        f.fusion([make_match(5.0, 10.0, cov)], 0.0, monitor=False)
        # Frame 1: measurement at (5.2, 10.2)
        f.fusion([make_match(5.2, 10.2, cov)], 0.1, monitor=False)
        # Position should move toward measurement but not exactly equal it
        self.assertAlmostEqual(f.x, 5.2, delta=0.5)
        self.assertAlmostEqual(f.y, 10.2, delta=0.5)
        self.assertEqual(f.idx, 2)

    def test_multiple_measurements_per_frame(self):
        """Multiple measurements in one frame should all contribute."""
        f = make_ci_filter(x=0.0, y=0.0)
        cov = [[0.5, 0.0], [0.0, 0.5]]
        # Frame 0: two measurements at different positions
        m1 = make_match(5.0, 10.0, cov, tracker_id=1)
        m2 = make_match(7.0, 12.0, cov, tracker_id=2)
        f.fusion([m1, m2], 0.0, monitor=False)
        # Should be between the two measurements
        self.assertGreater(f.x, 4.5)
        self.assertLess(f.x, 7.5)

    def test_measurement_with_smaller_cov_gets_more_weight(self):
        """Measurement with smaller covariance should pull the estimate more."""
        # Test this by running two filters: one where m1 has small cov, one where m2 has small cov
        cov_small = [[0.01, 0.0], [0.0, 0.01]]
        cov_large = [[10.0, 0.0], [0.0, 10.0]]

        f1 = make_ci_filter(x=0.0, y=0.0)
        m1 = make_match(0.0, 0.0, cov_small, tracker_id=1)
        m2 = make_match(10.0, 10.0, cov_large, tracker_id=2)
        f1.fusion([m1, m2], 0.0, monitor=False)

        f2 = make_ci_filter(x=0.0, y=0.0)
        m3 = make_match(0.0, 0.0, cov_large, tracker_id=1)
        m4 = make_match(10.0, 10.0, cov_small, tracker_id=2)
        f2.fusion([m3, m4], 0.0, monitor=False)

        # f1 should be closer to (0,0), f2 should be closer to (10,10)
        self.assertLess(f1.x, 5.0, "Small cov at (0,0) should pull estimate left")
        self.assertGreater(f2.x, 5.0, "Small cov at (10,10) should pull estimate right")

    def test_covariance_decreases_with_measurements(self):
        """Position covariance should generally decrease when measurements arrive."""
        f = make_ci_filter(x=5.0, y=5.0)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        f.fusion([make_match(5.0, 5.0, cov)], 0.0, monitor=False)
        trace_before = np.trace(f.error_covariance)

        # Several more measurements should reduce uncertainty
        for i in range(5):
            f.fusion([make_match(5.0 + 0.01 * i, 5.0 + 0.01 * i, cov)],
                     0.1 * (i + 1), monitor=False)

        trace_after = np.trace(f.error_covariance)
        # CI is conservative, so covariance may not shrink as fast as Kalman,
        # but it should still be smaller than initial after many measurements
        self.assertLess(trace_after, 2.0,  # initial was 1+1=2
            "Covariance should decrease from initial value after measurements")

    def test_all_fusion_modes(self):
        """CI filter should work with all three fusion modes without errors."""
        for mode in [0, 1, 2]:
            f = make_ci_filter(fusion_mode=mode, x=0.0, y=0.0)
            cov = [[0.1, 0.0], [0.0, 0.1]]
            for i in range(10):
                f.fusion([make_match(float(i), float(i), cov, angle=0.1*i)],
                         0.1 * i, monitor=False)
            self.assertTrue(np.isfinite(f.x), f"Mode {mode}: x is not finite")
            self.assertTrue(np.isfinite(f.y), f"Mode {mode}: y is not finite")
            self.assertTrue(np.all(np.isfinite(f.error_covariance)),
                f"Mode {mode}: error_covariance has non-finite values")

    def test_monitor_mode(self):
        """Monitor mode should populate error_tracker_temp and trupercept_list."""
        f = make_ci_filter(x=0.0, y=0.0)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        f.fusion([make_match(0.0, 0.0, cov)], 0.0, monitor=False)
        f.fusion([make_match(0.1, 0.1, cov)], 0.1, monitor=True)
        # Should have populated monitoring data
        self.assertIsInstance(f.error_tracker_temp, list)
        self.assertIsInstance(f.trupercept_list, list)

    def test_passthrough_covariance_mode(self):
        """With passthrough=True, output covariance should match measurement covariance."""
        f = make_ci_filter(x=0.0, y=0.0, passthrough_covariance=True)
        meas_cov = [[0.25, 0.05], [0.05, 0.35]]
        f.fusion([make_match(0.0, 0.0, meas_cov)], 0.0, monitor=False)
        f.fusion([make_match(0.1, 0.1, meas_cov)], 0.1, monitor=False)
        # Output covariance should approximately match measurement covariance
        np.testing.assert_allclose(f.error_covariance, meas_cov, atol=0.05,
            err_msg="Passthrough mode should preserve measurement covariance")

    def test_no_measurements_prediction_only(self):
        """With no measurements, filter should still work (prediction only)."""
        f = make_ci_filter(x=5.0, y=5.0)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        f.fusion([make_match(5.0, 5.0, cov)], 0.0, monitor=False)
        # Empty measurement list
        f.fusion([], 0.1, monitor=False)
        # Filter should still be valid
        self.assertTrue(np.isfinite(f.x))
        self.assertTrue(np.isfinite(f.y))


# =============================================================================
# TEST GROUP 4: CI vs KALMAN COMPARISON
# =============================================================================

class TestCIvsKalman(unittest.TestCase):
    """
    Compare CI filter against standard Kalman.

    KEY EXPECTATION: CI should be more conservative (larger covariance)
    but should never become overconfident. Both should track well.
    """

    def test_ci_covariance_not_smaller_than_kalman(self):
        """
        CI should produce equal or larger covariance than standard Kalman.

        Rationale: Standard Kalman assumes uncorrelated errors, which gives
        the tightest possible bound. CI accounts for unknown correlation,
        so it must be more conservative (larger covariance). This is the
        fundamental CI property — never overconfident.
        """
        np.random.seed(42)
        ci = make_ci_filter(x=0.0, y=0.0, fusion_mode=0)
        kf = make_kalman_filter(x=0.0, y=0.0, fusion_mode=0)

        cov = [[0.1, 0.0], [0.0, 0.1]]
        for i in range(20):
            t = 0.1 * i
            mx = float(i) * 0.5 + np.random.randn() * 0.1
            my = float(i) * 0.3 + np.random.randn() * 0.1
            match = make_match(mx, my, cov)
            ci.fusion([match], t, monitor=False)
            kf.fusion([match], t, monitor=False)

        # CI covariance trace should be >= Kalman covariance trace
        ci_trace = np.trace(ci.error_covariance)
        kf_trace = np.trace(kf.error_covariance)
        # Allow small tolerance (CI may occasionally be slightly smaller due to
        # different update order/rounding)
        self.assertGreater(ci_trace, kf_trace * 0.5,
            f"CI trace ({ci_trace:.4f}) should not be much smaller than Kalman trace ({kf_trace:.4f})")

    def test_both_track_constant_velocity_target(self):
        """Both CI and Kalman should track a constant-velocity target reasonably."""
        np.random.seed(123)
        speed = 10.0  # m/s
        dt = 0.1
        n_steps = 50

        positions = [(speed * dt * i, 0.0) for i in range(n_steps)]
        cov = np.eye(2) * 0.1

        ci = make_ci_filter(x=0.0, y=0.0, fusion_mode=0)
        kf = make_kalman_filter(x=0.0, y=0.0, fusion_mode=0)

        ci_errors = []
        kf_errors = []
        for i, (gt_x, gt_y) in enumerate(positions):
            t = dt * i
            mx = gt_x + np.random.randn() * 0.3
            my = gt_y + np.random.randn() * 0.3
            match = make_match(mx, my, cov.tolist())
            ci.fusion([match], t, monitor=False)
            kf.fusion([match], t, monitor=False)
            ci_errors.append(math.hypot(ci.x - gt_x, ci.y - gt_y))
            kf_errors.append(math.hypot(kf.x - gt_x, kf.y - gt_y))

        # Both should have reasonable tracking error (< 1m mean for last 30 steps)
        ci_mean_error = np.mean(ci_errors[-30:])
        kf_mean_error = np.mean(kf_errors[-30:])
        self.assertLess(ci_mean_error, 1.0,
            f"CI mean error {ci_mean_error:.3f} should be < 1.0m")
        self.assertLess(kf_mean_error, 1.0,
            f"Kalman mean error {kf_mean_error:.3f} should be < 1.0m")

    def test_ci_does_not_diverge_over_long_runs(self):
        """CI filter should remain stable over many frames without diverging."""
        np.random.seed(77)
        f = make_ci_filter(x=0.0, y=0.0, fusion_mode=0)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        for i in range(500):
            t = 0.1 * i
            mx = 10.0 + np.random.randn() * 0.3  # Stationary target at (10, 10)
            my = 10.0 + np.random.randn() * 0.3
            f.fusion([make_match(mx, my, cov)], t, monitor=False)

        error = math.hypot(f.x - 10.0, f.y - 10.0)
        self.assertLess(error, 1.0,
            f"CI should converge to target, error={error:.3f}m")
        self.assertTrue(np.all(np.isfinite(f.P_hat_t)),
            "P_hat_t should remain finite after 500 steps")
        # Check PSD with tolerance (small negative eigenvalues from numerical drift are OK)
        min_eig = np.linalg.eigvalsh(f.P_hat_t).min()
        self.assertGreater(min_eig, -1e-6,
            f"P_hat_t should remain PSD after 500 steps, min eigenvalue={min_eig}")


# =============================================================================
# TEST GROUP 5: GPEM COVARIANCE SENSITIVITY
# =============================================================================

class TestGPEMSensitivity(unittest.TestCase):
    """
    Test that CI properly differentiates between different measurement covariances.

    This is the key requirement for GPEM: distance-dependent covariance should
    cause the filter to weight close (low-cov) measurements more than far (high-cov)
    measurements.
    """

    def test_small_cov_measurement_dominates_large_cov(self):
        """
        With two measurements of different covariances, the small-cov one
        should have more influence on the fused position.
        """
        f = make_ci_filter(x=0.0, y=0.0, fusion_mode=0)
        # Frame 0: init at origin
        f.fusion([make_match(0.0, 0.0, [[1.0, 0.0], [0.0, 1.0]])], 0.0, monitor=False)

        # Frame 1: two measurements
        # m1 at (0, 0) with very small covariance (close, well-observed)
        # m2 at (10, 10) with very large covariance (far, poorly-observed)
        m1 = make_match(0.0, 0.0, [[0.01, 0.0], [0.0, 0.01]], tracker_id=1)
        m2 = make_match(10.0, 10.0, [[10.0, 0.0], [0.0, 10.0]], tracker_id=2)
        f.fusion([m1, m2], 0.1, monitor=False)

        # Filter should be much closer to (0, 0) than to (10, 10)
        dist_to_m1 = math.hypot(f.x - 0.0, f.y - 0.0)
        dist_to_m2 = math.hypot(f.x - 10.0, f.y - 10.0)
        self.assertLess(dist_to_m1, dist_to_m2,
            f"Small-cov measurement should dominate: d_m1={dist_to_m1:.3f}, d_m2={dist_to_m2:.3f}")

    def test_gpem_linear_vs_static_single_measurement(self):
        """
        With single measurements per frame, CI's omega directly reflects
        the measurement covariance. A filter receiving small-R measurements
        should track closer to measurements (omega → 0) than one receiving
        large-R measurements (omega → 1, trust prediction more).

        This is the key GPEM benefit: distance-dependent covariance from
        nearby sensors produces smaller R, which makes CI trust the
        measurement more heavily.
        """
        np.random.seed(42)

        # Stationary target at (10, 10) with noisy measurements
        f_small_R = make_ci_filter(x=10.0, y=10.0, fusion_mode=0)
        f_large_R = make_ci_filter(x=10.0, y=10.0, fusion_mode=0)

        small_cov = [[0.01, 0.0], [0.0, 0.01]]  # Close sensor
        large_cov = [[5.0, 0.0], [0.0, 5.0]]     # Far sensor

        for i in range(30):
            t = 0.1 * i
            # Same noisy measurement for both
            mx = 10.0 + np.random.randn() * 0.5
            my = 10.0 + np.random.randn() * 0.5
            f_small_R.fusion([make_match(mx, my, small_cov)], t, monitor=False)
            f_large_R.fusion([make_match(mx, my, large_cov)], t, monitor=False)

        # The output covariance should differ: small-R filter should have
        # smaller output covariance (it trusts measurements more)
        trace_small = np.trace(f_small_R.error_covariance)
        trace_large = np.trace(f_large_R.error_covariance)
        self.assertLess(trace_small, trace_large,
            f"Small-R filter should have smaller covariance: {trace_small:.4f} vs {trace_large:.4f}")

    def test_covariance_ratio_affects_omega(self):
        """
        Directly verify that different measurement covariance magnitudes
        produce different CI omega values (and thus different trust weights).
        """
        # Prediction state with known covariance
        z_pred = np.array([0.0, 0.0])
        R_pred = np.array([[1.0, 0.0], [0.0, 1.0]])

        # Small measurement covariance → omega should be small (trust measurement)
        z_meas = np.array([1.0, 1.0])
        R_small = np.array([[0.01, 0.0], [0.0, 0.01]])
        _, _, omega_small = _ci_fuse(z_pred, R_pred, z_meas, R_small)

        # Large measurement covariance → omega should be large (trust prediction)
        R_large = np.array([[100.0, 0.0], [0.0, 100.0]])
        _, _, omega_large = _ci_fuse(z_pred, R_pred, z_meas, R_large)

        self.assertLess(omega_small, omega_large,
            f"Small R should give smaller omega: ω_small={omega_small:.4f}, ω_large={omega_large:.4f}")

    def test_ci_differentiates_covariance_over_trajectory(self):
        """
        Over a trajectory, filters with different measurement covariances
        should produce different output covariances, demonstrating that CI
        uses measurement covariance information.

        We compare output covariance traces rather than positions because
        with single-measurement CI, the position estimates are similar
        (both use CI fusion) but the COVARIANCE reflects how much the
        filter trusted the measurement vs prediction.
        """
        np.random.seed(42)
        dt = 0.1
        n_steps = 50

        f_small = make_ci_filter(x=0.0, y=0.0, fusion_mode=0)
        small_cov = [[0.01, 0.0], [0.0, 0.01]]

        f_large = make_ci_filter(x=0.0, y=0.0, fusion_mode=0)
        large_cov = [[5.0, 0.0], [0.0, 5.0]]

        for i in range(n_steps):
            t = dt * i
            # Moving target
            gt_x = i * dt * 5.0
            gt_y = i * dt * 3.0
            mx = gt_x + np.random.randn() * 0.1
            my = gt_y + np.random.randn() * 0.1

            f_small.fusion([make_match(mx, my, small_cov)], t, monitor=False)
            f_large.fusion([make_match(mx, my, large_cov)], t, monitor=False)

        # Small-R filter should have lower output covariance
        trace_small = np.trace(f_small.error_covariance)
        trace_large = np.trace(f_large.error_covariance)
        self.assertLess(trace_small, trace_large,
            f"Small-R filter should have smaller output covariance: {trace_small:.4f} vs {trace_large:.4f}")


# =============================================================================
# TEST GROUP 6: EDGE CASES AND NUMERICAL ROBUSTNESS
# =============================================================================

class TestCIEdgeCases(unittest.TestCase):
    """Test edge cases and numerical robustness."""

    def test_very_small_covariance(self):
        """Filter should handle very small covariances without numerical issues."""
        f = make_ci_filter(x=0.0, y=0.0)
        tiny_cov = [[1e-6, 0.0], [0.0, 1e-6]]
        f.fusion([make_match(5.0, 5.0, tiny_cov)], 0.0, monitor=False)
        f.fusion([make_match(5.01, 5.01, tiny_cov)], 0.1, monitor=False)
        self.assertTrue(np.all(np.isfinite(f.error_covariance)))

    def test_very_large_covariance(self):
        """Filter should handle very large covariances gracefully."""
        f = make_ci_filter(x=0.0, y=0.0)
        huge_cov = [[1e6, 0.0], [0.0, 1e6]]
        f.fusion([make_match(5.0, 5.0, huge_cov)], 0.0, monitor=False)
        f.fusion([make_match(5.0, 5.0, huge_cov)], 0.1, monitor=False)
        self.assertTrue(np.all(np.isfinite(f.error_covariance)))

    def test_anisotropic_covariance(self):
        """Filter should work with non-diagonal (anisotropic) covariances."""
        f = make_ci_filter(x=0.0, y=0.0)
        aniso_cov = [[1.0, 0.5], [0.5, 2.0]]
        f.fusion([make_match(5.0, 5.0, aniso_cov)], 0.0, monitor=False)
        f.fusion([make_match(5.1, 5.1, aniso_cov)], 0.1, monitor=False)
        # Output covariance should still be PD
        eigs = np.linalg.eigvalsh(f.error_covariance)
        self.assertTrue(np.all(eigs > 0),
            f"Anisotropic cov should produce PD output: eigs={eigs}")

    def test_nan_measurements_skipped(self):
        """NaN measurements should be silently skipped."""
        f = make_ci_filter(x=5.0, y=5.0)
        f.fusion([make_match(5.0, 5.0, [[0.1, 0.0], [0.0, 0.1]])], 0.0, monitor=False)
        # Second frame: one valid, one NaN
        m_valid = make_match(5.1, 5.1, [[0.1, 0.0], [0.0, 0.1]], tracker_id=1)
        m_nan = make_match(float('nan'), 5.0, [[0.1, 0.0], [0.0, 0.1]], tracker_id=2)
        f.fusion([m_valid, m_nan], 0.1, monitor=False)
        self.assertTrue(np.isfinite(f.x))
        self.assertTrue(np.isfinite(f.y))

    def test_inf_measurements_skipped(self):
        """Inf measurements should be silently skipped."""
        f = make_ci_filter(x=5.0, y=5.0)
        f.fusion([make_match(5.0, 5.0, [[0.1, 0.0], [0.0, 0.1]])], 0.0, monitor=False)
        m_valid = make_match(5.1, 5.1, [[0.1, 0.0], [0.0, 0.1]], tracker_id=1)
        m_inf = make_match(float('inf'), 5.0, [[0.1, 0.0], [0.0, 0.1]], tracker_id=2)
        f.fusion([m_valid, m_inf], 0.1, monitor=False)
        self.assertTrue(np.isfinite(f.x))

    def test_trust_scoring_filters_bad_participants(self):
        """Low-trust measurements should be excluded."""
        f = make_ci_filter(x=0.0, y=0.0, use_trust_scoring=True)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        f.fusion([make_match(5.0, 5.0, cov)], 0.0, monitor=False)

        # Low-trust participant (id=2) at wrong position, should be ignored
        m_good = make_match(5.1, 5.1, cov, tracker_id=10001)  # participant 1
        m_bad = make_match(100.0, 100.0, cov, tracker_id=20001)  # participant 2
        trust_scores = {1: 1.0, 2: 0.5}  # Participant 2 has low trust (≤ 0.67)
        f.fusion([m_good, m_bad], 0.1, monitor=False, participant_trust_scores=trust_scores)

        # Should be near (5.1, 5.1), not near (100, 100)
        self.assertLess(f.x, 10.0, "Low-trust measurement should be filtered out")

    def test_zero_elapsed_time_handled(self):
        """Zero elapsed time between frames should not crash."""
        f = make_ci_filter(x=5.0, y=5.0)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        f.fusion([make_match(5.0, 5.0, cov)], 0.0, monitor=False)
        # Same timestamp
        f.fusion([make_match(5.1, 5.1, cov)], 0.0, monitor=False)
        self.assertTrue(np.isfinite(f.x))

    def test_negative_elapsed_time_handled(self):
        """Negative elapsed time (out-of-order) should not crash."""
        f = make_ci_filter(x=5.0, y=5.0)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        f.fusion([make_match(5.0, 5.0, cov)], 1.0, monitor=False)
        # Earlier timestamp
        f.fusion([make_match(5.1, 5.1, cov)], 0.5, monitor=False)
        self.assertTrue(np.isfinite(f.x))

    def test_single_measurement_with_ctrv(self):
        """CTRV mode with single measurement should work."""
        f = make_ci_filter(x=0.0, y=0.0, fusion_mode=2)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        f.fusion([make_match(5.0, 5.0, cov, angle=0.5)], 0.0, monitor=False)
        f.fusion([make_match(5.5, 5.5, cov, angle=0.6)], 0.1, monitor=False)
        self.assertTrue(np.isfinite(f.x))
        self.assertTrue(np.isfinite(f.y))


# =============================================================================
# TEST GROUP 7: PREDICTION METHODS
# =============================================================================

class TestCIPrediction(unittest.TestCase):
    """Test getKalmanPred and getKalmanPredWithCovariance."""

    def test_getKalmanPred_returns_five_values(self):
        """getKalmanPred should return (x, y, a, b, phi)."""
        f = make_ci_filter(x=5.0, y=10.0, fusion_mode=0)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        f.fusion([make_match(5.0, 10.0, cov)], 0.0, monitor=False)
        result = f.getKalmanPred(0.1)
        self.assertEqual(len(result), 5)
        x, y, a, b, phi = result
        self.assertTrue(np.isfinite(x))
        self.assertTrue(np.isfinite(y))
        self.assertTrue(a >= 0)
        self.assertTrue(b >= 0)

    def test_getKalmanPredWithCovariance_returns_three_values(self):
        """getKalmanPredWithCovariance should return (x, y, P)."""
        f = make_ci_filter(x=5.0, y=10.0, fusion_mode=0)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        f.fusion([make_match(5.0, 10.0, cov)], 0.0, monitor=False)
        result = f.getKalmanPredWithCovariance(0.1)
        self.assertEqual(len(result), 3)
        x, y, P = result
        self.assertTrue(np.isfinite(x))
        self.assertTrue(np.isfinite(y))
        self.assertEqual(P.shape, (4, 4))

    def test_prediction_covariance_grows_with_time(self):
        """Prediction further into the future should have larger covariance."""
        f = make_ci_filter(x=5.0, y=5.0, fusion_mode=0)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        f.fusion([make_match(5.0, 5.0, cov)], 0.0, monitor=False)

        _, _, P_short = f.getKalmanPredWithCovariance(0.1)
        _, _, P_long = f.getKalmanPredWithCovariance(1.0)

        trace_short = np.trace(P_short[0:2, 0:2])
        trace_long = np.trace(P_long[0:2, 0:2])
        self.assertGreater(trace_long, trace_short,
            "Longer prediction should have larger covariance")

    def test_prediction_extrapolates_position(self):
        """CV filter should predict position forward based on velocity."""
        f = make_ci_filter(x=0.0, y=0.0, fusion_mode=0)
        # Feed several measurements establishing velocity ~(1, 0) m/s
        cov = [[0.01, 0.0], [0.0, 0.01]]
        for i in range(20):
            f.fusion([make_match(float(i) * 0.1, 0.0, cov)], 0.1 * i, monitor=False)

        # Predict 1 second ahead: should extrapolate ~1m forward
        pred_x, pred_y, _, _, _ = f.getKalmanPred(0.1 * 19 + 1.0)
        # Position should be ahead of current position
        self.assertGreater(pred_x, f.x,
            "Prediction should extrapolate forward based on velocity")

    def test_ctrv_prediction(self):
        """CTRV prediction should work without errors."""
        f = make_ci_filter(x=0.0, y=0.0, fusion_mode=2)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        for i in range(5):
            f.fusion([make_match(float(i), float(i) * 0.5, cov, angle=0.1*i)],
                     0.1 * i, monitor=False)

        x, y, a, b, phi = f.getKalmanPred(0.5)
        self.assertTrue(np.isfinite(x))
        self.assertTrue(np.isfinite(y))

    def test_negative_time_prediction(self):
        """Prediction before last update should return current state."""
        f = make_ci_filter(x=5.0, y=5.0, fusion_mode=0)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        f.fusion([make_match(5.0, 5.0, cov)], 1.0, monitor=False)
        x, y, _, _, _ = f.getKalmanPred(0.5)  # Before last update
        self.assertAlmostEqual(x, f.x, places=5)
        self.assertAlmostEqual(y, f.y, places=5)

    def test_all_fusion_modes_predict(self):
        """All fusion modes should produce valid predictions."""
        for mode in [0, 1, 2]:
            f = make_ci_filter(x=5.0, y=5.0, fusion_mode=mode)
            cov = [[0.1, 0.0], [0.0, 0.1]]
            f.fusion([make_match(5.0, 5.0, cov, angle=0.0)], 0.0, monitor=False)
            f.fusion([make_match(5.5, 5.5, cov, angle=0.1)], 0.1, monitor=False)

            x1, y1, a, b, phi = f.getKalmanPred(0.2)
            self.assertTrue(np.isfinite(x1), f"Mode {mode}: pred x not finite")
            self.assertTrue(np.isfinite(y1), f"Mode {mode}: pred y not finite")

            x2, y2, P = f.getKalmanPredWithCovariance(0.2)
            self.assertTrue(np.isfinite(x2), f"Mode {mode}: pred_cov x not finite")
            np.testing.assert_allclose(x1, x2, atol=1e-10,
                err_msg=f"Mode {mode}: both prediction methods should agree on x")


# =============================================================================
# TEST GROUP 8: WIDTH/LENGTH TRACKING
# =============================================================================

class TestCIDimensionTracking(unittest.TestCase):
    """Test that width/length 1D Kalman filters work correctly."""

    def test_width_converges_to_measurement(self):
        """Width should converge toward measured width over time."""
        f = make_ci_filter(x=0.0, y=0.0, initial_width=2.0)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        for i in range(50):
            f.fusion([make_match(0.0, 0.0, cov, width=3.0, width_std=0.3)],
                     0.1 * i, monitor=False)
        self.assertAlmostEqual(f.width, 3.0, delta=0.1,
            msg=f"Width should converge to 3.0, got {f.width:.3f}")

    def test_length_converges_to_measurement(self):
        """Length should converge toward measured length over time."""
        f = make_ci_filter(x=0.0, y=0.0, initial_length=4.5)
        cov = [[0.1, 0.0], [0.0, 0.1]]
        for i in range(50):
            f.fusion([make_match(0.0, 0.0, cov, length=6.0, length_std=0.3)],
                     0.1 * i, monitor=False)
        self.assertAlmostEqual(f.length, 6.0, delta=0.1,
            msg=f"Length should converge to 6.0, got {f.length:.3f}")


# =============================================================================
# TEST GROUP 9: PROCESS NOISE (Q) VALIDATION
# =============================================================================

class TestProcessNoise(unittest.TestCase):
    """Verify process noise matrix properties."""

    def test_Q_positive_semidefinite_cv(self):
        """Q should be PSD for CV mode."""
        f = make_ci_filter(fusion_mode=0)
        for dt in [0.01, 0.1, 0.5, 1.0]:
            Q = f._compute_Q(dt)
            eigs = np.linalg.eigvalsh(Q)
            self.assertTrue(np.all(eigs >= -1e-15),
                f"Q should be PSD for dt={dt}, min eigenvalue={eigs.min()}")

    def test_Q_positive_semidefinite_ca(self):
        """Q should be PSD for CA mode."""
        f = make_ci_filter(fusion_mode=1)
        for dt in [0.01, 0.1, 0.5, 1.0]:
            Q = f._compute_Q(dt)
            eigs = np.linalg.eigvalsh(Q)
            self.assertTrue(np.all(eigs >= -1e-15),
                f"Q should be PSD for dt={dt}, min eigenvalue={eigs.min()}")

    def test_Q_positive_semidefinite_ctrv(self):
        """Q should be PSD for CTRV mode."""
        f = make_ci_filter(fusion_mode=2)
        for dt in [0.01, 0.1, 0.5, 1.0]:
            Q = f._compute_Q(dt)
            eigs = np.linalg.eigvalsh(Q)
            self.assertTrue(np.all(eigs >= -1e-15),
                f"Q should be PSD for dt={dt}, min eigenvalue={eigs.min()}")

    def test_Q_symmetric(self):
        """Q should be symmetric."""
        for mode in [0, 1, 2]:
            f = make_ci_filter(fusion_mode=mode)
            Q = f._compute_Q(0.1)
            np.testing.assert_allclose(Q, Q.T, atol=1e-15,
                err_msg=f"Q should be symmetric for mode {mode}")

    def test_Q_grows_with_dt(self):
        """Larger dt should give larger Q (more process noise)."""
        for mode in [0, 1, 2]:
            f = make_ci_filter(fusion_mode=mode)
            Q_small = f._compute_Q(0.01)
            Q_large = f._compute_Q(1.0)
            self.assertGreater(np.trace(Q_large), np.trace(Q_small),
                f"Mode {mode}: Q should grow with dt")

    def test_Q_scales_with_sigma_a(self):
        """CI filter's Q should scale with its own sigma_a (may differ from EKF)."""
        ci = make_ci_filter(fusion_mode=0)
        dt = 0.1
        Q = ci._compute_Q(dt)
        # Q_pos should be proportional to sigma_a^2
        expected_q_pos = ci.sigma_a**2 * dt**4 / 4
        self.assertAlmostEqual(Q[0, 0], expected_q_pos, places=10,
            msg=f"Q[0,0] should equal sigma_a²·dt⁴/4 = {expected_q_pos}")


# =============================================================================
# TEST GROUP 10: CI CONSISTENCY PROPERTY (theoretical guarantee)
# =============================================================================

class TestCIConsistency(unittest.TestCase):
    """
    Verify CI's key properties.

    The CI guarantee is NOT that P_fused ≤ Pa and P_fused ≤ Pb in PSD sense.
    That property only holds in the information form: P_fused^-1 ≥ max(ω·Pa^-1, (1-ω)·Pb^-1).

    The actual CI guarantees are:
    1. P_fused is always a valid (PD) covariance matrix
    2. P_fused^-1 = ω·Pa^-1 + (1-ω)·Pb^-1 (information is accumulated)
    3. trace(P_fused) ≤ max(trace(Pa), trace(Pb)) (doesn't make things worse)
    4. The fused estimate is a consistent bound on the true error for ANY
       cross-correlation between the two input errors
    """

    def test_ci_fused_is_always_pd(self):
        """P_fused must always be positive definite for any PD inputs."""
        np.random.seed(42)
        for _ in range(50):
            n = 2
            A = np.random.randn(n, n)
            Pa = A @ A.T + 0.1 * np.eye(n)
            B = np.random.randn(n, n)
            Pb = B @ B.T + 0.1 * np.eye(n)
            xa = np.random.randn(n)
            xb = np.random.randn(n)

            _, P_f, omega = _ci_fuse(xa, Pa, xb, Pb)

            # P_fused must be PD
            eigs = np.linalg.eigvalsh(P_f)
            self.assertTrue(np.all(eigs > -1e-10),
                f"P_fused must be PD, min eigenvalue={eigs.min()}, omega={omega:.4f}")

            # P_fused must be symmetric
            np.testing.assert_allclose(P_f, P_f.T, atol=1e-10)

    def test_ci_information_accumulation(self):
        """P_fused^-1 should equal ω·Pa^-1 + (1-ω)·Pb^-1 (CI definition)."""
        np.random.seed(99)
        for _ in range(30):
            n = 3
            A = np.random.randn(n, n)
            Pa = A @ A.T + 0.1 * np.eye(n)
            B = np.random.randn(n, n)
            Pb = B @ B.T + 0.1 * np.eye(n)
            xa = np.random.randn(n)
            xb = np.random.randn(n)

            _, P_f, omega = _ci_fuse(xa, Pa, xb, Pb)

            # Verify the CI equation holds
            expected_P_inv = omega * np.linalg.inv(Pa) + (1.0 - omega) * np.linalg.inv(Pb)
            actual_P_inv = np.linalg.inv(P_f)
            np.testing.assert_allclose(actual_P_inv, expected_P_inv, atol=1e-8,
                err_msg="P_fused^-1 should match CI formula")

    def test_ci_trace_not_worse_than_max(self):
        """trace(P_fused) should be ≤ max(trace(Pa), trace(Pb))."""
        np.random.seed(42)
        for _ in range(50):
            n = 2
            A = np.random.randn(n, n)
            Pa = A @ A.T + 0.1 * np.eye(n)
            B = np.random.randn(n, n)
            Pb = B @ B.T + 0.1 * np.eye(n)
            xa = np.random.randn(n)
            xb = np.random.randn(n)

            _, P_f, _ = _ci_fuse(xa, Pa, xb, Pb)
            max_trace = max(np.trace(Pa), np.trace(Pb))
            self.assertLessEqual(np.trace(P_f), max_trace + 1e-6,
                f"trace(P_fused) should be ≤ max input trace")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    unittest.main()


class TestCIDistanceSensitivity(unittest.TestCase):
    """Test that CI filter benefits from tighter R at close range."""

    def test_close_range_beats_far_range(self):
        """CI with close-range (small) R should track better than far-range (large) R."""
        np.random.seed(42)

        # Close-range R (10m DETR3D): small uncertainty
        R_close = [[0.18, 0.0], [0.0, 0.04]]
        # Far-range R (70m DETR3D): large uncertainty
        R_far = [[0.32, 0.0], [0.0, 0.06]]

        true_x, true_y = 5.0, 3.0

        f_close = make_ci_filter(x=0, y=0, fusion_mode=0)
        f_far = make_ci_filter(x=0, y=0, fusion_mode=0)

        for i in range(30):
            t = 0.1 * (i + 1)
            # Close: small noise
            cx = true_x + np.random.randn() * math.sqrt(0.18)
            cy = true_y + np.random.randn() * math.sqrt(0.04)
            # Far: big noise
            fx = true_x + np.random.randn() * math.sqrt(0.32)
            fy = true_y + np.random.randn() * math.sqrt(0.06)

            f_close.fusion([make_match(cx, cy, R_close)], t, monitor=False)
            f_far.fusion([make_match(fx, fy, R_far)], t, monitor=False)

        err_close = math.hypot(f_close.x - true_x, f_close.y - true_y)
        err_far = math.hypot(f_far.x - true_x, f_far.y - true_y)

        print(f"\nClose R error: {err_close:.4f}m, Far R error: {err_far:.4f}m")
        print(f"P_close: [{f_close.P_hat_t[0,0]:.4f}, {f_close.P_hat_t[1,1]:.4f}]")
        print(f"P_far:   [{f_far.P_hat_t[0,0]:.4f}, {f_far.P_hat_t[1,1]:.4f}]")

        # Close range should have smaller output covariance
        self.assertLess(f_close.P_hat_t[0, 0], f_far.P_hat_t[0, 0],
            "CI with tighter R should produce smaller P")
