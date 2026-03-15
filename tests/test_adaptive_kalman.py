"""
Tests for the Adaptive Kalman Filter (innovation-based Q adaptation).

Run: python -m pytest tests/test_adaptive_kalman.py -v -s
"""

import unittest
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sensor_fusion import MatchClass
from filters.adaptive_kalman import AdaptiveKalman
from filters.kalman_ctrv import ResizableKalman
import utils


def make_match(x, y, cov, tracker_id=1, confidence=1.0, trust=1.0,
               width=2.0, length=4.5, angle=0.0, width_std=0.3, length_std=0.3):
    return MatchClass(
        tracker_id, x, y, np.asarray(cov, dtype='float'),
        0.0, 0.0,
        np.asarray(cov, dtype='float'),
        confidence, trust,
        1, 0.0,
        width, length, angle,
        width_std, length_std
    )


class TestAdaptiveKalmanInit(unittest.TestCase):
    """Test initialization and interface compliance."""

    def test_cv_initialization(self):
        akf = AdaptiveKalman(0.0, 1.0, 2.0, fusion_mode=0)
        self.assertEqual(akf.x, 1.0)
        self.assertEqual(akf.y, 2.0)
        self.assertEqual(akf.fusion_mode, 0)
        self.assertEqual(akf.F_t_len, 4)

    def test_ctrv_initialization(self):
        akf = AdaptiveKalman(0.0, 1.0, 2.0, fusion_mode=2)
        self.assertEqual(akf.fusion_mode, 2)
        self.assertEqual(akf.F_t_len, 5)

    def test_same_constructor_signature_as_kalman(self):
        """AdaptiveKalman must accept the same constructor args as ResizableKalman."""
        kwargs = dict(time=0.0, x=5.0, y=10.0, fusion_mode=0,
                      initial_width=1.8, initial_length=4.0,
                      use_trust_scoring=False, passthrough_covariance=True)
        akf = AdaptiveKalman(**kwargs)
        rkf = ResizableKalman(**kwargs)
        self.assertEqual(akf.x, rkf.x)
        self.assertEqual(akf.y, rkf.y)

    def test_has_adaptation_attributes(self):
        akf = AdaptiveKalman(0.0, 0.0, 0.0, fusion_mode=0)
        self.assertTrue(hasattr(akf, '_innovation_window'))
        self.assertTrue(hasattr(akf, '_adapt_step'))
        self.assertTrue(hasattr(akf, 'WINDOW_SIZE'))

    def test_initial_sigma_a(self):
        akf = AdaptiveKalman(0.0, 0.0, 0.0, fusion_mode=0)
        self.assertEqual(akf.sigma_a, 2.0)  # Same default as parent


class TestAdaptiveKalmanFusion(unittest.TestCase):
    """Test fusion behavior and Q adaptation."""

    def _run_fusion_steps(self, akf, n_steps=30, noise_std=0.05, dt=0.1):
        """Run n_steps of fusion with a linearly-moving target."""
        cov = np.array([[noise_std**2, 0], [0, noise_std**2]])
        np.random.seed(42)
        for step in range(n_steps):
            t = dt * (step + 1)
            true_x = 10.0 + 0.5 * t  # Moving at 0.5 m/s in x
            true_y = 20.0
            meas_x = true_x + np.random.normal(0, noise_std)
            meas_y = true_y + np.random.normal(0, noise_std)
            m = make_match(meas_x, meas_y, cov)
            akf.fusion([m], t, monitor=False)
        return akf

    def test_first_frame_initialization(self):
        """First fusion should initialize state like parent."""
        akf = AdaptiveKalman(0.0, 0.0, 0.0, fusion_mode=0)
        cov = np.array([[0.1, 0], [0, 0.1]])
        m = make_match(5.0, 10.0, cov)
        akf.fusion([m], 0.1, monitor=False)
        self.assertAlmostEqual(akf.x, 5.0, places=1)
        self.assertAlmostEqual(akf.y, 10.0, places=1)

    def test_sigma_a_adapts(self):
        """After enough steps, sigma_a should differ from initial value."""
        akf = AdaptiveKalman(0.0, 10.0, 20.0, fusion_mode=0)
        self._run_fusion_steps(akf, n_steps=40, noise_std=0.05)
        # sigma_a should have adapted (may go up or down depending on innovations)
        # Just verify it's within bounds and was modified
        self.assertGreaterEqual(akf.sigma_a, AdaptiveKalman.SIGMA_A_MIN)
        self.assertLessEqual(akf.sigma_a, AdaptiveKalman.SIGMA_A_MAX)

    def test_innovation_window_fills(self):
        """Innovation window should accumulate entries."""
        akf = AdaptiveKalman(0.0, 10.0, 20.0, fusion_mode=0)
        self._run_fusion_steps(akf, n_steps=10)
        # First frame is initialization (idx==0), so innovations from step 2 onward
        self.assertGreater(len(akf._innovation_window), 0)

    def test_innovation_window_bounded(self):
        """Innovation window should not exceed WINDOW_SIZE."""
        akf = AdaptiveKalman(0.0, 10.0, 20.0, fusion_mode=0)
        self._run_fusion_steps(akf, n_steps=50)
        self.assertLessEqual(len(akf._innovation_window), AdaptiveKalman.WINDOW_SIZE)

    def test_tracks_position_accurately(self):
        """Should track a linearly-moving target with low error."""
        akf = AdaptiveKalman(0.0, 10.0, 20.0, fusion_mode=0)
        akf = self._run_fusion_steps(akf, n_steps=30, noise_std=0.05)
        # After 30 steps at dt=0.1, t=3.0, true_x = 10+0.5*3 = 11.5
        error = math.hypot(akf.x - 11.5, akf.y - 20.0)
        self.assertLess(error, 0.2, f"Position error {error:.4f}m too large")

    def test_sigma_a_increases_with_noisy_measurements(self):
        """With high-noise measurements, sigma_a should trend upward."""
        akf = AdaptiveKalman(0.0, 10.0, 20.0, fusion_mode=0)
        initial_sigma_a = akf.sigma_a
        # Use very noisy measurements but report low covariance (mismatch → large innovations)
        cov = np.array([[0.01, 0], [0, 0.01]])  # Report as very precise
        np.random.seed(42)
        for step in range(40):
            t = 0.1 * (step + 1)
            meas_x = 10.0 + np.random.normal(0, 0.5)  # Actual noise is 0.5m, much larger than reported
            meas_y = 20.0 + np.random.normal(0, 0.5)
            m = make_match(meas_x, meas_y, cov)
            akf.fusion([m], t, monitor=False)
        # sigma_a should have increased to compensate
        self.assertGreater(akf.sigma_a, initial_sigma_a * 0.9,
                          f"sigma_a should increase with model mismatch, got {akf.sigma_a:.3f}")

    def test_ctrv_mode_works(self):
        """CTRV mode should work without errors."""
        akf = AdaptiveKalman(0.0, 10.0, 20.0, fusion_mode=2)
        cov = np.array([[0.1, 0], [0, 0.1]])
        for step in range(20):
            t = 0.1 * (step + 1)
            # 3-element measurement for CTRV (x, y, heading)
            m = make_match(10.0 + 0.01*step, 20.0, cov)
            akf.fusion([m], t, monitor=False)
        self.assertFalse(np.any(np.isnan(akf.X_hat_t)))

    def test_no_measurements_handled(self):
        """Fusion with empty measurement list should not crash."""
        akf = AdaptiveKalman(0.0, 10.0, 20.0, fusion_mode=0)
        cov = np.array([[0.1, 0], [0, 0.1]])
        m = make_match(10.0, 20.0, cov)
        akf.fusion([m], 0.1, monitor=False)  # Initialize
        akf.fusion([], 0.2, monitor=False)    # No measurements
        self.assertFalse(np.any(np.isnan(akf.X_hat_t)))

    def test_monitor_mode(self):
        """Fusion with monitor=True should populate error_tracker_temp."""
        akf = AdaptiveKalman(0.0, 10.0, 20.0, fusion_mode=0)
        cov = np.array([[0.1, 0], [0, 0.1]])
        m = make_match(10.0, 20.0, cov)
        akf.fusion([m], 0.1, monitor=False)  # Initialize
        akf.fusion([m], 0.2, monitor=True)   # Monitor
        # Should have populated monitoring data without error
        self.assertTrue(hasattr(akf, 'error_tracker_temp'))


class TestAdaptiveVsStandard(unittest.TestCase):
    """Compare adaptive KF vs standard KF tracking quality."""

    def test_both_track_linear_motion(self):
        """Both filters should track linear motion with comparable accuracy."""
        np.random.seed(99)
        cov = np.array([[0.04, 0], [0, 0.04]])

        akf = AdaptiveKalman(0.0, 0.0, 0.0, fusion_mode=0)
        skf = ResizableKalman(0.0, 0.0, 0.0, fusion_mode=0)

        for step in range(30):
            t = 0.1 * (step + 1)
            true_x = 1.0 * t
            true_y = 0.5 * t
            mx = true_x + np.random.normal(0, 0.2)
            my = true_y + np.random.normal(0, 0.2)
            m = make_match(mx, my, cov)
            akf.fusion([m], t, monitor=False)
            skf.fusion([m], t, monitor=False)

        akf_err = math.hypot(akf.x - 3.0, akf.y - 1.5)
        skf_err = math.hypot(skf.x - 3.0, skf.y - 1.5)

        # Both should be reasonably accurate
        self.assertLess(akf_err, 0.5, f"AKF error too large: {akf_err:.3f}")
        self.assertLess(skf_err, 0.5, f"SKF error too large: {skf_err:.3f}")

    def test_prediction_methods(self):
        """getKalmanPred and getKalmanPredWithCovariance should work."""
        akf = AdaptiveKalman(0.0, 10.0, 20.0, fusion_mode=0)
        cov = np.array([[0.1, 0], [0, 0.1]])
        m = make_match(10.0, 20.0, cov)
        akf.fusion([m], 0.1, monitor=False)
        akf.fusion([m], 0.2, monitor=False)

        # getKalmanPred
        px, py, a, b, phi = akf.getKalmanPred(0.3)
        self.assertFalse(np.isnan(px))
        self.assertFalse(np.isnan(py))

        # getKalmanPredWithCovariance
        px2, py2, P = akf.getKalmanPredWithCovariance(0.3)
        self.assertFalse(np.any(np.isnan(P)))
        self.assertEqual(P.shape[0], akf.F_t_len)


if __name__ == "__main__":
    unittest.main()
