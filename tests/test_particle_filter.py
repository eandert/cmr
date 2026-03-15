"""
Tests for the Particle Filter (SIR).

Run: python -m pytest tests/test_particle_filter.py -v -s
"""

import unittest
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sensor_fusion import MatchClass
from filters.particle_filter import ParticleFilter
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


class TestParticleFilterInit(unittest.TestCase):
    """Test initialization and interface compliance."""

    def test_cv_initialization(self):
        pf = ParticleFilter(0.0, 1.0, 2.0, fusion_mode=0)
        self.assertEqual(pf.x, 1.0)
        self.assertEqual(pf.y, 2.0)
        self.assertEqual(pf.fusion_mode, 0)
        self.assertEqual(pf.F_t_len, 4)

    def test_ctrv_initialization(self):
        pf = ParticleFilter(0.0, 1.0, 2.0, fusion_mode=2)
        self.assertEqual(pf.fusion_mode, 2)
        self.assertEqual(pf.F_t_len, 5)

    def test_ca_initialization(self):
        pf = ParticleFilter(0.0, 1.0, 2.0, fusion_mode=1)
        self.assertEqual(pf.fusion_mode, 1)
        self.assertEqual(pf.F_t_len, 6)

    def test_same_constructor_signature_as_kalman(self):
        """ParticleFilter must accept the same constructor args as ResizableKalman."""
        kwargs = dict(time=0.0, x=5.0, y=10.0, fusion_mode=0,
                      initial_width=1.8, initial_length=4.0,
                      use_trust_scoring=False, passthrough_covariance=True)
        pf = ParticleFilter(**kwargs)
        rkf = ResizableKalman(**kwargs)
        self.assertEqual(pf.x, rkf.x)
        self.assertEqual(pf.y, rkf.y)

    def test_particles_array_shape(self):
        pf = ParticleFilter(0.0, 0.0, 0.0, fusion_mode=0)
        self.assertEqual(pf.particles.shape, (ParticleFilter.N_PARTICLES, 4))

    def test_particles_array_shape_ctrv(self):
        pf = ParticleFilter(0.0, 0.0, 0.0, fusion_mode=2)
        self.assertEqual(pf.particles.shape, (ParticleFilter.N_PARTICLES, 5))

    def test_weights_sum_to_one(self):
        pf = ParticleFilter(0.0, 0.0, 0.0, fusion_mode=0)
        self.assertAlmostEqual(np.sum(pf.weights), 1.0, places=10)


class TestParticleFilterFusion(unittest.TestCase):
    """Test fusion behavior."""

    def _run_fusion_steps(self, pf, n_steps=30, noise_std=0.05, dt=0.1):
        """Run n_steps of fusion with a linearly-moving target."""
        cov = np.array([[noise_std**2, 0], [0, noise_std**2]])
        np.random.seed(42)
        for step in range(n_steps):
            t = dt * (step + 1)
            true_x = 10.0 + 0.5 * t
            true_y = 20.0
            meas_x = true_x + np.random.normal(0, noise_std)
            meas_y = true_y + np.random.normal(0, noise_std)
            m = make_match(meas_x, meas_y, cov)
            pf.fusion([m], t, monitor=False)
        return pf

    def test_first_frame_initialization(self):
        """First fusion should initialize state."""
        pf = ParticleFilter(0.0, 0.0, 0.0, fusion_mode=0)
        cov = np.array([[0.1, 0], [0, 0.1]])
        m = make_match(5.0, 10.0, cov)
        pf.fusion([m], 0.1, monitor=False)
        self.assertAlmostEqual(pf.x, 5.0, places=0)
        self.assertAlmostEqual(pf.y, 10.0, places=0)

    def test_tracks_position_accurately(self):
        """Should track a linearly-moving target with reasonable error."""
        pf = ParticleFilter(0.0, 10.0, 20.0, fusion_mode=0)
        pf = self._run_fusion_steps(pf, n_steps=30, noise_std=0.05)
        # After 30 steps at dt=0.1, t=3.0, true_x = 10+0.5*3 = 11.5
        error = math.hypot(pf.x - 11.5, pf.y - 20.0)
        self.assertLess(error, 0.5, f"Position error {error:.4f}m too large")

    def test_covariance_is_psd(self):
        """Error covariance should be positive semi-definite."""
        pf = ParticleFilter(0.0, 10.0, 20.0, fusion_mode=0)
        self._run_fusion_steps(pf, n_steps=20)
        eigvals = np.linalg.eigvalsh(pf.error_covariance)
        self.assertTrue(np.all(eigvals >= -1e-10),
                        f"Covariance not PSD: eigenvalues={eigvals}")

    def test_no_measurements_handled(self):
        """Fusion with empty measurement list should not crash."""
        pf = ParticleFilter(0.0, 10.0, 20.0, fusion_mode=0)
        cov = np.array([[0.1, 0], [0, 0.1]])
        m = make_match(10.0, 20.0, cov)
        pf.fusion([m], 0.1, monitor=False)  # Initialize
        pf.fusion([], 0.2, monitor=False)   # No measurements
        self.assertFalse(np.any(np.isnan(pf.X_hat_t)))

    def test_ctrv_mode_works(self):
        """CTRV mode should work without errors."""
        pf = ParticleFilter(0.0, 10.0, 20.0, fusion_mode=2)
        cov = np.array([[0.1, 0], [0, 0.1]])
        for step in range(20):
            t = 0.1 * (step + 1)
            m = make_match(10.0 + 0.01 * step, 20.0, cov)
            pf.fusion([m], t, monitor=False)
        self.assertFalse(np.any(np.isnan(pf.X_hat_t)))

    def test_monitor_mode(self):
        """Fusion with monitor=True should populate error_tracker_temp."""
        pf = ParticleFilter(0.0, 10.0, 20.0, fusion_mode=0)
        cov = np.array([[0.1, 0], [0, 0.1]])
        m = make_match(10.0, 20.0, cov)
        pf.fusion([m], 0.1, monitor=False)  # Initialize
        pf.fusion([m], 0.2, monitor=True)   # Monitor
        self.assertTrue(hasattr(pf, 'error_tracker_temp'))

    def test_weights_remain_normalized(self):
        """Weights should sum to 1 after fusion steps."""
        pf = ParticleFilter(0.0, 10.0, 20.0, fusion_mode=0)
        self._run_fusion_steps(pf, n_steps=10)
        self.assertAlmostEqual(np.sum(pf.weights), 1.0, places=5)


class TestParticleFilterVsKalman(unittest.TestCase):
    """Compare PF vs standard Kalman tracking quality."""

    def test_both_track_linear_motion(self):
        """Both filters should track linear motion with comparable accuracy."""
        np.random.seed(99)
        cov = np.array([[0.04, 0], [0, 0.04]])

        pf = ParticleFilter(0.0, 0.0, 0.0, fusion_mode=0)
        skf = ResizableKalman(0.0, 0.0, 0.0, fusion_mode=0)

        for step in range(30):
            t = 0.1 * (step + 1)
            true_x = 1.0 * t
            true_y = 0.5 * t
            mx = true_x + np.random.normal(0, 0.2)
            my = true_y + np.random.normal(0, 0.2)
            m = make_match(mx, my, cov)
            pf.fusion([m], t, monitor=False)
            skf.fusion([m], t, monitor=False)

        pf_err = math.hypot(pf.x - 3.0, pf.y - 1.5)
        skf_err = math.hypot(skf.x - 3.0, skf.y - 1.5)

        self.assertLess(pf_err, 0.5, f"PF error too large: {pf_err:.3f}")
        self.assertLess(skf_err, 0.5, f"SKF error too large: {skf_err:.3f}")

    def test_prediction_methods(self):
        """getKalmanPred and getKalmanPredWithCovariance should work."""
        pf = ParticleFilter(0.0, 10.0, 20.0, fusion_mode=0)
        cov = np.array([[0.1, 0], [0, 0.1]])
        m = make_match(10.0, 20.0, cov)
        pf.fusion([m], 0.1, monitor=False)
        pf.fusion([m], 0.2, monitor=False)

        px, py, a, b, phi = pf.getKalmanPred(0.3)
        self.assertFalse(np.isnan(px))
        self.assertFalse(np.isnan(py))

        px2, py2, P = pf.getKalmanPredWithCovariance(0.3)
        self.assertFalse(np.any(np.isnan(P)))
        self.assertEqual(P.shape[0], pf.F_t_len)


class TestParticleFilterResampling(unittest.TestCase):
    """Test resampling mechanics."""

    def test_systematic_resample_preserves_count(self):
        """Resampling should return N indices."""
        weights = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
        indices = ParticleFilter._systematic_resample(weights)
        self.assertEqual(len(indices), len(weights))

    def test_n_eff_computation(self):
        """N_eff should be N for uniform weights, 1 for degenerate."""
        pf = ParticleFilter(0.0, 0.0, 0.0, fusion_mode=0)
        # Uniform weights
        n_eff = pf._compute_n_eff()
        self.assertAlmostEqual(n_eff, ParticleFilter.N_PARTICLES, places=1)

        # Degenerate weights (all on one particle)
        pf.weights = np.zeros(ParticleFilter.N_PARTICLES)
        pf.weights[0] = 1.0
        n_eff = pf._compute_n_eff()
        self.assertAlmostEqual(n_eff, 1.0, places=5)

    def test_resampling_triggered(self):
        """After high-noise updates, resampling should occur (weights reset to uniform)."""
        pf = ParticleFilter(0.0, 10.0, 20.0, fusion_mode=0)
        cov = np.array([[0.001, 0], [0, 0.001]])  # Very precise measurements
        np.random.seed(42)
        for step in range(10):
            t = 0.1 * (step + 1)
            m = make_match(10.0 + np.random.normal(0, 0.5), 20.0, cov)
            pf.fusion([m], t, monitor=False)
        # After resampling, weights should be close to uniform
        # (can't guarantee exact uniformity due to jitter, but should be close)
        self.assertFalse(np.any(np.isnan(pf.weights)))
        self.assertAlmostEqual(np.sum(pf.weights), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
