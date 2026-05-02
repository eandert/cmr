"""
Tests for the Localizer module (localizer.py).

Run: python -m pytest tests/test_localizer.py -v -s
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from localizer import (
    Localizer, VelocityBin, get_localizer, load_localizer_model,
    load_localizer_distributions, MAE_TO_STD, LOCALIZER_MODELS
)
import utils


def make_error_package():
    from error import ErrorPackage, ErrorType
    return ErrorPackage(ErrorType.LOCALIZATION, 0)


class TestLocalizerModelLoading(unittest.TestCase):
    """Test loading of localizer models from CSV."""

    def test_load_kiss_icp(self):
        model = load_localizer_model("kiss_icp")
        self.assertIn('longitudinal_linear', model)
        self.assertIn('lateral_linear', model)

    def test_load_kiss_icp_noisy(self):
        model = load_localizer_model("kiss_icp_noisy")
        self.assertIn('longitudinal_linear', model)
        self.assertIn('lateral_linear', model)

    def test_kiss_icp_noisy_in_models_dict(self):
        self.assertIn("kiss_icp_noisy", LOCALIZER_MODELS)

    def test_load_kiss_icp_distributions(self):
        dists = load_localizer_distributions("kiss_icp")
        self.assertIn('longitudinal', dists)
        self.assertIn('lateral', dists)
        # Should have specific bins (not just overall)
        specific = [b for b in dists['longitudinal'] if (b.max_vel - b.min_vel) <= 10]
        self.assertGreater(len(specific), 5)

    def test_load_kiss_icp_noisy_distributions(self):
        dists = load_localizer_distributions("kiss_icp_noisy")
        self.assertIn('longitudinal', dists)
        self.assertIn('lateral', dists)

    def test_longitudinal_naming_accepted(self):
        """New CSVs use 'longitudinal' instead of 'radial' — both should work."""
        model = load_localizer_model("kiss_icp")
        self.assertIn('longitudinal_linear', model)


class TestLocalizerInit(unittest.TestCase):
    """Test Localizer initialization and std-from-bins."""

    def test_get_localizer_kiss_icp(self):
        loc = get_localizer("kiss_icp", make_error_package(), use_gpem_model=True)
        self.assertIsInstance(loc, Localizer)
        self.assertTrue(len(loc.longitudinal_bins) > 0)
        self.assertTrue(len(loc.lateral_bins) > 0)

    def test_mae_to_std_is_1_with_bins(self):
        """When bins are available, refit sets mae_to_std = 1.0."""
        loc = get_localizer("kiss_icp", make_error_package(), use_gpem_model=True)
        self.assertEqual(loc.mae_to_std, 1.0)

    def test_mae_to_std_is_1_for_noisy(self):
        loc = get_localizer("kiss_icp_noisy", make_error_package(), use_gpem_model=True)
        self.assertEqual(loc.mae_to_std, 1.0)

    def test_refit_changes_polynomials(self):
        """After refit, polynomials should predict std directly (not MAE)."""
        loc = get_localizer("kiss_icp", make_error_package(), use_gpem_model=True)
        # At 10 m/s, the polynomial should give a value close to the bin std
        poly_val = abs(loc.longitudinal_error_polynomial.evaluate(10.0))
        # Should be in a reasonable range for std (0.01-0.5 for kiss_icp)
        self.assertGreater(poly_val, 0.01)
        self.assertLess(poly_val, 0.5)


class TestLocalizerCovariance(unittest.TestCase):
    """Test covariance estimation."""

    def test_gpem_covariance_kiss_icp(self):
        loc = get_localizer("kiss_icp", make_error_package(), use_gpem_model=True)
        cov = loc.get_localization_covariance(10.0, 0.0)
        self.assertEqual(cov.shape, (2, 2))
        # Should be positive definite
        self.assertGreater(cov[0, 0], 0)
        self.assertGreater(cov[1, 1], 0)
        # Should be symmetric
        np.testing.assert_allclose(cov[0, 1], cov[1, 0])

    def test_gpem_covariance_kiss_icp_noisy(self):
        loc = get_localizer("kiss_icp_noisy", make_error_package(), use_gpem_model=True)
        cov = loc.get_localization_covariance(10.0, 0.0)
        self.assertGreater(cov[0, 0], 0)

    def test_noisy_covariance_larger_than_accurate(self):
        """Noisy localizer should have larger covariance than accurate."""
        loc_acc = get_localizer("kiss_icp", make_error_package(), use_gpem_model=True)
        loc_noisy = get_localizer("kiss_icp_noisy", make_error_package(), use_gpem_model=True)
        cov_acc = loc_acc.get_localization_covariance(10.0, 0.0)
        cov_noisy = loc_noisy.get_localization_covariance(10.0, 0.0)
        # Noisy should have significantly larger variances
        self.assertGreater(cov_noisy[0, 0], cov_acc[0, 0] * 2,
                          f"Noisy cov {cov_noisy[0,0]:.6f} should be >> accurate {cov_acc[0,0]:.6f}")

    def test_static_covariance_uses_avg_variance(self):
        """Static mode should use avg-variance method (Jensen's fix)."""
        loc = get_localizer("kiss_icp", make_error_package(), use_gpem_model=False)
        # Get the static std
        lat_std = loc._get_lateral_std_average()
        long_std = loc._get_longitudinal_std_average()
        # Should be positive and reasonable
        self.assertGreater(lat_std, 0.01)
        self.assertLess(lat_std, 1.0)
        self.assertGreater(long_std, 0.01)
        self.assertLess(long_std, 1.0)

    def test_static_vs_gpem_covariance(self):
        """Static and GPEM should give different covariances at non-average velocities."""
        loc_static = get_localizer("kiss_icp", make_error_package(), use_gpem_model=False)
        loc_gpem = get_localizer("kiss_icp", make_error_package(), use_gpem_model=True)
        cov_static = loc_static.get_localization_covariance(5.0, 0.0)
        cov_gpem = loc_gpem.get_localization_covariance(5.0, 0.0)
        # They should differ (GPEM is velocity-dependent)
        self.assertFalse(np.allclose(cov_static, cov_gpem),
                        "Static and GPEM should differ at non-average velocity")


class TestLocalizerSampling(unittest.TestCase):
    """Test error sampling."""

    def test_sample_lateral_error(self):
        loc = get_localizer("kiss_icp", make_error_package(), use_gpem_model=True)
        np.random.seed(42)
        errors = [loc.sample_lateral_error(10.0) for _ in range(1000)]
        # Mean should be near 0
        self.assertAlmostEqual(np.mean(errors), 0.0, places=1)
        # Std should be in a reasonable range
        self.assertGreater(np.std(errors), 0.01)
        self.assertLess(np.std(errors), 0.5)

    def test_sample_longitudinal_error(self):
        loc = get_localizer("kiss_icp", make_error_package(), use_gpem_model=True)
        np.random.seed(42)
        errors = [loc.sample_longitudinal_error(10.0) for _ in range(1000)]
        self.assertAlmostEqual(np.mean(errors), 0.0, places=1)

    def test_velocity_clamping(self):
        """Velocities beyond bin range should clamp to edge bins, not error."""
        loc = get_localizer("kiss_icp", make_error_package(), use_gpem_model=True)
        # High velocity (beyond 22 m/s max bin)
        std_high = loc.get_longitudinal_localization_std(30.0)
        self.assertGreater(std_high, 0)
        # Low velocity
        std_low = loc.get_longitudinal_localization_std(0.0)
        self.assertGreater(std_low, 0)


class TestVelocityBin(unittest.TestCase):
    """Test VelocityBin class."""

    def test_contains(self):
        b = VelocityBin("normal", {"mean": 0.1, "std": 0.05}, 4.0, 6.0)
        self.assertTrue(b.contains(5.0))
        self.assertFalse(b.contains(3.0))
        self.assertFalse(b.contains(6.0))

    def test_get_std(self):
        b = VelocityBin("normal", {"mean": 0.1, "std": 0.05}, 0.0, 2.0)
        self.assertEqual(b.get_std(), 0.05)

    def test_sample(self):
        b = VelocityBin("normal", {"mean": 0.0, "std": 0.1}, 0.0, 2.0)
        np.random.seed(42)
        samples = [b.sample() for _ in range(1000)]
        self.assertAlmostEqual(np.std(samples), 0.1, places=1)


class TestLocalizerModelDataMapping(unittest.TestCase):
    """Regression test: all localizer names used in experiments must resolve
    to valid model_data (with bias/var regressions for MSE computation).
    Without this, get_longitudinal_mse() silently falls back to std² which
    can underestimate localization variance by 4×+."""

    EXPERIMENT_LOCALIZER_NAMES = [
        "kiss_icp", "kiss_icp_noisy",
        "kiss_icp_clean_seed", "kiss_icp_noisy_seed",
        "CT_ICP", "orb_slam3",
    ]

    def test_all_localizer_names_in_models_dict(self):
        """Every localizer name used in experiments must map in LOCALIZER_MODELS."""
        for name in self.EXPERIMENT_LOCALIZER_NAMES:
            self.assertIn(name, LOCALIZER_MODELS,
                         f"'{name}' missing from LOCALIZER_MODELS — model_data will be None, "
                         f"MSE falls back to std² silently")

    def test_all_localizers_have_model_data(self):
        """get_localizer() must produce non-None _model_data for all experiment localizers."""
        ep = make_error_package()
        for name in self.EXPERIMENT_LOCALIZER_NAMES:
            loc = get_localizer(name, ep, use_gpem_model=True)
            self.assertIsNotNone(loc._model_data,
                                f"'{name}' has _model_data=None — MSE will use std² fallback")

    def test_mse_uses_var_regression_not_std_squared(self):
        """MSE should use bias²+var regression, not just std². They should differ."""
        ep = make_error_package()
        for name in ["kiss_icp_noisy_seed", "kiss_icp_clean_seed"]:
            loc = get_localizer(name, ep, use_gpem_model=True)
            mse = loc.get_longitudinal_mse(15.0)
            std = loc.get_longitudinal_localization_std(15.0)
            # MSE and std² should NOT be identical (that would mean fallback)
            self.assertNotAlmostEqual(mse, std**2, places=4,
                                     msg=f"'{name}' MSE ({mse:.6f}) == std² ({std**2:.6f}) — "
                                         f"model_data may not be loading (using std² fallback)")

    def test_noisy_seed_localization_variance_is_significant(self):
        """Noisy seed localizer should have variance >> 0.01 (not tiny std² fallback)."""
        ep = make_error_package()
        loc = get_localizer("kiss_icp_noisy_seed", ep, use_gpem_model=True)
        cov = loc.get_localization_covariance(15.0, 0.0)
        avg_var = np.mean(np.diag(cov))
        self.assertGreater(avg_var, 0.05,
                          f"Noisy seed variance {avg_var:.6f} is suspiciously small — "
                          f"model_data may not be loaded (expected ~0.25)")


if __name__ == "__main__":
    unittest.main()
