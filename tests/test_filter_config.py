"""
Tests for centralized filter configuration (filters/filter_config.py).
Ensures all filters read sigma_a from the shared config, not hardcoded values.

Run: python -m pytest tests/test_filter_config.py -v -s
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Load filter_config directly to avoid circular import through filters/__init__.py
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "filter_config",
    os.path.join(os.path.dirname(__file__), "../src/filters/filter_config.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SIGMA_A = _mod.SIGMA_A
get_sigma_a = _mod.get_sigma_a


class TestFilterConfigExists(unittest.TestCase):
    """Verify all expected filter types are in the config."""

    def test_all_filter_types_present(self):
        expected = ["ekf", "ci", "bici", "akf", "pf"]
        for ft in expected:
            self.assertIn(ft, SIGMA_A, f"'{ft}' missing from SIGMA_A config")

    def test_all_values_positive(self):
        for ft, val in SIGMA_A.items():
            self.assertGreater(val, 0, f"sigma_a for '{ft}' must be positive")

    def test_bici_independent_from_ci(self):
        """BICI and CI should have separate config entries (even if same default)."""
        self.assertIn("bici", SIGMA_A)
        self.assertIn("ci", SIGMA_A)
        # They can be equal but must be separately addressable


class TestFilterConfigOverride(unittest.TestCase):
    """Verify runtime override mechanism."""

    def test_override_works(self):
        config = {"sigma_a_override": {"ekf": 99.0}}
        self.assertEqual(get_sigma_a("ekf", config), 99.0)

    def test_override_doesnt_affect_others(self):
        config = {"sigma_a_override": {"ekf": 99.0}}
        self.assertEqual(get_sigma_a("ci", config), SIGMA_A["ci"])

    def test_no_override_returns_default(self):
        self.assertEqual(get_sigma_a("ekf"), SIGMA_A["ekf"])

    def test_empty_config_returns_default(self):
        self.assertEqual(get_sigma_a("ekf", {}), SIGMA_A["ekf"])


class TestFiltersReadFromConfig(unittest.TestCase):
    """Verify each filter class actually reads sigma_a from filter_config."""

    def _make_filter(self, module_path, class_name):
        """Import and instantiate a filter, bypassing __init__.py circular import."""
        spec = importlib.util.spec_from_file_location(
            module_path.split('/')[-1].replace('.py', ''),
            os.path.join(os.path.dirname(__file__), '..', 'src', module_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, class_name)
        return cls(0.0, 0.0, 0.0, fusion_mode=0)

    def test_ekf_uses_config(self):
        ekf = self._make_filter('filters/kalman_ctrv.py', 'ResizableKalman')
        self.assertEqual(ekf.sigma_a, SIGMA_A["ekf"],
                        f"EKF sigma_a={ekf.sigma_a} doesn't match config {SIGMA_A['ekf']}")

    def test_ci_uses_config(self):
        ci = self._make_filter('filters/covariance_intersection.py', 'CovarianceIntersectionFilter')
        self.assertEqual(ci.sigma_a, SIGMA_A["ci"],
                        f"CI sigma_a={ci.sigma_a} doesn't match config {SIGMA_A['ci']}")

    def test_bici_uses_own_config(self):
        bici = self._make_filter('filters/bici_filter.py', 'BICIFilter')
        self.assertEqual(bici.sigma_a, SIGMA_A["bici"],
                        f"BICI sigma_a={bici.sigma_a} doesn't match config {SIGMA_A['bici']}")

    def test_bici_independent_from_ci(self):
        """BICI and CI read their own config keys."""
        bici = self._make_filter('filters/bici_filter.py', 'BICIFilter')
        ci = self._make_filter('filters/covariance_intersection.py', 'CovarianceIntersectionFilter')
        self.assertEqual(bici.sigma_a, SIGMA_A["bici"])
        self.assertEqual(ci.sigma_a, SIGMA_A["ci"])

    def test_akf_uses_config(self):
        akf = self._make_filter('filters/adaptive_kalman.py', 'AdaptiveKalman')
        self.assertEqual(akf.sigma_a, SIGMA_A["akf"],
                        f"AKF sigma_a={akf.sigma_a} doesn't match config {SIGMA_A['akf']}")

    def test_pf_uses_config(self):
        pf = self._make_filter('filters/particle_filter.py', 'ParticleFilter')
        self.assertEqual(pf.sigma_a, SIGMA_A["pf"],
                        f"PF sigma_a={pf.sigma_a} doesn't match config {SIGMA_A['pf']}")


if __name__ == "__main__":
    unittest.main()
