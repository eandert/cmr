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
sigma_a_from_R = _mod.sigma_a_from_R
SIGMA_A_K = _mod.SIGMA_A_K
set_sigma_a_overrides = _mod.set_sigma_a_overrides
clear_sigma_a_overrides = _mod.clear_sigma_a_overrides


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


class TestAutoQLaw(unittest.TestCase):
    """The auto-Q law sigma_a = k*sqrt(mean R) (qr-ratio autotune lead)."""

    def test_calibration_anchor(self):
        # Calibrated so sigma_a ~= 2.0 at the DAIR detector R ~= 0.025.
        self.assertAlmostEqual(sigma_a_from_R(0.025), 2.0, delta=0.05)

    def test_k_value(self):
        # k chosen as 2.0 / sqrt(0.025); sigma_a_from_R(R) == k*sqrt(R).
        self.assertAlmostEqual(SIGMA_A_K, 2.0 / (0.025 ** 0.5), delta=0.1)
        self.assertAlmostEqual(sigma_a_from_R(0.09), SIGMA_A_K * 0.3, places=6)

    def test_monotonic_in_R(self):
        self.assertLess(sigma_a_from_R(0.01), sigma_a_from_R(0.025))
        self.assertLess(sigma_a_from_R(0.025), sigma_a_from_R(0.2))

    def test_large_R_inflates(self):
        # An inflated localizer cov (R ~= 0.195) should land sigma_a ~= 5.6, not ~2.
        self.assertAlmostEqual(sigma_a_from_R(0.195), 5.56, delta=0.2)

    def test_nonnegative_R_clamped(self):
        self.assertEqual(sigma_a_from_R(-1.0), 0.0)


class TestProcessLocalOverride(unittest.TestCase):
    """The process-local sigma_a override (the config->filter bridge for auto-Q)."""

    def tearDown(self):
        clear_sigma_a_overrides()

    def test_override_applied(self):
        set_sigma_a_overrides({"sabre": 3.5})
        self.assertEqual(get_sigma_a("sabre"), 3.5)

    def test_override_does_not_leak_to_others(self):
        set_sigma_a_overrides({"sabre": 3.5})
        self.assertEqual(get_sigma_a("ekf"), SIGMA_A["ekf"])

    def test_clear_restores_default(self):
        set_sigma_a_overrides({"sabre": 3.5})
        clear_sigma_a_overrides()
        self.assertEqual(get_sigma_a("sabre"), SIGMA_A["sabre"])

    def test_config_arg_beats_process_override(self):
        set_sigma_a_overrides({"ekf": 3.5})
        self.assertEqual(get_sigma_a("ekf", {"sigma_a_override": {"ekf": 9.0}}), 9.0)

    def test_env_beats_process_override(self):
        set_sigma_a_overrides({"ekf": 3.5})
        os.environ["CMR_SIGMA_A_EKF"] = "8.0"
        try:
            self.assertEqual(get_sigma_a("ekf"), 8.0)
        finally:
            del os.environ["CMR_SIGMA_A_EKF"]

    def test_idempotent_replace(self):
        set_sigma_a_overrides({"sabre": 3.5})
        set_sigma_a_overrides({"sabre": 4.5})  # replaces, not merges
        self.assertEqual(get_sigma_a("sabre"), 4.5)


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

    def test_filter_reads_process_local_override(self):
        """A filter built AFTER set_sigma_a_overrides reads the auto-Q value.

        This is the config->filter bridge: filters call get_sigma_a() with no config
        (they are constructed deep in the fusion stack), so the run entry point installs
        a process-local override that the filter's own filter_config module exposes.
        The override must be set on the SAME module the filter imports
        (sys.modules['filters.filter_config']), not the standalone test copy.
        """
        # Loading any filter populates sys.modules['filters.filter_config'].
        self._make_filter('filters/kalman_ctrv.py', 'ResizableKalman')
        fc = sys.modules['filters.filter_config']
        try:
            fc.set_sigma_a_overrides({"ekf": 7.0})
            ekf = self._make_filter('filters/kalman_ctrv.py', 'ResizableKalman')
            self.assertEqual(ekf.sigma_a, 7.0)
        finally:
            fc.clear_sigma_a_overrides()
        # And cleared: a fresh filter is back to the default.
        ekf2 = self._make_filter('filters/kalman_ctrv.py', 'ResizableKalman')
        self.assertEqual(ekf2.sigma_a, SIGMA_A["ekf"])


if __name__ == "__main__":
    unittest.main()
