"""
Tests for SourceReliabilityLUT — the per-source innovation tracker for SABRE.

Run: python -m pytest tests/test_source_reliability_lut.py -v -s
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "source_reliability_lut",
    os.path.join(os.path.dirname(__file__), "../src/filters/source_reliability_lut.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SourceReliabilityLUT = _mod.SourceReliabilityLUT


class TestLUTBasics(unittest.TestCase):

    def test_empty_lut(self):
        lut = SourceReliabilityLUT()
        self.assertEqual(lut.active_sources, 0)

    def test_unknown_source_returns_default(self):
        lut = SourceReliabilityLUT()
        self.assertEqual(lut.get_reliability(999), 1.0)

    def test_omega_priors_default_for_unknown(self):
        lut = SourceReliabilityLUT()
        priors = lut.get_omega_priors([1, 2, 3])
        self.assertEqual(priors, [1.0, 1.0, 1.0])

    def test_update_adds_source(self):
        lut = SourceReliabilityLUT()
        lut.update(1, 2.0)
        self.assertEqual(lut.active_sources, 1)

    def test_update_increments_count(self):
        lut = SourceReliabilityLUT()
        for _ in range(5):
            lut.update(1, 2.0)
        log = lut.get_log()
        self.assertEqual(log['sources'][1]['n_observations'], 5)


class TestReliabilityComputation(unittest.TestCase):

    def test_well_calibrated_source(self):
        """NIS ≈ meas_dim → reliability ≈ 1.0."""
        lut = SourceReliabilityLUT(meas_dim=2, adapt_rate=1.0)  # instant adaptation
        for _ in range(10):
            lut.update(1, 2.0)  # NIS = dim = 2 → ratio = 1 → reliability = 1
        rel = lut.get_reliability(1)
        self.assertAlmostEqual(rel, 1.0, places=1)

    def test_unreliable_source_gets_low_score(self):
        """NIS >> meas_dim → reliability < 1.0."""
        lut = SourceReliabilityLUT(meas_dim=2, adapt_rate=0.5)
        for _ in range(15):
            lut.update(1, 10.0)  # NIS = 10, dim = 2 → ratio = 5 → reliability = 0.2
        rel = lut.get_reliability(1)
        self.assertLess(rel, 0.5)

    def test_very_reliable_source_gets_high_score(self):
        """NIS << meas_dim → reliability > 1.0."""
        lut = SourceReliabilityLUT(meas_dim=2, adapt_rate=0.5)
        for _ in range(15):
            lut.update(1, 0.5)  # NIS = 0.5, dim = 2 → ratio = 0.25 → reliability = 4.0
        rel = lut.get_reliability(1)
        self.assertGreater(rel, 2.0)

    def test_reliability_bounded(self):
        """Reliability should be clipped to [0.1, 10.0]."""
        lut = SourceReliabilityLUT(meas_dim=2, adapt_rate=1.0)
        # Extreme NIS
        for _ in range(10):
            lut.update(1, 1000.0)
        self.assertGreaterEqual(lut.get_reliability(1), 0.1)
        for _ in range(10):
            lut.update(2, 0.001)
        self.assertLessEqual(lut.get_reliability(2), 10.0)

    def test_smoothing_prevents_oscillation(self):
        """With low adapt_rate, reliability should change slowly."""
        lut = SourceReliabilityLUT(meas_dim=2, adapt_rate=0.1)
        # Start with good NIS
        for _ in range(10):
            lut.update(1, 2.0)
        good_rel = lut.get_reliability(1)
        # Sudden bad NIS
        lut.update(1, 20.0)
        after_one_bad = lut.get_reliability(1)
        # Should barely change
        self.assertAlmostEqual(after_one_bad, good_rel, places=0)


class TestOmegaPriors(unittest.TestCase):

    def test_priors_differentiate_sources(self):
        """Good and bad sources should get different priors."""
        lut = SourceReliabilityLUT(meas_dim=2, adapt_rate=0.5)
        for _ in range(15):
            lut.update(1, 2.0)   # Good source
            lut.update(2, 10.0)  # Bad source
        priors = lut.get_omega_priors([1, 2])
        self.assertGreater(priors[0], priors[1])

    def test_priors_new_source_is_neutral(self):
        """New source mixed with established source."""
        lut = SourceReliabilityLUT(meas_dim=2, adapt_rate=0.5)
        for _ in range(15):
            lut.update(1, 2.0)
        priors = lut.get_omega_priors([1, 99])  # 99 is new
        self.assertEqual(priors[1], 1.0)

    def test_needs_min_observations(self):
        """Source with < 3 observations should return default."""
        lut = SourceReliabilityLUT(meas_dim=2)
        lut.update(1, 2.0)
        lut.update(1, 2.0)
        self.assertEqual(lut.get_reliability(1), 1.0)  # Only 2 obs, not enough


class TestEviction(unittest.TestCase):

    def test_stale_source_evicted(self):
        lut = SourceReliabilityLUT(stale_timeout=5)
        lut.update(1, 2.0)
        for _ in range(10):
            lut.step()
        self.assertEqual(lut.active_sources, 0)

    def test_active_source_not_evicted(self):
        lut = SourceReliabilityLUT(stale_timeout=5)
        for _ in range(10):
            lut.update(1, 2.0)
            lut.step()
        self.assertEqual(lut.active_sources, 1)

    def test_evicted_source_resets_on_return(self):
        lut = SourceReliabilityLUT(stale_timeout=5, adapt_rate=1.0)
        for _ in range(10):
            lut.update(1, 10.0)  # Bad source
        bad_rel = lut.get_reliability(1)
        self.assertLess(bad_rel, 0.5)
        # Evict
        for _ in range(10):
            lut.step()
        self.assertEqual(lut.active_sources, 0)
        # Source returns — should be default
        self.assertEqual(lut.get_reliability(1), 1.0)


class TestLog(unittest.TestCase):

    def test_log_structure(self):
        lut = SourceReliabilityLUT()
        for _ in range(5):
            lut.update(1, 2.0)
            lut.update(2, 5.0)
            lut.step()
        log = lut.get_log()
        self.assertIn('frame', log)
        self.assertIn('n_sources', log)
        self.assertIn('sources', log)
        self.assertEqual(log['n_sources'], 2)
        self.assertIn(1, log['sources'])
        self.assertIn('reliability', log['sources'][1])


class TestInvalidInputs(unittest.TestCase):

    def test_nan_nis_ignored(self):
        lut = SourceReliabilityLUT()
        lut.update(1, float('nan'))
        self.assertEqual(lut.active_sources, 0)

    def test_inf_nis_ignored(self):
        lut = SourceReliabilityLUT()
        lut.update(1, float('inf'))
        self.assertEqual(lut.active_sources, 0)

    def test_negative_nis_ignored(self):
        lut = SourceReliabilityLUT()
        lut.update(1, -5.0)
        self.assertEqual(lut.active_sources, 0)


if __name__ == "__main__":
    unittest.main()
