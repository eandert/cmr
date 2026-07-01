"""Unit tests for the GPEM-derived birth-gate plumbing.

Covers:
  1. Per-bin gate math: TP-preserving formula, Bayes-optimal crossing, edges.
  2. Curve loader: lookup hit/miss, caching across calls, evaluate clipping,
     quadratic vs linear evaluation paths.
  3. Auto-creation pipeline: analyze_detector on synthetic calibration CSV
     yields the expected scalar and curve fit.

These tests deliberately avoid loading any real calibration data — synthetic
fixtures are written to a temp dir per test so the suite stays hermetic.

Run: PYTHONPATH=src python -m pytest tests/test_birth_gate.py -v
"""
import csv
import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


# ============================================================================
# 1. Per-bin gate math
# ============================================================================

class TestPerBinGateMath:
    """Validates the closed-form TP-preserving formula and Bayes crossing."""

    @pytest.fixture
    def script(self):
        # Import the analysis script as a module so we can call its helpers.
        sys.path.insert(0, str(REPO / "scripts"))
        return importlib.import_module("derive_optimal_birth_gate")

    def test_tp_preserving_clean_separation(self, script):
        """When TP and FP score distributions are well-separated, gate sits at
        FP+α·σ_fp (reject all FPs without losing TPs)."""
        # μ_fp=0.20 σ_fp=0.05 → fp_upper(α=1) = 0.25
        # μ_tp=0.70 σ_tp=0.05 → tp_lower(α=1) = 0.65
        # Clean separation → gate = min(0.25, 0.65) = 0.25
        gate = script._tp_preserving_score_for_bin(
            mu_tp=0.70, sig_tp=0.05, mu_fp=0.20, sig_fp=0.05, alpha=1.0)
        assert gate == pytest.approx(0.25, abs=1e-6)

    def test_tp_preserving_overlap_pushes_gate_down(self, script):
        """When TP and FP score distributions overlap, gate is pulled down to
        TP−α·σ_tp to keep ~84% of TPs at the cost of letting some FPs in."""
        # μ_fp=0.30 σ_fp=0.10 → fp_upper(α=1) = 0.40
        # μ_tp=0.40 σ_tp=0.10 → tp_lower(α=1) = 0.30
        # Overlap → gate = min(0.40, 0.30) = 0.30 (pushed down)
        gate = script._tp_preserving_score_for_bin(
            mu_tp=0.40, sig_tp=0.10, mu_fp=0.30, sig_fp=0.10, alpha=1.0)
        assert gate == pytest.approx(0.30, abs=1e-6)

    def test_tp_preserving_alpha_2_is_more_permissive(self, script):
        """α=2 should produce a smaller gate than α=1 in the overlap regime
        (TP-1σ → TP-2σ pushes the lower edge further down)."""
        a1 = script._tp_preserving_score_for_bin(
            mu_tp=0.40, sig_tp=0.10, mu_fp=0.30, sig_fp=0.10, alpha=1.0)
        a2 = script._tp_preserving_score_for_bin(
            mu_tp=0.40, sig_tp=0.10, mu_fp=0.30, sig_fp=0.10, alpha=2.0)
        assert a2 < a1
        # μ_tp − 2σ_tp = 0.20
        assert a2 == pytest.approx(0.20, abs=1e-6)

    def test_tp_preserving_clipped_to_unit_interval(self, script):
        """Pathological inputs cannot produce gates outside [0, 1]."""
        # Extreme negative tail: μ_tp=0.05, σ_tp=0.10 → tp_lower=-0.05
        gate = script._tp_preserving_score_for_bin(
            mu_tp=0.05, sig_tp=0.10, mu_fp=0.10, sig_fp=0.10, alpha=1.0)
        assert 0.0 <= gate <= 1.0

    def test_bayes_optimal_finds_crossing(self, script):
        """At equal counts and unit-variance Gaussians at 0.3 and 0.7, the
        Bayes 50/50 crossing is the midpoint 0.5."""
        s_star = script._bayes_optimal_score_for_bin(
            n_tp=100, n_fp=100,
            mu_tp=0.70, sig_tp=0.10,
            mu_fp=0.30, sig_fp=0.10,
        )
        assert s_star is not None
        assert s_star == pytest.approx(0.50, abs=0.005)

    def test_bayes_optimal_count_imbalance_shifts_crossing(self, script):
        """When TPs vastly outnumber FPs, the Bayes crossing shifts toward
        the FP mean (you'd accept lower scores because TPs are abundant)."""
        balanced = script._bayes_optimal_score_for_bin(
            n_tp=100, n_fp=100,
            mu_tp=0.70, sig_tp=0.10,
            mu_fp=0.30, sig_fp=0.10,
        )
        imbalanced = script._bayes_optimal_score_for_bin(
            n_tp=1000, n_fp=10,
            mu_tp=0.70, sig_tp=0.10,
            mu_fp=0.30, sig_fp=0.10,
        )
        assert imbalanced < balanced

    def test_bayes_optimal_returns_none_for_empty_bins(self, script):
        """Bins with no TPs or no FPs can't define a crossing."""
        assert script._bayes_optimal_score_for_bin(
            n_tp=0, n_fp=10, mu_tp=0.5, sig_tp=0.1, mu_fp=0.5, sig_fp=0.1) is None
        assert script._bayes_optimal_score_for_bin(
            n_tp=10, n_fp=0, mu_tp=0.5, sig_tp=0.1, mu_fp=0.5, sig_fp=0.1) is None


# ============================================================================
# 2. Curve loader (BirthGateCurve)
# ============================================================================

class TestBirthGateCurve:
    """Loader contract: cache hits/misses, evaluate paths, clipping."""

    @pytest.fixture
    def synthetic_curve_csv(self, tmp_path):
        """Build a one-row birth_gate_curves.csv with known coefficients."""
        csv_path = tmp_path / "birth_gate_curves.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "detector", "alpha", "fit_type",
                "a_quad", "b_quad", "c_quad",
                "a_lin", "b_lin",
                "n_range_bins",
            ])
            # gate(d) = 0.0001·d² − 0.01·d + 0.5
            # at d=10: 0.01 − 0.1 + 0.5 = 0.41
            # at d=50: 0.25 − 0.5 + 0.5 = 0.25
            w.writerow([
                "test_detector", 1.0, "tp_preserve_a1",
                0.0001, -0.01, 0.5,
                -0.005, 0.5,
                10,
            ])
            w.writerow([
                "test_detector", 2.0, "tp_preserve_a2",
                0.0, -0.005, 0.3,
                -0.005, 0.3,
                10,
            ])
        return csv_path

    def test_loads_curve_for_existing_detector(self, synthetic_curve_csv):
        from birth_gate_curves import load_birth_gate_curve, _cache
        _cache.clear()
        curve = load_birth_gate_curve("test_detector", 1.0,
                                       curve_csv=synthetic_curve_csv)
        assert curve is not None
        assert curve.detector == "test_detector"
        assert curve.alpha == 1.0
        assert curve.a_quad == pytest.approx(0.0001)
        assert curve.c_quad == pytest.approx(0.5)

    def test_returns_none_for_missing_detector(self, synthetic_curve_csv):
        from birth_gate_curves import load_birth_gate_curve, _cache
        _cache.clear()
        curve = load_birth_gate_curve("ghost_detector", 1.0,
                                       curve_csv=synthetic_curve_csv)
        assert curve is None

    def test_returns_none_for_unknown_alpha(self, synthetic_curve_csv):
        from birth_gate_curves import load_birth_gate_curve, _cache
        _cache.clear()
        curve = load_birth_gate_curve("test_detector", 3.0,
                                       curve_csv=synthetic_curve_csv)
        assert curve is None

    def test_evaluate_quadratic_path(self, synthetic_curve_csv):
        from birth_gate_curves import load_birth_gate_curve, _cache
        _cache.clear()
        c = load_birth_gate_curve("test_detector", 1.0, curve_csv=synthetic_curve_csv)
        # 0.0001·100 − 0.01·10 + 0.5 = 0.01 − 0.1 + 0.5 = 0.41
        assert c.evaluate(10.0) == pytest.approx(0.41, abs=1e-6)
        # 0.0001·2500 − 0.01·50 + 0.5 = 0.25 − 0.5 + 0.5 = 0.25
        assert c.evaluate(50.0) == pytest.approx(0.25, abs=1e-6)

    def test_evaluate_linear_path(self, synthetic_curve_csv):
        from birth_gate_curves import load_birth_gate_curve, _cache
        _cache.clear()
        c = load_birth_gate_curve("test_detector", 1.0, curve_csv=synthetic_curve_csv)
        # −0.005·10 + 0.5 = 0.45
        assert c.evaluate(10.0, fit="linear") == pytest.approx(0.45, abs=1e-6)
        # −0.005·100 + 0.5 = 0.0
        assert c.evaluate(100.0, fit="linear") == pytest.approx(0.0, abs=1e-6)

    def test_evaluate_clipped_to_unit_interval(self, synthetic_curve_csv):
        """Evaluations far outside the fit range may extrapolate <0 or >1;
        the loader must clip so downstream gate logic sees a valid prob."""
        from birth_gate_curves import load_birth_gate_curve, _cache
        _cache.clear()
        c = load_birth_gate_curve("test_detector", 1.0, curve_csv=synthetic_curve_csv)
        # Far extrapolation: linear at d=200 → −0.005·200 + 0.5 = -0.5 → clipped 0.0
        assert c.evaluate(200.0, fit="linear") == 0.0

    def test_cache_reuse_within_same_csv(self, synthetic_curve_csv):
        """Repeat calls for the same CSV should reuse the cache."""
        from birth_gate_curves import load_birth_gate_curve, _cache
        _cache.clear()
        c1 = load_birth_gate_curve("test_detector", 1.0, curve_csv=synthetic_curve_csv)
        c2 = load_birth_gate_curve("test_detector", 1.0, curve_csv=synthetic_curve_csv)
        assert c1 is c2  # same object reference → cache hit

    def test_alpha_1_and_alpha_2_are_distinct(self, synthetic_curve_csv):
        from birth_gate_curves import load_birth_gate_curve, _cache
        _cache.clear()
        c1 = load_birth_gate_curve("test_detector", 1.0, curve_csv=synthetic_curve_csv)
        c2 = load_birth_gate_curve("test_detector", 2.0, curve_csv=synthetic_curve_csv)
        assert c1 is not c2
        assert c1.c_quad != c2.c_quad


# ============================================================================
# 3. End-to-end pipeline (analyze_detector → curve → evaluate)
# ============================================================================

class TestEndToEndPipeline:
    """analyze_detector reads a calibration CSV and returns the right scalar
    and curve. This exercises CSV parsing, weighting, and the curve fit."""

    @pytest.fixture
    def script(self):
        sys.path.insert(0, str(REPO / "scripts"))
        return importlib.import_module("derive_optimal_birth_gate")

    @pytest.fixture
    def synthetic_calibration_csv(self, tmp_path, script):
        """Build a polar_calibration.csv with two range buckets and write it
        to where analyze_detector will look (script.SENSOR_MODELS)."""
        csv_path = tmp_path / "synth_detector_polar_calibration.csv"
        # Schema cribbed from real CSV — only the fields analyze_detector
        # actually reads need to be valid.
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["range_lo", "range_hi", "angle_lo", "angle_hi",
                        "gt_count", "missed", "miss_rate", "source",
                        "n_fp", "fp_per_frame",
                        "score_mean", "score_std",
                        "fp_score_mean", "fp_score_std"])
            # Bucket 1: clean separation, TP-dominant
            for ang_lo in (-30.0, 0.0, 30.0):
                w.writerow([0.0, 10.0, ang_lo, ang_lo + 30.0,
                            100, 5, 0.05, "real",
                            10, 0.001,
                            0.75, 0.10,
                            0.20, 0.10])
            # Bucket 2: overlap, gate must drop
            for ang_lo in (-30.0, 0.0, 30.0):
                w.writerow([10.0, 20.0, ang_lo, ang_lo + 30.0,
                            50, 10, 0.20, "real",
                            30, 0.003,
                            0.45, 0.10,
                            0.35, 0.10])
        # Redirect script's lookup directory to our tmp.
        script.SENSOR_MODELS = tmp_path
        return csv_path

    def test_analyze_detector_returns_scalar_and_per_range(self, synthetic_calibration_csv, script):
        r = script.analyze_detector("synth_detector")
        assert r is not None
        # 6 bins (2 range buckets × 3 angle buckets)
        assert r["n_bins"] == 6
        # α=1 scalar should sit between the two buckets' gates.
        # Bucket 1 (clean): tp_preserve = min(0.20+0.10, 0.75-0.10) = 0.30
        # Bucket 2 (overlap): tp_preserve = min(0.35+0.10, 0.45-0.10) = 0.35
        sc = r["scalar"]["tp_preserve_a1"]
        assert 0.30 <= sc <= 0.35

    def test_analyze_detector_emits_per_range_aggregates(self, synthetic_calibration_csv, script):
        r = script.analyze_detector("synth_detector")
        # Both range buckets must be present in per_range output.
        per_range = r["per_range"]
        rng_buckets = sorted(per_range.keys())
        assert (0.0, 10.0) in per_range
        assert (10.0, 20.0) in per_range
        # Bucket 1 gate = 0.30 (clean sep, fp_upper)
        assert per_range[(0.0, 10.0)]["tp_preserve_a1"] == pytest.approx(0.30, abs=1e-6)
        # Bucket 2 gate = 0.35 (overlap pushes to fp_upper since TP-1σ=0.35 too)
        assert per_range[(10.0, 20.0)]["tp_preserve_a1"] == pytest.approx(0.35, abs=1e-6)

    def test_analyze_detector_emits_curve_fit(self, synthetic_calibration_csv, script):
        r = script.analyze_detector("synth_detector")
        fit = r["fits"]["tp_preserve_a1"]
        # Two range bins → linear is well-defined.
        assert fit["n_range_bins"] == 2
        # gate(5) ≈ 0.30, gate(15) ≈ 0.35 → linear slope ≈ +0.005
        a, b = fit["lin_a"], fit["lin_b"]
        # lin: at d=5, 0.005·5 + b ≈ 0.30 → b ≈ 0.275
        assert a == pytest.approx(0.005, abs=1e-3)
        # Quadratic with only 2 distinct x's reduces to linear; the script
        # falls back gracefully.
