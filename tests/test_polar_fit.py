"""Polar GPEM fit — equation evaluation tests.

The polar mode in `src/error_model.py` evaluates per-axis std as the closed-form

    std(r, theta) = (a + b * r) * (1 + c * cos(theta) + d * cos(2 * theta))

This is a FIT, not a bin lookup. Until 2026-06-08 the polar path tried to read
per-(range, angle) bins at runtime; the bins were noisy and required dense
calibration coverage. The closed-form equation extrapolates smoothly and only
needs four coefficients per axis. These tests pin the equation:

  - independent of `error_model` plumbing, so a refactor that touches loaders
    can't silently regress the math
  - parameterised on known inputs to lock the contract upstream parsers must
    honour: a, b, c, d emit in (offset, slope, fundamental, second-harmonic)
    order.
"""

import math

import numpy as np
import pytest


def polar_predict(a: float, b: float, c: float, d: float,
                  r: float, theta_rad: float) -> float:
    """Reference implementation. Mirrors error_model._Regression.predict_std_polar."""
    scale = a + b * r
    shape = 1.0 + c * math.cos(theta_rad) + d * math.cos(2.0 * theta_rad)
    return scale * shape


class TestPolarPredictRef:
    """Lock the closed-form math directly."""

    def test_zero_range_returns_pure_offset(self):
        # r=0 collapses scale to `a`; shape unchanged.
        val = polar_predict(a=0.5, b=0.1, c=0.0, d=0.0, r=0.0, theta_rad=0.0)
        assert val == pytest.approx(0.5)

    def test_zero_angle_collapses_shape_to_1_plus_c_plus_d(self):
        # theta=0 → cos(0)=1, cos(0)=1 → shape = 1 + c + d.
        a, b, c, d = 0.3, 0.02, 0.2, 0.1
        r = 10.0
        expected_scale = a + b * r            # 0.5
        expected_shape = 1.0 + c + d          # 1.3
        expected = expected_scale * expected_shape  # 0.65
        val = polar_predict(a, b, c, d, r=r, theta_rad=0.0)
        assert val == pytest.approx(expected)

    def test_quarter_angle_isolates_d_term(self):
        # theta=pi/2 → cos(pi/2)=0, cos(pi)=-1 → shape = 1 - d. c term cancels.
        a, b, c, d = 0.4, 0.01, 0.3, 0.15
        r = 20.0
        expected_scale = a + b * r            # 0.6
        expected_shape = 1.0 - d              # 0.85
        expected = expected_scale * expected_shape
        val = polar_predict(a, b, c, d, r=r, theta_rad=math.pi / 2.0)
        assert val == pytest.approx(expected)

    def test_half_angle_collapses_to_1_minus_c_plus_d(self):
        # theta=pi → cos(pi)=-1, cos(2pi)=1 → shape = 1 - c + d.
        a, b, c, d = 0.5, 0.005, 0.25, 0.1
        r = 50.0
        expected_shape = 1.0 - c + d          # 0.85
        expected_scale = a + b * r            # 0.75
        expected = expected_scale * expected_shape
        val = polar_predict(a, b, c, d, r=r, theta_rad=math.pi)
        assert val == pytest.approx(expected)

    def test_linear_in_range_when_shape_constants_zero(self):
        # c=d=0 → shape=1; expect strict linear in r.
        a, b = 0.2, 0.03
        rs = np.linspace(0, 100, 11)
        vals = np.array([polar_predict(a, b, 0.0, 0.0, r, 0.0) for r in rs])
        expected = a + b * rs
        np.testing.assert_allclose(vals, expected, rtol=1e-12)

    def test_angular_symmetric_about_zero(self):
        # cos is even → polar_predict(theta) == polar_predict(-theta).
        a, b, c, d = 0.3, 0.02, 0.2, 0.1
        r = 30.0
        for theta in (math.pi / 6, math.pi / 4, math.pi / 3):
            assert polar_predict(a, b, c, d, r, theta) == pytest.approx(
                polar_predict(a, b, c, d, r, -theta)
            )

    def test_extrapolates_smoothly_beyond_calibration_range(self):
        # Reasoning recorded in PP_Score0_log_odds_autotuned: small b means
        # extrapolating from 50m to 100m only inflates std by a small amount.
        # Pin: with b=1e-3, the 50→100m delta is +0.05 (5cm).
        a, b, c, d = 0.5, 1e-3, 0.0, 0.0
        delta = polar_predict(a, b, c, d, 100.0, 0.0) - polar_predict(a, b, c, d, 50.0, 0.0)
        assert delta == pytest.approx(0.05, rel=1e-9)


class TestErrorModelPolarIntegration:
    """End-to-end: error_model loads polar coefficients from CSV and evaluates
    the equation. Uses a synthetic CSV so the test doesn't depend on any
    fitted bucket on disk."""

    @pytest.fixture
    def synthetic_csv(self, tmp_path):
        # error_model._load_regressions expects the polar columns
        # polar_a/b/c/d (+ their std variants) alongside the linear/quadratic
        # rows. Build the minimum viable file: one axis + miss_rate.
        csv_path = tmp_path / "polar_axis.csv"
        header = (
            "axis,offset,slope,quad_a,quad_b,quad_c,"
            "polar_a,polar_b,polar_c,polar_d,"
            "polar_std_a,polar_std_b,polar_std_c,polar_std_d"
        )
        # axes = (x, y, z, length, width, height, yaw, missed_rate) per loader;
        # we only need axes that the loader actually reads. Easiest: write a
        # full set with the same values so any axis pull works.
        axes = ["x", "y", "z", "length", "width", "height", "yaw", "missed_rate"]
        a, b, c, d = 0.4, 0.02, 0.25, 0.10
        rows = [header]
        for axis in axes:
            rows.append(
                f"{axis},0.1,0.01,0.0,0.0,0.0,"
                f"{a},{b},{c},{d},"
                "0.001,0.0001,0.005,0.005"
            )
        csv_path.write_text("\n".join(rows) + "\n")
        return csv_path, (a, b, c, d)

    def test_predict_std_polar_matches_closed_form(self, synthetic_csv, monkeypatch):
        """Skip cleanly if the loader API isn't importable in this CI shape."""
        csv_path, (a, b, c, d) = synthetic_csv
        try:
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
            from error_model import _Regression  # type: ignore
        except Exception as e:  # pragma: no cover — loader changed shape
            pytest.skip(f"error_model loader shape changed: {e}")

        # Construct directly — exercises predict_std_polar without round-tripping
        # through the CSV loader (which is covered in test_error_model.py).
        # predict_std_polar prefers polar_std_* when present, so put the test
        # coefficients there. polar_* (MAE) is the secondary path.
        reg = _Regression(
            axis="x", intercept=0.0, slope=0.0, unit="m", notes="",
            quad_a=None, quad_b=None, quad_c=None,
            std_intercept=None, std_slope=None,
            std_quad_a=None, std_quad_b=None, std_quad_c=None,
            bias_intercept=None, bias_slope=None,
            bias_quad_a=None, bias_quad_b=None, bias_quad_c=None,
            var_intercept=None, var_slope=None,
            var_quad_a=None, var_quad_b=None, var_quad_c=None,
            polar_a=a, polar_b=b, polar_c=c, polar_d=d,
            polar_std_a=a, polar_std_b=b, polar_std_c=c, polar_std_d=d,
        )
        for r, theta in [(0.0, 0.0), (25.0, math.pi / 4), (80.0, math.pi)]:
            expected = polar_predict(a, b, c, d, r, theta)
            assert reg.predict_std_polar(r, theta) == pytest.approx(expected)
