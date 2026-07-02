"""Regression test for ``ErrorModel.sample_all_errors`` / ``sample_errors``.

The error-model refactor (commit afacc61) dropped the aggregate samplers that
the SUMO error-injection path (`sensor.py:create_detected_bounding_boxes`)
depends on, leaving only the per-axis shims. The sweeps crashed with
``AttributeError: 'ErrorModel' object has no attribute 'sample_all_errors'``
(masked downstream as a TraCI "peer shutdown"). The only test that exercised it,
``test_sensing_errors_gauntlet``, is module-skipped (old-API import), so nothing
caught it. This pins the restored aggregate contract across every GPEM mode.

Run: PYTHONPATH=src python -m pytest tests/test_error_model_sample_all.py -v
"""
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent

pytest.importorskip("error_model")
from error_model import ErrorModel  # noqa: E402

# The simulator's three detectors (validated end-to-end on fast_city).
SIM_DETECTORS = ["detr3d", "bev_fusion", "centerpoint"]
MODES = ["static", "linear", "quadratic", "polar"]

# Keys sensor.py + the gauntlet test consume from the returned dict.
REQUIRED_KEYS = {
    "x_error", "y_error", "width_error", "length_error", "yaw_error",
    "distal_std", "perp_std", "width_std", "length_std", "yaw_std",
}


def _available(name):
    return (REPO / "src" / "data" / "sensor_models" / f"{name}.csv").exists()


@pytest.mark.parametrize("name", SIM_DETECTORS)
@pytest.mark.parametrize("mode", MODES)
def test_sample_all_errors_contract(name, mode):
    if not _available(name):
        pytest.skip(f"{name}.csv not present")
    em = ErrorModel(name, mode=mode, max_range=100.0)
    out = em.sample_all_errors(40.0, math.radians(30.0))
    assert REQUIRED_KEYS <= set(out), f"missing keys: {REQUIRED_KEYS - set(out)}"
    assert all(np.isfinite(v) for v in out.values()), out
    # predicted stds must be non-negative
    for k in ("distal_std", "perp_std", "width_std", "length_std", "yaw_std"):
        assert out[k] >= 0.0, (k, out[k])


@pytest.mark.parametrize("name", SIM_DETECTORS)
def test_sample_errors_rotation_matches_stds(name):
    """sample_errors returns (x, y, distal_std, perp_std); the stds must equal the
    direct get_*_std calls (same distance/angle), i.e. no drift between the
    sampled-position helper and the covariance path."""
    if not _available(name):
        pytest.skip(f"{name}.csv not present")
    em = ErrorModel(name, mode="polar", max_range=100.0)
    ang = math.radians(45.0)
    _, _, distal_std, perp_std = em.sample_errors(30.0, ang)
    assert distal_std == pytest.approx(em.get_distal_std(30.0, math.degrees(ang)))
    assert perp_std == pytest.approx(em.get_perpendicular_std(30.0, math.degrees(ang)))


def test_polar_uses_fitted_coefficients_not_bins():
    """GPEM polar must be the smooth fit: predict_std_polar (coefficients), never
    a bin lookup. Pin that the sim detectors carry populated polar_std coeffs so
    mode='polar' resolves through the fit."""
    if not _available("detr3d"):
        pytest.skip("detr3d.csv not present")
    em = ErrorModel("detr3d", mode="polar", max_range=100.0)
    # Two nearby angles should give a smooth (continuous) change, unlike bin steps.
    s1 = em.get_perpendicular_std(40.0, 10.0)
    s2 = em.get_perpendicular_std(40.0, 12.0)
    assert s1 > 0 and s2 > 0
    assert abs(s1 - s2) < 0.05, "polar std should vary smoothly with angle (fit, not bins)"
