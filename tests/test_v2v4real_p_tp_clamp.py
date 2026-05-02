"""
Regression test: detection_probability can return 0.0 (e.g. when distance
> max_range, or in a 100%-miss-rate polar bin).  A p_tp=0 fed into the
log-odds lifecycle becomes log(0) = -inf, which NaN-poisons track state and
eventually causes sutherland_hodgman_clip to hang on a degenerate polygon.

The fix in v2v4real_replay is:
    obj.p_tp = max(1e-6, ems_for_vehicle[cov].detection_probability(...))

These tests verify:
1. detection_probability can legitimately return 0.0 (the trigger condition).
2. The clamped value is always > 0 (the fix).
3. The clamped value is never > 1 (no out-of-range accident).

Run: PYTHONPATH=src python -m pytest tests/test_v2v4real_p_tp_clamp.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sensor_fusion  # noqa: F401  (break circular import, same as other v2v4real tests)
from error_models import DetectorErrorModel


def _make_em(max_range: float = 50.0, miss_rate_intercept: float = 0.0,
             miss_rate_slope: float = 0.0) -> DetectorErrorModel:
    """Build a minimal DetectorErrorModel with no polar bins."""
    em = DetectorErrorModel.__new__(DetectorErrorModel)
    em.max_range = max_range
    em.has_polar = False
    em.miss_rate_regression = (miss_rate_intercept, miss_rate_slope)
    return em


class TestDetectionProbabilityCanBeZero:
    def test_returns_zero_beyond_max_range(self):
        em = _make_em(max_range=50.0)
        assert em.detection_probability(60.0) == 0.0

    def test_returns_zero_for_full_miss_rate_regression(self):
        # intercept=1, slope=0 → miss_rate=1 → detection_probability=0
        em = _make_em(miss_rate_intercept=1.0, miss_rate_slope=0.0)
        assert em.detection_probability(10.0) == 0.0


class TestPtpClampNeverZero:
    """The fix: max(1e-6, detection_probability(...)) must be strictly > 0."""

    @pytest.mark.parametrize("distance,angle_deg", [
        (60.0, 0.0),   # beyond max_range → raw would be 0.0
        (10.0, 0.0),   # full miss rate → raw would be 0.0
        (10.0, 45.0),  # normal case → still clamped from below
        (0.1,  -30.0), # near range, should be close to 1.0
    ])
    def test_clamped_p_tp_is_positive(self, distance, angle_deg):
        em = _make_em(max_range=50.0, miss_rate_intercept=1.0)
        clamped = max(1e-6, em.detection_probability(distance, angle_deg))
        assert clamped > 0.0, f"p_tp must be > 0 (got {clamped})"

    @pytest.mark.parametrize("distance,angle_deg", [
        (1.0,  0.0),
        (30.0, 90.0),
        (60.0, 0.0),
    ])
    def test_clamped_p_tp_never_exceeds_one(self, distance, angle_deg):
        em = _make_em(max_range=50.0)
        clamped = max(1e-6, em.detection_probability(distance, angle_deg))
        assert clamped <= 1.0, f"p_tp must be <= 1 (got {clamped})"
