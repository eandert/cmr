"""Unit tests for the per-track vertical (z, height) estimator.

CMR fuses in 2D BEV; z (gravity-center) and box-height ride alongside the planar
state as a lightweight EMA on each track (sensor_fusion._blend_vertical), fed by
the matched detections' z/height. These guard (a) the EMA math, (b) that the
field survives the ingest → cov-mode-helper → measurement plumbing, and (c) that
absent vertical data degrades to None (B3 then falls back to re-attach) — so the
2D BEV fusion/metrics are provably unaffected.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sensor_fusion as sf           # noqa: E402  (import first: breaks the sensor<->sensor_fusion cycle)
import sensor as cmr_sensor          # noqa: E402
import traci_interface as ti         # noqa: E402


def test_blend_vertical_ema():
    assert sf._blend_vertical(None, 1.5) == 1.5            # first obs initialises
    assert sf._blend_vertical(2.0, None) == 2.0            # missing obs -> unchanged
    assert sf._blend_vertical(None, None) is None          # nothing known yet
    # EMA with the documented gain.
    g = sf.VERTICAL_EMA_GAIN
    assert abs(sf._blend_vertical(1.0, 3.0) - (1.0 + g * (3.0 - 1.0))) < 1e-12
    # Repeated identical obs converge to that value.
    z = None
    for _ in range(50):
        z = sf._blend_vertical(z, 20.0)
    assert abs(z - 20.0) < 1e-6


def test_as_float_or_none():
    assert sf._as_float_or_none(None) is None
    assert sf._as_float_or_none(float("nan")) is None
    assert sf._as_float_or_none("nope") is None
    assert sf._as_float_or_none(np.float64(1.5)) == 1.5


def _raw(z=20.1, height=1.6):
    return cmr_sensor.DetectedObject(
        vehicle_id="veh", vehicle_type="car", detected_bbox=None,
        centroid=[3.0, 4.0], width=1.8, length=4.3, angle=0.1,
        expected_error_gaussian=None, velocity_vector=[0.0, 0.0],
        z=z, height=height,
    )


def test_detected_object_carries_z_height():
    d = _raw()
    assert d.z == 20.1 and d.height == 1.6
    # Default (detector without vertical info) is None, not 0.
    d0 = cmr_sensor.DetectedObject("v", "car", None, [0, 0], 2, 4, 0, None,
                                   velocity_vector=[0, 0])
    assert d0.z is None and d0.height is None


def test_cov_helpers_propagate_z_height():
    """The per-cov-mode rebuild must carry z/height to the measurement the filter sees."""
    flat = ti._detection_with_flat_covariance(_raw(), np.eye(2) * 0.5)
    assert flat.z == 20.1 and flat.height == 1.6
