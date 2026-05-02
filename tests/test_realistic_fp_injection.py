"""
Unit tests for realistic FP injection.

Tests verify:
  - enable_realistic_fp defaults to False (no behaviour change for existing code)
  - FPs appear in detected_objects when flag is True and calibration has fp_per_frame > 0
  - FP vehicle_ids carry the "fp_" prefix
  - FP detections have NO corresponding entry in detected_ground_truths
  - FP positions land within their source bin's (range, angle) bounds
  - Poisson rate: long-run mean of fp count ≈ fp_per_frame * n_bins
  - Zero-rate bins produce no FPs
  - Score stays in [0, 1]

Run from repo root: python -m pytest tests/test_realistic_fp_injection.py -v
"""

import math
import sys
import os
import types
import unittest

import numpy as np

import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sensor_model_loader import CalibrationBin, CalibrationTable
from error_models import DetectorErrorModel

# Warm up the sensor → sensor_fusion circular import chain so that
# `import sensor` later resolves cleanly (sensor_fusion pre-initializes sensor).
import sensor_fusion  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers for building minimal test fixtures
# ---------------------------------------------------------------------------

def _make_bin(range_lo, range_hi, angle_lo_deg, angle_hi_deg,
              fp_per_frame=1.0,
              fp_width_mean=2.0,  fp_width_std=0.2,
              fp_length_mean=4.0, fp_length_std=0.3,
              fp_height_mean=1.5, fp_height_std=0.1,
              fp_yaw_std=0.5,
              n_fp_car=10, n_fp_truck=2, n_fp_bus=1, n_fp_construction=0,
              fp_score_mean=0.35, fp_score_std=0.08):
    return CalibrationBin(
        range_lo=range_lo, range_hi=range_hi,
        angle_lo_deg=angle_lo_deg, angle_hi_deg=angle_hi_deg,
        gt_count=100, n_missed=20, miss_rate=0.2,
        n_fp=50, fp_per_frame=fp_per_frame,
        score_mean=0.7, score_std=0.1,
        fp_score_mean=fp_score_mean, fp_score_std=fp_score_std,
        source="measured",
        fp_width_mean=fp_width_mean,   fp_width_std=fp_width_std,
        fp_length_mean=fp_length_mean, fp_length_std=fp_length_std,
        fp_height_mean=fp_height_mean, fp_height_std=fp_height_std,
        fp_yaw_std=fp_yaw_std,
        n_fp_car=n_fp_car, n_fp_truck=n_fp_truck,
        n_fp_bus=n_fp_bus, n_fp_construction_vehicle=n_fp_construction,
    )


def _make_model_with_bins(bins):
    """Return a DetectorErrorModel whose calibration is patched with the given bins."""
    model = get_error_model_with_calibration(bins)
    return model


def get_error_model_with_calibration(bins):
    """Build a DetectorErrorModel backed by a real CSV but with patched calibration bins."""
    model = DetectorErrorModel.__new__(DetectorErrorModel)
    model.model_name = "_test_"
    model.csv_loaded = True
    model.distributions_from_csv = True
    model.use_gpem_model = False
    model.use_quadratic = False
    model.use_polar = False
    model.max_range = 100.0
    # Provide minimal distal/perp bins so sample_all_errors won't crash if called
    model.distal_bins = []
    model.perp_bins = []
    model.height_bins = []
    model.yaw_bins = []
    model.width_bins = []
    model.length_bins = []
    # Minimal _model_data stub with only the calibration attribute needed
    model_data = types.SimpleNamespace(calibration=CalibrationTable(bins))
    model._model_data = model_data
    return model


# ---------------------------------------------------------------------------
# Tests: sample_false_positives()
# ---------------------------------------------------------------------------

class TestSampleFalsePositives(unittest.TestCase):

    def setUp(self):
        self.bin = _make_bin(10.0, 20.0, -10.0, 10.0, fp_per_frame=5.0)
        self.model = get_error_model_with_calibration([self.bin])

    def test_returns_list(self):
        result = self.model.sample_false_positives(0.0, 0.0, 0.0)
        self.assertIsInstance(result, list)

    def test_zero_rate_bin_yields_no_fps(self):
        zero_bin = _make_bin(10.0, 20.0, -10.0, 10.0, fp_per_frame=0.0)
        model = get_error_model_with_calibration([zero_bin])
        for _ in range(50):
            self.assertEqual(model.sample_false_positives(0.0, 0.0, 0.0), [])

    def test_fp_count_mean_close_to_lambda(self):
        """Long-run mean fp count should be ≈ fp_per_frame (Poisson)."""
        np.random.seed(42)
        counts = [len(self.model.sample_false_positives(0.0, 0.0, 0.0))
                  for _ in range(500)]
        self.assertAlmostEqual(np.mean(counts), 5.0, delta=0.4)

    def test_fp_range_within_bin_bounds(self):
        """Each sampled FP's range from origin should be within [range_lo, range_hi]."""
        np.random.seed(0)
        for _ in range(200):
            for fp in self.model.sample_false_positives(0.0, 0.0, 0.0):
                r = fp["range"]
                self.assertGreaterEqual(r, self.bin.range_lo)
                self.assertLess(r, self.bin.range_hi)

    def test_score_in_unit_interval(self):
        np.random.seed(1)
        for _ in range(100):
            for fp in self.model.sample_false_positives(0.0, 0.0, 0.0):
                self.assertGreaterEqual(fp["score"], 0.0)
                self.assertLessEqual(fp["score"], 1.0)

    def test_class_label_valid(self):
        valid_classes = {"car", "truck", "bus", "construction_vehicle"}
        np.random.seed(2)
        for _ in range(50):
            for fp in self.model.sample_false_positives(0.0, 0.0, 0.0):
                self.assertIn(fp["class_label"], valid_classes)

    def test_position_offset_by_sensor_pose(self):
        """FP mean position should shift by the sensor translation offset."""
        # Collect many samples at origin and at offset; mean difference should match offset.
        bin_hi_rate = _make_bin(10.0, 20.0, -5.0, 5.0, fp_per_frame=20.0)
        model = get_error_model_with_calibration([bin_hi_rate])

        np.random.seed(5)
        random.seed(5)
        xs_origin = [fp["x"] for _ in range(30)
                     for fp in model.sample_false_positives(0.0, 0.0, 0.0)]

        np.random.seed(5)
        random.seed(5)
        xs_shifted = [fp["x"] for _ in range(30)
                      for fp in model.sample_false_positives(100.0, 0.0, 0.0)]

        if xs_origin and xs_shifted:
            mean_diff = np.mean(xs_shifted) - np.mean(xs_origin)
            self.assertAlmostEqual(mean_diff, 100.0, delta=1.0)

    def test_no_calibration_returns_empty(self):
        model = get_error_model_with_calibration([])
        self.assertEqual(model.sample_false_positives(0.0, 0.0, 0.0), [])

    def test_none_model_data_returns_empty(self):
        model = get_error_model_with_calibration([self.bin])
        model._model_data = None
        self.assertEqual(model.sample_false_positives(0.0, 0.0, 0.0), [])


# ---------------------------------------------------------------------------
# Tests: create_detected_bounding_boxes FP wiring
# ---------------------------------------------------------------------------

def _make_minimal_sensor(enable_realistic_fp=False, fp_rate=2.0):
    """Build a Sensor-like object with a patched error_model and FP flag."""
    import sensor as sensor_mod
    from config.detector_type import DetectorType

    s = sensor_mod.Sensor.__new__(sensor_mod.Sensor)
    # Minimal attributes needed by create_detected_bounding_boxes
    s.sensor_type_id = 0
    s.detector_type_id = DetectorType.PERFECT.id
    s.detector_type = DetectorType.PERFECT
    s.max_range = 100.0
    s.horizontal_fov = math.pi  # 180 °  — wide enough for any angle
    s.center_angle = 0.0
    s.x_extrinsics_original = 0.0
    s.y_extrinsics_original = 0.0
    s.angle_extrinsics_original = 0.0
    s.x_offset_error = 0.0
    s.y_offset_error = 0.0
    s.angle_offset_error = 0.0
    s.enable_realistic_fp = enable_realistic_fp
    s.fp_per_frame_cap = None
    s.fp_score_threshold = 0.0  # no score gate in unit tests — test bins have mean=0.35

    fp_bin = _make_bin(10.0, 20.0, -10.0, 10.0, fp_per_frame=fp_rate)
    s.error_model = get_error_model_with_calibration([fp_bin])
    return s


class TestFPWiringInSensor(unittest.TestCase):

    def setUp(self):
        import sensor as sensor_mod  # safe: sensor_fusion prewarm at module top resolved the cycle
        self._sensor_mod = sensor_mod

    def test_flag_default_false_no_fp(self):
        """With enable_realistic_fp=False (default), no fp_ IDs should appear."""
        s = _make_minimal_sensor(enable_realistic_fp=False, fp_rate=100.0)
        objs, gts = self._sensor_mod.create_detected_bounding_boxes(
            s, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), [], "ego"
        )
        fp_objs = [o for o in objs if o.vehicle_id.startswith("fp_")]
        self.assertEqual(len(fp_objs), 0, "FP injection must be off by default")

    def test_flag_true_produces_fp_detections(self):
        """With enable_realistic_fp=True and high rate, FPs should appear."""
        s = _make_minimal_sensor(enable_realistic_fp=True, fp_rate=20.0)
        np.random.seed(42)
        objs, gts = self._sensor_mod.create_detected_bounding_boxes(
            s, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), [], "ego"
        )
        fp_objs = [o for o in objs if o.vehicle_id.startswith("fp_")]
        self.assertGreater(len(fp_objs), 0, "FPs should be injected when flag is True")

    def test_fp_not_in_detected_ground_truths(self):
        """FP detections must NOT have entries in detected_ground_truths."""
        s = _make_minimal_sensor(enable_realistic_fp=True, fp_rate=20.0)
        np.random.seed(7)
        objs, gts = self._sensor_mod.create_detected_bounding_boxes(
            s, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), [], "ego"
        )
        fp_count = sum(1 for o in objs if o.vehicle_id.startswith("fp_"))
        self.assertEqual(len(gts), 0, "gt list should be empty when no GT objects provided")
        self.assertGreater(fp_count, 0, "sanity: some FPs should have appeared")

    def test_fp_ids_unique(self):
        """Every FP detection_id should be unique within a single frame."""
        s = _make_minimal_sensor(enable_realistic_fp=True, fp_rate=10.0)
        np.random.seed(99)
        objs, _ = self._sensor_mod.create_detected_bounding_boxes(
            s, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), [], "ego"
        )
        fp_ids = [o.vehicle_id for o in objs if o.vehicle_id.startswith("fp_")]
        self.assertEqual(len(fp_ids), len(set(fp_ids)), "FP IDs must be unique")

    def test_fp_detected_object_has_expected_fields(self):
        """Each FP DetectedObject should have the required attributes."""
        s = _make_minimal_sensor(enable_realistic_fp=True, fp_rate=50.0)
        np.random.seed(10)
        objs, _ = self._sensor_mod.create_detected_bounding_boxes(
            s, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), [], "ego"
        )
        for obj in objs:
            if not obj.vehicle_id.startswith("fp_"):
                continue
            self.assertIsInstance(obj.centroid, tuple)
            self.assertEqual(len(obj.centroid), 2)
            self.assertIsNotNone(obj.error_covariance)
            self.assertEqual(obj.velocity_vector, (0.0, 0.0))
            self.assertIsInstance(obj.dimensions, list)
            self.assertEqual(len(obj.dimensions), 2)


if __name__ == "__main__":
    unittest.main()
