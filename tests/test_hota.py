"""
Tests for the HOTA (Higher Order Tracking Accuracy) metric.

Run: python -m pytest tests/test_hota.py -v -s
"""

import unittest
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from metrics.hota import HotaAccumulator


class MockDetection:
    """Minimal detection object for testing."""
    def __init__(self, vehicle_id, centroid, bbox=None, width=4.0, length=2.0, angle=0.0):
        self.vehicle_id = vehicle_id
        self.centroid = centroid
        self.detected_bbox = bbox or self._make_bbox(centroid, width, length, angle)
        self.dimensions = [width, length]
        self.width = width
        self.length = length
        self.angle = angle

    def _make_bbox(self, centroid, w, l, angle):
        cx, cy = centroid
        hw, hl = w / 2, l / 2
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        corners = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        return [(cx + cos_a * x - sin_a * y, cy + sin_a * x + cos_a * y) for x, y in corners]


class MockGroundTruth:
    """Minimal ground truth object for testing."""
    def __init__(self, vehicle_id, centroid, bbox=None, width=4.0, length=2.0, angle=0.0):
        self.vehicle_id = vehicle_id
        self.centroid = centroid
        self.bbox = bbox or self._make_bbox(centroid, width, length, angle)
        self.dimensions = [width, length]
        self.width = width
        self.length = length
        self.angle = angle

    def _make_bbox(self, centroid, w, l, angle):
        cx, cy = centroid
        hw, hl = w / 2, l / 2
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        corners = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        return [(cx + cos_a * x - sin_a * y, cy + sin_a * x + cos_a * y) for x, y in corners]


class TestHotaAccumulatorBasic(unittest.TestCase):
    """Test basic HOTA computation."""

    def test_empty_accumulator(self):
        acc = HotaAccumulator()
        result = acc.compute()
        self.assertEqual(result['hota'], 0.0)
        self.assertEqual(result['deta'], 0.0)
        self.assertEqual(result['assa'], 0.0)

    def test_perfect_tracking(self):
        """Perfect 1:1 matching across all frames should give HOTA=1."""
        acc = HotaAccumulator()
        for t in range(10):
            dets = [MockDetection(1, (10.0 + t, 20.0)), MockDetection(2, (30.0 + t, 40.0))]
            gts = [MockGroundTruth("A", (10.0 + t, 20.0)), MockGroundTruth("B", (30.0 + t, 40.0))]
            acc.update(dets, gts)
        result = acc.compute()
        self.assertAlmostEqual(result['deta'], 1.0, places=2)
        self.assertAlmostEqual(result['assa'], 1.0, places=2)
        self.assertAlmostEqual(result['hota'], 1.0, places=2)

    def test_no_detections(self):
        """All FN, no detections."""
        acc = HotaAccumulator()
        for t in range(5):
            gts = [MockGroundTruth("A", (10.0, 20.0))]
            acc.update([], gts)
        result = acc.compute()
        self.assertEqual(result['deta'], 0.0)
        self.assertEqual(result['assa'], 0.0)
        self.assertEqual(result['hota'], 0.0)

    def test_all_false_positives(self):
        """Detections with no GT — all FP."""
        acc = HotaAccumulator()
        for t in range(5):
            dets = [MockDetection(1, (10.0, 20.0))]
            acc.update(dets, [])
        result = acc.compute()
        self.assertEqual(result['deta'], 0.0)
        self.assertEqual(result['hota'], 0.0)


class TestHotaAccumulatorAssociation(unittest.TestCase):
    """Test that AssA correctly measures identity consistency."""

    def test_id_switch_reduces_assa(self):
        """An ID switch mid-sequence should reduce AssA below 1.0."""
        acc_perfect = HotaAccumulator()
        acc_switch = HotaAccumulator()

        # Perfect: det 1 -> gt A, det 2 -> gt B for all frames
        for t in range(10):
            dets = [MockDetection(1, (10.0, 20.0)), MockDetection(2, (30.0, 40.0))]
            gts = [MockGroundTruth("A", (10.0, 20.0)), MockGroundTruth("B", (30.0, 40.0))]
            acc_perfect.update(dets, gts)

        # Switch: det 1 -> gt A for frames 0-4, then det 1 -> gt B for frames 5-9
        for t in range(10):
            if t < 5:
                dets = [MockDetection(1, (10.0, 20.0)), MockDetection(2, (30.0, 40.0))]
                gts = [MockGroundTruth("A", (10.0, 20.0)), MockGroundTruth("B", (30.0, 40.0))]
            else:
                # Swap positions so det 1 matches gt B and det 2 matches gt A
                dets = [MockDetection(1, (30.0, 40.0)), MockDetection(2, (10.0, 20.0))]
                gts = [MockGroundTruth("A", (10.0, 20.0)), MockGroundTruth("B", (30.0, 40.0))]
            acc_switch.update(dets, gts)

        result_perfect = acc_perfect.compute()
        result_switch = acc_switch.compute()

        # DetA should be the same (all matched in both cases)
        self.assertAlmostEqual(result_perfect['deta'], result_switch['deta'], places=2)
        # AssA should be lower with the switch
        self.assertGreater(result_perfect['assa'], result_switch['assa'])
        # HOTA should be lower with the switch
        self.assertGreater(result_perfect['hota'], result_switch['hota'])

    def test_intermittent_detection_reduces_assa(self):
        """A track that disappears for some frames should have lower AssA."""
        acc = HotaAccumulator()
        for t in range(10):
            gts = [MockGroundTruth("A", (10.0, 20.0))]
            if t % 2 == 0:
                dets = [MockDetection(1, (10.0, 20.0))]
            else:
                dets = []
            acc.update(dets, gts)
        result = acc.compute()
        # DetA < 1 because of misses
        self.assertLess(result['deta'], 1.0)
        # AssA should still be decent since det 1 always matches gt A when present
        self.assertGreater(result['assa'], 0.3)
        # But not perfect because of the gaps
        self.assertLess(result['assa'], 1.0)


class TestHotaDecomposition(unittest.TestCase):
    """Test that HOTA = sqrt(DetA * AssA)."""

    def test_hota_is_geometric_mean(self):
        acc = HotaAccumulator()
        for t in range(10):
            if t < 5:
                dets = [MockDetection(1, (10.0, 20.0)), MockDetection(2, (30.0, 40.0))]
                gts = [MockGroundTruth("A", (10.0, 20.0)), MockGroundTruth("B", (30.0, 40.0))]
            else:
                dets = [MockDetection(1, (30.0, 40.0)), MockDetection(2, (10.0, 20.0))]
                gts = [MockGroundTruth("A", (10.0, 20.0)), MockGroundTruth("B", (30.0, 40.0))]
            acc.update(dets, gts)
        result = acc.compute()
        expected_hota = math.sqrt(result['deta'] * result['assa'])
        self.assertAlmostEqual(result['hota'], expected_hota, places=6)

    def test_reset(self):
        acc = HotaAccumulator()
        acc.update([MockDetection(1, (10.0, 20.0))], [MockGroundTruth("A", (10.0, 20.0))])
        acc.reset()
        result = acc.compute()
        self.assertEqual(result['hota'], 0.0)


if __name__ == "__main__":
    unittest.main()
