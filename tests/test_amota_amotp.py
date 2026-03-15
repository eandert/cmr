"""
Tests for AMOTA and AMOTP metric implementations.
Tests the corrected formulas:
  MOTA = 1 - (FP + FN) / max(1, GT)
  MOTP = average Euclidean center distance of matched TP pairs (meters, lower is better)
"""
import sys
import os
import math
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from metrics.amota import (
    calculate_amota, calculate_amotp, _match_detections_to_gt, _expand_bbox,
    iou, polygon_area, convex_polygon_intersection_area
)


def make_bbox(cx, cy, w, h, angle=0.0):
    """Create an axis-aligned bounding box as 4 corner tuples."""
    hw, hh = w / 2, h / 2
    if angle == 0.0:
        return [
            (cx - hw, cy - hh),
            (cx + hw, cy - hh),
            (cx + hw, cy + hh),
            (cx - hw, cy + hh),
        ]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + cos_a * x - sin_a * y, cy + sin_a * x + cos_a * y) for x, y in corners]


class FakeGT:
    """Minimal ground truth object for testing."""
    def __init__(self, cx, cy, w=4.0, h=2.0):
        self.centroid = [cx, cy]
        self.bbox = make_bbox(cx, cy, w, h)
        self.width = w
        self.length = h
        self.angle = 0.0


class FakeDet:
    """Minimal detected object for testing."""
    def __init__(self, cx, cy, w=4.0, h=2.0):
        self.centroid = [cx, cy]
        self.detected_bbox = make_bbox(cx, cy, w, h)
        self.width = w
        self.length = h
        self.angle = 0.0


class TestIoU(unittest.TestCase):
    def test_identical_boxes(self):
        b = make_bbox(0, 0, 4, 2)
        self.assertAlmostEqual(iou(b, b), 1.0, places=5)

    def test_no_overlap(self):
        b1 = make_bbox(0, 0, 2, 2)
        b2 = make_bbox(100, 100, 2, 2)
        self.assertAlmostEqual(iou(b1, b2), 0.0, places=5)

    def test_partial_overlap(self):
        b1 = make_bbox(0, 0, 4, 4)
        b2 = make_bbox(2, 0, 4, 4)  # shifted right by 2, 50% overlap on x
        # intersection = 2x4 = 8, union = 16+16-8 = 24
        self.assertAlmostEqual(iou(b1, b2), 8.0 / 24.0, places=4)

    def test_contained_box(self):
        b1 = make_bbox(0, 0, 10, 10)
        b2 = make_bbox(0, 0, 2, 2)  # small box inside big box
        # intersection = 4, union = 100+4-4 = 100
        self.assertAlmostEqual(iou(b1, b2), 4.0 / 100.0, places=4)

    def test_none_inputs(self):
        self.assertEqual(iou(None, make_bbox(0, 0, 2, 2)), 0.0)
        self.assertEqual(iou(make_bbox(0, 0, 2, 2), None), 0.0)


class TestMatching(unittest.TestCase):
    def test_perfect_match(self):
        gts = [FakeGT(0, 0), FakeGT(10, 0)]
        dets = [FakeDet(0, 0), FakeDet(10, 0)]
        matches, fp, fn = _match_detections_to_gt(dets, gts, iou_threshold=0.5, bbox_tolerance=0.0)
        self.assertEqual(len(matches), 2)
        self.assertEqual(len(fp), 0)
        self.assertEqual(len(fn), 0)

    def test_no_detections(self):
        gts = [FakeGT(0, 0), FakeGT(10, 0)]
        matches, fp, fn = _match_detections_to_gt([], gts, iou_threshold=0.5, bbox_tolerance=0.0)
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(fp), 0)
        self.assertEqual(len(fn), 2)

    def test_extra_detections(self):
        gts = [FakeGT(0, 0)]
        dets = [FakeDet(0, 0), FakeDet(50, 50)]
        matches, fp, fn = _match_detections_to_gt(dets, gts, iou_threshold=0.5, bbox_tolerance=0.0)
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(fp), 1)  # one false positive
        self.assertEqual(len(fn), 0)

    def test_missed_gt(self):
        gts = [FakeGT(0, 0), FakeGT(50, 50)]
        dets = [FakeDet(0, 0)]
        matches, fp, fn = _match_detections_to_gt(dets, gts, iou_threshold=0.5, bbox_tolerance=0.0)
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(fp), 0)
        self.assertEqual(len(fn), 1)  # one false negative

    def test_below_iou_threshold(self):
        gts = [FakeGT(0, 0, w=4, h=2)]
        # Shift detection far enough that even with tolerance expansion, IoU is below threshold
        dets = [FakeDet(10, 0, w=4, h=2)]  # no overlap even with expanded boxes
        matches, fp, fn = _match_detections_to_gt(dets, gts, iou_threshold=0.1, bbox_tolerance=0.5)
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(fp), 1)
        self.assertEqual(len(fn), 1)

    def test_best_iou_matching(self):
        """Ensure greedy matching picks the best IoU, not the first match."""
        gt = [FakeGT(0, 0, w=4, h=4)]
        # det1 overlaps slightly, det2 overlaps perfectly
        det1 = FakeDet(1.5, 0, w=4, h=4)  # partial overlap
        det2 = FakeDet(0, 0, w=4, h=4)    # perfect overlap
        dets = [det1, det2]  # det1 is checked first
        matches, fp, fn = _match_detections_to_gt(dets, gt, iou_threshold=0.3, bbox_tolerance=0.0)
        # Both should try to match gt[0], but only one can win
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(fp), 1)
        self.assertEqual(len(fn), 0)


class TestAMOTA(unittest.TestCase):
    def test_perfect_detection(self):
        """All GT detected perfectly => MOTA = 1.0"""
        gts = [FakeGT(0, 0), FakeGT(10, 0), FakeGT(20, 0)]
        dets = [FakeDet(0, 0), FakeDet(10, 0), FakeDet(20, 0)]
        mota = calculate_amota(dets, gts)
        self.assertAlmostEqual(mota, 1.0, places=5)

    def test_no_detections(self):
        """No detections => FN = GT => MOTA = 1 - GT/GT = 0.0"""
        gts = [FakeGT(0, 0), FakeGT(10, 0)]
        mota = calculate_amota([], gts)
        self.assertAlmostEqual(mota, 0.0, places=5)

    def test_no_ground_truth(self):
        """No GT => returns None (undefined)"""
        result = calculate_amota([], [])
        self.assertIsNone(result)

    def test_all_false_positives(self):
        """Detections with no GT => returns None"""
        dets = [FakeDet(0, 0), FakeDet(10, 0)]
        result = calculate_amota(dets, [])
        self.assertIsNone(result)

    def test_negative_mota(self):
        """More FP than GT => MOTA goes negative"""
        gts = [FakeGT(0, 0)]  # 1 GT
        # 1 matched + 5 false positives far away => MOTA = 1 - 5/1 = -4.0
        dets = [FakeDet(0, 0), FakeDet(50, 50), FakeDet(60, 60),
                FakeDet(70, 70), FakeDet(80, 80), FakeDet(90, 90)]
        mota = calculate_amota(dets, gts)
        self.assertAlmostEqual(mota, -4.0, places=5)

    def test_half_detected(self):
        """2 of 4 GT detected, no FP => FN=2 => MOTA = 1 - 2/4 = 0.5"""
        gts = [FakeGT(0, 0), FakeGT(10, 0), FakeGT(50, 50), FakeGT(60, 60)]
        dets = [FakeDet(0, 0), FakeDet(10, 0)]
        mota = calculate_amota(dets, gts)
        self.assertAlmostEqual(mota, 0.5, places=5)

    def test_mota_zero_is_not_skipped(self):
        """MOTA = 0.0 is a valid score (FP+FN == GT), not a sentinel."""
        # 1 GT, 0 matches, 1 FP => FP=1, FN=1 => MOTA = 1 - 2/1 = -1.0
        # Actually for MOTA = 0: need FP+FN = GT. E.g., 2 GT, 0 matched, 0 FP => FN=2, MOTA = 1-2/2 = 0
        gts = [FakeGT(0, 0), FakeGT(50, 50)]
        dets = []  # no detections => FN=2, FP=0 => MOTA = 1 - 2/2 = 0.0
        mota = calculate_amota(dets, gts)
        self.assertAlmostEqual(mota, 0.0, places=5)
        self.assertIsNotNone(mota)  # Should NOT be None


class TestAMOTP(unittest.TestCase):
    def test_perfect_localization(self):
        """Exactly matched => distance = 0.0"""
        gts = [FakeGT(0, 0), FakeGT(10, 0)]
        dets = [FakeDet(0, 0), FakeDet(10, 0)]
        motp = calculate_amotp(dets, gts)
        self.assertAlmostEqual(motp, 0.0, places=5)

    def test_known_offset(self):
        """Detection shifted by known amount => MOTP = that distance."""
        gts = [FakeGT(0, 0, w=20, h=20)]
        dets = [FakeDet(0.6, 0.8, w=20, h=20)]  # distance = 1.0m, boxes still overlap
        motp = calculate_amotp(dets, gts)
        self.assertAlmostEqual(motp, 1.0, places=4)

    def test_larger_offset(self):
        """Detection shifted by known amount with large boxes => MOTP = that distance."""
        gts = [FakeGT(0, 0, w=20, h=20)]
        dets = [FakeDet(3, 4, w=20, h=20)]  # distance = 5.0, large boxes still overlap
        motp = calculate_amotp(dets, gts)
        self.assertAlmostEqual(motp, 5.0, places=4)

    def test_average_distance(self):
        """Two matches with different offsets => average distance."""
        gts = [FakeGT(0, 0, w=20, h=20), FakeGT(40, 0, w=20, h=20)]
        dets = [FakeDet(0.6, 0.8, w=20, h=20), FakeDet(40, 0, w=20, h=20)]
        # dist1 = 1.0, dist2 = 0.0 => avg = 0.5
        motp = calculate_amotp(dets, gts)
        self.assertAlmostEqual(motp, 0.5, places=4)

    def test_no_matches(self):
        """No IoU matches => returns 0.0"""
        gts = [FakeGT(0, 0)]
        dets = [FakeDet(100, 100)]  # far away, no overlap
        motp = calculate_amotp(dets, gts)
        self.assertAlmostEqual(motp, 0.0, places=5)

    def test_empty_inputs(self):
        self.assertAlmostEqual(calculate_amotp([], []), 0.0)
        self.assertAlmostEqual(calculate_amotp([], [FakeGT(0, 0)]), 0.0)
        self.assertAlmostEqual(calculate_amotp([FakeDet(0, 0)], []), 0.0)

    def test_gpem_improvement_visible(self):
        """
        Key test: AMOTP must distinguish between a detection at (0.5, 0) vs (1.0, 0)
        relative to GT at (0, 0). This is what GPEM improves — localization quality.
        The old AMOTP (precision) was blind to this difference.
        """
        gts = [FakeGT(0, 0, w=10, h=10)]
        det_close = [FakeDet(0.5, 0, w=10, h=10)]
        det_far = [FakeDet(1.0, 0, w=10, h=10)]

        motp_close = calculate_amotp(det_close, gts)
        motp_far = calculate_amotp(det_far, gts)

        self.assertAlmostEqual(motp_close, 0.5, places=4)
        self.assertAlmostEqual(motp_far, 1.0, places=4)
        self.assertLess(motp_close, motp_far)


class TestPolygonHelpers(unittest.TestCase):
    def test_square_area(self):
        square = [(0, 0), (2, 0), (2, 2), (0, 2)]
        self.assertAlmostEqual(polygon_area(square), 4.0, places=5)

    def test_intersection_identical(self):
        sq = [(0, 0), (2, 0), (2, 2), (0, 2)]
        self.assertAlmostEqual(convex_polygon_intersection_area(sq, sq), 4.0, places=4)

    def test_intersection_half_overlap(self):
        sq1 = [(0, 0), (2, 0), (2, 2), (0, 2)]
        sq2 = [(1, 0), (3, 0), (3, 2), (1, 2)]
        # overlap = 1x2 = 2
        self.assertAlmostEqual(convex_polygon_intersection_area(sq1, sq2), 2.0, places=4)

    def test_no_intersection(self):
        sq1 = [(0, 0), (1, 0), (1, 1), (0, 1)]
        sq2 = [(10, 10), (11, 10), (11, 11), (10, 11)]
        self.assertAlmostEqual(convex_polygon_intersection_area(sq1, sq2), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
