"""Tests for fast vectorized polygon geometry operations."""

import sys
import os
import numpy as np
import math
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from geometry import (
    polygon_area_signed, polygon_area, ensure_ccw,
    sutherland_hodgman_clip, get_rotated_box_corners, rotated_box_iou,
)


class TestPolygonArea(unittest.TestCase):

    def test_unit_square_ccw(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        self.assertAlmostEqual(polygon_area(sq), 1.0, places=10)
        self.assertAlmostEqual(polygon_area_signed(sq), 1.0, places=10)

    def test_unit_square_cw(self):
        sq = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        self.assertAlmostEqual(polygon_area(sq), 1.0, places=10)
        self.assertAlmostEqual(polygon_area_signed(sq), -1.0, places=10)

    def test_triangle(self):
        tri = np.array([[0, 0], [4, 0], [0, 3]], dtype=float)
        self.assertAlmostEqual(polygon_area(tri), 6.0, places=10)

    def test_empty(self):
        self.assertEqual(polygon_area(np.array([])), 0.0)
        self.assertEqual(polygon_area(np.array([[1, 2]])), 0.0)
        self.assertEqual(polygon_area(np.array([[1, 2], [3, 4]])), 0.0)

    def test_large_rectangle(self):
        rect = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=float)
        self.assertAlmostEqual(polygon_area(rect), 5000.0, places=5)

    def test_pentagon(self):
        # Regular pentagon with known area
        angles = np.linspace(0, 2 * np.pi, 6)[:-1]  # 5 vertices
        r = 10.0
        verts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        expected = 0.5 * 5 * r**2 * np.sin(2 * np.pi / 5)
        self.assertAlmostEqual(polygon_area(verts), expected, places=5)


class TestEnsureCCW(unittest.TestCase):

    def test_already_ccw(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        result = ensure_ccw(sq)
        np.testing.assert_array_almost_equal(result, sq)

    def test_cw_to_ccw(self):
        cw = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        result = ensure_ccw(cw)
        self.assertGreater(polygon_area_signed(result), 0)

    def test_empty(self):
        result = ensure_ccw(np.array([]))
        self.assertEqual(len(result), 0)

    def test_two_points(self):
        result = ensure_ccw(np.array([[0, 0], [1, 1]]))
        self.assertEqual(len(result), 2)


class TestSutherlandHodgmanClip(unittest.TestCase):

    def test_identical_squares(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        result = sutherland_hodgman_clip(sq, sq)
        self.assertGreater(len(result), 2)
        self.assertAlmostEqual(polygon_area(result), 1.0, places=5)

    def test_half_overlap(self):
        sq1 = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        sq2 = np.array([[1, 0], [3, 0], [3, 2], [1, 2]], dtype=float)
        result = sutherland_hodgman_clip(sq1, sq2)
        self.assertGreater(len(result), 2)
        self.assertAlmostEqual(polygon_area(result), 2.0, places=5)

    def test_no_overlap(self):
        sq1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        sq2 = np.array([[5, 5], [6, 5], [6, 6], [5, 6]], dtype=float)
        result = sutherland_hodgman_clip(sq1, sq2)
        self.assertEqual(len(result), 0)

    def test_contained(self):
        outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        inner = np.array([[2, 2], [4, 2], [4, 4], [2, 4]], dtype=float)
        result = sutherland_hodgman_clip(inner, outer)
        self.assertAlmostEqual(polygon_area(result), 4.0, places=5)

    def test_containing(self):
        outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        inner = np.array([[2, 2], [4, 2], [4, 4], [2, 4]], dtype=float)
        result = sutherland_hodgman_clip(outer, inner)
        self.assertAlmostEqual(polygon_area(result), 4.0, places=5)

    def test_corner_overlap(self):
        sq1 = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        sq2 = np.array([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=float)
        result = sutherland_hodgman_clip(sq1, sq2)
        self.assertAlmostEqual(polygon_area(result), 1.0, places=5)

    def test_triangle_square(self):
        tri = np.array([[0, 0], [4, 0], [2, 4]], dtype=float)
        sq = np.array([[1, 0], [3, 0], [3, 2], [1, 2]], dtype=float)
        result = sutherland_hodgman_clip(tri, sq)
        self.assertGreater(len(result), 2)
        area = polygon_area(result)
        self.assertGreater(area, 0)
        self.assertLessEqual(area, 4.0 + 1e-6)  # at most square area

    def test_cw_input_handled(self):
        """CW polygons should be normalized and produce correct results."""
        sq1_cw = np.array([[0, 0], [0, 2], [2, 2], [2, 0]], dtype=float)
        sq2_cw = np.array([[1, 1], [1, 3], [3, 3], [3, 1]], dtype=float)
        result = sutherland_hodgman_clip(sq1_cw, sq2_cw)
        self.assertAlmostEqual(polygon_area(result), 1.0, places=5)

    def test_empty_input(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        result = sutherland_hodgman_clip(np.array([]), sq)
        self.assertEqual(len(result), 0)
        result = sutherland_hodgman_clip(sq, np.array([]))
        self.assertEqual(len(result), 0)

    def test_thin_overlap(self):
        """Edge-to-edge touching should produce near-zero area."""
        sq1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        sq2 = np.array([[1, 0], [2, 0], [2, 1], [1, 1]], dtype=float)
        result = sutherland_hodgman_clip(sq1, sq2)
        if len(result) >= 3:
            self.assertAlmostEqual(polygon_area(result), 0.0, places=3)


class TestGetRotatedBoxCorners(unittest.TestCase):

    def test_axis_aligned(self):
        corners = get_rotated_box_corners(0, 0, 2, 4, 0)
        self.assertEqual(corners.shape, (4, 2))
        # Convention: w=2 (width, perpendicular), h=4 (length, along heading)
        # At angle=0: x spans ±length/2 = ±2, y spans ±width/2 = ±1
        self.assertAlmostEqual(corners[:, 0].min(), -2.0, places=10)
        self.assertAlmostEqual(corners[:, 0].max(), 2.0, places=10)
        self.assertAlmostEqual(corners[:, 1].min(), -1.0, places=10)
        self.assertAlmostEqual(corners[:, 1].max(), 1.0, places=10)

    def test_rotated_90(self):
        corners = get_rotated_box_corners(0, 0, 2, 4, math.pi / 2)
        # After 90° rotation: x spans ±width/2 = ±1, y spans ±length/2 = ±2
        self.assertAlmostEqual(corners[:, 0].min(), -1.0, places=5)
        self.assertAlmostEqual(corners[:, 0].max(), 1.0, places=5)
        self.assertAlmostEqual(corners[:, 1].min(), -2.0, places=5)
        self.assertAlmostEqual(corners[:, 1].max(), 2.0, places=5)

    def test_area_preserved(self):
        for angle in [0, 0.5, 1.0, 1.5, math.pi]:
            corners = get_rotated_box_corners(5, 10, 3, 7, angle)
            area = polygon_area(corners)
            self.assertAlmostEqual(area, 21.0, places=5,
                                   msg=f"Area not preserved at angle={angle}")


class TestRotatedBoxIoU(unittest.TestCase):

    def test_identical_boxes(self):
        box = (0, 0, 2, 4, 0)
        self.assertAlmostEqual(rotated_box_iou(box, box), 1.0, places=5)

    def test_no_overlap(self):
        box_a = (0, 0, 2, 2, 0)
        box_b = (10, 10, 2, 2, 0)
        self.assertAlmostEqual(rotated_box_iou(box_a, box_b), 0.0, places=5)

    def test_half_overlap_axis_aligned(self):
        box_a = (0, 0, 2, 2, 0)
        box_b = (1, 0, 2, 2, 0)
        iou = rotated_box_iou(box_a, box_b)
        # Overlap = 1x2 = 2, union = 4+4-2 = 6, IoU = 2/6 = 0.333
        self.assertAlmostEqual(iou, 1.0 / 3.0, places=3)

    def test_quarter_overlap(self):
        box_a = (0, 0, 2, 2, 0)
        box_b = (1, 1, 2, 2, 0)
        iou = rotated_box_iou(box_a, box_b)
        # Overlap = 1x1 = 1, union = 4+4-1 = 7, IoU = 1/7 = 0.143
        self.assertAlmostEqual(iou, 1.0 / 7.0, places=3)

    def test_contained_box(self):
        outer = (0, 0, 10, 10, 0)
        inner = (0, 0, 2, 2, 0)
        iou = rotated_box_iou(outer, inner)
        # Overlap = 4, union = 100+4-4 = 100, IoU = 4/100 = 0.04
        self.assertAlmostEqual(iou, 4.0 / 100.0, places=3)

    def test_rotated_identical(self):
        box_a = (5, 5, 3, 4, 0.5)
        self.assertAlmostEqual(rotated_box_iou(box_a, box_a), 1.0, places=4)

    def test_rotated_overlap(self):
        box_a = (0, 0, 4, 2, 0)
        box_b = (0, 0, 4, 2, math.pi / 4)  # 45 degree rotation
        iou = rotated_box_iou(box_a, box_b)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_zero_area_box(self):
        box_a = (0, 0, 0, 2, 0)
        box_b = (0, 0, 2, 2, 0)
        self.assertAlmostEqual(rotated_box_iou(box_a, box_b), 0.0)

    def test_symmetric(self):
        box_a = (1, 2, 3, 4, 0.3)
        box_b = (2, 3, 4, 5, 0.7)
        self.assertAlmostEqual(
            rotated_box_iou(box_a, box_b),
            rotated_box_iou(box_b, box_a),
            places=8
        )

    def test_many_angles(self):
        """IoU of identical box should be 1.0 regardless of rotation."""
        for angle in np.linspace(0, 2 * np.pi, 20):
            box = (10, 20, 5, 3, angle)
            iou = rotated_box_iou(box, box)
            self.assertAlmostEqual(iou, 1.0, places=4,
                                   msg=f"Self-IoU failed at angle={angle:.3f}")


class TestConsistencyWithOldImplementation(unittest.TestCase):
    """Compare new geometry module against the old utils.py implementations."""

    def setUp(self):
        # Import old implementations
        from utils import (
            polygon_area_signed as old_polygon_area_signed,
            polygon_area as old_polygon_area,
            sutherland_hodgman_clip as old_sutherland_hodgman_clip,
            rotated_box_iou as old_rotated_box_iou,
        )
        self.old_area_signed = old_polygon_area_signed
        self.old_area = old_polygon_area
        self.old_clip = old_sutherland_hodgman_clip
        self.old_iou = old_rotated_box_iou

    def test_area_consistency(self):
        rng = np.random.RandomState(42)
        for _ in range(50):
            n = rng.randint(3, 8)
            angles = np.sort(rng.uniform(0, 2 * np.pi, n))
            r = rng.uniform(1, 10)
            verts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
            new_area = polygon_area(verts)
            old_area = self.old_area(verts)
            self.assertAlmostEqual(new_area, old_area, places=8,
                                   msg=f"Area mismatch for {n}-gon")

    def test_iou_consistency(self):
        """New IoU should match old IoU for many random box pairs."""
        rng = np.random.RandomState(123)
        for _ in range(100):
            cx1, cy1 = rng.uniform(-10, 10, 2)
            w1, h1 = rng.uniform(0.5, 5, 2)
            a1 = rng.uniform(0, np.pi)
            cx2, cy2 = cx1 + rng.uniform(-3, 3), cy1 + rng.uniform(-3, 3)
            w2, h2 = rng.uniform(0.5, 5, 2)
            a2 = rng.uniform(0, np.pi)

            box_a = (cx1, cy1, w1, h1, a1)
            box_b = (cx2, cy2, w2, h2, a2)

            new_iou = rotated_box_iou(box_a, box_b)
            old_iou = self.old_iou(box_a, box_b)

            self.assertAlmostEqual(new_iou, old_iou, places=4,
                                   msg=f"IoU mismatch: new={new_iou:.6f} old={old_iou:.6f} "
                                       f"boxes={box_a} {box_b}")

    def test_clip_consistency(self):
        """Clipped polygon area should match between old and new."""
        rng = np.random.RandomState(77)
        for _ in range(50):
            # Generate two random convex polygons (rotated rectangles)
            for angle in [0, 0.3, 0.7, 1.2]:
                cx, cy = rng.uniform(-5, 5, 2)
                w, h = rng.uniform(1, 5, 2)
                corners1 = get_rotated_box_corners(cx, cy, w, h, angle)

                cx2, cy2 = cx + rng.uniform(-2, 2), cy + rng.uniform(-2, 2)
                w2, h2 = rng.uniform(1, 5, 2)
                a2 = angle + rng.uniform(-0.5, 0.5)
                corners2 = get_rotated_box_corners(cx2, cy2, w2, h2, a2)

                new_result = sutherland_hodgman_clip(corners1, corners2)
                old_result = self.old_clip(corners1, corners2)

                new_area = polygon_area(new_result) if len(new_result) >= 3 else 0.0
                old_area = self.old_area(old_result) if len(old_result) >= 3 else 0.0

                self.assertAlmostEqual(new_area, old_area, places=4,
                                       msg=f"Clip area mismatch: new={new_area:.6f} old={old_area:.6f}")


if __name__ == '__main__':
    unittest.main()
