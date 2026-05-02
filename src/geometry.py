"""
Fast vectorized polygon geometry operations for IoU computation.

Replaces per-vertex Python loops with numpy array operations.
Used by metrics (AMOTA, HOTA) and sensor fusion matching.
"""

import numpy as np
from typing import Optional


def polygon_area_signed(vertices: np.ndarray) -> float:
    """Signed area of polygon via shoelace formula (positive = CCW)."""
    if len(vertices) < 3:
        return 0.0
    v = np.asarray(vertices, dtype=np.float64)
    x, y = v[:, 0], v[:, 1]
    # Avoid np.roll — use direct indexing for shifted arrays
    x_next = np.empty_like(x)
    x_next[:-1] = x[1:]
    x_next[-1] = x[0]
    y_next = np.empty_like(y)
    y_next[:-1] = y[1:]
    y_next[-1] = y[0]
    return 0.5 * float(np.dot(x, y_next) - np.dot(y, x_next))


def polygon_area(vertices: np.ndarray) -> float:
    """Absolute area of polygon."""
    return abs(polygon_area_signed(vertices))


def ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    """Ensure polygon vertices are in counter-clockwise order."""
    v = np.asarray(vertices, dtype=np.float64)
    if len(v) < 3:
        return v
    if polygon_area_signed(v) < 0:
        return v[::-1].copy()
    return v


def _clip_edge(polygon: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> np.ndarray:
    """Clip polygon against a single edge using vectorized Sutherland-Hodgman step.

    Args:
        polygon: (N, 2) vertices
        edge_start, edge_end: (2,) edge endpoints

    Returns:
        Clipped polygon vertices (M, 2)
    """
    n = len(polygon)
    if n == 0:
        return polygon

    # Vectorized inside test for all vertices:
    ex, ey = edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]
    cross = ex * (polygon[:, 1] - edge_start[1]) - ey * (polygon[:, 0] - edge_start[0])
    inside = cross >= -1e-10  # (N,) bool

    # Previous vertex for each vertex (avoid np.roll)
    prev_polygon = np.empty_like(polygon)
    prev_polygon[0] = polygon[-1]
    prev_polygon[1:] = polygon[:-1]
    prev_cross = np.empty_like(cross)
    prev_cross[0] = cross[-1]
    prev_cross[1:] = cross[:-1]
    prev_inside = prev_cross >= -1e-10

    # Classify each vertex pair
    # Case 1: both inside -> keep current
    # Case 2: prev outside, current inside -> add intersection + current
    # Case 3: prev inside, current outside -> add intersection
    # Case 4: both outside -> skip

    enters = inside & ~prev_inside   # crossing in
    exits = ~inside & prev_inside     # crossing out

    # Compute intersections for all edges that cross
    needs_intersect = enters | exits
    if not np.any(needs_intersect):
        if np.all(inside):
            return polygon.copy()
        else:
            return np.empty((0, 2), dtype=np.float64)

    # Line intersection: prev->current vs edge_start->edge_end
    # Using parametric form: P + t*(C-P) intersects E0 + s*(E1-E0)
    d_poly = polygon - prev_polygon  # direction vectors for each polygon edge
    d_edge = edge_end - edge_start   # direction for clip edge

    # denom = d_poly x d_edge (2D cross product)
    denom = d_poly[:, 0] * d_edge[1] - d_poly[:, 1] * d_edge[0]

    # Avoid division by zero
    safe_denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)

    # t = (edge_start - prev) x d_edge / denom
    diff = edge_start - prev_polygon
    t = (diff[:, 0] * d_edge[1] - diff[:, 1] * d_edge[0]) / safe_denom
    t = np.clip(t, 0, 1)

    # Intersection points
    intersections = prev_polygon + t[:, np.newaxis] * d_poly  # (N, 2)

    # Build output: for each vertex, emit 0, 1, or 2 points
    # Pre-allocate max possible size (2*N)
    out = np.empty((2 * n, 2), dtype=np.float64)
    count = 0

    for i in range(n):
        if enters[i]:
            out[count] = intersections[i]
            count += 1
            out[count] = polygon[i]
            count += 1
        elif inside[i] and prev_inside[i]:
            out[count] = polygon[i]
            count += 1
        elif exits[i]:
            out[count] = intersections[i]
            count += 1

    return out[:count].copy()


def sutherland_hodgman_clip(subject_polygon: np.ndarray, clip_polygon: np.ndarray) -> np.ndarray:
    """Clip subject polygon against clip polygon using Sutherland-Hodgman algorithm.

    Both polygons are normalized to CCW order. Uses vectorized per-edge clipping.

    Args:
        subject_polygon: (N, 2) polygon to be clipped
        clip_polygon: (M, 2) clipping polygon

    Returns:
        (K, 2) clipped polygon vertices, or empty array if no intersection
    """
    subject = ensure_ccw(np.asarray(subject_polygon, dtype=np.float64))
    clip = ensure_ccw(np.asarray(clip_polygon, dtype=np.float64))

    if len(subject) < 3 or len(clip) < 3:
        return np.empty((0, 2), dtype=np.float64)

    output = subject
    m = len(clip)
    for i in range(m):
        if len(output) < 3:
            return np.empty((0, 2), dtype=np.float64)
        output = _clip_edge(output, clip[i], clip[(i + 1) % m])

    if len(output) < 3:
        return np.empty((0, 2), dtype=np.float64)
    return output


def get_rotated_box_corners(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    """Get corners of a rotated bounding box as (4, 2) array.

    Convention: w=width (perpendicular to heading), h=length (along heading).
    At angle=0, length (h) is along x-axis, width (w) is along y-axis.
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    hw, hh = w / 2.0, h / 2.0
    # Corners relative to center: x=length, y=width (swapped)
    corners_rel = np.array([
        [-hh, -hw],
        [hh, -hw],
        [hh, hw],
        [-hh, hw],
    ])
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return corners_rel @ rot.T + np.array([cx, cy])


def rotated_box_iou(box_a, box_b, convex_hull: bool = False) -> float:
    """IoU between two rotated bounding boxes (cx, cy, w, h, angle).

    Args:
        box_a, box_b: tuple/array of (cx, cy, w, h, angle)
        convex_hull: if True, compute intersection area via scipy.spatial.ConvexHull
            instead of the shoelace formula.  Matches DMSTrack's evaluator exactly
            at the cost of ~3× slower execution.

    Returns:
        IoU value between 0 and 1
    """
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    if area_a <= 0 or area_b <= 0:
        return 0.0

    corners_a = get_rotated_box_corners(box_a[0], box_a[1], box_a[2], box_a[3], box_a[4])
    corners_b = get_rotated_box_corners(box_b[0], box_b[1], box_b[2], box_b[3], box_b[4])

    intersection = sutherland_hodgman_clip(corners_a, corners_b)
    if len(intersection) < 3:
        return 0.0

    if convex_hull:
        from scipy.spatial import ConvexHull
        try:
            intersection_area = ConvexHull(intersection).volume  # .volume = area in 2D
        except Exception:
            intersection_area = polygon_area(intersection)
    else:
        intersection_area = polygon_area(intersection)

    union_area = area_a + area_b - intersection_area
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def iou_3d_box(box_a, box_b) -> float:
    """3D IoU between two boxes: BEV rotated-rectangle overlap × vertical interval overlap.

    Args:
        box_a, box_b: (cx, cy, w, l, yaw, z_center, height)
            cx, cy    — BEV centre (same ground-plane axes as rotated_box_iou)
            w, l      — box width and length in the BEV plane
            yaw       — rotation about the vertical axis (radians)
            z_center  — centre of the box along the vertical axis
            height    — full box height

    Returns:
        3D IoU in [0, 1].
    """
    cx_a, cy_a, w_a, l_a, yaw_a, z_a, h_a = box_a
    cx_b, cy_b, w_b, l_b, yaw_b, z_b, h_b = box_b

    bev_area_a = w_a * l_a
    bev_area_b = w_b * l_b
    if bev_area_a <= 0 or bev_area_b <= 0 or h_a <= 0 or h_b <= 0:
        return 0.0

    # BEV footprint intersection
    corners_a = get_rotated_box_corners(cx_a, cy_a, w_a, l_a, yaw_a)
    corners_b = get_rotated_box_corners(cx_b, cy_b, w_b, l_b, yaw_b)
    inter_poly = sutherland_hodgman_clip(corners_a, corners_b)
    if len(inter_poly) < 3:
        return 0.0
    bev_inter = polygon_area(inter_poly)
    if bev_inter <= 0:
        return 0.0

    # Vertical interval intersection.
    # z_center is the camera-down Y coordinate of the box bottom; interval = [z_center−h, z_center].
    # In camera-down convention: smaller value = higher up, so:
    #   top of box (higher in world) = z_center − h  (more negative)
    #   bottom of box (ground)       = z_center       (less negative / more positive)
    # Overlap = min(bottom_a, bottom_b) − max(top_a, top_b)
    z_inter = max(0.0, min(z_a, z_b) - max(z_a - h_a, z_b - h_b))
    if z_inter <= 0:
        return 0.0

    inter_3d = bev_inter * z_inter
    union_3d = bev_area_a * h_a + bev_area_b * h_b - inter_3d
    if union_3d <= 0:
        return 0.0

    return inter_3d / union_3d
